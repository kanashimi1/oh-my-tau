//! Headless e2e harness for the TUI controller — the `WaitSettle` /
//! `WaitRender` barrier pair from the prime-agent headless plans
//! (`.ref` `agents_view_wait_render_barrier.rs`), ported onto the
//! [`omp_tui::controller::run_headless`] seam.
//!
//! The harness owns the three sides the running loop talks to, each shared
//! with the loop through an `Arc` (the loop itself lives in a spawned task on
//! the test's current-thread runtime and keeps the `Shell` to itself):
//!
//! - [`KeyFeed`]: the crossterm input stream. Synthetic events are queued by
//!   the test; the stream hands out one per controller iteration and returns
//!   `Poll::Pending` once empty, so a drained loop parks in its `select!`
//!   instead of spinning (a `None`-terminated stream would busy-loop).
//! - [`RenderLog`]: every ratatui draw over the wrapped `TestBackend`
//!   appends the full frame buffer. Tests read it two ways: *N consecutive
//!   identical buffers* (the render barrier, `TuiHarness::wait_rendered`)
//!   and a content-quiet window (the settle budget,
//!   `TuiHarness::wait_settled`).
//! - [`TestPort`]: a recording [`RuntimePort`] double — no network, no LLM.
//!   Loop dispatches are synchronous, so an empty recording means the agent
//!   side is idle.
//!
//! Idle detection is logical, not wall-clock: the queue is drained, every
//! sent runtime frame was consumed (each send carries a paint watermark)
//! and no draw changed the buffer for [`QUIET_WINDOW`] (longer than the
//! controller's 33 ms coalesce tick, so a pending message-update pump
//! always lands inside the window). The only bounded animator at startup is
//! the clean-theme welcome shimmer (`SHIMMER_MS` = 1.5 s), well inside the
//! 5 s settle budget.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use futures::Stream;
use omp_tui::action::QueueKind;
use omp_tui::controller::run_headless;
use omp_tui::render::{RawOut, Shell};
use omp_tui::RuntimePort;
use omp_tui::{RunConfig, SubmitBehavior};
use parking_lot::Mutex;
use ratatui::backend::{Backend, ClearType, TestBackend, WindowSize};
use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::{Position, Size};
use ratatui::{Terminal, TerminalOptions, Viewport};
use serde_json::Value;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;

/// Rows of the live region (chat viewport + editor + status line). Mirrors
/// the controller's private `LIVE_HEIGHT`.
const LIVE_ROWS: u16 = 12;

/// Total budget for one wait (settle or barrier). Everything the startup
/// path animates (welcome shimmer 1.5 s) fits comfortably inside.
const WAIT_BUDGET: Duration = Duration::from_secs(5);

/// Content-quiet window a settled frame must hold. Longer than the
/// controller's 33 ms pump tick, so a trailing coalesced repaint always
/// lands inside the window and restarts it.
const QUIET_WINDOW: Duration = Duration::from_millis(60);

/// Poll cadence while a wait is outstanding.
const POLL_STEP: Duration = Duration::from_millis(5);

/// Bound on how long [`TuiHarness::quit`] waits for the loop to actually
/// exit after the `app.exit` chord (CI machines can be slow; a lost exit is
/// a harness bug and must fail loudly, not hang the job).
const JOIN_BUDGET: Duration = Duration::from_secs(10);

/// Render log shared between the harness backend (producer) and the waits
/// (consumers).
type RenderLogHandle = Arc<Mutex<RenderLog>>;

/// Every full-frame buffer the backend drew, newest last, plus the wall
/// clock of the last *content* change (identical watchdog repaints do not
/// count as change).
#[derive(Debug)]
struct RenderLog {
    frames: VecDeque<Buffer>,
    last_change: Instant,
    /// Total draws recorded — the watermark source for "the loop consumed
    /// everything sent so far": every consumed frame ends a controller
    /// iteration, and every iteration paints. Monotonic, unlike `frames`
    /// (which the ring cap can shrink).
    paints: u64,
}

impl RenderLog {
    /// Ring bound. Identical repaints append forever (the controller's 1 s
    /// raw-mode watchdog repaints each second); e2e traffic stays orders of
    /// magnitude below this, and barrier arm indices are only used within
    /// one wait.
    const CAP: usize = 512;

    fn new() -> Self {
        Self {
            frames: VecDeque::new(),
            last_change: Instant::now(),
            paints: 0,
        }
    }

    fn record(&mut self, frame: Buffer) {
        let changed = self.frames.back().is_none_or(|prev| prev != &frame);
        self.frames.push_back(frame);
        self.paints = self.paints.saturating_add(1);
        if changed {
            self.last_change = Instant::now();
        }
        while self.frames.len() > Self::CAP {
            self.frames.pop_front();
        }
    }

    /// Total draws so far (monotonic).
    fn paints(&self) -> u64 {
        self.paints
    }

    fn len(&self) -> usize {
        self.frames.len()
    }

    /// `true` when no draw changed the frame content for `window`.
    fn is_quiet(&self, window: Duration) -> bool {
        self.last_change.elapsed() >= window
    }

    /// Newest snapshot, if anything rendered yet.
    fn snapshot(&self) -> Option<Buffer> {
        self.frames.back().cloned()
    }

    /// The render barrier: the last run of `n` consecutive identical frames
    /// whose start sits at or after `start - 1` (the settled pre-arm tail
    /// counts — the nudges queued by the caller make it identical to the
    /// post-arm repaints). Returns the run's final buffer.
    fn consecutive_identical_from(&self, start: usize, n: usize) -> Option<Buffer> {
        if n == 0 || self.frames.len() < n {
            return None;
        }
        let last_start = self.frames.len() - n;
        let first = start.saturating_sub(1).min(last_start);
        for i in (first..=last_start).rev() {
            let head = &self.frames[i];
            if (1..n).all(|k| &self.frames[i + k] == head) {
                return Some(self.frames[i + n - 1].clone());
            }
        }
        None
    }
}

/// `RawOut` sink that drops the escape sequences written outside ratatui's
/// diff engine (paint brackets, OSC marks). A `TestBackend` has no tty to
/// style.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullRaw;

impl RawOut for NullRaw {
    fn raw(&mut self, _s: &str) -> io::Result<()> {
        Ok(())
    }
}

/// `TestBackend` wrapper that appends every drawn frame to the shared
/// [`RenderLog`]. The snapshot is taken after the update reached the inner
/// buffer, so `TuiHarness::frame_text` always sees post-draw state,
/// committed transcript rows included.
#[derive(Debug)]
pub struct HarnessBackend {
    inner: TestBackend,
    log: RenderLogHandle,
}

impl HarnessBackend {
    fn new(width: u16, height: u16, log: RenderLogHandle) -> Self {
        Self {
            inner: TestBackend::new(width, height),
            log,
        }
    }
}

impl Backend for HarnessBackend {
    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        self.inner.draw(content)?;
        self.log.lock().record(self.inner.buffer().clone());
        Ok(())
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        self.inner.hide_cursor()
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        self.inner.show_cursor()
    }

    fn get_cursor_position(&mut self) -> io::Result<Position> {
        self.inner.get_cursor_position()
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        self.inner.set_cursor_position(position)
    }

    fn clear(&mut self) -> io::Result<()> {
        self.inner.clear()
    }

    fn clear_region(&mut self, clear_type: ClearType) -> io::Result<()> {
        self.inner.clear_region(clear_type)
    }

    fn append_lines(&mut self, n: u16) -> io::Result<()> {
        self.inner.append_lines(n)
    }

    fn size(&self) -> io::Result<Size> {
        self.inner.size()
    }

    fn window_size(&mut self) -> io::Result<WindowSize> {
        self.inner.window_size()
    }

    fn flush(&mut self) -> io::Result<()> {
        Backend::flush(&mut self.inner)
    }
}

/// Queue + wake state behind the [`KeyFeed`] input stream.
#[derive(Debug, Default)]
struct FeedState {
    queue: Mutex<VecDeque<Event>>,
    /// Waker of the suspended `select!`, registered on the drained poll;
    /// a push wakes the loop immediately.
    waker: Mutex<Option<Waker>>,
    pushed: AtomicUsize,
    delivered: AtomicUsize,
}

impl FeedState {
    fn push(&self, event: Event) {
        self.queue.lock().push_back(event);
        self.pushed.fetch_add(1, Ordering::SeqCst);
        if let Some(waker) = self.waker.lock().take() {
            waker.wake();
        }
    }

    fn pushed_count(&self) -> usize {
        self.pushed.load(Ordering::SeqCst)
    }

    fn delivered_count(&self) -> usize {
        self.delivered.load(Ordering::SeqCst)
    }
}

/// The controller's crossterm input stream: queued events, then `Pending`
/// forever (never `None` — an exhausted stream would make the biased
/// `select!` hot-spin).
#[derive(Debug)]
pub struct KeyFeed {
    state: Arc<FeedState>,
}

impl KeyFeed {
    fn new_with_state(state: Arc<FeedState>) -> Self {
        Self { state }
    }
}

impl Stream for KeyFeed {
    type Item = io::Result<Event>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let state = &self.state;
        let mut next = state.queue.lock().pop_front();
        if next.is_none() {
            *state.waker.lock() = Some(cx.waker().clone());
            // A push may have landed between the queue check and the waker
            // store; re-check so an event is never left behind a registered
            // but never-woken waker.
            next = state.queue.lock().pop_front();
        }
        match next {
            Some(event) => {
                state.delivered.fetch_add(1, Ordering::SeqCst);
                Poll::Ready(Some(Ok(event)))
            }
            None => Poll::Pending,
        }
    }
}

/// Recording [`RuntimePort`] double: the "runtime" without network or LLM.
/// Every dispatch is appended to a shared log the test can assert on; calls
/// are synchronous, so when the loop is idle the port is idle.
#[derive(Debug, Default, Clone)]
pub struct TestPort {
    calls: Arc<Mutex<Vec<String>>>,
}

impl TestPort {
    fn record(&self, call: String) {
        self.calls.lock().push(call);
    }

    /// Recorded dispatches, oldest first.
    #[must_use]
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().clone()
    }
}

impl RuntimePort for TestPort {
    fn submit(&self, id: Option<String>, text: String, behavior: Option<SubmitBehavior>) {
        self.record(format!("submit({id:?}, {text:?}, {behavior:?})"));
    }

    fn steer(&self, text: String) {
        self.record(format!("steer({text:?})"));
    }

    fn follow_up(&self, text: String) {
        self.record(format!("follow_up({text:?})"));
    }

    fn remove_queued(&self, text: String, queue: QueueKind) {
        self.record(format!("remove_queued({text:?}, {queue:?})"));
    }

    fn promote_queued(&self, text: String) {
        self.record(format!("promote_queued({text:?})"));
    }

    fn abort(&self) {
        self.record("abort()".into());
    }

    fn set_model(&self, provider: String, id: String) {
        self.record(format!("set_model({provider}, {id})"));
    }

    fn set_thinking_level(&self, level: String) {
        self.record(format!("set_thinking_level({level})"));
    }

    fn run_slash(&self, name: String, args: String) {
        self.record(format!("run_slash({name}, {args:?})"));
    }

    fn extension_ui_response(&self, id: String, value: Value) {
        self.record(format!("extension_ui_response({id}, {value})"));
    }

    fn resume_session(&self, path: PathBuf, fork: bool) {
        // `Path`'s `Debug` (quoted) is the established message form.
        #[allow(clippy::unnecessary_debug_formatting)]
        self.record(format!("resume_session({path:?}, fork={fork})"));
    }

    fn continue_run(&self) {
        self.record("continue_run()".into());
    }

    fn abort_retry(&self) {
        self.record("abort_retry()".into());
    }

    fn rewind_entry(&self, entry_id: String) {
        self.record(format!("rewind_entry({entry_id})"));
    }

    fn abort_compaction(&self) {
        self.record("abort_compaction()".into());
    }

    fn refresh_models(&self) {
        self.record("refresh_models()".into());
    }

    fn settings_get(&self, ids: &[String]) -> Vec<(String, Value)> {
        self.record(format!("settings_get({ids:?})"));
        // Defaults only: never read the process-global HOME (unlike the
        // real port).
        ids.iter().map(|id| (id.clone(), Value::Null)).collect()
    }

    fn settings_set(&self, id: String, value: Value) -> Result<(), String> {
        self.record(format!("settings_set({id}, {value})"));
        // Recorded, not persisted: writes must not reach the real config.yml.
        Ok(())
    }

    fn request_session_info(&self) {
        self.record("request_session_info()".into());
    }

    fn request_usage(&self) {
        self.record("request_usage()".into());
    }
}

/// Handle over one headless [`omp_tui::controller::run_headless`] session.
///
/// Text rendering of the current frame is grid-faithful (each cell's symbol,
/// trailing blanks trimmed per row); assertions should keep to `contains`
/// on that text, not on exact rows — the status line's exact composition is
/// preset-dependent.
#[derive(Debug)]
pub struct TuiHarness {
    model: String,
    model_name: Option<String>,
    version: String,
    cwd: PathBuf,
    width: u16,
    height: u16,
    feed: Arc<FeedState>,
    log: RenderLogHandle,
    /// Paint watermark per `send_frame`: a frame counts as consumed once a
    /// later draw landed (`RenderLog::paints` moved past the mark), so
    /// `wait_settled` never returns with runtime work still queued.
    pending_frames: Mutex<Vec<u64>>,
    frames_tx: UnboundedSender<Value>,
    frames_rx: Option<UnboundedReceiver<Value>>,
    port: TestPort,
    run: Option<JoinHandle<anyhow::Result<()>>>,
}

impl TuiHarness {
    /// A harness with an empty session: no models, no session sources, no
    /// provider — the minimal [`RunConfig`] that raises the UI.
    #[must_use]
    pub fn new(model: &str, width: u16, height: u16) -> Self {
        let (frames_tx, frames_rx) = tokio::sync::mpsc::unbounded_channel();
        Self {
            model: model.to_owned(),
            model_name: None,
            version: "0.0.0-test".to_owned(),
            cwd: PathBuf::from("/repo"),
            width,
            height,
            feed: Arc::default(),
            log: Arc::new(Mutex::new(RenderLog::new())),
            pending_frames: Mutex::new(Vec::new()),
            // The sender stays alive for the whole session: a closed frame
            // channel would make the controller's `frames.recv()` arm of the
            // biased `select!` return `None` every iteration and hot-spin.
            frames_tx,
            frames_rx: Some(frames_rx),
            port: TestPort::default(),
            run: None,
        }
    }

    /// Spawn the controller loop over this harness's `TestBackend` shell.
    ///
    /// The inline viewport is anchored at the screen bottom (cursor parked
    /// on the last row), exactly like the production CPR fallback in
    /// `ShellBackend`, so committed transcript rows land *above* the live
    /// region instead of being overdrawn by it.
    ///
    /// # Panics
    ///
    /// Panics if called twice, or if the headless terminal cannot be created
    /// (never on a `TestBackend`).
    pub fn start(&mut self) {
        assert!(
            self.run.is_none(),
            "start: the controller loop is already running"
        );
        let mut backend = HarnessBackend::new(self.width, self.height, Arc::clone(&self.log));
        Backend::set_cursor_position(
            &mut backend,
            Position {
                x: 0,
                y: self.height.saturating_sub(1),
            },
        )
        .expect("park the TestBackend cursor on the last row");
        let term = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(LIVE_ROWS),
            },
        )
        .expect("create the headless terminal");
        let mut shell = Shell::new_for_backend(term, NullRaw, LIVE_ROWS, false);
        let cfg = RunConfig {
            frames: self.frames_rx.take().expect("frames receiver"),
            model: self.model.clone(),
            model_name: self.model_name.clone(),
            version: self.version.clone(),
            cwd: self.cwd.clone(),
            models: Vec::new(),
            session_sources: Vec::new(),
            osc11_bg: None,
        };
        let keys = KeyFeed::new_with_state(Arc::clone(&self.feed));
        let port = self.port.clone();
        self.run = Some(tokio::spawn(async move {
            run_headless(cfg, port, &mut shell, keys).await
        }));
    }

    /// Queue one synthetic terminal event.
    fn push_event(&self, event: Event) {
        self.feed.push(event);
    }

    /// Queue one key event.
    pub fn push_key(&self, code: KeyCode, modifiers: KeyModifiers) {
        self.push_event(Event::Key(KeyEvent::new(code, modifiers)));
    }

    /// Queue `text` as plain (unmodified) character keystrokes.
    pub fn type_text(&self, text: &str) {
        for ch in text.chars() {
            self.push_key(KeyCode::Char(ch), KeyModifiers::NONE);
        }
    }

    /// Send one runtime frame (the `Runtime::start` channel feeding
    /// [`RunConfig::frames`]).
    pub fn send_frame(&self, frame: Value) {
        self.pending_frames.lock().push(self.log.lock().paints());
        self.frames_tx
            .send(frame)
            .expect("frame channel closed: the loop task is gone");
    }

    /// `true` when every frame sent so far was consumed by the loop. The
    /// watermark reads "a draw landed after the send": the loop never parks
    /// with a queued frame (the channel wakes it), so by the time the quiet
    /// window holds — the loop suspended, nothing left to paint — every
    /// marked frame has been through an iteration.
    fn frames_settled(&self) -> bool {
        let paints = self.log.lock().paints();
        let mut pending = self.pending_frames.lock();
        pending.retain(|mark| paints <= *mark);
        pending.is_empty()
    }

    /// The settle barrier (`WaitSettle`): returns once every queued event
    /// was delivered, every sent runtime frame was consumed by the loop and
    /// no draw changed the frame for [`QUIET_WINDOW`]. The port needs no
    /// check of its own — its dispatches are synchronous.
    ///
    /// # Panics
    ///
    /// Panics if the loop task exited before settling (startup failure) or
    /// the budget ran out.
    pub async fn wait_settled(&self) {
        self.wait_settled_for(WAIT_BUDGET).await;
    }

    async fn wait_settled_for(&self, budget: Duration) {
        assert!(
            self.run.is_some(),
            "wait_settled: the controller loop was never started"
        );
        let deadline = Instant::now() + budget;
        loop {
            assert!(
                !self.aborted(),
                "wait_settled: the controller loop exited before settling"
            );
            let drained = self.feed.pushed_count() == self.feed.delivered_count();
            if drained && self.frames_settled() && self.log.lock().is_quiet(QUIET_WINDOW) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "wait_settled: the loop never went idle within {budget:?}"
            );
            tokio::time::sleep(POLL_STEP).await;
        }
    }

    /// The render barrier (`WaitRender`): settle, then require `n`
    /// consecutive identical frame buffers (nudged with same-size resizes —
    /// repaint without state change — so the identical run is guaranteed to
    /// exist) and return the run's final snapshot.
    ///
    /// # Panics
    ///
    /// Same as [`Self::wait_settled`], plus when the barrier never
    /// satisfies within the budget.
    pub async fn wait_rendered(&self, n_consecutive: usize) -> Buffer {
        assert!(
            n_consecutive >= 1,
            "wait_rendered: n_consecutive must be at least 1"
        );
        self.wait_settled().await;
        let arm = self.log.lock().len();
        for _ in 1..n_consecutive {
            self.push_event(Event::Resize(self.width, self.height));
        }
        let deadline = Instant::now() + WAIT_BUDGET;
        loop {
            assert!(
                !self.aborted(),
                "wait_rendered: the controller loop exited before rendering"
            );
            if let Some(frame) = self
                .log
                .lock()
                .consecutive_identical_from(arm, n_consecutive)
            {
                return frame;
            }
            assert!(
                Instant::now() < deadline,
                "wait_rendered: no run of {n_consecutive} identical frames within the budget"
            );
            tokio::time::sleep(POLL_STEP).await;
        }
    }

    /// Whether the spawned loop task has ended (normally or by panic) while
    /// a wait is still outstanding.
    fn aborted(&self) -> bool {
        self.run
            .as_ref()
            .is_some_and(tokio::task::JoinHandle::is_finished)
    }

    /// Text of the newest frame: one line per terminal row, trailing blanks
    /// trimmed.
    ///
    /// # Panics
    ///
    /// Panics if nothing rendered yet (before [`Self::start`]).
    #[must_use]
    pub fn frame_text(&self) -> String {
        let frame = self
            .log
            .lock()
            .snapshot()
            .expect("frame_text: nothing rendered yet");
        buffer_text(&frame)
    }

    /// Dispatches the loop made to the runtime so far.
    #[must_use]
    pub fn port_calls(&self) -> Vec<String> {
        self.port.calls()
    }

    /// End the session and join the loop. The exit chord is ctrl+d
    /// (`app.exit`): Esc is `app.interrupt`, which while idle is a ladder
    /// no-op and can never end `run()`.
    ///
    /// # Errors
    ///
    /// Forwards the controller's own result (startup/draw failures).
    ///
    /// # Panics
    ///
    /// Panics if `start` was never called, the loop fails to exit within
    /// [`JOIN_BUDGET`], or the loop task panicked.
    pub async fn quit(mut self) -> anyhow::Result<()> {
        self.push_key(KeyCode::Char('d'), KeyModifiers::CONTROL);
        let run = self.run.take().expect("quit: run() was never started");
        let joined = tokio::time::timeout(JOIN_BUDGET, run)
            .await
            .expect("quit: the controller loop did not exit after the app.exit chord");
        joined.expect("quit: the controller task panicked")
    }
}

/// Grid-faithful text of `frame`: each cell's symbol, one row per line,
/// trailing blanks trimmed.
#[must_use]
pub fn buffer_text(frame: &Buffer) -> String {
    let area = frame.area;
    (0..area.height)
        .map(|y| {
            (0..area.width)
                .map(|x| frame[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}
