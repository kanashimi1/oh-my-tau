//! Global push-to-talk capture, live transcription and the no-focus overlay.

use crate::dictation::{stable_preview, Action, ButtonState, Event, LinearResampler};
use crate::settings::DictationSettings;
use crate::{log_line, SERVICE_URL};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use serde_json::json;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindow, WebviewWindowBuilder};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_KEYUP,
    KEYEVENTF_UNICODE, MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, MOUSEINPUT, VK_ESCAPE,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, GetMessageW, SetWindowsHookExW, ShowWindow, LLMHF_INJECTED, MSG,
    MSLLHOOKSTRUCT, SW_HIDE, SW_SHOWNOACTIVATE, WH_MOUSE_LL, WM_XBUTTONDOWN, WM_XBUTTONUP,
};

const SAMPLE_RATE: u32 = 16_000;
const XBUTTON1_DATA: u16 = 1;
const PARTIAL_EVERY: Duration = Duration::from_millis(550);
const MIN_AUDIO_SAMPLES: usize = SAMPLE_RATE as usize / 5;
const PARTIAL_WINDOW_SAMPLES: usize = SAMPLE_RATE as usize * 12;
const ERROR_VISIBLE: Duration = Duration::from_millis(2600);
static ENABLED: AtomicBool = AtomicBool::new(true);
static HOOK_TX: OnceLock<Sender<Message>> = OnceLock::new();
static SETTINGS: OnceLock<Mutex<DictationSettings>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestKind {
    Warmup,
    Partial,
    Final,
}

struct InferenceResult {
    generation: u64,
    kind: RequestKind,
    result: Result<String, String>,
}

enum Message {
    Button(Event, u64),
    Cancel,
    Samples(Vec<f32>, f32),
    Inference(InferenceResult),
}

struct Job {
    generation: u64,
    kind: RequestKind,
    audio: Vec<u8>,
    language: String,
}

pub fn update(settings: &DictationSettings) {
    ENABLED.store(settings.enabled, Ordering::Release);
    if let Some(current) = SETTINGS.get() {
        if let Ok(mut current) = current.lock() {
            *current = settings.clone();
        }
    }
}

fn settings() -> DictationSettings {
    SETTINGS
        .get()
        .and_then(|s| s.lock().ok().map(|s| s.clone()))
        .unwrap_or_default()
}

pub fn register(app: &AppHandle) -> tauri::Result<()> {
    let initial = crate::settings::load().dictation;
    let _ = SETTINGS.set(Mutex::new(initial.clone()));
    update(&initial);
    let (tx, rx) = mpsc::channel();
    let _ = HOOK_TX.set(tx);
    std::thread::spawn(mouse_hook_loop);
    std::thread::spawn(keyboard_escape_loop);
    let worker_app = app.clone();
    std::thread::spawn(move || coordinate(worker_app, rx));
    let initial_app = app.clone();
    app.run_on_main_thread(move || {
        if let Err(error) = create_overlay(&initial_app) {
            log_line(&format!("dictation overlay: {error}"));
        }
    })?;
    Ok(())
}

fn create_overlay(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    let monitor = app
        .primary_monitor()?
        .or_else(|| app.available_monitors().ok()?.into_iter().next());
    let (x, y) = monitor.map_or((100, 100), |m| {
        let width = 760_i32;
        let height = 140_i32;
        (
            m.position().x + (m.size().width as i32 - width).max(0) / 2,
            m.position().y + (m.size().height as i32 - height - 72).max(0),
        )
    });
    let window =
        WebviewWindowBuilder::new(app, "dictation", WebviewUrl::App("dictation.html".into()))
            .title("Tau dictation")
            .inner_size(760.0, 140.0)
            .additional_browser_args(&crate::settings::browser_args(
                crate::settings::devtools_port(&crate::settings::load()),
            ))
            .position(x as f64, y as f64)
            .decorations(false)
            .transparent(true)
            .always_on_top(true)
            .skip_taskbar(true)
            .resizable(false)
            .focused(false)
            .visible(false)
            .build()?;
    Ok(window)
}

fn overlay(app: &AppHandle, stable: &str, draft: &str, status: &str, mode: &str) {
    let Some(window) = app.get_webview_window("dictation") else {
        return;
    };
    let payload = json!({
        "stable": stable,
        "draft": draft,
        "status": status,
        "mode": mode,
    });
    let _ = window.eval(&format!("window.updateDictation({payload})"));
    if mode != "hidden" {
        if let Ok(hwnd) = window.hwnd() {
            unsafe { ShowWindow(hwnd.0 as HWND, SW_SHOWNOACTIVATE) };
        }
    }
}

fn overlay_level(app: &AppHandle, level: f32) {
    if let Some(window) = app.get_webview_window("dictation") {
        let _ = window.eval(&format!("window.updateDictationLevel({level})"));
    }
}

fn hide_overlay(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("dictation") {
        if let Ok(hwnd) = window.hwnd() {
            unsafe { ShowWindow(hwnd.0 as HWND, SW_HIDE) };
        }
    }
}

fn coordinate(app: AppHandle, rx: Receiver<Message>) {
    let mut button = ButtonState::default();
    let mut generation = 0_u64;
    let mut _stream = None;
    let mut resampler = None;
    let mut samples = Vec::new();
    let mut previous = String::new();
    let mut recording = false;
    let mut finalizing = false;
    let mut in_flight: Option<(u64, RequestKind)> = None;
    let mut pending_final: Option<Job> = None;
    let tx = HOOK_TX.get().expect("dictation channel").clone();
    let mut last_partial = Instant::now() - PARTIAL_EVERY;
    let mut hide_at: Option<Instant> = None;

    loop {
        let timeout = if recording && !finalizing {
            PARTIAL_EVERY.saturating_sub(last_partial.elapsed())
        } else {
            Duration::from_millis(100)
        };
        match rx.recv_timeout(timeout) {
            Ok(Message::Button(event, now_ms)) => {
                match button.event(event, now_ms, false, settings().hold_ms) {
                    None => {}
                    Some(Action::Start) => {
                        hide_at = None;
                        if app.get_webview_window("dictation").is_none() {
                            if let Err(error) = create_overlay(&app) {
                                log_line(&format!("dictation overlay: {error}"));
                                overlay(&app, "", "", "Ошибка окна диктовки", "error");
                                continue;
                            }
                        }
                        generation = generation.wrapping_add(1);
                        pending_final = None;
                        finalizing = false;
                        recording = true;
                        previous.clear();
                        samples.clear();
                        let (new_stream, new_resampler) = match start_capture(tx.clone()) {
                            Ok(capture) => (Some(capture.0), Some(capture.1)),
                            Err(error) => {
                                log_line(&format!("dictation capture: {error}"));
                                recording = false;
                                hide_at = Some(Instant::now() + ERROR_VISIBLE);
                                overlay(
                                    &app,
                                    "",
                                    "",
                                    "Микрофон недоступен · Esc — закрыть",
                                    "error",
                                );
                                (None, None)
                            }
                        };
                        _stream = new_stream;
                        resampler = new_resampler;
                        if _stream.is_some() {
                            overlay(&app, "", "", "Слушаю…", "listening");
                        }
                        if in_flight.is_none() {
                            in_flight = submit(
                                Job {
                                    generation,
                                    kind: RequestKind::Warmup,
                                    audio: Vec::new(),
                                    language: settings().language,
                                },
                                tx.clone(),
                            );
                        }
                        last_partial = Instant::now();
                    }
                    Some(Action::DiscardAndReinjectClick) => {
                        generation = generation.wrapping_add(1);
                        recording = false;
                        finalizing = false;
                        pending_final = None;
                        _stream = None;
                        resampler = None;
                        samples.clear();
                        hide_overlay(&app);
                        reinject_click();
                    }
                    Some(Action::FinishAndTranscribe) => {
                        recording = false;
                        finalizing = true;
                        hide_at = None;
                        _stream = None;
                        resampler = None;
                        if samples.len() < MIN_AUDIO_SAMPLES {
                            finalizing = false;
                            hide_overlay(&app);
                            continue;
                        }
                        overlay(&app, "", "", "Фиксирую фразу…", "thinking");
                        let job = Job {
                            generation,
                            kind: RequestKind::Final,
                            audio: wav(&samples),
                            language: settings().language,
                        };
                        if in_flight.is_none() {
                            in_flight = submit(job, tx.clone());
                        } else {
                            pending_final = Some(job);
                        }
                    }
                }
            }
            Ok(Message::Cancel) => {
                // A visible overlay may outlive its worker state after a stale or failed
                // request. Escape is always a dismissal, never gated on worker flags.
                generation = generation.wrapping_add(1);
                recording = false;
                finalizing = false;
                pending_final = None;
                _stream = None;
                resampler = None;
                samples.clear();
                hide_at = None;
                hide_overlay(&app);
            }
            Ok(Message::Samples(input, level)) => {
                if recording {
                    overlay_level(&app, level);
                    if let Some(resampler) = resampler.as_mut() {
                        resampler.push(&input, &mut samples);
                    }
                }
            }
            Ok(Message::Inference(result)) => {
                if in_flight == Some((result.generation, result.kind)) {
                    in_flight = None;
                }
                if result.generation == generation {
                    match (result.kind, result.result) {
                        (RequestKind::Warmup, _) => {}
                        (RequestKind::Partial, Ok(text)) if recording => {
                            let (stable, draft) = stable_preview(&previous, &text);
                            previous = text;
                            overlay(&app, &stable, &draft, "Слушаю…", "listening");
                        }
                        (RequestKind::Final, Ok(text)) if finalizing => {
                            finalizing = false;
                            if text.trim().is_empty() {
                                overlay(&app, "", "", "Речь не распознана", "error");
                                hide_at = Some(Instant::now() + ERROR_VISIBLE);
                            } else {
                                type_unicode(text.trim());
                                hide_overlay(&app);
                            }
                        }
                        (RequestKind::Partial | RequestKind::Final, Err(error)) => {
                            log_line(&format!("dictation transcription: {error}"));
                            if result.kind == RequestKind::Final {
                                finalizing = false;
                            }
                            overlay(
                                &app,
                                "",
                                "",
                                "Ошибка распознавания · Esc — закрыть",
                                "error",
                            );
                            hide_at = Some(Instant::now() + ERROR_VISIBLE);
                        }
                        _ => {}
                    }
                }
                if in_flight.is_none() {
                    if let Some(job) = pending_final.take() {
                        if job.generation == generation && finalizing {
                            in_flight = submit(job, tx.clone());
                        }
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        if hide_at.is_some_and(|deadline| Instant::now() >= deadline) {
            hide_at = None;
            hide_overlay(&app);
        }

        if recording
            && !finalizing
            && in_flight.is_none()
            && last_partial.elapsed() >= PARTIAL_EVERY
            && samples.len() >= MIN_AUDIO_SAMPLES
        {
            in_flight = submit(
                Job {
                    generation,
                    kind: RequestKind::Partial,
                    audio: wav(&samples[samples.len().saturating_sub(PARTIAL_WINDOW_SAMPLES)..]),
                    language: settings().language,
                },
                tx.clone(),
            );
            last_partial = Instant::now();
        }
    }
}

fn submit(job: Job, tx: Sender<Message>) -> Option<(u64, RequestKind)> {
    let id = (job.generation, job.kind);
    std::thread::spawn(move || {
        let result = request(&job);
        let _ = tx.send(Message::Inference(InferenceResult {
            generation: job.generation,
            kind: job.kind,
            result,
        }));
    });
    Some(id)
}

fn request(job: &Job) -> Result<String, String> {
    if job.kind == RequestKind::Warmup {
        return ureq::post(&format!("{SERVICE_URL}/api/transcribe/warmup"))
            .timeout(Duration::from_secs(120))
            .call()
            .map(|_| String::new())
            .map_err(|e| e.to_string());
    }
    let partial = usize::from(job.kind == RequestKind::Partial);
    let url = format!(
        "{SERVICE_URL}/api/transcribe?ext=wav&lang={}&partial={partial}",
        job.language
    );
    let response = ureq::post(&url)
        .set("Content-Type", "audio/wav")
        .timeout(Duration::from_secs(120))
        .send_bytes(&job.audio)
        .map_err(|e| e.to_string())?;
    let value: serde_json::Value = response.into_json().map_err(|e| e.to_string())?;
    value["text"].as_str().map(str::to_owned).ok_or_else(|| {
        value["error"]
            .as_str()
            .unwrap_or("invalid STT response")
            .to_owned()
    })
}

fn wav(samples: &[f32]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("WAV in memory");
        for &sample in samples {
            let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(value).expect("WAV in memory");
        }
        writer.finalize().expect("finalize WAV in memory");
    }
    cursor.into_inner()
}

fn start_capture(tx: Sender<Message>) -> Result<(Stream, LinearResampler), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "no default input device".to_owned())?;
    let supported = device.default_input_config().map_err(|e| e.to_string())?;
    let source_rate = supported.sample_rate().0;
    let config: StreamConfig = supported.clone().into();
    let channels = config.channels as usize;
    let error = |error| log_line(&format!("dictation audio stream: {error}"));
    let stream = match supported.sample_format() {
        SampleFormat::F32 => {
            let tx = tx.clone();
            device.build_input_stream(
                &config,
                move |data: &[f32], _| send_samples(data, channels, &tx),
                error,
                None,
            )
        }
        SampleFormat::I16 => {
            let tx = tx.clone();
            device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    send_samples(&floats, channels, &tx);
                },
                error,
                None,
            )
        }
        SampleFormat::U16 => {
            let tx = tx.clone();
            device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    let floats: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                        .collect();
                    send_samples(&floats, channels, &tx);
                },
                error,
                None,
            )
        }
        format => return Err(format!("unsupported input format {format:?}")),
    }
    .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok((stream, LinearResampler::new(source_rate, SAMPLE_RATE)))
}

fn send_samples(data: &[f32], channels: usize, tx: &Sender<Message>) {
    if channels == 0 {
        return;
    }
    // AT2020 is on the first UAC-2 input channel; selecting it also avoids
    // mixing in the electrically quiet second channel as background noise.
    let samples: Vec<f32> = data.chunks_exact(channels).map(|frame| frame[0]).collect();
    let level = (samples.iter().map(|sample| sample * sample).sum::<f32>()
        / samples.len().max(1) as f32)
        .sqrt();
    let _ = tx.send(Message::Samples(samples, level));
}

fn mouse_hook_loop() {
    unsafe {
        let module = GetModuleHandleW(std::ptr::null());
        let hook = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_hook), module, 0);
        if hook.is_null() {
            log_line("dictation: could not install XBUTTON1 mouse hook");
            return;
        }
        let mut message: MSG = std::mem::zeroed();
        while GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) > 0 {}
    }
}

fn keyboard_escape_loop() {
    let mut was_pressed = false;
    loop {
        let pressed = unsafe { GetAsyncKeyState(VK_ESCAPE as i32) } as u16 & 0x8000 != 0;
        if pressed && !was_pressed {
            if let Some(tx) = HOOK_TX.get() {
                let _ = tx.send(Message::Cancel);
            }
        }
        was_pressed = pressed;
        std::thread::sleep(std::time::Duration::from_millis(8));
    }
}

unsafe extern "system" fn mouse_hook(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code >= 0 && ENABLED.load(Ordering::Acquire) {
        let data = &*(lparam as *const MSLLHOOKSTRUCT);
        let xbutton1 = (data.mouseData >> 16) as u16 == XBUTTON1_DATA;
        let pressed = wparam as u32 == WM_XBUTTONDOWN;
        let released = wparam as u32 == WM_XBUTTONUP;
        if xbutton1 && (pressed || released) && data.flags & LLMHF_INJECTED == 0 {
            if let Some(tx) = HOOK_TX.get() {
                let _ = tx.send(Message::Button(
                    if pressed {
                        Event::Press
                    } else {
                        Event::Release
                    },
                    windows_sys::Win32::System::SystemInformation::GetTickCount64(),
                ));
            }
            return 1;
        }
    }
    CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam)
}

fn reinject_click() {
    for flags in [MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP] {
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: XBUTTON1_DATA as u32,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
    }
}

fn type_unicode(text: &str) {
    let units: Vec<u16> = text.encode_utf16().collect();
    for chunk in units.chunks(32) {
        let mut inputs = Vec::with_capacity(chunk.len() * 2);
        for &unit in chunk {
            for flags in [KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP] {
                inputs.push(INPUT {
                    r#type: windows_sys::Win32::UI::Input::KeyboardAndMouse::INPUT_KEYBOARD,
                    Anonymous: INPUT_0 {
                        ki: KEYBDINPUT {
                            wVk: 0,
                            wScan: unit,
                            dwFlags: flags,
                            time: 0,
                            dwExtraInfo: 0,
                        },
                    },
                });
            }
        }
        unsafe {
            SendInput(
                inputs.len() as u32,
                inputs.as_ptr(),
                std::mem::size_of::<INPUT>() as i32,
            )
        };
    }
}
