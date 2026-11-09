//! Shared mock helpers for the `eval_agents` tests (runner stand-ins and
//! handle registration), used by the parent and `workpool` test modules.

use super::*;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub(crate) fn run_output(
    id: &str,
    status: omp_task::RunStatus,
    output: &str,
) -> omp_task::RunOutput {
    omp_task::RunOutput {
        id: id.to_string(),
        agent: "task".to_string(),
        status,
        exit_code: Some(0),
        output: output.to_string(),
        stderr: String::new(),
        schema: omp_task::SchemaValidation::NoSchema,
        duration: Duration::ZERO,
        truncated: false,
    }
}

/// Runner stand-in: records steers, settles on `finish`/`cancel`.
pub(crate) struct MockRun {
    pub(crate) id: String,
    pub(crate) steered: Mutex<Vec<String>>,
    pub(crate) cancelled: AtomicBool,
    pub(crate) done: Mutex<Option<tokio::sync::oneshot::Sender<omp_task::RunOutput>>>,
}

impl MockRun {
    pub(crate) fn finish(&self, status: omp_task::RunStatus, output: &str) {
        if let Some(tx) = self.done.lock().take() {
            let _ = tx.send(run_output(&self.id, status, output));
        }
    }

    pub(crate) fn live(&self) -> bool {
        self.done.lock().is_some()
    }
}

impl RunCtl for MockRun {
    fn steer<'a>(
        &'a self,
        message: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), omp_task::SteerError>> + Send + 'a>> {
        Box::pin(async move {
            if !self.live() || self.cancelled.load(Ordering::SeqCst) {
                return Err(omp_task::SteerError::NotRunning);
            }
            self.steered.lock().push(message.to_string());
            Ok(())
        })
    }

    fn cancel(&self) -> bool {
        if !self.live() || self.cancelled.swap(true, Ordering::SeqCst) {
            return false;
        }
        self.finish(omp_task::RunStatus::Cancelled, "");
        true
    }

    fn output_tokens(&self) -> u64 {
        0
    }
}

#[derive(Default)]
pub(crate) struct MockLauncher {
    pub(crate) runs: Mutex<Vec<(omp_task::TaskSpec, Arc<MockRun>)>>,
}

impl MockLauncher {
    pub(crate) fn run(&self, index: usize) -> Arc<MockRun> {
        self.runs.lock()[index].1.clone()
    }

    pub(crate) fn launched(&self) -> usize {
        self.runs.lock().len()
    }
}

impl Launcher for MockLauncher {
    fn launch(&self, spec: omp_task::TaskSpec) -> (Arc<dyn RunCtl>, RunFuture) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let id = spec.id.clone().unwrap_or_default();
        let run = Arc::new(MockRun {
            id: id.clone(),
            steered: Mutex::new(Vec::new()),
            cancelled: AtomicBool::new(false),
            done: Mutex::new(Some(tx)),
        });
        self.runs.lock().push((spec, run.clone()));
        let future: RunFuture = Box::pin(async move {
            rx.await
                .unwrap_or_else(|_| run_output(&id, omp_task::RunStatus::Cancelled, ""))
        });
        let run: Arc<dyn RunCtl> = run;
        (run, future)
    }
}

pub(crate) fn spec(id: &str) -> omp_task::TaskSpec {
    omp_task::TaskSpec {
        id: Some(id.to_string()),
        agent: "task".to_string(),
        prompt: "p".to_string(),
        cwd: PathBuf::from("."),
        output_schema: None,
        schema_mode: omp_task::SchemaMode::Permissive,
        model: None,
        tools: None,
        depth: 0,
        extra_args: Vec::new(),
        env: Vec::new(),
        worktree: false,
    }
}

/// Register an agent handle backed by a mock run, settled by a
/// background task exactly like `run_agent` does.
pub(crate) fn mock_agent(launcher: &MockLauncher, id: &str) -> (Arc<Handle>, Arc<MockRun>) {
    let handle = Handle::new("agent", id.to_string());
    AGENT_HANDLES.lock().insert(id.to_string(), handle.clone());
    let run = launch_agent(&handle, spec(id), launcher);
    let settle = handle.clone();
    tokio::spawn(async move {
        let output = run.await;
        settle_agent(&settle, &output, omp_task::SchemaMode::Permissive, false);
        settle.mark_run_finished();
    });
    let mock = launcher.run(launcher.launched() - 1);
    (handle, mock)
}

pub(crate) async fn settled(handle: &Handle) {
    for _ in 0..200 {
        if handle.is_settled() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("handle {} never settled", handle.id);
}

/// Serialize the tests that mutate the recursion-depth env: the env is
/// process-global and `create_workpool` reads it through the gate.
pub(crate) static DEPTH_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
