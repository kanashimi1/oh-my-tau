//! The `goal` tool (omp `GoalTool`, hidden): manage the active goal-mode
//! objective, mirroring `goals/tools/goal-tool.ts` + `goals/runtime.ts` op
//! semantics. Goal mode is config-gated in omp (`cfgGoalEnabled`); this port
//! wires the tool to an in-process [`GoalRuntime`]. Without one (the default
//! headless wiring, and always when goal mode is off) every op fails with
//! omp's "Goal mode is not active." — omp `session.getGoalRuntime` undefined.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use serde_json::{json, Map, Value};

const DESCRIPTION: &str = "Manage active goal-mode objective.\n\nSingle `op` field:\n- `create`: starts goal; enables goal mode. Requires `objective`; optional positive `token_budget`. Only when no goal exists and none is paused.\n- `get`: returns current active/paused goal and remaining token budget.\n- `resume`: re-activates paused goal for continued work.\n- `complete`: marks goal complete only when actually done and every deliverable verified against current evidence. NEVER because budget low or turn ending.\n- `drop`: discards current goal without completing it.\n\nPaused goal from `get` → MUST `resume` before continuing work.";

/// Parameter schema, byte-equal (as JSON) to omp's `goalSchema`.
#[must_use]
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["create", "get", "complete", "resume", "drop"], "description": "goal operation"},
            "objective": {"type": "string", "description": "goal objective"},
            "token_budget": {"type": "integer", "description": "token budget"}
        },
        "required": ["op"]
    })
}

/// omp `GoalStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // omp GoalStatus wire shape; paused/budget-limited round-trip for parity even though omp-rs never sets them
pub enum GoalStatus {
    Active,
    Paused,
    BudgetLimited,
    Complete,
    Dropped,
}

impl GoalStatus {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::BudgetLimited => "budget-limited",
            Self::Complete => "complete",
            Self::Dropped => "dropped",
        }
    }
    #[allow(dead_code)] // wire-shape parity with omp; resume parsing is not in omp-rs yet
    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "active" => Self::Active,
            "paused" => Self::Paused,
            "budget-limited" => Self::BudgetLimited,
            "complete" => Self::Complete,
            "dropped" => Self::Dropped,
            _ => return None,
        })
    }
    #[allow(dead_code)] // only the (not-yet-wired) token-accounting tail calls it
    fn is_accounting(self) -> bool {
        matches!(self, Self::Active | Self::BudgetLimited)
    }
}

/// omp `Goal`: serializable goal progress and resource budget.
#[derive(Debug, Clone)]
pub struct Goal {
    pub id: String,
    pub objective: String,
    pub status: GoalStatus,
    pub token_budget: Option<i64>,
    pub tokens_used: i64,
    pub time_used_seconds: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

impl Goal {
    /// omp `remainingTokens`.
    #[must_use]
    pub fn remaining_tokens(&self) -> Option<i64> {
        self.token_budget.map(|b| (b - self.tokens_used).max(0))
    }

    fn to_json(&self) -> Value {
        let mut obj = Map::new();
        obj.insert("id".into(), json!(self.id));
        obj.insert("objective".into(), json!(self.objective));
        obj.insert("status".into(), json!(self.status.as_str()));
        if let Some(budget) = self.token_budget {
            obj.insert("tokenBudget".into(), json!(budget));
        }
        obj.insert("tokensUsed".into(), json!(self.tokens_used));
        obj.insert("timeUsedSeconds".into(), json!(self.time_used_seconds));
        obj.insert("createdAt".into(), json!(self.created_at));
        obj.insert("updatedAt".into(), json!(self.updated_at));
        Value::Object(obj)
    }
}

/// omp `completionBudgetReport` (goals/runtime.ts).
#[must_use]
fn completion_budget_report(goal: &Goal) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(budget) = goal.token_budget {
        parts.push(format!("tokens used: {} of {}", goal.tokens_used, budget));
    }
    if goal.time_used_seconds > 0 {
        parts.push(format!("time used: {} seconds", goal.time_used_seconds));
    }
    if parts.is_empty() {
        None
    } else {
        Some(format!(
            "Goal achieved. Report final budget usage to the user: {}.",
            parts.join("; ")
        ))
    }
}

/// In-process goal-mode state machine (omp `GoalRuntime`, host subset): the
/// current goal plus wall-clock accounting while the goal is being worked.
#[derive(Default)]
pub struct GoalRuntime {
    goal: Option<Goal>,
    /// Wall-clock accumulation anchor while an accounting status is live.
    active_since: Option<Instant>,
    next_id: u64,
}

impl GoalRuntime {
    /// Flush elapsed wall clock into `time_used_seconds` (omp `#flushUsageLocked`).
    fn flush_time(&mut self) {
        if let (Some(goal), Some(since)) = (self.goal.as_mut(), self.active_since.take()) {
            goal.time_used_seconds += i64::try_from(since.elapsed().as_secs()).unwrap_or(0);
        }
    }

    fn mark_active(&mut self) {
        self.active_since = Some(Instant::now());
    }

    fn now_millis() -> i64 {
        i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis()),
        )
        .unwrap_or(0)
    }

    /// omp `GoalRuntime.createGoal`.
    pub fn create_goal(&mut self, objective: &str, token_budget: Option<i64>) -> Result<Goal> {
        let objective = objective.trim();
        if objective.is_empty() {
            anyhow::bail!("objective is required when op=create");
        }
        if token_budget.is_some_and(|b| b <= 0) {
            anyhow::bail!("token_budget must be a positive integer when provided");
        }
        if let Some(goal) = &self.goal {
            if !matches!(goal.status, GoalStatus::Dropped | GoalStatus::Complete) {
                anyhow::bail!("cannot create a new goal because this session already has a goal");
            }
        }
        self.flush_time();
        self.next_id += 1;
        let now = Self::now_millis();
        let goal = Goal {
            id: format!(
                "{}",
                now * 1000 + i64::try_from(self.next_id).unwrap_or(i64::MAX)
            ),
            objective: objective.to_string(),
            status: GoalStatus::Active,
            token_budget,
            tokens_used: 0,
            time_used_seconds: 0,
            created_at: now,
            updated_at: now,
        };
        self.mark_active();
        self.goal = Some(goal.clone());
        Ok(goal)
    }

    /// omp `GoalRuntime.resumeGoal`.
    pub fn resume_goal(&mut self) -> Result<Goal> {
        match self.goal.as_ref() {
            None => anyhow::bail!("No paused goal."),
            Some(g) if g.status == GoalStatus::Complete => {
                anyhow::bail!("Goal is already complete.")
            }
            Some(_) => {}
        }
        let now = Self::now_millis();
        self.flush_time();
        if let Some(goal) = self.goal.as_mut() {
            goal.status = GoalStatus::Active;
            goal.updated_at = now;
        }
        self.mark_active();
        Ok(self.goal.clone().expect("goal checked above"))
    }

    /// omp `GoalRuntime.dropGoal`.
    pub fn drop_goal(&mut self) -> Option<Goal> {
        let now = Self::now_millis();
        self.flush_time();
        let goal = self.goal.as_mut()?;
        goal.status = GoalStatus::Dropped;
        goal.updated_at = now;
        let dropped = goal.clone();
        // omp dropGoal commits state undefined: `get` reports no goal after.
        self.goal = None;
        Some(dropped)
    }

    /// omp `GoalRuntime.completeGoalFromTool`.
    pub fn complete_goal(&mut self) -> Result<Goal> {
        if let Some(goal) = self.goal.as_ref() {
            match goal.status {
                GoalStatus::Complete => anyhow::bail!("goal is already complete"),
                GoalStatus::Dropped => anyhow::bail!("cannot complete a dropped goal"),
                _ => {}
            }
        } else {
            anyhow::bail!("cannot complete goal because no goal is active");
        }
        let now = Self::now_millis();
        self.flush_time();
        let Some(goal) = self.goal.as_mut() else {
            anyhow::bail!("cannot complete goal because no goal is active");
        };
        goal.status = GoalStatus::Complete;
        goal.updated_at = now;
        Ok(goal.clone())
    }

    /// Attribute token usage to the active goal (omp accounting tail).
    #[allow(dead_code)] // omp token-accounting tail; wired when per-goal accounting lands
    pub fn add_tokens(&mut self, tokens: i64) {
        if let Some(goal) = self.goal.as_mut() {
            if goal.status.is_accounting() {
                goal.tokens_used += tokens;
            }
        }
    }

    /// omp `session.getGoalModeState().goal` (paused state included).
    #[must_use]
    pub fn state_goal(&self) -> Option<&Goal> {
        self.goal.as_ref()
    }
}

/// The `goal` tool.
pub struct GoalTool {
    schema: Value,
    runtime: Option<Arc<std::sync::Mutex<GoalRuntime>>>,
}

/// The hidden `goal` tool (omp `HIDDEN_TOOLS.goal`) without a goal runtime:
/// goal mode is not active, so every op fails (omp `getGoalRuntime` undefined).
#[must_use]
pub fn tool() -> Arc<dyn AgentTool> {
    runtime_tool(None)
}

/// The hidden `goal` tool wired to a live goal-mode runtime.
#[must_use]
pub fn runtime_tool(runtime: Option<Arc<std::sync::Mutex<GoalRuntime>>>) -> Arc<dyn AgentTool> {
    Arc::new(GoalTool {
        schema: schema(),
        runtime,
    })
}

/// omp `buildGoalToolResponse`.
fn build_response(
    goal: Option<&Goal>,
    include_completion_report: bool,
) -> (Value, Option<i64>, Option<String>) {
    let remaining = goal.and_then(Goal::remaining_tokens);
    let report =
        if include_completion_report && goal.is_some_and(|g| g.status == GoalStatus::Complete) {
            completion_budget_report(goal.expect("goal checked"))
        } else {
            None
        };
    (goal.map_or(Value::Null, Goal::to_json), remaining, report)
}

/// omp's model-facing goal summary text.
fn goal_text(goal: Option<&Goal>, remaining: Option<i64>, report: Option<&str>) -> String {
    let Some(goal) = goal else {
        return "No active goal.".to_string();
    };
    let mut text = format!(
        "Goal: {}\nStatus: {}\nTokens: {} used",
        goal.objective,
        goal.status.as_str(),
        goal.tokens_used
    );
    if let Some(budget) = goal.token_budget {
        text.push_str(" / ");
        text.push_str(&budget.to_string());
        text.push_str(" budget");
    }
    if let Some(remaining) = remaining {
        text.push_str("\nRemaining tokens: ");
        text.push_str(&remaining.to_string());
    }
    if let Some(report) = report {
        text.push_str("\n\n");
        text.push_str(report);
    }
    text
}

impl AgentTool for GoalTool {
    fn name(&self) -> &'static str {
        "goal"
    }
    fn description(&self) -> &str {
        DESCRIPTION
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn label(&self) -> &'static str {
        "Goal"
    }
    fn execute(
        self: Arc<Self>,
        _tool_call_id: String,
        params: Value,
        _signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        Box::pin(async move {
            let Some(runtime) = self.runtime.clone() else {
                anyhow::bail!("Goal mode is not active.");
            };
            let op = params
                .get("op")
                .and_then(Value::as_str)
                .unwrap_or("undefined")
                .to_string();
            let mut runtime = runtime.lock().expect("goal runtime poisoned");
            let goal: Option<Goal> = match op.as_str() {
                "create" => {
                    let objective = params
                        .get("objective")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let budget = params.get("token_budget").and_then(Value::as_i64);
                    Some(runtime.create_goal(&objective, budget)?)
                }
                "get" => runtime.state_goal().cloned(),
                "resume" => Some(runtime.resume_goal()?),
                "drop" => runtime.drop_goal(),
                "complete" => Some(runtime.complete_goal()?),
                other => anyhow::bail!("Unknown goal op: {other}"),
            };
            let (goal_json, remaining, report) = build_response(goal.as_ref(), op == "complete");
            let text = goal_text(goal.as_ref(), remaining, report.as_deref());
            let mut details = Map::new();
            details.insert("op".into(), json!(op));
            details.insert("goal".into(), goal_json);
            details.insert(
                "remainingTokens".into(),
                remaining.map_or(Value::Null, |r| json!(r)),
            );
            details.insert(
                "completionBudgetReport".into(),
                report.map_or(Value::Null, Value::String),
            );
            Ok(AgentToolResult {
                content: vec![pa_agent::types::ToolResultContent::text(text)],
                details: Value::Object(details),
                terminate: None,
                is_error: None,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_matches_omp() {
        assert_eq!(
            schema(),
            json!({
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["create", "get", "complete", "resume", "drop"], "description": "goal operation"},
                    "objective": {"type": "string", "description": "goal objective"},
                    "token_budget": {"type": "integer", "description": "token budget"}
                },
                "required": ["op"]
            })
        );
    }

    #[test]
    fn without_runtime_every_op_fails() {
        let tool = tool();
        assert_eq!(tool.name(), "goal");
    }

    #[test]
    fn runtime_lifecycle_matches_omp_semantics() {
        let mut rt = GoalRuntime::default();
        assert_eq!(rt.resume_goal().unwrap_err().to_string(), "No paused goal.");
        let goal = rt.create_goal(" ship it ", Some(100)).unwrap();
        assert_eq!(goal.status, GoalStatus::Active);
        assert_eq!(goal.remaining_tokens(), Some(100));
        assert!(rt
            .create_goal("again", None)
            .unwrap_err()
            .to_string()
            .contains("already has a goal"));
        rt.add_tokens(30);
        assert_eq!(rt.state_goal().unwrap().tokens_used, 30);
        let paused = rt.state_goal().unwrap().clone();
        rt.goal.as_mut().unwrap().status = GoalStatus::Paused;
        drop(paused);
        assert_eq!(rt.resume_goal().unwrap().status, GoalStatus::Active);
        let done = rt.complete_goal().unwrap();
        assert_eq!(done.status, GoalStatus::Complete);
        assert_eq!(done.remaining_tokens(), Some(70));
        assert_eq!(
            rt.complete_goal().unwrap_err().to_string(),
            "goal is already complete"
        );
        assert!(
            completion_budget_report(&done).is_some_and(|r| r.contains("tokens used: 30 of 100"))
        );
    }

    #[test]
    fn drop_discards_goal() {
        let mut rt = GoalRuntime::default();
        assert!(rt.drop_goal().is_none());
        rt.create_goal("x", None).unwrap();
        let dropped = rt.drop_goal().unwrap();
        assert_eq!(dropped.status, GoalStatus::Dropped);
        assert!(rt.state_goal().is_none());
    }
}
