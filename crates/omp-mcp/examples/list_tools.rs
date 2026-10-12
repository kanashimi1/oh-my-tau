//! List the MCP tools omp-rs would register from the real config locations,
//! exactly as the agent startup does. Usage: `cargo run --example list_tools`.

use std::sync::Arc;

use omp_mcp::McpManager;

#[tokio::main]
async fn main() {
    let cwd = std::env::current_dir().expect("cwd");
    let manager = McpManager::from_default_configs(&cwd);
    let tools = manager.tools().await;
    if tools.is_empty() {
        println!("(no MCP tools)");
    }
    for tool in tools {
        println!("{}", tool.registry_name);
    }
    for (name, status) in manager.statuses().await {
        println!("{name}: {status:?}");
    }
    // Keep the runtime alive so in-flight closes finish.
    drop(Arc::new(()));
}
