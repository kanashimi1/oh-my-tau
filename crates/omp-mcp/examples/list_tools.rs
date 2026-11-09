#![allow(clippy::disallowed_methods)] // tests/entry points: throwaway runtimes drive async code
//! List the MCP tools omp-rs would register from the real config locations,
//! exactly as the agent startup does. Usage: `cargo run --example list_tools`.

use std::sync::Arc;

use omp_mcp::McpManager;

#[tokio::main]
async fn main() {
    let Ok(cwd) = std::env::current_dir() else {
        eprintln!("cannot determine current directory");
        std::process::exit(1);
    };
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
