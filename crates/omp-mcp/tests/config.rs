//! Config loading tests: user config via `OMP_CONFIG_DIR`, project configs,
//! disabled/forced servers. Env-mutating cases live in ONE test because
//! cargo runs tests of a binary in parallel threads sharing the process env.

use std::collections::BTreeMap;
use std::path::PathBuf;

use omp_mcp::{load_configs, McpServerConfig, ServerKind};

fn write(dir: &std::path::Path, name: &str, body: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(name), body).unwrap();
}

#[test]
fn config_loading_matches_omp_rules() {
    let root = tempfile::tempdir().unwrap();
    let user_dir = root.path().join("usercfg");

    // Case 1: merge + precedence (project overrides user per name).
    let project = root.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    write(
        &user_dir.join(".omp").join("agent"),
        "mcp.json",
        r#"{"mcpServers": {
            "user-a": {"command": "a", "enabled": false},
            "user-b": {"command": "b"},
            "hidden": {"command": "h"}
        }}"#,
    );
    write(
        &project,
        ".mcp.json",
        r#"{"mcpServers": {
            "user-b": {"command": "b-override"},
            "proj": {"type": "http", "url": "http://127.0.0.1:9/mcp"}
        }}"#,
    );
    write(
        &project.join(".omp"),
        "mcp.json",
        r#"{"mcpServers": {"proj2": {"command": "p2"}}}"#,
    );
    std::env::set_var("OMP_CONFIG_DIR", &user_dir);
    let configs = load_configs(&project);

    assert!(configs.contains_key("user-b"));
    assert!(configs.contains_key("proj"));
    assert!(configs.contains_key("proj2"));
    assert!(!configs.contains_key("user-a"));
    assert_eq!(configs["user-b"].command.as_deref(), Some("b-override"));
    assert_eq!(configs["proj"].kind, ServerKind::Http);
    assert_eq!(
        configs["proj"].url.as_deref(),
        Some("http://127.0.0.1:9/mcp")
    );

    // Case 2: disabledServers hide; enabledServers force-enable.
    let user2 = root.path().join("user2");
    write(
        &user2.join(".omp").join("agent"),
        "mcp.json",
        r#"{
            "disabledServers": ["off"],
            "enabledServers": ["forced"],
            "mcpServers": {
                "off": {"command": "x"},
                "forced": {"command": "y", "enabled": false},
                "on": {"command": "z"}
            }
        }"#,
    );
    std::env::set_var("OMP_CONFIG_DIR", &user2);
    let configs = load_configs(&project);
    assert!(!configs.contains_key("off"));
    assert!(configs.contains_key("forced"));
    assert!(configs.contains_key("on"));

    // Case 3: malformed user config is ignored, not fatal.
    let user3 = root.path().join("user3");
    write(&user3.join(".omp").join("agent"), "mcp.json", "{ not json");
    let project3 = root.path().join("project3");
    std::fs::create_dir_all(&project3).unwrap();
    write(
        &project3,
        "mcp.json",
        r#"{"mcpServers": {"ok": {"command": "ok"}}}"#,
    );
    std::env::set_var("OMP_CONFIG_DIR", &user3);
    let configs: BTreeMap<String, McpServerConfig> = load_configs(&project3);
    std::env::remove_var("OMP_CONFIG_DIR");
    assert_eq!(configs.len(), 1);
    assert!(configs.contains_key("ok"));
}

#[test]
fn project_config_paths_lists_all_candidates() {
    let cwd = PathBuf::from("/tmp/x");
    let paths = omp_mcp::config::project_config_paths(&cwd);
    assert_eq!(
        paths,
        vec![
            PathBuf::from("/tmp/x/mcp.json"),
            PathBuf::from("/tmp/x/.mcp.json"),
            PathBuf::from("/tmp/x/.omp/mcp.json")
        ]
    );
}
