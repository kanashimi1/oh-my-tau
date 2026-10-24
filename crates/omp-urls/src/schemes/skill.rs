//! `skill://` protocol handler (TS `internal-urls/skill-protocol.ts`).
//!
//! URL forms:
//! - `skill://<name>` — reads the skill's SKILL.md
//! - `skill://<name>/<path>` — reads a relative path within the skill's base dir
//! - `skill://<namespace>/<name>[/<path>]` — collision-namespaced skill

use std::path::{Path, PathBuf};

use crate::{InternalResource, InternalUrl, UrlCompletion, UrlCtx, UrlProvider};

use super::{
    content_type_for_path, decode_component, format_directory_listing, validate_relative_path,
    SkillEntry,
};

/// Discover skills the way omp-rs renders them (TS `extensibility/skills.ts`):
/// project `.omp/skills` walk-up (closest first), then the agent dir's `skills/`.
pub fn discover_skills(agent_dir: &Path, cwd: &Path) -> Vec<SkillEntry> {
    let mut skills = Vec::new();
    for dir in cwd.ancestors() {
        load_skills_dir(&dir.join(".omp/skills"), &mut skills);
    }
    load_skills_dir(&agent_dir.join("skills"), &mut skills);
    skills
}

fn load_skills_dir(dir: &Path, skills: &mut Vec<SkillEntry>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let file = path.join("SKILL.md");
        let Ok(content) = std::fs::read_to_string(&file) else {
            continue;
        };
        let (meta, _body) = super::parse_frontmatter(&content);
        let get = |key: &str| meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        let name = get("name").unwrap_or_else(|| entry.file_name().to_string_lossy().into_owned());
        let description = get("description").unwrap_or_default();
        let hide = get("hide").is_some_and(|v| v == "true");
        if hide {
            continue;
        }
        skills.push(SkillEntry {
            name,
            file_path: file.to_string_lossy().into_owned(),
            base_dir: path.to_string_lossy().into_owned(),
            description,
            // Plain discovered skills have no plugin package root (TS
            // `skill.containRoot` is agent-plugin-only); lexical rules apply.
            contain_root: None,
        });
    }
}

/// On-disk target of a `skill://` URL. `directory` maps a bare `skill://<name>`
/// to the skill base dir (TS `skillTargetPath`).
pub fn skill_target_path(
    url: &InternalUrl,
    skills: &[SkillEntry],
    directory: bool,
) -> Result<PathBuf, String> {
    let skill_name = url
        .host
        .is_empty()
        .then_some(())
        .map_or(url.host.as_str(), |_| "");
    if url.host.is_empty() {
        let _ = skill_name;
        return Err("skill:// URL requires a skill name: skill://<name>".to_string());
    }
    let mut path_part = url.pathname.trim_start_matches('/');
    let mut skill: Option<&SkillEntry> = None;
    if !path_part.is_empty() {
        // A namespaced skill (`<plugin>/<name>`) spans the host and the first
        // path segment; an exact match is unambiguous and wins.
        let slash = path_part.find('/').unwrap_or(path_part.len());
        let namespaced = format!("{}/{}", url.host, decode_component(&path_part[..slash]));
        if let Some(s) = skills.iter().find(|s| s.name == namespaced) {
            skill = Some(s);
            path_part = if slash == path_part.len() {
                ""
            } else {
                &path_part[slash + 1..]
            };
        }
    }
    let skill = match skill.or_else(|| skills.iter().find(|s| s.name == url.host)) {
        Some(s) => s,
        None => {
            let mut available = skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>();
            available.sort();
            let available_str = if available.is_empty() {
                "none".to_string()
            } else {
                available.join(", ")
            };
            return Err(format!(
                "Unknown skill: {}\nAvailable: {available_str}",
                url.host
            ));
        }
    };

    let resolved = if path_part.is_empty() {
        PathBuf::from(if directory {
            &skill.base_dir
        } else {
            &skill.file_path
        })
    } else {
        let relative = decode_component(path_part);
        validate_relative_path(&relative, "skill")?;
        let resolved = Path::new(&skill.base_dir).join(&relative);
        let base = Path::new(&skill.base_dir);
        if resolved != base && !resolved.starts_with(base) {
            return Err("Path traversal is not allowed".to_string());
        }
        resolved
    };
    // Agent Plugin skills (§4.1, TS `resolveContainedPath`): every target,
    // including the bare instruction file and base directory, must resolve
    // canonically within the plugin root; symlinks may only target files
    // inside the same package. Fails closed on unresolvable targets.
    if let Some(root) = &skill.contain_root {
        let escape = format!(
            "skill:// path resolves outside the plugin root: {}",
            url.raw_href
        );
        let real_root = std::fs::canonicalize(root).map_err(|_| escape.clone())?;
        let real = std::fs::canonicalize(&resolved).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => e.to_string(),
            _ => escape.clone(),
        })?;
        if real != real_root && !real.starts_with(&real_root) {
            return Err(escape);
        }
        return Ok(real);
    }
    Ok(resolved)
}

/// Resolve a `skill://` URL to a resource (TS `SkillProtocolHandler.resolve`).
pub fn resolve(url: &InternalUrl, skills: &[SkillEntry]) -> Result<InternalResource, String> {
    // A trailing slash lists the skill's base directory (TS `skill://demo/`).
    let directory = url.pathname == "/";
    let target = skill_target_path(url, skills, directory)?;
    let meta =
        std::fs::metadata(&target).map_err(|_| format!("File not found: {}", target.display()))?;
    let path_str = target.to_string_lossy().into_owned();
    if meta.is_dir() {
        let entries: Vec<(String, bool)> = std::fs::read_dir(&target)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| {
                (
                    e.file_name().to_string_lossy().into_owned(),
                    e.path().is_dir(),
                )
            })
            .collect();
        let content = format_directory_listing(entries);
        return Ok(InternalResource {
            content,
            is_directory: true,
            immutable: Some(true),
            path: Some(path_str),
        });
    }
    if !meta.is_file() {
        return Err(format!(
            "skill:// URL must resolve to a file or directory: {}",
            url.raw_href
        ));
    }
    let content = std::fs::read_to_string(&target).map_err(|e| e.to_string())?;
    let _ = content_type_for_path(&path_str);
    Ok(InternalResource {
        content,
        is_directory: false,
        immutable: Some(true),
        path: Some(path_str),
    })
}

/// Skill file or directory path; `None` for a missing entry (TS `locate`).
pub fn locate(
    url: &InternalUrl,
    skills: &[SkillEntry],
    directory: bool,
) -> Result<Option<PathBuf>, String> {
    let target = skill_target_path(url, skills, directory)?;
    Ok(target.exists().then_some(target))
}

/// Autocomplete candidates (TS `complete`).
pub fn complete(skills: &[SkillEntry]) -> Vec<UrlCompletion> {
    skills
        .iter()
        .map(|s| UrlCompletion {
            label: s.name.clone(),
            detail: Some(s.description.clone()),
        })
        .collect()
}

/// Read-only `skill://` provider over the discovered skill directories.
#[derive(Default)]
pub struct SkillProvider;

impl UrlProvider for SkillProvider {
    fn scheme(&self) -> &'static str {
        "skill"
    }

    fn spec(&self) -> &crate::SchemeSpec {
        static SPEC: crate::SchemeSpec = crate::SchemeSpec {
            backing: crate::SchemeBacking::File,
            selectors: crate::SchemeSelectors::Lines,
            immutable: true,
            single_slash_alias: false,
            linkable: true,
            port_authority: false,
            write: None,
        };
        &SPEC
    }

    fn read(&self, url: &InternalUrl, ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        let skills = active_skills(ctx);
        resolve(url, &skills).map_err(anyhow::Error::msg)
    }

    fn locate(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        locate(url, &active_skills(ctx), false).ok().flatten()
    }

    fn complete(&self, _query: &str, ctx: &UrlCtx) -> Vec<UrlCompletion> {
        complete(&active_skills(ctx))
    }

    fn prompt_doc(&self) -> Option<String> {
        Some("`skill://<name>`: instructions; append `/<path>` for a file.".to_string())
    }
}

/// Skills visible to the calling session: discovered fresh from the agent dir
/// and the caller's cwd (TS `getActiveSkills()`).
fn active_skills(ctx: &UrlCtx) -> Vec<SkillEntry> {
    let agent_dir = omp_providers::AgentDir::default().0;
    let cwd = ctx
        .cwd
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    discover_skills(&agent_dir, &cwd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_internal_url;

    fn skill(name: &str) -> SkillEntry {
        SkillEntry {
            name: name.to_string(),
            file_path: format!("/tmp/skills/{name}/SKILL.md"),
            base_dir: format!("/tmp/skills/{name}"),
            description: String::new(),
            contain_root: None,
        }
    }

    /// Review #3: agent-plugin skills (`contain_root`) must canonically
    /// contain every target — a symlink escaping the package is rejected
    /// (TS `resolveContainedPath`), while plain skills stay lexical.
    #[test]
    fn plugin_containment_rejects_escaping_symlinks() {
        let root = std::env::temp_dir().join(format!("omp-skill-contain-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let skill_dir = root.join("pkg/skill");
        let outside = root.join("secret.txt");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(&outside, "top secret").unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "skill").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, skill_dir.join("leak")).unwrap();

        let mut entry = skill("demo");
        entry.base_dir = skill_dir.to_string_lossy().into_owned();
        entry.file_path = skill_dir.join("SKILL.md").to_string_lossy().into_owned();
        entry.contain_root = Some(root.join("pkg").to_string_lossy().into_owned());

        // A symlink pointing outside the plugin root is refused.
        #[cfg(unix)]
        {
            let url = parse_internal_url("skill://demo/leak").unwrap();
            let err = skill_target_path(&url, &[entry.clone()], false).unwrap_err();
            assert!(err.contains("outside the plugin root"), "{err}");
        }
        // An in-package file resolves (to its canonical path).
        let url = parse_internal_url("skill://demo/SKILL.md").unwrap();
        let got = skill_target_path(&url, &[entry.clone()], false).unwrap();
        assert_eq!(
            got,
            std::fs::canonicalize(skill_dir.join("SKILL.md")).unwrap()
        );
        // Missing target surfaces as not-found, not as a containment pass.
        let url = parse_internal_url("skill://demo/nope.md").unwrap();
        assert!(skill_target_path(&url, &[entry], false).is_err());

        // Plain skills (no contain_root) keep the lexical rule: the same
        // symlink resolves, matching TS non-plugin behavior.
        #[cfg(unix)]
        {
            let mut plain = skill("demo");
            plain.base_dir = skill_dir.to_string_lossy().into_owned();
            let url = parse_internal_url("skill://demo/leak").unwrap();
            assert_eq!(
                skill_target_path(&url, &[plain], false).unwrap(),
                skill_dir.join("leak")
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn requires_name() {
        let url = parse_internal_url("skill://").unwrap();
        assert_eq!(
            skill_target_path(&url, &[skill("a")], false).unwrap_err(),
            "skill:// URL requires a skill name: skill://<name>"
        );
    }

    #[test]
    fn unknown_skill_lists_available() {
        let url = parse_internal_url("skill://zzz").unwrap();
        assert_eq!(
            skill_target_path(&url, &[skill("a"), skill("b")], false).unwrap_err(),
            "Unknown skill: zzz\nAvailable: a, b"
        );
        let url = parse_internal_url("skill://zzz").unwrap();
        assert_eq!(
            skill_target_path(&url, &[], false).unwrap_err(),
            "Unknown skill: zzz\nAvailable: none"
        );
    }

    #[test]
    fn bare_skill_maps_to_skill_md() {
        let url = parse_internal_url("skill://foo").unwrap();
        assert_eq!(
            skill_target_path(&url, &[skill("foo")], false).unwrap(),
            PathBuf::from("/tmp/skills/foo/SKILL.md")
        );
        assert_eq!(
            skill_target_path(&url, &[skill("foo")], true).unwrap(),
            PathBuf::from("/tmp/skills/foo")
        );
    }

    #[test]
    fn namespaced_skill_wins() {
        let url = parse_internal_url("skill://ns/foo/extra.md").unwrap();
        assert_eq!(
            skill_target_path(&url, &[skill("foo"), skill("ns/foo")], false).unwrap(),
            PathBuf::from("/tmp/skills/ns/foo/extra.md")
        );
    }

    #[test]
    fn traversal_rejected() {
        let url = parse_internal_url("skill://foo/../etc").unwrap();
        assert_eq!(
            skill_target_path(&url, &[skill("foo")], false).unwrap_err(),
            "Path traversal is not allowed"
        );
    }

    #[test]
    fn resource_reads_file() {
        let dir = std::env::temp_dir().join(format!("ompurls-skill-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".omp/skills/demo")).unwrap();
        std::fs::write(
            dir.join(".omp/skills/demo/SKILL.md"),
            "---\nname: demo\ndescription: A demo\n---\nBody",
        )
        .unwrap();
        let skills = discover_skills(&dir, &dir);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "demo");
        let url = parse_internal_url("skill://demo").unwrap();
        let res = resolve(&url, &skills).unwrap();
        assert!(!res.is_directory);
        assert!(res.content.ends_with("Body"));
        let url = parse_internal_url("skill://demo/").unwrap();
        let list = resolve(&url, &skills).unwrap();
        assert_eq!(list.content, "SKILL.md");
        std::fs::remove_dir_all(&dir).ok();
    }
}
