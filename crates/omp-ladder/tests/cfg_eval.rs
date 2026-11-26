//! `cfg(...)` / `cfg_attr(...)` attribute evaluation and `cfg!`
//! macro interaction.

use omp_ladder::{parse_top_level, project, CfgContext, CfgExpr, ItemStatus};

fn ctx(flags: &[&str], kv: &[(&str, &str)]) -> CfgContext {
    let mut c = CfgContext::default();
    for f in flags {
        c.flags.insert((*f).to_owned());
    }
    for (k, v) in kv {
        c.kv.insert((*k).to_owned(), (*v).to_owned());
    }
    c
}

const SRC: &str = r#"
#[cfg(feature = "serde")]
fn with_serde() {}

#[cfg(all(unix, not(test)))]
fn unix_not_test() {}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn unixy() {}

#[cfg_attr(feature = "serde", cfg(test))]
fn attr_gated() {}

#[derive(Debug)]
#[cfg(feature = "never-on")]
/// doc between attrs
pub struct Gated { x: u8 }

fn uses_cfg_macro() {
    if cfg!(feature = "serde") { }
    let _ = cfg!(all(unix, target_pointer_width = "64"));
}
"#;

#[test]
fn cfg_expression_eval() {
    let e = CfgExpr::All(vec![
        CfgExpr::Atom("unix".into()),
        CfgExpr::Not(Box::new(CfgExpr::Atom("test".into()))),
    ]);
    assert!(e.eval(&ctx(&["unix"], &[])));
    assert!(!e.eval(&ctx(&["unix", "test"], &[])));
    let kv = CfgExpr::KeyValue("target_os".into(), "linux".into());
    assert!(kv.eval(&ctx(&[], &[("target_os", "linux")])));
    assert!(!kv.eval(&ctx(&[], &[("target_os", "windows")])));
}

#[test]
fn cfg_attributes_gate_items() {
    let cst = parse_top_level(SRC);

    let on = project(&cst, &ctx(&["unix"], &[("feature", "serde"), ("target_os", "linux")]));
    let by_name = |name: &str| {
        on.items
            .iter()
            .find(|i| i.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    assert_eq!(by_name("with_serde").status, ItemStatus::Active);
    assert_eq!(by_name("unix_not_test").status, ItemStatus::Active);
    assert_eq!(by_name("unixy").status, ItemStatus::Active);
    // cfg_attr fires: serde on -> requires cfg(test), test off -> disabled.
    assert_eq!(by_name("attr_gated").status, ItemStatus::CfgDisabled);
    assert_eq!(by_name("Gated").status, ItemStatus::CfgDisabled);
    // `cfg!(...)` inside a fn body is a macro expression; it must not
    // gate the enclosing item.
    assert_eq!(by_name("uses_cfg_macro").status, ItemStatus::Active);

    let off = project(&cst, &CfgContext::empty());
    let by_name = |name: &str| {
        off.items
            .iter()
            .find(|i| i.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    assert_eq!(by_name("with_serde").status, ItemStatus::CfgDisabled);
    assert_eq!(by_name("unix_not_test").status, ItemStatus::CfgDisabled);
    assert_eq!(by_name("unixy").status, ItemStatus::CfgDisabled);
    // cfg_attr condition false -> no constraint -> active.
    assert_eq!(by_name("attr_gated").status, ItemStatus::Active);
    // cfg-disabled items are still indexed (with their spans), just
    // marked not-live.
    assert!(!by_name("with_serde").span.is_empty());

    let full = project(
        &cst,
        &ctx(&["test"], &[("feature", "serde"), ("target_os", "macos")]),
    );
    let gated = full
        .items
        .iter()
        .find(|i| i.name.as_deref() == Some("attr_gated"))
        .map(|i| i.status);
    assert_eq!(gated, Some(ItemStatus::Active));
}

#[test]
fn cfg_value_with_commas_and_parens_stays_intact() {
    let src = "#[cfg(feature = \"a,b(c)\")]\nfn f() {}\n";
    let cst = parse_top_level(src);
    let idx = project(&cst, &ctx(&[], &[("feature", "a,b(c)")]));
    assert_eq!(idx.items[0].status, ItemStatus::Active);
    let idx = project(&cst, &ctx(&[], &[("feature", "a")]));
    assert_eq!(idx.items[0].status, ItemStatus::CfgDisabled);
}

#[test]
fn macro_call_items_are_opaque_even_when_cfg_enabled() {
    let src = "#[cfg(unix)]\nsome_macro! { fn fake() {} struct Fake; }\n";
    let cst = parse_top_level(src);
    let idx = project(&cst, &ctx(&["unix"], &[]));
    assert_eq!(idx.items.len(), 1);
    let item = &idx.items[0];
    assert_eq!(item.kind, omp_ladder::ItemKind::MacroCall);
    assert_eq!(item.status, ItemStatus::Opaque);
    // cfg-disabled macro call is still opaque, not "safely parsed".
    let idx = project(&cst, &CfgContext::empty());
    assert_eq!(idx.items[0].status, ItemStatus::Opaque);
    assert!(!idx.items[0].cfg_enabled);
}
