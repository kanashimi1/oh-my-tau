//! Host/Origin guard (hardening beyond TS): a loopback dashboard answers only
//! requests addressed to itself, which defeats DNS-rebinding and cross-site
//! reads of `/api/*`. An explicit non-loopback `--host` opts out (the user
//! chose to expose it), as the legitimate `Host` values are then unknowable.

use std::net::IpAddr;

use hyper::header::{HOST, ORIGIN};
use hyper::HeaderMap;

use crate::port_conflict::normalize_host;

fn is_loopback(host: &str) -> bool {
    let host = normalize_host(host);
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// `host[:port]` authority with IPv6 literals bracketed.
fn authority(host: &str, port: u16) -> String {
    let host = normalize_host(host);
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// Whether the dashboard bound to `hostname` enforces the guard.
#[must_use]
pub(super) fn enforced(hostname: &str) -> bool {
    is_loopback(hostname)
}

/// `Err(reason)` (answer 403) unless `Host` ∈ {configured hostname, localhost,
/// 127.0.0.1, [::1]}`:port` and any `Origin` is `http://` one of those.
pub(super) fn check(headers: &HeaderMap, hostname: &str, port: u16) -> Result<(), &'static str> {
    let allowed: Vec<String> = [hostname, "localhost", "127.0.0.1", "::1"]
        .iter()
        .map(|h| authority(h, port))
        .collect();
    let is_allowed = |candidate: &str| allowed.iter().any(|a| a.eq_ignore_ascii_case(candidate));
    let host = headers
        .get(HOST)
        .and_then(|v| v.to_str().ok())
        .ok_or("Forbidden: missing Host header")?;
    if !is_allowed(host) {
        return Err("Forbidden: unexpected Host header");
    }
    if let Some(origin) = headers.get(ORIGIN) {
        let origin = origin
            .to_str()
            .ok()
            .and_then(|o| o.strip_prefix("http://"))
            .ok_or("Forbidden: unexpected Origin")?;
        if !is_allowed(origin) {
            return Err("Forbidden: unexpected Origin");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, value.parse().expect("header value"));
        }
        map
    }

    #[test]
    fn loopback_hosts_with_the_bound_port_pass() {
        for host in [
            "localhost:3847",
            "127.0.0.1:3847",
            "[::1]:3847",
            "LOCALHOST:3847",
        ] {
            assert!(
                check(&headers(&[("host", host)]), "127.0.0.1", 3847).is_ok(),
                "{host}"
            );
        }
        let ok = headers(&[
            ("host", "localhost:3847"),
            ("origin", "http://127.0.0.1:3847"),
        ]);
        assert!(check(&ok, "127.0.0.1", 3847).is_ok());
        assert!(check(&headers(&[("host", "[::1]:3847")]), "::1", 3847).is_ok());
        assert!(check(&headers(&[("host", "[::1]:3847")]), "[::1]", 3847).is_ok());
        assert!(check(
            &headers(&[("host", "[::1]:3847"), ("origin", "http://[::1]:3847")]),
            "[::1]",
            3847
        )
        .is_ok());
    }

    #[test]
    fn bracketed_and_bare_ipv6_guard_the_same_authorities() {
        for hostname in ["[::1]", "::1"] {
            assert!(
                check(&headers(&[("host", "[::1]:3847")]), hostname, 3847).is_ok(),
                "{hostname}"
            );
            assert!(
                check(&headers(&[("host", "[::1]:1")]), hostname, 3847).is_err(),
                "{hostname} wrong port"
            );
            assert!(
                check(&headers(&[("host", "[::2]:3847")]), hostname, 3847).is_err(),
                "{hostname} other ip"
            );
        }
        assert!(enforced("[::1]") == enforced("::1"));
    }

    #[test]
    fn foreign_host_origin_port_or_missing_host_are_rejected() {
        let bad =
            |pairs: &[(&'static str, &str)]| check(&headers(pairs), "127.0.0.1", 3847).is_err();
        assert!(bad(&[("host", "evil.example:3847")]), "DNS rebinding host");
        assert!(bad(&[("host", "localhost")]), "port required");
        assert!(bad(&[("host", "localhost:1")]), "wrong port");
        assert!(bad(&[]), "no host");
        assert!(
            bad(&[
                ("host", "localhost:3847"),
                ("origin", "http://evil.example")
            ]),
            "foreign origin"
        );
        assert!(
            bad(&[("host", "localhost:3847"), ("origin", "null")]),
            "null origin"
        );
        assert!(
            bad(&[
                ("host", "localhost:3847"),
                ("origin", "https://localhost:3847")
            ]),
            "scheme"
        );
    }

    #[test]
    fn only_explicit_non_loopback_hosts_skip_the_guard() {
        assert!(
            enforced("127.0.0.1") && enforced("localhost") && enforced("::1") && enforced("[::1]")
        );
        assert!(!enforced("0.0.0.0") && !enforced("192.168.1.5") && !enforced("::"));
    }
}
