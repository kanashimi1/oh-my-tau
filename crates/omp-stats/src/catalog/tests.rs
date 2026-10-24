use super::*;

/// Unix-ms timestamp for a weekday (Sunday = 0) at `hour:minute` UTC.
fn ts_for(weekday: u8, hour: i64, minute: i64) -> i64 {
    // Day 19723 was a Monday (2024-01-01).
    let day = 19_723 + (weekday as i64 + 6).rem_euclid(7);
    day * 86_400_000 + hour * 3_600_000 + minute * 60_000
}

#[test]
fn plain_cost() {
    let cost = ModelCost {
        base: TokenCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        },
        long_context: None,
        time_based: None,
    };
    let mut usage = Usage {
        input: 1_000,
        output: 500,
        cache_read: 2_000,
        cache_write: 400,
        ..Default::default()
    };
    let c = calculate_usage_cost(&cost, &mut usage, None);
    assert!((c.input - 0.003).abs() < 1e-12);
    assert!((c.output - 0.0075).abs() < 1e-12);
    assert!((c.cache_read - 0.0006).abs() < 1e-12);
    assert!((c.cache_write - 0.0015).abs() < 1e-12);
    assert!((c.total - (0.003 + 0.0075 + 0.0006 + 0.0015)).abs() < 1e-12);
    assert_eq!(usage.cost, c);
}

#[test]
fn long_context_tier_switch() {
    let cost = ModelCost {
        base: TokenCost {
            input: 0.1,
            output: 0.5,
            cache_read: 0.01,
            cache_write: 0.125,
        },
        long_context: Some(LongContextTokenCost {
            base: TokenCost {
                input: 0.5,
                output: 2.5,
                cache_read: 0.05,
                cache_write: 0.625,
            },
            input_threshold: 100_000,
            input_threshold_inclusive: None,
        }),
        time_based: None,
    };
    // Below threshold: base rates (0.1/M * 99_999).
    assert!((calculate_uncached_input_cost(&cost, 99_999, None) - 0.009_999_9).abs() < 1e-9);
    // Above threshold: long-context rates (0.5/M * 200_000).
    assert!((calculate_uncached_input_cost(&cost, 200_000, None) - 0.1).abs() < 1e-9);
    // Exactly at threshold without inclusive: still base (0.1/M * 100_000).
    assert!((calculate_uncached_input_cost(&cost, 100_000, None) - 0.01).abs() < 1e-9);
}

#[test]
fn cache_write_with_cttl() {
    let cost = ModelCost {
        base: TokenCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        },
        long_context: None,
        time_based: None,
    };
    let mut usage = Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 1_000,
        cttl: Some(CttlUsage {
            ephemeral5m: Some(600),
            ephemeral1h: Some(400),
        }),
        ..Default::default()
    };
    let c = calculate_usage_cost(&cost, &mut usage, None);
    // 600 @ 3.75/M + 400 @ (3*2)/M = 0.00225 + 0.0024
    assert!((c.cache_write - (0.00225 + 0.0024)).abs() < 1e-12);
    // Residual (breakdown under-sums) bills at the flat 5m rate.
    let mut usage2 = Usage {
        cache_write: 1_200,
        cttl: Some(CttlUsage {
            ephemeral5m: Some(500),
            ephemeral1h: Some(300),
        }),
        ..Default::default()
    };
    let c2 = calculate_usage_cost(&cost, &mut usage2, None);
    // 500 + residual 400 at 3.75/M, 300 at 6/M
    assert!((c2.cache_write - ((900.0 * 3.75 + 300.0 * 6.0) / 1_000_000.0)).abs() < 1e-12);
}

#[test]
fn time_based_off_peak_and_peak() {
    let cost = ModelCost {
        base: TokenCost {
            input: 0.3,
            output: 1.2,
            cache_read: 0.006,
            cache_write: 0.0,
        },
        long_context: None,
        time_based: Some(TimeBasedCost {
            off_peak_multiplier: 0.5,
            peak_windows: vec![
                // Mon-Fri 01:00-04:00 UTC
                PeakPricingWindow {
                    weekdays: vec![1, 2, 3, 4, 5],
                    start_minute: 60,
                    end_minute: 240,
                },
                // Mon-Fri 06:00-10:00 UTC
                PeakPricingWindow {
                    weekdays: vec![1, 2, 3, 4, 5],
                    start_minute: 360,
                    end_minute: 600,
                },
            ],
            effective_rates: None,
        }),
    };
    // Monday 02:00 is peak -> multiplier 1: 0.3/M * 1M = 0.3.
    let peak_ts = ts_for(1, 2, 0);
    assert!(is_peak_pricing_period(
        cost.time_based.as_ref().unwrap(),
        peak_ts
    ));
    assert!((calculate_uncached_input_cost(&cost, 1_000_000, Some(peak_ts)) - 0.3).abs() < 1e-9);
    // Sunday 02:00 is off-peak -> multiplier 0.5: 0.3 * 0.5 = 0.15.
    let off_ts = ts_for(0, 2, 0);
    assert!(!is_peak_pricing_period(
        cost.time_based.as_ref().unwrap(),
        off_ts
    ));
    assert!((calculate_uncached_input_cost(&cost, 1_000_000, Some(off_ts)) - 0.15).abs() < 1e-9);
}

#[test]
fn catalog_cost_fallback() {
    // xai-oauth has no grok-2 row; the xai fallback is billable.
    assert!(bundled_model_cost("xai-oauth", "grok-2").is_none());
    let cost = get_catalog_cost("xai-oauth", "grok-2").expect("fallback to xai");
    assert!((cost.input() - 2.0).abs() < 1e-12);
    assert!((cost.output() - 10.0).abs() < 1e-12);
    // Unknown provider/model stays null.
    assert!(get_catalog_cost("nope", "nope").is_none());
}

#[test]
fn bundled_anthropic_sonnet() {
    let cost = bundled_model_cost("anthropic", "claude-sonnet-4-5").expect("row exists");
    assert!((cost.input() - 3.0).abs() < 1e-12);
    assert!((cost.output() - 15.0).abs() < 1e-12);
    assert!((cost.cache_read() - 0.3).abs() < 1e-12);
    assert!((cost.cache_write() - 3.75).abs() < 1e-12);
    assert!(cost.time_based.is_none());
    assert!(has_billable_cost(&cost));
    assert!(!is_scheduled_catalog_model(
        "anthropic",
        "claude-sonnet-4-5"
    ));
    // Prices a real request from the bundled card: 1k in + 500 out.
    let mut usage = Usage {
        input: 1_000,
        output: 500,
        ..Default::default()
    };
    let c = calculate_usage_cost(&cost, &mut usage, None);
    assert!((c.total - (1_000.0 * 3.0 + 500.0 * 15.0) / 1_000_000.0).abs() < 1e-12);
}

#[test]
fn unpriced_sql_matches_ts() {
    let s = unpriced_request_sql("");
    assert!(s.contains("total_tokens > 0"));
    assert!(s.contains("cost_total = 0"));
    assert!(s.contains("provider = 'xai-oauth'"));
    assert!(s.contains("cost_unpriced = 1"));
    let p = unpriced_request_sql("m.");
    assert!(p.contains("m.total_tokens"));
    assert!(p.contains("m.cost_unpriced"));
}

#[test]
fn scheduled_catalog_detection() {
    assert!(is_scheduled_catalog_model("deepseek", "deepseek-flash"));
    assert!(!is_scheduled_catalog_model(
        "anthropic",
        "claude-sonnet-4-5"
    ));
}
