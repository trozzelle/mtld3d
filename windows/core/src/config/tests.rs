//! Unit tests for the `mtld3d.conf` parser.
//!
//! `parse` is driven with file text plus optional env-override text and the result checked
//! field by field: documented defaults, both values of every boolean, later assignments and
//! env segments winning in that order, and malformed or unknown entries keeping the previous
//! value instead of derailing the rest of the input. Per-value shapes (MiB caps, decimal
//! render scale and its bounds, hex id lists, quoted strings) get their own cases.
//!
//! Every case here passes no app profile; the profile layer and its precedence against these
//! two live in `app_profile/tests.rs`, where a profile can be built.

use mtld3d_shared::mtl::{ColorSpacePolicy, SoftwareCursorPolicy};

use super::{AdapterSpoof, CursorScale, Mtld3dConfig, parse};

#[test]
fn empty_input_returns_defaults() {
    assert_eq!(parse(None, "", None), Mtld3dConfig::default());
}

#[test]
fn defaults_match_documented_values() {
    let d = Mtld3dConfig::default();
    assert!(!d.caps_all);
    assert!(!d.main_thread_checker);
    assert!(!d.fail_next_submit);
    assert!(!d.expand_packed16);
    assert!(!d.deny_float32_filtering);
    assert!(!d.managed_memory);
    assert!(!d.linear_align256);
    assert!(d.hdr_enable);
    assert_eq!(d.color_space, ColorSpacePolicy::Passthrough);
    assert_eq!(d.cursor_scale, CursorScale::Auto);
    assert_eq!(d.cursor_software, SoftwareCursorPolicy::Off);
    assert!(d.shader_cache_enable);
    assert!(d.shader_async_compile);
    assert_eq!(d.log_dir, "");
    assert_eq!(d.bytecode_dump_dir, "");
    assert_eq!(d.skip_shaders, [] as [u64; 0]);
    assert_eq!(d.present_gate_file, "");
    assert!(!d.present_occluded);
    assert!(!d.query_flush_immediate);
    assert!(!d.query_event_immediate);
    assert!(!d.buffer_ignore_lock_bounds);
    assert_eq!(d.vbib_retention_cap_bytes, 512 * 1024 * 1024);
    assert_eq!(d.pagebox_pool_cap_bytes, 128 * 1024 * 1024);
    assert_eq!(d.present_max_fps, 0);
    assert_eq!(d.present_max_frame_latency, 0, "auto");
    assert_eq!(d.render_scale_percent, 100);
}

/// The advertised video-memory ceiling defaults by guest width.
///
/// A 32-bit guest runs out of process address space long before a
/// unified-memory Mac runs out of GPU memory, and an engine that sizes its
/// streaming pool from the advertised figure will commit until it does.
#[test]
fn vram_budget_defaults_to_a_ceiling_only_on_a_32_bit_guest() {
    let d = parse(None, "", None);
    if cfg!(target_pointer_width = "32") {
        assert_eq!(d.vram_budget_cap_bytes, 1024 * 1024 * 1024);
    } else {
        assert_eq!(
            d.vram_budget_cap_bytes, 0,
            "a 64-bit guest keeps no ceiling"
        );
    }
}

#[test]
fn vram_budget_parses_as_mib_and_zero_lifts_the_ceiling() {
    assert_eq!(
        parse(None, "memory.vramBudgetMB = 512\n", None).vram_budget_cap_bytes,
        512 * 1024 * 1024
    );
    assert_eq!(
        parse(None, "memory.vramBudgetMB = 0\n", None).vram_budget_cap_bytes,
        0
    );
}

#[test]
fn pagebox_pool_cap_parses_as_mib() {
    let cfg = parse(None, "memory.pageboxPoolCapMB = 96\n", None);
    assert_eq!(cfg.pagebox_pool_cap_bytes, 96 * 1024 * 1024);
}

#[test]
fn pagebox_pool_cap_zero_disables() {
    let cfg = parse(
        None,
        "memory.pageboxPoolCapMB = 96\nmemory.pageboxPoolCapMB = 0\n",
        None,
    );
    assert_eq!(cfg.pagebox_pool_cap_bytes, 0);
}

#[test]
fn pagebox_pool_cap_garbage_keeps_default() {
    let cfg = parse(None, "memory.pageboxPoolCapMB = lots\n", None);
    assert_eq!(cfg.pagebox_pool_cap_bytes, 128 * 1024 * 1024);
}

#[test]
fn present_max_fps_positive_integer_parses() {
    let cfg = parse(None, "present.maxFps = 60\n", None);
    assert_eq!(cfg.present_max_fps, 60);
}

#[test]
fn present_max_fps_zero_means_uncapped() {
    let cfg = parse(None, "present.maxFps = 60\npresent.maxFps = 0\n", None);
    assert_eq!(cfg.present_max_fps, 0);
}

#[test]
fn present_max_fps_garbage_keeps_default() {
    let cfg = parse(None, "present.maxFps = fast\n", None);
    assert_eq!(cfg.present_max_fps, 0);
}

#[test]
fn present_max_frame_latency_takes_a_frame_count() {
    assert_eq!(
        parse(None, "present.maxFrameLatency = 1\n", None).present_max_frame_latency,
        1
    );
    assert_eq!(
        parse(None, "present.maxFrameLatency = 30\n", None).present_max_frame_latency,
        30
    );
}

#[test]
fn present_max_frame_latency_auto_follows_the_game() {
    let cfg = parse(
        None,
        "present.maxFrameLatency = 2\npresent.maxFrameLatency = AUTO\n",
        None,
    );
    assert_eq!(cfg.present_max_frame_latency, 0);
}

#[test]
fn present_max_frame_latency_out_of_range_or_garbage_keeps_the_value() {
    for invalid in ["0", "31", "-1", "low"] {
        let body = format!("present.maxFrameLatency = 4\npresent.maxFrameLatency = {invalid}\n");
        assert_eq!(
            parse(None, &body, None).present_max_frame_latency,
            4,
            "{invalid} is refused"
        );
    }
}

#[test]
fn render_scale_float_becomes_a_percentage() {
    assert_eq!(
        parse(None, "render.scale = 0.5\n", None).render_scale_percent,
        50
    );
    assert_eq!(
        parse(None, "render.scale = 0.75\n", None).render_scale_percent,
        75
    );
    assert_eq!(
        parse(None, "render.scale = 1\n", None).render_scale_percent,
        100
    );
}

#[test]
fn render_scale_accepts_its_exact_bounds() {
    assert_eq!(
        parse(None, "render.scale = 0.01\n", None).render_scale_percent,
        1
    );
    assert_eq!(
        parse(None, "render.scale = 1.0\n", None).render_scale_percent,
        100
    );
}

#[test]
fn render_scale_out_of_range_keeps_default() {
    for src in [
        // Above 1.0 has no path home: the present-side scaler only
        // enlarges, so a render bigger than the drawable cannot resolve.
        "render.scale = 1.5\n",
        "render.scale = 2.5\n",
        "render.scale = 0\n",
        "render.scale = -1\n",
        // Far outside u32: must be rejected, not saturated into range.
        "render.scale = 1e20\n",
        "render.scale = inf\n",
        "render.scale = NaN\n",
        "render.scale = half\n",
    ] {
        assert_eq!(
            parse(None, src, None).render_scale_percent,
            100,
            "{src:?} must keep the default"
        );
    }
}

#[test]
fn query_flush_immediate_round_trips_false() {
    let cfg = parse(None, "query.flushImmediate = false\n", None);
    assert!(!cfg.query_flush_immediate);
}

#[test]
fn query_flush_immediate_round_trips_true() {
    let cfg = parse(None, "query.flushImmediate = true\n", None);
    assert!(cfg.query_flush_immediate);
}

#[test]
fn query_event_immediate_round_trips_false() {
    let cfg = parse(None, "query.eventImmediate = false\n", None);
    assert!(!cfg.query_event_immediate);
}

#[test]
fn query_event_immediate_round_trips_true() {
    let cfg = parse(None, "query.eventImmediate = true\n", None);
    assert!(cfg.query_event_immediate);
}

#[test]
fn depth_alias_same_size_defaults_off_and_round_trips() {
    let cfg = parse(None, "", None);
    assert!(!cfg.depth_alias_same_size);
    let cfg = parse(None, "depth.aliasSameSize = true\n", None);
    assert!(cfg.depth_alias_same_size);
}

#[test]
fn buffer_ignore_lock_bounds_defaults_off_and_round_trips() {
    let cfg = parse(None, "", None);
    assert!(!cfg.buffer_ignore_lock_bounds);
    let cfg = parse(None, "buffer.ignoreLockBounds = true\n", None);
    assert!(cfg.buffer_ignore_lock_bounds);
    let cfg = parse(None, "buffer.ignoreLockBounds = false\n", None);
    assert!(!cfg.buffer_ignore_lock_bounds);
}

#[test]
fn cursor_scale_auto_keyword_case_insensitive() {
    let cfg = parse(None, "cursor.scale = Auto\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Auto);
}

#[test]
fn cursor_scale_positive_integer_parses_to_fixed() {
    let cfg = parse(None, "cursor.scale = 3\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Fixed(3));
}

#[test]
fn cursor_scale_zero_keeps_default() {
    let cfg = parse(None, "cursor.scale = 0\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Auto);
}

#[test]
fn cursor_scale_garbage_keeps_default() {
    let cfg = parse(None, "cursor.scale = jumbo\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Auto);
}

#[test]
fn cursor_scale_auto_follows_the_display() {
    // Resolved again on every display move, so both readings of the same
    // policy have to answer for the display they were handed.
    assert_eq!(CursorScale::Auto.resolve(1), 1);
    assert_eq!(CursorScale::Auto.resolve(2), 2);
}

#[test]
fn cursor_scale_override_ignores_the_display() {
    assert_eq!(CursorScale::Fixed(3).resolve(1), 3);
    assert_eq!(CursorScale::Fixed(3).resolve(2), 3);
}

#[test]
fn cursor_scale_resolves_into_the_hcursor_range() {
    assert_eq!(CursorScale::Auto.resolve(0), 1);
    assert_eq!(CursorScale::Auto.resolve(99), 8);
    assert_eq!(CursorScale::Fixed(0).resolve(2), 1);
    assert_eq!(CursorScale::Fixed(99).resolve(2), 8);
}

#[test]
fn cursor_software_accepts_all_three_values_case_insensitive() {
    let cfg = parse(None, "cursor.software = TRUE\n", None);
    assert_eq!(cfg.cursor_software, SoftwareCursorPolicy::On);

    let cfg = parse(None, "cursor.software = Auto\n", None);
    assert_eq!(cfg.cursor_software, SoftwareCursorPolicy::Auto);

    let cfg = parse(
        None,
        "cursor.software = true\ncursor.software = False\n",
        None,
    );
    assert_eq!(cfg.cursor_software, SoftwareCursorPolicy::Off);
}

#[test]
fn cursor_software_garbage_keeps_default() {
    let cfg = parse(None, "cursor.software = maybe\n", None);
    assert_eq!(cfg.cursor_software, SoftwareCursorPolicy::Off);
}

#[test]
fn cursor_software_auto_follows_the_hdr_path() {
    assert!(SoftwareCursorPolicy::Auto.resolve(true));
    assert!(!SoftwareCursorPolicy::Auto.resolve(false));
}

#[test]
fn cursor_software_overrides_ignore_the_hdr_path() {
    assert!(SoftwareCursorPolicy::On.resolve(false));
    assert!(SoftwareCursorPolicy::On.resolve(true));
    assert!(!SoftwareCursorPolicy::Off.resolve(false));
    assert!(!SoftwareCursorPolicy::Off.resolve(true));
}

#[test]
fn color_space_accepts_both_policies_case_insensitive() {
    let cfg = parse(None, "color.space = accurate\n", None);
    assert_eq!(cfg.color_space, ColorSpacePolicy::Accurate);

    let cfg = parse(None, "color.space = PASSTHROUGH\n", None);
    assert_eq!(cfg.color_space, ColorSpacePolicy::Passthrough);
}

#[test]
fn color_space_unknown_value_keeps_default() {
    let cfg = parse(None, "color.space = vivid\n", None);
    assert_eq!(cfg.color_space, ColorSpacePolicy::Passthrough);
}

#[test]
fn comments_and_blank_lines_are_skipped() {
    let src = "\
        # comment\n\
        \n\
        \t  \n\
        color.hdr.enable = true\n\
        # debug.capsAll = true\n\
    ";
    let cfg = parse(None, src, None);
    assert!(cfg.hdr_enable);
    assert!(!cfg.caps_all);
}

#[test]
fn main_thread_checker_defaults_off_and_parses_on() {
    assert!(!parse(None, "", None).main_thread_checker, "off by default");
    assert!(parse(None, "debug.mainThreadChecker = true\n", None).main_thread_checker);
    assert!(!parse(None, "debug.mainThreadChecker = false\n", None).main_thread_checker);
    assert!(
        !parse(None, "debug.mainThreadChecker = maybe\n", None).main_thread_checker,
        "an unparsable value keeps the default"
    );
    assert!(
        parse(None, "", Some("debug.mainThreadChecker=true")).main_thread_checker,
        "the env override reaches the key"
    );
}

#[test]
fn fail_next_submit_defaults_off_and_parses_on() {
    assert!(!parse(None, "", None).fail_next_submit, "off by default");
    assert!(parse(None, "debug.failNextSubmit = true\n", None).fail_next_submit);
    assert!(!parse(None, "debug.failNextSubmit = false\n", None).fail_next_submit);
    assert!(
        !parse(None, "debug.failNextSubmit = maybe\n", None).fail_next_submit,
        "an unparsable value keeps the default"
    );
    assert!(
        parse(None, "", Some("debug.failNextSubmit=true")).fail_next_submit,
        "the env override reaches the key"
    );
}

#[test]
fn render_lod_bias_defaults_on_and_parses_off() {
    assert!(parse(None, "", None).render_lod_bias, "on by default");
    assert!(!parse(None, "render.lodBias = false\n", None).render_lod_bias);
    assert!(parse(None, "render.lodBias = true\n", None).render_lod_bias);
    assert!(
        parse(None, "render.lodBias = maybe\n", None).render_lod_bias,
        "an unparsable value keeps the default"
    );
}

#[test]
fn boolean_keys_round_trip_both_values() {
    let cfg = parse(
        None,
        "debug.capsAll = true\ncolor.hdr.enable = false\nshaderCache.enable = false\n\
         intel.expandPacked16 = true\nintel.denyFloat32Filtering = true\n\
         intel.managedMemory = true\nintel.linearAlign256 = true\n\
         debug.mainThreadChecker = true\nshader.asyncCompile = false\n",
        None,
    );
    assert!(cfg.caps_all);
    assert!(cfg.main_thread_checker);
    assert!(!cfg.hdr_enable);
    assert!(!cfg.shader_cache_enable);
    assert!(!cfg.shader_async_compile);
    assert!(cfg.expand_packed16);
    assert!(cfg.deny_float32_filtering);
    assert!(cfg.managed_memory);
    assert!(cfg.linear_align256);
}

#[test]
fn retired_debug_keys_are_ignored() {
    // The two keys moved into the `intel.*` family; their old names are
    // unknown keys now and must not reach the fields they used to set.
    let cfg = parse(
        None,
        "debug.expandPacked16 = true\ndebug.float32Filtering = false\n",
        None,
    );
    assert_eq!(cfg, Mtld3dConfig::default());
}

#[test]
fn booleans_are_case_insensitive() {
    let cfg = parse(
        None,
        "debug.capsAll = TRUE\ncolor.hdr.enable = False\n",
        None,
    );
    assert!(cfg.caps_all);
    assert!(!cfg.hdr_enable);
}

#[test]
fn whitespace_around_equals_is_tolerated() {
    let cfg = parse(
        None,
        "  debug.capsAll=true  \ncolor.hdr.enable\t=\ttrue\n",
        None,
    );
    assert!(cfg.caps_all);
    assert!(cfg.hdr_enable);
}

#[test]
fn quoted_string_value_preserves_inner_whitespace() {
    let cfg = parse(None, "debug.bytecodeDumpDir = \" /tmp/x \"\n", None);
    assert_eq!(cfg.bytecode_dump_dir, " /tmp/x ");
}

#[test]
fn unquoted_string_value_is_trimmed() {
    let cfg = parse(None, "debug.bytecodeDumpDir = /tmp/x\n", None);
    assert_eq!(cfg.bytecode_dump_dir, "/tmp/x");
}

#[test]
fn log_dir_is_a_plain_string() {
    let cfg = parse(None, "log.dir = logs\\mtld3d\n", None);
    assert_eq!(cfg.log_dir, "logs\\mtld3d");
    let cfg = parse(None, "", Some("log.dir=C:\\mtld3d-logs"));
    assert_eq!(cfg.log_dir, "C:\\mtld3d-logs");
}

#[test]
fn present_gate_file_is_a_plain_string() {
    let cfg = parse(
        None,
        "debug.presentGateFile = C:\\users\\x\\Temp\\gate\n",
        None,
    );
    assert_eq!(cfg.present_gate_file, "C:\\users\\x\\Temp\\gate");
    let cfg = parse(None, "", Some("debug.presentGateFile=gate.txt"));
    assert_eq!(
        cfg.present_gate_file, "gate.txt",
        "the env override reaches the key"
    );
    let cfg = parse(None, "debug.presentGateFile =\n", None);
    assert!(cfg.present_gate_file.is_empty(), "empty disables the gate");
}

#[test]
fn present_occluded_defaults_off_and_parses_on() {
    assert!(!parse(None, "", None).present_occluded, "off by default");
    assert!(parse(None, "debug.presentOccluded = true\n", None).present_occluded);
    assert!(!parse(None, "debug.presentOccluded = false\n", None).present_occluded);
    assert!(
        !parse(None, "debug.presentOccluded = maybe\n", None).present_occluded,
        "an unparsable value keeps the default"
    );
    assert!(
        parse(None, "", Some("debug.presentOccluded=true")).present_occluded,
        "the env override reaches the key"
    );
}

#[test]
fn empty_string_disables_bytecode_dump() {
    let cfg = parse(None, "debug.bytecodeDumpDir =\n", None);
    assert_eq!(cfg.bytecode_dump_dir, "");
}

#[test]
fn hex_list_parses_with_and_without_0x_prefix() {
    let cfg = parse(None, "debug.skipShaders = 0xabc, def, 0x12345\n", None);
    assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef, 0x1_2345]);
}

#[test]
fn hex_list_drops_unparseable_entries_silently() {
    let cfg = parse(None, "debug.skipShaders = abc, gggg, def,, , 0\n", None);
    assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef, 0]);
}

#[test]
fn unknown_key_does_not_corrupt_other_assignments() {
    let cfg = parse(None, "bogusKey = whatever\ncolor.hdr.enable = true\n", None);
    assert!(cfg.hdr_enable);
}

#[test]
fn missing_equals_line_is_skipped() {
    let cfg = parse(
        None,
        "not a key value pair\ncolor.hdr.enable = true\n",
        None,
    );
    assert!(cfg.hdr_enable);
}

#[test]
fn non_boolean_value_keeps_default() {
    // Canary is a key that defaults to `false`, so a parser that
    // wrongly assigned `true` on garbage would fail here; with a
    // `true`-defaulting key the two outcomes are indistinguishable.
    let cfg = parse(None, "debug.capsAll = maybe\n", None);
    assert!(!cfg.caps_all, "default must be preserved");
}

#[test]
fn later_assignment_wins() {
    let cfg = parse(None, "debug.capsAll = false\ndebug.capsAll = true\n", None);
    assert!(cfg.caps_all);
}

#[test]
fn env_override_after_file_wins() {
    let cfg = parse(
        None,
        "color.hdr.enable = false\n",
        Some("color.hdr.enable=true"),
    );
    assert!(cfg.hdr_enable);
}

#[test]
fn env_override_merges_with_file() {
    let cfg = parse(None, "cursor.scale = 2\n", Some("color.hdr.enable=true"));
    assert_eq!(cfg.cursor_scale, CursorScale::Fixed(2));
    assert!(cfg.hdr_enable);
}

#[test]
fn env_override_supports_all_keys() {
    let env = "debug.capsAll=true\
        ;color.hdr.enable=true\
        ;color.space=accurate\
        ;cursor.scale=4\
        ;shaderCache.enable=false\
        ;debug.bytecodeDumpDir=/tmp/x\
        ;debug.skipShaders=0xabc,def\
        ;query.flushImmediate=false\
        ;query.eventImmediate=true\
        ;memory.vbibRetentionCapMB=256\
        ;memory.pageboxPoolCapMB=96\
        ;present.maxFps=72\
        ;render.scale=0.5";
    let cfg = parse(None, "", Some(env));
    assert!(cfg.caps_all);
    assert!(cfg.hdr_enable);
    assert_eq!(cfg.color_space, ColorSpacePolicy::Accurate);
    assert_eq!(cfg.cursor_scale, CursorScale::Fixed(4));
    assert!(!cfg.shader_cache_enable);
    assert_eq!(cfg.bytecode_dump_dir, "/tmp/x");
    assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef]);
    assert!(!cfg.query_flush_immediate);
    assert!(cfg.query_event_immediate);
    assert_eq!(cfg.vbib_retention_cap_bytes, 256 * 1024 * 1024);
    assert_eq!(cfg.pagebox_pool_cap_bytes, 96 * 1024 * 1024);
    assert_eq!(cfg.present_max_fps, 72);
    assert_eq!(cfg.render_scale_percent, 50);
}

#[test]
fn env_override_empty_segments_skipped() {
    let cfg = parse(None, "", Some(";;color.hdr.enable=true;;"));
    assert!(cfg.hdr_enable);
}

#[test]
fn env_override_lists_keep_comma_separator() {
    let from_file = parse(None, "debug.skipShaders = 0xabc, 0xdef\n", None);
    let from_env = parse(None, "", Some("debug.skipShaders=0xabc,0xdef"));
    assert_eq!(from_file.skip_shaders, from_env.skip_shaders);
    assert_eq!(from_env.skip_shaders, vec![0xabc, 0xdef]);
}

#[test]
fn env_override_unknown_key_keeps_other_assignments() {
    let cfg = parse(None, "", Some("bogus.key=foo;color.hdr.enable=true"));
    assert!(cfg.hdr_enable);
}

#[test]
fn env_override_none_matches_file_only() {
    let src = "debug.capsAll = true\ncursor.scale = 3\n";
    assert_eq!(parse(None, src, None), parse(None, src, Some("")));
}

#[test]
fn adapter_spoof_parses_all_values() {
    assert_eq!(
        parse(None, "adapter.spoof = nvidia\n", None).adapter_spoof,
        AdapterSpoof::Nvidia
    );
    assert_eq!(
        parse(None, "adapter.spoof = AMD\n", None).adapter_spoof,
        AdapterSpoof::Amd
    );
    assert_eq!(
        parse(None, "adapter.spoof = ati\n", None).adapter_spoof,
        AdapterSpoof::Amd
    );
    assert_eq!(
        parse(None, "adapter.spoof = none\n", None).adapter_spoof,
        AdapterSpoof::None
    );
}

#[test]
fn adapter_spoof_rejects_unknown_vendor() {
    let cfg = parse(None, "adapter.spoof = matrox\n", None);
    assert_eq!(cfg.adapter_spoof, AdapterSpoof::None, "default preserved");
}

#[test]
fn df_formats_defaults_on_and_parses_off() {
    assert!(parse(None, "", None).df_formats);
    assert!(!parse(None, "caps.dfFormats = false\n", None).df_formats);
}

#[test]
fn parse_ppl_keys() {
    use crate::ppl::{PplConfig, PplFalloff, PplMode, PplModel};
    let cfg = parse(
        None,
        "lighting.perPixel = points\nlighting.perPixelMixed = true\n\
         lighting.perPixelModel = halflambert\nlighting.perPixelFalloff = smooth\n\
         lighting.perPixelDebug = term\n",
        None,
    );
    assert_eq!(
        cfg.ppl,
        PplConfig {
            mode: PplMode::Points,
            mixed: true,
            model: PplModel::HalfLambert,
            falloff: PplFalloff::Smooth,
            debug_term: true,
            shoulder_pct: 0,
            figures: false,
            ..PplConfig::default()
        }
    );
}

#[test]
fn parse_ppl_figures() {
    let cfg = parse(None, "lighting.perPixelFigures = true\n", None);
    assert!(cfg.ppl.figures);
    let cfg = parse(None, "lighting.perPixelFigures = false\n", None);
    assert!(!cfg.ppl.figures);
    for bad in ["yes", "1", "on"] {
        let cfg = parse(None, &format!("lighting.perPixelFigures = {bad}\n"), None);
        assert_eq!(cfg.ppl, crate::ppl::PplConfig::default(), "{bad}");
    }
}

#[test]
fn parse_fog_figures_key() {
    assert!(!Mtld3dConfig::default().fog.figures);
    let cfg = parse(None, "fog.figures = on\n", None);
    assert!(cfg.fog.figures);
    let cfg = parse(None, "fog.figures = on\nfog.figures = off\n", None);
    assert!(!cfg.fog.figures);
    for bad in ["yes", "1", "true", "maybe"] {
        let cfg = parse(None, &format!("fog.figures = {bad}\n"), None);
        assert_eq!(cfg.fog, crate::fog::FogConfig::default(), "{bad}");
    }
    // A bad value keeps the earlier assignment, not just the default.
    let cfg = parse(None, "fog.figures = on\nfog.figures = maybe\n", None);
    assert!(cfg.fog.figures);
}

#[test]
fn parse_scene_float_key() {
    assert!(!Mtld3dConfig::default().scene_float);
    let cfg = parse(None, "color.sceneFloat = on\n", None);
    assert!(cfg.scene_float);
    let cfg = parse(
        None,
        "color.sceneFloat = on\ncolor.sceneFloat = off\n",
        None,
    );
    assert!(!cfg.scene_float);
    for bad in ["yes", "1", "true", "maybe"] {
        let cfg = parse(None, &format!("color.sceneFloat = {bad}\n"), None);
        assert!(!cfg.scene_float, "{bad}");
    }
    // A bad value keeps the earlier assignment, not just the default.
    let cfg = parse(
        None,
        "color.sceneFloat = on\ncolor.sceneFloat = maybe\n",
        None,
    );
    assert!(cfg.scene_float);
    let cfg = parse(None, "", Some("color.sceneFloat=on"));
    assert!(cfg.scene_float);
}

#[test]
fn parse_fog_curve_keys() {
    use crate::fog::FogCurve;
    let stock = Mtld3dConfig::default().fog;
    assert_eq!((stock.curve, stock.desaturate_pct), (FogCurve::Linear, 0));
    let cfg = parse(None, "fog.curve = exp\nfog.desaturate = 0.35\n", None);
    assert_eq!((cfg.fog.curve, cfg.fog.desaturate_pct), (FogCurve::Exp, 35));
    assert_eq!(
        parse(None, "fog.curve = EXP2\n", None).fog.curve,
        FogCurve::Exp2
    );
    assert_eq!(
        parse(None, "fog.curve = exp2\nfog.curve = linear\n", None)
            .fog
            .curve,
        FogCurve::Linear
    );
    assert_eq!(
        parse(None, "fog.desaturate = 1\n", None).fog.desaturate_pct,
        100
    );
    assert_eq!(
        parse(None, "fog.desaturate = 0.5\n", None)
            .fog
            .desaturate_pct,
        50
    );
    assert_eq!(
        parse(None, "fog.desaturate = 0.5\nfog.desaturate = off\n", None)
            .fog
            .desaturate_pct,
        0
    );
    assert_eq!(
        parse(None, "fog.desaturate = 0.5\nfog.desaturate = 0\n", None)
            .fog
            .desaturate_pct,
        0
    );
    for bad in ["cubic", "1", "on"] {
        let cfg = parse(None, &format!("fog.curve = {bad}\n"), None);
        assert_eq!(cfg.fog, crate::fog::FogConfig::default(), "{bad}");
    }
    for bad in ["1.5", "-0.2", "half", "on", "2"] {
        let cfg = parse(None, &format!("fog.desaturate = {bad}\n"), None);
        assert_eq!(cfg.fog, crate::fog::FogConfig::default(), "{bad}");
    }
    // A bad value keeps the earlier assignment, not just the default.
    let cfg = parse(
        None,
        "fog.curve = exp\nfog.curve = cubic\nfog.desaturate = 0.4\nfog.desaturate = 7\n",
        None,
    );
    assert_eq!((cfg.fog.curve, cfg.fog.desaturate_pct), (FogCurve::Exp, 40));
}

#[test]
fn parse_water_keys() {
    use crate::water::WaterConfig;
    assert_eq!(Mtld3dConfig::default().water, WaterConfig::default());
    assert!(!WaterConfig::default().enabled());
    let cfg = parse(None, "water.fresnel = on\nwater.glint = 0.75\n", None);
    assert_eq!(
        cfg.water,
        WaterConfig {
            fresnel: true,
            glint_pct: 75,
        }
    );
    // `off` is a zero strength; both ends of `[0, 4]` parse.
    let cfg = parse(None, "water.glint = 0.75\nwater.glint = off\n", None);
    assert_eq!(cfg.water.glint_pct, 0);
    assert_eq!(parse(None, "water.glint = 4\n", None).water.glint_pct, 400);
    assert_eq!(parse(None, "water.glint = 0\n", None).water.glint_pct, 0);
    let cfg = parse(None, "water.fresnel = on\nwater.fresnel = off\n", None);
    assert!(!cfg.water.fresnel);
    for bad in ["yes", "1", "true", "maybe"] {
        let cfg = parse(None, &format!("water.fresnel = {bad}\n"), None);
        assert_eq!(cfg.water, WaterConfig::default(), "{bad}");
    }
    for bad in ["4.01", "-0.5", "on", "lots"] {
        let cfg = parse(None, &format!("water.glint = {bad}\n"), None);
        assert_eq!(cfg.water, WaterConfig::default(), "{bad}");
    }
    // A bad value keeps the earlier assignment, not just the default.
    let cfg = parse(
        None,
        "water.fresnel = on\nwater.fresnel = maybe\nwater.glint = 1.5\nwater.glint = 9\n",
        None,
    );
    assert_eq!(
        cfg.water,
        WaterConfig {
            fresnel: true,
            glint_pct: 150,
        }
    );
}

#[test]
fn parse_post_debug() {
    use crate::post::PostConfig;
    assert_eq!(Mtld3dConfig::default().post, PostConfig::default());
    assert!(!PostConfig::default().boundary_wanted());
    let cfg = parse(None, "post.debug = on\n", None);
    assert_eq!(
        cfg.post,
        PostConfig {
            debug: true,
            ..PostConfig::default()
        }
    );
    assert!(cfg.post.boundary_wanted());
    let cfg = parse(None, "post.debug = on\npost.debug = off\n", None);
    assert!(!cfg.post.debug);
    // A bad value keeps the earlier assignment.
    let cfg = parse(None, "post.debug = on\npost.debug = maybe\n", None);
    assert!(cfg.post.debug);
}

#[test]
fn parse_post_ao_key() {
    use crate::post::PostConfig;
    assert_eq!(PostConfig::default().ao_pct, 0);
    let cfg = parse(None, "post.ao = 0.75\n", None);
    assert_eq!(
        cfg.post,
        PostConfig {
            ao_pct: 75,
            ..PostConfig::default()
        }
    );
    assert!(cfg.post.boundary_wanted(), "AO needs the boundary");
    // Both ends of `[0, 2]` parse; `off` and `0` are the same: no AO, no boundary.
    assert_eq!(parse(None, "post.ao = 2\n", None).post.ao_pct, 200);
    for off in ["0", "off", "OFF"] {
        let cfg = parse(None, &format!("post.ao = 1\npost.ao = {off}\n"), None);
        assert_eq!(cfg.post.ao_pct, 0, "{off}");
        assert!(!cfg.post.boundary_wanted(), "{off}");
    }
    // A bad value keeps the earlier assignment.
    for bad in ["2.01", "-0.5", "on", "lots", "4"] {
        let cfg = parse(None, &format!("post.ao = 1.25\npost.ao = {bad}\n"), None);
        assert_eq!(cfg.post.ao_pct, 125, "{bad}");
    }
}

#[test]
fn parse_post_godrays_key() {
    use crate::post::PostConfig;
    assert_eq!(PostConfig::default().god_rays_pct, 0);
    let cfg = parse(None, "post.godRays = 0.5\n", None);
    assert_eq!(
        cfg.post,
        PostConfig {
            god_rays_pct: 50,
            ..PostConfig::default()
        }
    );
    assert!(cfg.post.boundary_wanted(), "god rays need the boundary");
    assert_eq!(
        parse(None, "post.godRays = 2\n", None).post.god_rays_pct,
        200
    );
    for off in ["0", "off", "OFF"] {
        let cfg = parse(
            None,
            &format!("post.godRays = 1\npost.godRays = {off}\n"),
            None,
        );
        assert_eq!(cfg.post.god_rays_pct, 0, "{off}");
        assert!(!cfg.post.boundary_wanted(), "{off}");
    }
    for bad in ["2.01", "-0.5", "on", "lots", "4"] {
        let cfg = parse(
            None,
            &format!("post.godRays = 1.25\npost.godRays = {bad}\n"),
            None,
        );
        assert_eq!(cfg.post.god_rays_pct, 125, "{bad}");
    }
    // The key is its own: it leaves AO off.
    assert_eq!(parse(None, "post.godRays = 1\n", None).post.ao_pct, 0);
}

#[test]
fn parse_ppl_bad_value_keeps_default() {
    let cfg = parse(None, "lighting.perPixel = bogus\n", None);
    assert_eq!(cfg.ppl, crate::ppl::PplConfig::default());
}

#[test]
fn parse_ppl_shoulder() {
    let cfg = parse(None, "lighting.perPixelShoulder = 0.8\n", None);
    assert_eq!(cfg.ppl.shoulder_pct, 80);
    let cfg = parse(None, "lighting.perPixelShoulder = off\n", None);
    assert_eq!(cfg.ppl.shoulder_pct, 0);
    for bad in ["0.4", "1.0", "x"] {
        let cfg = parse(None, &format!("lighting.perPixelShoulder = {bad}\n"), None);
        assert_eq!(cfg.ppl, crate::ppl::PplConfig::default(), "{bad}");
    }
}

#[test]
fn env_overrides_ppl() {
    let cfg = parse(
        None,
        "lighting.perPixel = points\n",
        Some("lighting.perPixel=all"),
    );
    assert_eq!(cfg.ppl.mode, crate::ppl::PplMode::All);
}

#[test]
fn parse_hemi_ambient() {
    let cfg = parse(None, "lighting.hemiAmbient = 1.25,0.6\n", None);
    assert_eq!(
        cfg.hemi,
        Some(crate::ppl::HemiAmbient {
            sky_pct: 125,
            ground_pct: 60,
        })
    );
    assert_eq!(parse(None, "lighting.hemiAmbient = off\n", None).hemi, None);
}

#[test]
fn parse_hemi_bad_keeps_off() {
    for value in ["2.5,0.6", "x", "1.0"] {
        let cfg = parse(None, &format!("lighting.hemiAmbient = {value}\n"), None);
        assert_eq!(cfg.hemi, None, "{value}");
    }
}

#[test]
fn parse_light_controls() {
    use crate::light_controls::{LightControls, Scaled};
    let cfg = parse(
        None,
        "lighting.sunScale = 1.5\nlighting.sunTint = 1,0.9,0.8\n\
         lighting.ambientScale = 0.5\nlighting.ambientTint = 1,1,1\n\
         lighting.nightFloor = 0.25\nlighting.torchScale = 2\n\
         lighting.torchTint = 1,0.8,0.6\nlighting.placedScale = 1.25\n\
         lighting.placedRange = 2\nlighting.figureScale = 0.75\n",
        None,
    );
    assert_eq!(
        cfg.lights,
        LightControls {
            sun: Scaled {
                scale_pct: 150,
                tint_pct: [100, 90, 80],
            },
            ambient: Scaled {
                scale_pct: 50,
                tint_pct: [100, 100, 100],
            },
            night_floor_pct: 25,
            torch: Scaled {
                scale_pct: 200,
                tint_pct: [100, 80, 60],
            },
            placed_scale_pct: 125,
            placed_range_pct: 200,
            figure_scale_pct: 75,
        }
    );
    // The default is the identity.
    assert!(Mtld3dConfig::default().lights.is_identity());
    assert!(parse(None, "", None).lights.is_identity());
}

#[test]
fn bad_light_values_keep_defaults() {
    let d = Mtld3dConfig::default();
    // Negative, over 4.0, malformed, and a wrong-count tint each warn and
    // keep the default — the whole struct stays at the identity.
    for (line, key) in [
        ("lighting.sunScale = -1", "sunScale"),
        ("lighting.sunScale = 4.01", "sunScale"),
        ("lighting.sunScale = x", "sunScale"),
        ("lighting.ambientScale = -0.1", "ambientScale"),
        ("lighting.ambientScale = 5", "ambientScale"),
        ("lighting.nightFloor = 4.5", "nightFloor"),
        ("lighting.torchScale = abc", "torchScale"),
        ("lighting.placedScale = 1e20", "placedScale"),
        ("lighting.placedRange = -2", "placedRange"),
        ("lighting.figureScale = 1.2.3", "figureScale"),
        ("lighting.sunTint = 1,0.8", "sunTint"),
        ("lighting.sunTint = 1,0.8,0.6,0.5", "sunTint"),
        ("lighting.sunTint = 1,x,0.6", "sunTint"),
        ("lighting.torchTint = -1,0.8,0.6", "torchTint"),
        ("lighting.ambientTint = 1,0.8,9", "ambientTint"),
        ("lighting.sunScale = nan", "sunScale"),
        ("lighting.sunScale = inf", "sunScale"),
        ("lighting.sunTint = 1,nan,1", "sunTint"),
        ("lighting.torchTint = 1,inf,1", "torchTint"),
    ] {
        let cfg = parse(None, &format!("{line}\n"), None);
        assert_eq!(cfg.lights, d.lights, "{key}: {line}");
        // The rest of the config is unaffected.
        assert_eq!(cfg.caps_all, d.caps_all);
    }
}

/// A light value converts to hundredths by rounding, not truncation:
/// `0.29` (0.28999… as f32) is 29.
#[test]
fn light_value_rounds_to_hundredths() {
    let cfg = parse(None, "lighting.sunScale = 0.29\n", None);
    assert_eq!(cfg.lights.sun.scale_pct, 29);
}

#[test]
fn env_overrides_lights() {
    // The env form is the same `key=value` body, semicolon-separated.
    let env = "lighting.sunScale=1.5\
        ;lighting.sunTint=1,0.9,0.8\
        ;lighting.nightFloor=0.25\
        ;lighting.torchTint=1,0.8,0.6";
    let cfg = parse(None, "", Some(env));
    use crate::light_controls::{LightControls, Scaled};
    assert_eq!(
        cfg.lights,
        LightControls {
            sun: Scaled {
                scale_pct: 150,
                tint_pct: [100, 90, 80],
            },
            night_floor_pct: 25,
            torch: Scaled {
                scale_pct: 100,
                tint_pct: [100, 80, 60],
            },
            ..LightControls::default()
        }
    );
    // The env override wins over the file, as for every other key.
    let cfg = parse(
        None,
        "lighting.sunScale = 2\n",
        Some("lighting.sunScale=1.5"),
    );
    assert_eq!(cfg.lights.sun.scale_pct, 150);
}

#[test]
fn parse_specular_keys() {
    let cfg = parse(
        None,
        "lighting.perPixelSpecular = on\nlighting.specularScale = 1.5\n\
         lighting.specularPower = 32\nlighting.specularDefault = 0.2,0.25,1\n",
        None,
    );
    assert!(cfg.ppl.specular);
    assert_eq!(cfg.ppl.specular_scale_pct, 150);
    assert_eq!(cfg.ppl.specular_power, Some(32));
    assert_eq!(cfg.ppl.specular_default_pct, Some([20, 25, 100]));
    let cfg = parse(
        None,
        "lighting.perPixelSpecular = ON\nlighting.perPixelSpecular = off\n\
         lighting.specularPower = 32\nlighting.specularPower = auto\n",
        None,
    );
    assert!(!cfg.ppl.specular);
    assert_eq!(cfg.ppl.specular_power, None);
    let cfg = parse(
        None,
        "lighting.specularScale = 0\nlighting.specularPower = 1\n",
        None,
    );
    assert_eq!(cfg.ppl.specular_scale_pct, 0);
    assert_eq!(cfg.ppl.specular_power, Some(1));
    let cfg = parse(
        None,
        "lighting.specularScale = 4\nlighting.specularPower = 128\n",
        None,
    );
    assert_eq!(cfg.ppl.specular_scale_pct, 400);
    assert_eq!(cfg.ppl.specular_power, Some(128));
    let d = crate::ppl::PplConfig::default();
    assert!(!d.specular);
    assert_eq!(d.specular_scale_pct, 100);
    assert_eq!(d.specular_power, None);
    assert_eq!(d.specular_default_pct, None);
}

#[test]
fn bad_specular_values_keep_defaults() {
    for line in [
        "lighting.perPixelSpecular = yes",
        "lighting.perPixelSpecular = true",
        "lighting.specularScale = -0.1",
        "lighting.specularScale = 4.01",
        "lighting.specularScale = big",
        "lighting.specularPower = 0",
        "lighting.specularPower = 129",
        "lighting.specularPower = 16.5",
        "lighting.specularPower = sharp",
        "lighting.specularDefault = 0.5,0.5",
        "lighting.specularDefault = 0.5,0.5,1.1",
        "lighting.specularDefault = 0.5,x,0.5",
        "lighting.specularDefault = 0.5,0.5,0.5,0.5",
    ] {
        let cfg = parse(None, &format!("{line}\n"), None);
        assert_eq!(cfg.ppl, crate::ppl::PplConfig::default(), "{line}");
    }
}

#[test]
fn parse_post_bloom_keys() {
    use crate::post::PostConfig;
    let default = PostConfig::default();
    assert_eq!((default.bloom_pct, default.bloom_threshold_pct), (0, 80));
    let cfg = parse(None, "post.bloom = 0.6\n", None);
    assert_eq!(
        cfg.post,
        PostConfig {
            bloom_pct: 60,
            ..PostConfig::default()
        }
    );
    assert!(cfg.post.boundary_wanted(), "bloom needs the boundary");
    assert_eq!(parse(None, "post.bloom = 2\n", None).post.bloom_pct, 200);
    for off in ["0", "off", "OFF"] {
        let cfg = parse(None, &format!("post.bloom = 1\npost.bloom = {off}\n"), None);
        assert_eq!(cfg.post.bloom_pct, 0, "{off}");
        assert!(!cfg.post.boundary_wanted(), "{off}");
    }
    for bad in ["2.01", "-0.5", "on", "lots", "4"] {
        let cfg = parse(
            None,
            &format!("post.bloom = 1.25\npost.bloom = {bad}\n"),
            None,
        );
        assert_eq!(cfg.post.bloom_pct, 125, "{bad}");
    }
    // The threshold alone enables nothing.
    let cfg = parse(None, "post.bloomThreshold = 0.65\n", None);
    assert_eq!(cfg.post.bloom_threshold_pct, 65);
    assert!(!cfg.post.boundary_wanted());
    assert_eq!(
        parse(None, "post.bloomThreshold = 0.5\n", None)
            .post
            .bloom_threshold_pct,
        50
    );
    assert_eq!(
        parse(None, "post.bloomThreshold = 1\n", None)
            .post
            .bloom_threshold_pct,
        100
    );
    for bad in ["0.49", "1.01", "-0.8", "off", "high"] {
        let cfg = parse(None, &format!("post.bloomThreshold = {bad}\n"), None);
        assert_eq!(cfg.post.bloom_threshold_pct, 80, "{bad}");
        let cfg = parse(
            None,
            &format!("post.bloomThreshold = 0.9\npost.bloomThreshold = {bad}\n"),
            None,
        );
        assert_eq!(cfg.post.bloom_threshold_pct, 90, "{bad}");
    }
    // The keys are their own: AO and god rays stay off.
    let cfg = parse(None, "post.bloom = 1\n", None).post;
    assert_eq!((cfg.ao_pct, cfg.god_rays_pct), (0, 0));
}
