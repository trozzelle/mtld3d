use mtld3d_shared::encoder_wire::{FrameSlab, WireError, WireReader};

use super::{
    AdapterSpoof, ColorSpacePolicy, CursorScale, DeviceCapsFlags, GpuCaps, Mtld3dConfig,
    SoftwareCursorPolicy, WireValue,
};

fn round_trip<T: WireValue>(value: &T) -> T {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| value.write_wire(writer))
        .unwrap();
    let mut outer = WireReader::new(slab.as_bytes());
    let mut record = outer.next_record().unwrap().unwrap();
    let result = T::read_wire(&mut record.payload).unwrap();
    assert!(record.payload.is_empty());
    result
}

#[test]
fn resolved_configuration_preserves_nondefaults() {
    let mut value = Mtld3dConfig::default();
    value.caps_all = !value.caps_all;
    value.main_thread_checker = !value.main_thread_checker;
    value.fail_next_submit = !value.fail_next_submit;
    value.expand_packed16 = !value.expand_packed16;
    value.deny_float32_filtering = !value.deny_float32_filtering;
    value.managed_memory = !value.managed_memory;
    value.linear_align256 = !value.linear_align256;
    value.hdr_enable = !value.hdr_enable;
    value.scene_float = !value.scene_float;
    value.color_space = ColorSpacePolicy::Accurate;
    value.cursor_scale = CursorScale::Fixed(7);
    value.cursor_software = SoftwareCursorPolicy::On;
    value.shader_cache_enable = !value.shader_cache_enable;
    value.shader_async_compile = !value.shader_async_compile;
    value.log_dir = "log_dir/utf8-ä".into();
    value.bytecode_dump_dir = "bytecode_dump_dir/utf8-ä".into();
    value.skip_shaders = vec![0, u64::MAX, 55];
    value.present_gate_file = "present_gate_file/utf8-ä".into();
    value.present_occluded = !value.present_occluded;
    value.query_flush_immediate = !value.query_flush_immediate;
    value.query_event_immediate = !value.query_event_immediate;
    value.depth_alias_same_size = !value.depth_alias_same_size;
    value.buffer_ignore_lock_bounds = !value.buffer_ignore_lock_bounds;
    value.vbib_retention_cap_bytes = u64::MAX - 123;
    value.vram_budget_cap_bytes = u64::MAX - 123;
    value.pagebox_pool_cap_bytes = u64::MAX - 123;
    value.present_max_fps = 123;
    value.present_max_frame_latency = 17;
    value.render_scale_percent = 123;
    value.render_lod_bias = !value.render_lod_bias;
    value.adapter_spoof = AdapterSpoof::Amd;
    value.df_formats = !value.df_formats;
    value.ppl = crate::ppl::PplConfig {
        mode: crate::ppl::PplMode::All,
        mixed: true,
        model: crate::ppl::PplModel::HalfLambert,
        falloff: crate::ppl::PplFalloff::Smooth,
        debug_term: true,
        shoulder_pct: 80,
        figures: true,
        specular: true,
        specular_scale_pct: 250,
        specular_power: Some(32),
        specular_default_pct: Some([20, 30, 100]),
    };
    value.lights = crate::light_controls::LightControls {
        sun: crate::light_controls::Scaled {
            scale_pct: 150,
            tint_pct: [100, 90, 80],
        },
        ambient: crate::light_controls::Scaled {
            scale_pct: 50,
            tint_pct: [100, 100, 100],
        },
        night_floor_pct: 25,
        torch: crate::light_controls::Scaled {
            scale_pct: 200,
            tint_pct: [100, 80, 60],
        },
        placed_scale_pct: 125,
        placed_range_pct: 200,
        figure_scale_pct: 75,
    };
    value.fog = crate::fog::FogConfig {
        figures: true,
        curve: crate::fog::FogCurve::Exp2,
        desaturate_pct: 35,
    };
    value.water = crate::water::WaterConfig {
        fresnel: true,
        glint_pct: 275,
    };
    value.post = crate::post::PostConfig {
        debug: true,
        ao_pct: 150,
        god_rays_pct: 175,
        bloom_pct: 125,
        bloom_threshold_pct: 65,
    };
    value.shadows = crate::shadows::ShadowConfig {
        sun: true,
        resolution: 4096,
        cascades: 4,
        distance: 12345,
        softness_pct: 275,
        debug: crate::shadows::ShadowDebug::Visibility,
    };
    value.rt = crate::rt::RtConfig {
        shadows: true,
        sun_size_pct: 333,
        ao_pct: 150,
        ao_radius: 499,
        gi_pct: 175,
        point_shadows: true,
        quality: crate::rt::RtQuality::High,
        debug: crate::rt::RtDebug::Points,
    };
    assert_eq!(round_trip(&value), value);
}

#[test]
fn gpu_caps_preserve_native_overrides() {
    let value = GpuCaps {
        unified_memory: false,
        min_linear_texture_align: 256,
        device_caps: DeviceCapsFlags::RESOLVE_NEEDS_RETIRE,
    };
    let decoded = round_trip(&value);
    assert_eq!(decoded.unified_memory, value.unified_memory);
    assert_eq!(
        decoded.min_linear_texture_align,
        value.min_linear_texture_align
    );
    assert_eq!(decoded.device_caps, value.device_caps);
}

#[test]
fn invalid_enum_tags_are_rejected() {
    let bytes = [255];
    assert_eq!(
        AdapterSpoof::read_wire(&mut WireReader::new(&bytes)),
        Err(WireError::InvalidValue)
    );
    assert_eq!(
        ColorSpacePolicy::read_wire(&mut WireReader::new(&bytes)),
        Err(WireError::InvalidValue)
    );
    assert_eq!(
        SoftwareCursorPolicy::read_wire(&mut WireReader::new(&bytes)),
        Err(WireError::InvalidValue)
    );
    assert_eq!(
        CursorScale::read_wire(&mut WireReader::new(&bytes)),
        Err(WireError::InvalidValue)
    );
}

/// Every value of each `shadows.*` / `rt.*` enum crosses the wire, alone and
/// in a whole configuration, and a tag past the last one is rejected.
#[test]
fn shadow_and_rt_keys_round_trip() {
    use crate::{
        rt::{RtConfig, RtDebug, RtQuality},
        shadows::{ShadowConfig, ShadowDebug},
    };
    for debug in [
        ShadowDebug::Off,
        ShadowDebug::Cascades,
        ShadowDebug::Visibility,
    ] {
        let value = ShadowConfig {
            debug,
            softness_pct: 7,
            ..ShadowConfig::default()
        };
        assert_eq!(round_trip(&value), value);
    }
    for quality in [RtQuality::Low, RtQuality::Medium, RtQuality::High] {
        for debug in [
            RtDebug::Off,
            RtDebug::Visibility,
            RtDebug::Ao,
            RtDebug::Gi,
            RtDebug::Points,
        ] {
            let value = RtConfig {
                quality,
                debug,
                gi_pct: 3,
                ..RtConfig::default()
            };
            assert_eq!(round_trip(&value), value);
        }
    }
    // The groups sit right after `color.sceneFloat`, so a stock config's
    // later fields still decode in place.
    let mut value = Mtld3dConfig::default();
    value.shadows.sun = true;
    value.rt.gi_pct = 50;
    value.color_space = ColorSpacePolicy::Accurate;
    assert_eq!(round_trip(&value), value);

    // `ShadowConfig` is sun, resolution, cascades, distance, softness, debug;
    // `RtConfig` ends with quality, debug. One past the last tag of each enum.
    let shadow = [0, 0, 8, 3, 0x40, 0x1f, 150, 0, 3];
    assert_eq!(
        ShadowConfig::read_wire(&mut WireReader::new(&shadow)),
        Err(WireError::InvalidValue)
    );
    let mut good = shadow;
    good[8] = 0;
    assert_eq!(
        ShadowConfig::read_wire(&mut WireReader::new(&good)),
        Ok(ShadowConfig::default())
    );
    let rt = |quality: u8, debug: u8| [0, 100, 0, 0, 0, 160, 0, 0, 0, 0, quality, debug];
    assert_eq!(
        RtConfig::read_wire(&mut WireReader::new(&rt(1, 0))),
        Ok(RtConfig::default())
    );
    for bad in [rt(3, 0), rt(1, 5)] {
        assert_eq!(
            RtConfig::read_wire(&mut WireReader::new(&bad)),
            Err(WireError::InvalidValue)
        );
    }
}
