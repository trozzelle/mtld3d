use super::{SceneReceiver, sun_receivers_wanted};
use crate::{
    config::{Mtld3dConfig, parse},
    encoder_data::PostBoundary,
};

fn boundary(ambient: [f32; 3], sun_color: [f32; 3]) -> PostBoundary {
    PostBoundary {
        ambient,
        sun_color,
        sun_dir_eye: Some([0.0, 0.0, 1.0]),
        sun_above_horizon: true,
        ..PostBoundary::default()
    }
}

/// `k` is the ambient's share of the ambient and the sun, by luminance; a
/// submission without its boundary, or with no light at all, darkens nothing.
#[test]
fn receiver_k_is_the_ambient_share() {
    let rt = crate::rt::RtConfig::default();
    let new = |ambient, sun| SceneReceiver::new(Some(&boundary(ambient, sun)), &rt).k;
    let k = new([0.2; 3], [0.6; 3]);
    assert!((k - 0.25).abs() < 1e-6, "{k}");
    // Luminance weighs green over blue: a green sun leaves less in shadow
    // than a blue one of the same value.
    let green = new([0.2; 3], [0.0, 0.6, 0.0]);
    let blue = new([0.2; 3], [0.0, 0.0, 0.6]);
    assert!(green < blue, "green {green} blue {blue}");
    // No sun: everything is ambient.
    assert_eq!(new([0.3; 3], [0.0; 3]), 1.0);
    // No light at all, or no boundary.
    assert_eq!(new([0.0; 3], [0.0; 3]), 1.0);
    assert_eq!(SceneReceiver::new(None, &rt), SceneReceiver::UNSHADOWED);
    assert_eq!(SceneReceiver::UNSHADOWED.k, 1.0);
    let bytes = SceneReceiver {
        k: 0.25,
        ..SceneReceiver::UNSHADOWED
    }
    .to_bytes();
    assert_eq!(
        bytes[..4],
        0.25f32.to_le_bytes(),
        "the pixel stage reads k as the first float"
    );
}

/// The ray-traced receivers read the `rt.ao` and `rt.gi` strengths and the
/// boundary's point lights (w 1 for a light, 0 for an empty entry) in the
/// layout `SCENE_RECEIVER_RT_MSL` declares: k, ao, gi, pad, then four
/// `float4` points.
#[test]
fn receiver_carries_strengths_and_points() {
    let rt = crate::config::parse(None, "rt.ao = 1.5\nrt.gi = 0.75\n", None).rt;
    let points = crate::rt::test_points();
    let with_points = PostBoundary {
        points,
        ..boundary([0.2; 3], [0.6; 3])
    };
    let rx = SceneReceiver::new(Some(&with_points), &rt);
    assert_eq!((rx.ao, rx.gi), (1.5, 0.75));
    assert_eq!(rx.points[1], [-5.0, 6.0, 7.0, 1.0]);
    assert_eq!(rx.points[3], [0.0; 4], "an empty entry matches nothing");
    let bytes = rx.to_bytes();
    let float = |i: usize| f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!([float(1), float(2)], [1.5, 0.75]);
    assert_eq!(
        [float(8), float(9), float(10), float(11)],
        [-5.0, 6.0, 7.0, 1.0]
    );
    // Without the boundary: the strengths stay, no light is matched.
    let none = SceneReceiver::new(None, &rt);
    assert_eq!((none.k, none.ao, none.gi), (1.0, 1.5, 0.75));
    assert!(none.points.iter().all(|p| p[3] == 0.0));
    let msl = super::SCENE_RECEIVER_RT_MSL;
    assert!(msl.starts_with("struct SceneReceiver {\n    float k;\n    float ao;\n    float gi;"));
    assert!(msl.contains("float4 points[4];"));
    assert!(msl.contains("distance(p, rx.points[k].xyz) <= 1.0"));
    assert_eq!(crate::rt::SCENE_POINT_MATCH, 1.0);
}

/// The receivers follow `shadows.sun` and `rt.shadows` only.
#[test]
fn sun_receivers_follow_the_sun_keys() {
    assert!(!sun_receivers_wanted(&Mtld3dConfig::default()));
    for key in ["shadows.sun = on\n", "rt.shadows = on\n"] {
        assert!(sun_receivers_wanted(&parse(None, key, None)), "{key}");
    }
    for key in ["rt.ao = 1\n", "rt.gi = 1\n", "rt.pointShadows = on\n"] {
        let cfg = parse(None, key, None);
        assert!(super::scene_capture_wanted(&cfg), "{key} is on");
        assert!(!sun_receivers_wanted(&cfg), "{key}");
    }
}

/// R39: every tracing key forces the directional lights (and the figure
/// path) per pixel, `rt.gi` alone included, so its bounce light reaches
/// the same draws as the sun's visibility; only `rt.pointShadows` forces
/// the point lights; with every key absent nothing is forced.
#[test]
fn scene_keys_force_lights_per_pixel() {
    use crate::{
        dxso::{FfVsFlags, FfVsKey},
        ppl::{PplConfig, PplForce, PplMasks, PsPairing, figures_enabled, ppl_masks},
    };
    assert_eq!(super::ppl_force(&Mtld3dConfig::default()), PplForce::NONE);
    for key in [
        "shadows.sun = on\n",
        "rt.shadows = on\n",
        "rt.ao = 1\n",
        "rt.gi = 1\n",
    ] {
        assert_eq!(
            super::ppl_force(&parse(None, key, None)),
            PplForce::DIRECTIONAL,
            "{key}"
        );
    }
    let points = super::ppl_force(&parse(None, "rt.pointShadows = on\n", None));
    assert!(points.directional && points.points);
    // A lit draw with one directional (slot 0) and one point light (slot 1):
    // `rt.gi` alone puts the sun per pixel with `lighting.perPixel` off.
    let lit = FfVsKey {
        flags: FfVsFlags::HAS_NORMAL | FfVsFlags::LIGHTING_ENABLED,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0b11,
        light_directional_mask: 0b01,
        light_spot_mask: 0,
        diffuse_source: 0,
        ambient_source: 0,
        specular_source: 0,
        emissive_source: 0,
        fog_mode: 0,
        tci: [0; 8],
        passthrough: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
        clip_plane_count: 0,
        ppl_mask: 0,
        ppl_model: 0,
        ppl_knee: 0,
        color_style: 0,
        scene: 0,
    };
    let off = PplConfig::default();
    let gi = super::ppl_force(&parse(None, "rt.gi = 1\n", None));
    assert_eq!(
        ppl_masks(&off, gi, &lit, PsPairing::FixedFunction),
        PplMasks {
            lights: 0b01,
            directional: 0b01,
        }
    );
    assert!(figures_enabled(&off, gi.directional));
    assert_eq!(
        ppl_masks(&off, points, &lit, PsPairing::FixedFunction).lights,
        0b11
    );
}

/// The Classic blob: a quad multiplied into the back buffer (`ZERO` /
/// `INVSRCCOLOR`), unlit, depth write off.
fn blob() -> super::BlobFacts {
    super::BlobFacts {
        pipeline_rs: crate::pipeline_state::PipelineRsBits {
            flags: crate::pipeline_state::PipelineRsFlags::BLEND_ENABLE,
            src_blend: mtld3d_types::D3DBLEND_ZERO as u8,
            dst_blend: mtld3d_types::D3DBLEND_INVSRCCOLOR as u8,
            ..crate::pipeline_state::PipelineRsBits::default()
        },
        depth_write: false,
        lit: false,
        vertices: 4,
    }
}

#[test]
fn blob_detected() {
    assert!(super::is_classic_blob(&blob()));
}

/// Other draws that multiply by the inverse source colour are not blobs:
/// more vertices, depth write on, lit, blending off or another factor.
#[test]
fn blob_not_detected_for_other_zero_invsrccolor() {
    let mut cases = Vec::new();
    for vertices in [3, 6, 24] {
        cases.push(super::BlobFacts { vertices, ..blob() });
    }
    cases.push(super::BlobFacts {
        depth_write: true,
        ..blob()
    });
    cases.push(super::BlobFacts {
        lit: true,
        ..blob()
    });
    let mut off = blob();
    off.pipeline_rs.flags = crate::pipeline_state::PipelineRsFlags::empty();
    cases.push(off);
    let mut src = blob();
    src.pipeline_rs.src_blend = mtld3d_types::D3DBLEND_ONE as u8;
    cases.push(src);
    let mut dst = blob();
    dst.pipeline_rs.dst_blend = mtld3d_types::D3DBLEND_INVSRCALPHA as u8;
    cases.push(dst);
    for case in &cases {
        assert!(!super::is_classic_blob(case), "{:?}", case);
    }
}
