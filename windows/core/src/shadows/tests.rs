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
    let k = SceneReceiver::new(Some(&boundary([0.2; 3], [0.6; 3]))).k;
    assert!((k - 0.25).abs() < 1e-6, "{k}");
    // Luminance weighs green over blue: a green sun leaves less in shadow
    // than a blue one of the same value.
    let green = SceneReceiver::new(Some(&boundary([0.2; 3], [0.0, 0.6, 0.0]))).k;
    let blue = SceneReceiver::new(Some(&boundary([0.2; 3], [0.0, 0.0, 0.6]))).k;
    assert!(green < blue, "green {green} blue {blue}");
    // No sun: everything is ambient.
    assert_eq!(
        SceneReceiver::new(Some(&boundary([0.3; 3], [0.0; 3]))).k,
        1.0
    );
    // No light at all, or no boundary.
    assert_eq!(
        SceneReceiver::new(Some(&boundary([0.0; 3], [0.0; 3]))).k,
        1.0
    );
    assert_eq!(SceneReceiver::new(None), SceneReceiver::UNSHADOWED);
    assert_eq!(SceneReceiver::UNSHADOWED.k, 1.0);
    assert_eq!(
        SceneReceiver { k: 0.25 }.to_bytes(),
        0.25f32.to_le_bytes(),
        "the pixel stage reads k as the first float"
    );
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
