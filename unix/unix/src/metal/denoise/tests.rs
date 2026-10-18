//! The denoiser's CPU side: camera cuts, the reprojection, the history's state and the passes' layout.

use mtld3d_core::{rt::RtQuality, shadows::cascades::Mat4};
use mtld3d_shared::MetalHandle;

use super::{
    DenoiseKind, FrameCamera, HistoryTextures, TemporalHistory, atrous_steps, camera_cut, passes,
    reprojection,
};
use crate::metal::scene_vis::Kernel;

/// The row-vector view of a camera at world `position`, turned `yaw` degrees about world z.
fn view(position: [f64; 3], yaw: f64) -> Mat4 {
    let (s, c) = yaw.to_radians().sin_cos();
    // Rows of R: world axes in eye space; p_eye = (p - position) · R.
    let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let t: [f64; 3] =
        core::array::from_fn(|col| -(0..3).map(|k| position[k] * r[k][col]).sum::<f64>());
    let mut m = [[0.0f32; 4]; 4];
    for row in 0..3 {
        for col in 0..3 {
            m[row][col] = r[row][col] as f32;
        }
    }
    for col in 0..3 {
        m[3][col] = t[col] as f32;
    }
    m[3][3] = 1.0;
    m
}

/// `p · m` for the row vector `(p, 1)`.
fn apply(m: &Mat4, p: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|c| {
        (0..3).map(|r| p[r] * f64::from(m[r][c])).sum::<f64>() + f64::from(m[3][c])
    })
}

/// A camera cuts past 500 units of travel or 30° of turn, never under,
/// mirrored views included; a singular view always cuts.
#[test]
fn camera_cut_thresholds() {
    let at = view([100.0, -40.0, 7.0], 15.0);
    for (dx, cut) in [(0.0, false), (499.0, false), (501.0, true)] {
        assert_eq!(
            camera_cut(&at, &view([100.0 + dx, -40.0, 7.0], 15.0)),
            cut,
            "moved {dx}"
        );
    }
    for (turn, cut) in [(29.0, false), (31.0, true), (-31.0, true)] {
        assert_eq!(
            camera_cut(&at, &view([100.0, -40.0, 7.0], 15.0 + turn)),
            cut,
            "turned {turn}"
        );
    }
    let mirror = |mut m: Mat4| {
        for row in &mut m {
            row[0] = -row[0];
        }
        m
    };
    assert!(!camera_cut(
        &mirror(at),
        &mirror(view([110.0, -40.0, 7.0], 25.0))
    ));
    assert!(camera_cut(
        &mirror(at),
        &mirror(view([110.0, -40.0, 7.0], 55.0))
    ));
    assert!(camera_cut(&at, &[[0.0; 4]; 4]));
}

/// The reprojection takes a world point from this frame's eye space to the
/// previous frame's.
#[test]
fn reprojection_maps_into_the_previous_eye() {
    let previous = view([3.0, 5.0, -2.0], 10.0);
    let current = view([30.0, -12.0, 4.0], 32.0);
    let m = reprojection(&previous, &current).expect("invertible");
    for world in [[0.0, 0.0, 0.0], [120.0, -33.0, 7.5], [-50.0, 80.0, 150.0]] {
        let got = apply(&m, apply(&current, world));
        let want = apply(&previous, world);
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() < 1e-3,
                "{world:?}: {got:?} vs {want:?}"
            );
        }
    }
}

fn camera(view: Mat4) -> FrameCamera {
    FrameCamera {
        view,
        proj: [1.0, 1.0, 1.0, -1.0],
        viewport: [0.0, 0.0, 64.0, 64.0],
    }
}

/// A kind's history serves the next frame once it ran; a cut, a reset or a
/// frame it skipped starts it over. The textures swap every frame and the
/// sequence index runs on through a reset.
#[test]
fn history_state_follows_frames_cuts_and_resets() {
    let still = camera(view([0.0; 3], 0.0));
    let mut history = TemporalHistory::default();
    let first = history.begin(still);
    assert!(
        !history.valid(DenoiseKind::Sun),
        "nothing before the first frame"
    );
    assert_eq!(first.reproject, still_identity());
    history.ran(DenoiseKind::Sun);
    let second = history.begin(still);
    assert_eq!(second.previous, first.current);
    assert_ne!(second.current, first.current);
    assert_eq!(second.frame, first.frame + 1);
    assert!(history.valid(DenoiseKind::Sun));
    assert!(!history.valid(DenoiseKind::Ao), "AO never ran");
    // Skipped a frame: no history.
    history.begin(still);
    assert!(!history.valid(DenoiseKind::Sun));
    history.ran(DenoiseKind::Sun);
    // A cut.
    history.begin(camera(view([600.0, 0.0, 0.0], 0.0)));
    assert!(!history.valid(DenoiseKind::Sun));
    history.ran(DenoiseKind::Sun);
    // A reset.
    let before = history.begin(camera(view([600.0, 0.0, 0.0], 0.0))).frame;
    assert!(history.valid(DenoiseKind::Sun));
    history.ran(DenoiseKind::Sun);
    history.reset();
    let after = history.begin(still);
    assert!(!history.valid(DenoiseKind::Sun));
    assert_eq!(after.frame, before + 1);
    assert_eq!(
        after.previous_camera, still,
        "no previous camera after a reset"
    );
}

fn still_identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// The temporal pass reads the raw buffer, the previous history and
/// geometry and writes this frame's; the à-trous passes run at steps 1, 2,
/// 4 (low and high quality) from the history through the output and the
/// raw buffer into the output, at medium at steps 1 and 2 from the history
/// through the raw buffer into the output.
#[test]
fn passes_ping_pong_into_the_output() {
    // SAFETY: opaque values the plan only names.
    let h = |raw: u64| unsafe { MetalHandle::new(raw) };
    let history = HistoryTextures {
        value: [h(0x10), h(0x11)],
        length: [h(0x20), h(0x21)],
    };
    let mut state = TemporalHistory::default();
    let frame = state.begin(camera(view([0.0; 3], 0.0)));
    let (now, before) = (frame.current, frame.previous);
    let geometry = [h(0x30), h(0x31)];
    let (raw, out) = (h(0x40), h(0x50));
    let plan = |quality| -> Vec<_> {
        passes(
            DenoiseKind::Gi,
            raw,
            &history,
            geometry,
            out,
            &frame,
            atrous_steps(quality),
        )
        .collect()
    };
    let atrous = |plan: &[super::DenoisePass]| -> Vec<_> {
        plan.iter()
            .map(|p| {
                (
                    p.kernel,
                    p.textures[0],
                    p.textures[1],
                    p.textures[2],
                    p.textures[3],
                    p.step,
                )
            })
            .collect()
    };
    for quality in [RtQuality::Low, RtQuality::Medium, RtQuality::High] {
        let plan = plan(quality);
        assert_eq!(plan[0].kernel, Kernel::TemporalGi);
        assert_eq!(
            plan[0].textures,
            [
                raw,
                history.value[now],
                geometry[now],
                history.value[before],
                geometry[before],
                history.length[before],
                history.length[now],
                MetalHandle::NULL
            ]
        );
        let (g, length) = (geometry[now], history.length[now]);
        let expected = if quality == RtQuality::Medium {
            vec![
                (Kernel::AtrousGi, history.value[now], raw, g, length, 1),
                (Kernel::AtrousGi, raw, out, g, length, 2),
            ]
        } else {
            vec![
                (Kernel::AtrousGi, history.value[now], out, g, length, 1),
                (Kernel::AtrousGi, out, raw, g, length, 2),
                (Kernel::AtrousGi, raw, out, g, length, 4),
            ]
        };
        assert_eq!(atrous(&plan[1..]), expected, "{quality:?}");
    }
}

/// The caps and formats of the four kinds.
#[test]
fn kinds_cap_their_history() {
    let caps: Vec<u32> = DenoiseKind::ALL.iter().map(|k| k.cap()).collect();
    assert_eq!(caps, [16, 16, 32, 16]);
    assert_eq!(DenoiseKind::Gi.empty(), 0.0);
    assert_eq!(DenoiseKind::Sun.empty(), 1.0);
}
