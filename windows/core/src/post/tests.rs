use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT9, D3DMATRIX, D3DTS_PROJECTION, D3DVECTOR};

use super::*;
use crate::{encoder_data::SetViewportOp, encoder_packet::FrameRecorder, scratch::ScratchArena};

const PERSPECTIVE: (bool, bool) = (true, false);
const ORTHO: (bool, bool) = (true, true);
const OFFSCREEN_WORLD: (bool, bool) = (false, false);
const OFFSCREEN_ORTHO: (bool, bool) = (false, true);

fn fired_at(state: &mut BoundaryState, draws: &[(bool, bool)]) -> Vec<usize> {
    draws
        .iter()
        .enumerate()
        .filter(|&(_, &(rt0, ortho))| state.on_draw(rt0, ortho))
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn no_boundary_without_perspective_draw() {
    let mut state = BoundaryState::default();
    // Login and loading screens: only orthographic or RHW back-buffer draws.
    assert_eq!(fired_at(&mut state, &[ORTHO, ORTHO, ORTHO]), []);
    assert!(!state.fired());
}

#[test]
fn boundary_fires_once_per_frame() {
    let mut state = BoundaryState::default();
    // World, then the UI; a second UI draw and a late world draw (a flare)
    // fire nothing more.
    assert_eq!(
        fired_at(
            &mut state,
            &[
                ORTHO,
                PERSPECTIVE,
                PERSPECTIVE,
                ORTHO,
                ORTHO,
                PERSPECTIVE,
                ORTHO
            ]
        ),
        [3]
    );
    assert!(state.fired());
}

#[test]
fn boundary_resets_on_present() {
    let mut state = BoundaryState::default();
    assert_eq!(fired_at(&mut state, &[PERSPECTIVE, ORTHO]), [1]);
    state.on_present();
    assert!(!state.fired());
    // The world seen last frame does not carry over: a UI-only frame stays clean.
    assert_eq!(fired_at(&mut state, &[ORTHO]), []);
    state.on_present();
    assert_eq!(fired_at(&mut state, &[PERSPECTIVE, ORTHO]), [1]);
}

#[test]
fn offscreen_draws_never_fire() {
    let mut state = BoundaryState::default();
    // Render-to-texture world and ortho draws neither count as world nor fire.
    assert_eq!(
        fired_at(&mut state, &[OFFSCREEN_WORLD, OFFSCREEN_ORTHO, ORTHO]),
        []
    );
    // A back-buffer world draw arms it; an off-screen ortho draw still does not fire.
    assert_eq!(
        fired_at(&mut state, &[PERSPECTIVE, OFFSCREEN_ORTHO, ORTHO]),
        [2]
    );
}

fn perspective_ff() -> FfState {
    let mut ff = FfState::new();
    let mut proj = D3DMATRIX::IDENTITY;
    proj.m[0] = 1.5;
    proj.m[5] = 2.0;
    proj.m[10] = 1.001;
    proj.m[11] = 1.0;
    proj.m[14] = -0.1;
    proj.m[15] = 0.0;
    ff.set_transform(D3DTS_PROJECTION, &proj);
    ff
}

fn ortho_ff() -> FfState {
    FfState::new()
}

/// One scripted draw: a stand-in for the draw's own records, then the post hook.
fn script_frame(
    cfg: PostConfig,
    frame: &mut PostFrame,
    recorder: &mut FrameRecorder,
    scratch: &mut ScratchArena,
    hook: bool,
) {
    let world = perspective_ff();
    let ui = ortho_ff();
    let draws: [(bool, &FfState); 5] = [
        (false, &world), // shadow map
        (true, &world),
        (true, &world),
        (true, &ui),
        (true, &ui),
    ];
    for (rt0, ff) in draws {
        recorder
            .record_typed(
                scratch,
                SetViewportOp {
                    x: 0,
                    y: 0,
                    width: 64,
                    height: 64,
                    min_z: 0.0,
                    max_z: 1.0,
                },
            )
            .unwrap();
        if hook
            && let Some(op) = frame.note_draw(&cfg, || DrawSite {
                rt0_is_backbuffer: rt0,
                ortho_or_rhw: ff.projection_is_ortho(),
                ff,
            })
        {
            recorder.record_typed(scratch, op).unwrap();
        }
    }
    frame.on_present();
}

#[test]
fn post_off_records_no_op() {
    let cfg = PostConfig::default();
    assert!(!cfg.boundary_wanted());
    let (mut base_rec, mut base_scratch) = (FrameRecorder::new(), ScratchArena::new());
    script_frame(
        cfg,
        &mut PostFrame::default(),
        &mut base_rec,
        &mut base_scratch,
        false,
    );
    let mut frame = PostFrame::default();
    let (mut rec, mut scratch) = (FrameRecorder::new(), ScratchArena::new());
    script_frame(cfg, &mut frame, &mut rec, &mut scratch, true);
    assert_eq!(rec.len(), base_rec.len());
    assert_eq!(scratch.bytes_used(), base_scratch.bytes_used());
    // Off, the hook never even looked at the draw.
    let mut untouched = PostFrame::default();
    assert!(
        untouched
            .note_draw(&cfg, || panic!("the site is not probed while post is off"))
            .is_none()
    );
    assert_eq!(frame, PostFrame::default());

    // The same script with the debug effect on records exactly one op.
    let on = PostConfig {
        debug: true,
        ..PostConfig::default()
    };
    let (mut on_rec, mut on_scratch) = (FrameRecorder::new(), ScratchArena::new());
    script_frame(
        on,
        &mut PostFrame::default(),
        &mut on_rec,
        &mut on_scratch,
        true,
    );
    assert_eq!(on_rec.len(), base_rec.len() + 1);
}

fn site<'a>(rt0: bool, ff: &'a FfState) -> impl FnOnce() -> DrawSite<'a> {
    let ortho = ff.projection_is_ortho();
    move || DrawSite {
        rt0_is_backbuffer: rt0,
        ortho_or_rhw: ortho,
        ff,
    }
}

#[test]
fn boundary_carries_the_last_world_draws_constants() {
    let cfg = PostConfig {
        debug: true,
        ..PostConfig::default()
    };
    let mut world = perspective_ff();
    world.set_light(
        0,
        &D3DLIGHT9 {
            type_: D3DLIGHT_DIRECTIONAL,
            direction: D3DVECTOR {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            },
            diffuse: mtld3d_types::D3DCOLORVALUE {
                r: 0.9,
                g: 0.8,
                b: 0.7,
                a: 1.0,
            },
            ..D3DLIGHT9::default()
        },
    );
    world.set_light_enabled(0, true);
    let ui = ortho_ff();
    let mut frame = PostFrame::default();
    assert_eq!(frame.note_draw(&cfg, site(true, &world)), None);
    let op = frame
        .note_draw(&cfg, site(true, &ui))
        .expect("the first UI draw after the world is the boundary");
    assert_eq!(op.proj, [1.5, 2.0, 1.001, -0.1]);
    assert_eq!(op.sun_color, [0.9, 0.8, 0.7]);
    let sun = op.sun_dir_eye.expect("an enabled directional light");
    assert_eq!(sun, world.scene_sun_eye().unwrap().dir_eye);
    // Without a sun the colour is black and the direction absent.
    let constants = world_constants(&perspective_ff());
    assert_eq!(constants.sun_dir_eye, None);
    assert_eq!(constants.sun_color, [0.0; 3]);
}
