//! Scene passes: the scene textures' cache and what a submission splices in ahead of the world.

use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    config::Mtld3dConfig,
    dxso::FfVsFlags,
    encoder_data::PostBoundary,
    passes::{ColorLoad, DepthLoad, Pass, PassColorAttachment, StoreAction},
    rt::RtConfig,
    shadows::{
        ShadowConfig,
        cascades::{SLOPE_BIAS, caster_depth_bias, projection, reproject_for, with_depth_offset},
    },
};
use mtld3d_shared::{
    Command, CommandType, MetalHandle,
    mtl::{PrimitiveType, VS_REPROJECT_SLOT},
    mtl_handle::MTLTextureKind,
};
use objc2::rc::Retained;
use objc2_metal::{MTLPixelFormat, MTLTexture, MTLTextureType};

use super::{SceneTextures, assemble};
use crate::{
    encoder::{
        FrameEncoder,
        replay::{
            neutral_pos_fixup,
            tests::{FAR, Fixture, Gpu, NEAR, SIZE, quad, replayed, snapshot},
        },
        scene_capture::{
            CapturedDraw,
            tests::{Bytes, Declaration, Draw, buffer_binds, ff_ps, ff_vs, run},
        },
        tests::{encoder_with_config, post_boundary_op, post_reset_frame},
    },
    metal::handle::IntoRetained,
};

const BACK_BUFFER: u64 = 0xB000;

fn sun() -> Mtld3dConfig {
    Mtld3dConfig {
        shadows: ShadowConfig {
            sun: true,
            ..ShadowConfig::default()
        },
        ..Mtld3dConfig::default()
    }
}

/// An encoder with `config` on a fresh `size` back buffer with `samples`, its scene textures prepared.
fn scene_encoder(
    coherent: &AtomicU64,
    retained: &AtomicU64,
    config: Mtld3dConfig,
    size: (u32, u32),
    samples: u8,
) -> Option<(impl Sized, FrameEncoder)> {
    let (device, mut enc) = encoder_with_config(coherent, retained, config)?;
    begin(&mut enc, size, samples, false);
    Some((device, enc))
}

/// What `begin_frame` does for the scene: a fresh pass state, a submission, the textures.
fn begin(enc: &mut FrameEncoder, size: (u32, u32), samples: u8, continues_frame: bool) {
    post_reset_frame(enc, size, samples, true);
    enc.scene.begin_submission(continues_frame);
    enc.prepare_scene_textures(size, samples);
}

/// One captured world draw in an open back-buffer pass, then the pass ends as `finalize_submit` ends it.
fn world_draw(enc: &mut FrameEncoder) {
    let (decl, bytes) = (Declaration::new(), Bytes::new());
    let (vs, ps) = (
        ff_vs(FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_NORMAL, 0),
        ff_ps(),
    );
    enc.pass_state
        .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
    run(enc, &vs, &ps, &decl, &bytes, &buffer_binds(), Draw::OPAQUE);
    enc.pass_state.end_current_pass("test");
}

fn handles(textures: &SceneTextures) -> Vec<u64> {
    textures.handles().map(MetalHandle::raw).collect()
}

fn retired_since(enc: &FrameEncoder, before: usize) -> Vec<u64> {
    enc.pending_resource_retention
        .iter()
        .skip(before)
        .map(|entry| entry.handle)
        .collect()
}

/// `(width, height, array length, pixel format, texture type)` of a scene texture.
fn shape(
    handle: MetalHandle<MTLTextureKind>,
) -> (usize, usize, usize, MTLPixelFormat, MTLTextureType) {
    let texture = handle.into_retained().expect("a live scene texture");
    (
        texture.width(),
        texture.height(),
        texture.arrayLength(),
        texture.pixelFormat(),
        texture.textureType(),
    )
}

/// The textures are made at the raster size the first time, reused while the
/// key holds, and replaced on a resize, the old ones retiring behind the GPU.
#[test]
fn scene_textures_rekeyed_on_resize() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, sun(), (64, 48), 1) else {
        return;
    };
    let first = *enc
        .scene_textures()
        .expect("scene textures under shadows.sun");
    let flat = |format| (64, 48, 1, format, MTLTextureType::Type2D);
    assert_eq!(shape(first.vis), flat(MTLPixelFormat::R16Float));
    assert_eq!(shape(first.ao), flat(MTLPixelFormat::R16Float));
    assert_eq!(shape(first.gi), flat(MTLPixelFormat::RGBA16Float));
    assert_eq!(shape(first.points), flat(MTLPixelFormat::RGBA16Float));
    assert_eq!(shape(first.predepth), flat(MTLPixelFormat::Depth32Float));
    assert_eq!(
        shape(first.cascades.expect("cascades under shadows.sun")),
        (
            2048,
            2048,
            3,
            MTLPixelFormat::Depth32Float,
            MTLTextureType::Type2DArray
        )
    );

    let before = enc.pending_resource_retention.len();
    begin(&mut enc, (64, 48), 1, false);
    assert_eq!(
        enc.scene_textures(),
        Some(&first),
        "the same key keeps them"
    );
    assert_eq!(retired_since(&enc, before), []);

    begin(&mut enc, (128, 48), 1, false);
    let second = *enc.scene_textures().expect("scene textures after a resize");
    assert_eq!(shape(second.vis).0, 128);
    assert_eq!(shape(second.predepth).0, 128);
    let mut retired = retired_since(&enc, before);
    retired.sort_unstable();
    let mut old = handles(&first);
    old.sort_unstable();
    assert_eq!(retired, old, "every old texture waits for the GPU");
    for handle in handles(&second) {
        assert!(!old.contains(&handle));
    }
    enc.shutdown_cleanup();
}

/// `rt.shadows` supersedes the cascades: none is made.
#[test]
fn ray_traced_shadows_make_no_cascades() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let config = Mtld3dConfig {
        rt: RtConfig {
            shadows: true,
            ..RtConfig::default()
        },
        ..sun()
    };
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, config, (64, 64), 1) else {
        return;
    };
    let textures = enc
        .scene_textures()
        .expect("scene textures under rt.shadows");
    assert_eq!(textures.cascades, None);
    assert_eq!(shape(textures.vis).0, 64);
    enc.shutdown_cleanup();
}

/// Draws into a multisampled back buffer land in its companion: no scene
/// textures, nothing spliced in, even for a split world.
#[test]
fn msaa_backbuffer_steps_aside() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, sun(), (64, 64), 4) else {
        return;
    };
    assert!(enc.scene_textures().is_none());
    world_draw(&mut enc);
    assert_eq!(enc.scene_draws().len(), 1, "capture itself still runs");
    assemble(&mut enc, true);
    assert_eq!(enc.pass_state.passes().len(), 1, "only the world pass");
    assert!(!enc.scene_split());

    // A single-sampled frame afterwards gets its textures.
    begin(&mut enc, (64, 64), 1, false);
    assert!(enc.scene_textures().is_some());
    enc.shutdown_cleanup();
}

/// The scene passes' clears: `(attachments, clear value)` per pass.
fn clears(passes: &[Pass]) -> Vec<(Vec<u64>, f32)> {
    passes
        .iter()
        .map(|pass| {
            let ColorLoad::Clear { r, g, b, a } = pass.color_load() else {
                panic!("a scene clear pass clears");
            };
            assert!([g, b, a].iter().all(|&c| c == r), "one value per pass");
            assert_eq!(pass.color_store(), StoreAction::Store);
            assert!(pass.commands().is_empty(), "no replay");
            let attached = core::iter::once(pass.color_texture())
                .chain(pass.extra_color().iter().map(PassColorAttachment::texture))
                .filter(|texture| !texture.is_null())
                .map(MetalHandle::raw)
                .collect();
            (attached, f32::from_bits(r))
        })
        .collect()
}

/// A world cut by a mid-frame flush before the boundary reads cleared
/// buffers in both submissions, with no replay pass; the next whole frame is
/// normal again.
#[test]
fn split_frame_reads_cleared_visibility() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, sun(), (64, 64), 1) else {
        return;
    };
    let textures = *enc.scene_textures().expect("scene textures");
    let expected = vec![
        (
            vec![textures.vis.raw(), textures.ao.raw(), textures.points.raw()],
            1.0,
        ),
        (vec![textures.gi.raw()], 0.0),
    ];

    // The flushed submission: world draws, no boundary yet.
    world_draw(&mut enc);
    assemble(&mut enc, true);
    assert!(enc.scene_split());
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 3);
    assert_eq!(clears(&passes[..2]), expected);
    assert_eq!(passes[2].color_texture().raw(), BACK_BUFFER);
    assert!(passes[2].leading_blits().is_empty());

    // Its continuation: more world draws, then the boundary.
    begin(&mut enc, (64, 64), 1, true);
    world_draw(&mut enc);
    enc.scene.note_boundary(&post_boundary_op());
    assemble(&mut enc, false);
    assert!(enc.scene_split(), "the continuation is split too");
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 3);
    assert_eq!(clears(&passes[..2]), expected);

    // The next whole frame.
    begin(&mut enc, (64, 64), 1, false);
    assert!(!enc.scene_split());
    world_draw(&mut enc);
    enc.scene.note_boundary(&post_boundary_op());
    assemble(&mut enc, false);
    assert!(!enc.scene_split());
    assert_eq!(enc.pass_state.passes().len(), 1, "nothing to clear");
    enc.shutdown_cleanup();
}

/// A flush after the boundary leaves a whole world behind it: no split.
#[test]
fn flush_after_the_boundary_is_no_split() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, sun(), (64, 64), 1) else {
        return;
    };
    world_draw(&mut enc);
    enc.scene.note_boundary(&post_boundary_op());
    assemble(&mut enc, true);
    assert!(!enc.scene_split());
    assert_eq!(enc.pass_state.passes().len(), 1);

    // The continuation holds only the UI: no world draw, no split.
    begin(&mut enc, (64, 64), 1, true);
    enc.pass_state
        .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
    enc.pass_state.end_current_pass("test");
    assemble(&mut enc, false);
    assert!(!enc.scene_split());
    assert_eq!(enc.pass_state.passes().len(), 1);
    enc.shutdown_cleanup();
}

/// With every `shadows.*`/`rt.*` key absent nothing is allocated, nothing is
/// retired and a submission, split or not, keeps exactly its own passes.
#[test]
fn default_off_assembles_nothing() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) =
        scene_encoder(&coherent, &retained, Mtld3dConfig::default(), (64, 64), 1)
    else {
        return;
    };
    assert!(enc.scene_textures().is_none());
    assert!(enc.scene_targets.held.is_none(), "nothing allocated");
    let retained_before = enc.pending_resource_retention.len();
    world_draw(&mut enc);
    assemble(&mut enc, true);
    assert_eq!(enc.pass_state.passes().len(), 1);
    assert!(!enc.scene_split());
    begin(&mut enc, (64, 64), 1, true);
    world_draw(&mut enc);
    enc.post_boundary(&sun_boundary(Some(OVERHEAD), true));
    assemble(&mut enc, false);
    assert_eq!(enc.pass_state.passes().len(), 1);
    assert!(!enc.scene_split());
    assert!(enc.scene_targets.held.is_none());
    assert_eq!(enc.pending_resource_retention.len(), retained_before);
    enc.shutdown_cleanup();
}

/// A sun overhead in the fixture's world: the identity view looks along
/// world +Z, the game's down, and the sun travels the same way.
const OVERHEAD: [f32; 3] = [0.0, 0.0, 1.0];

/// The boundary of the replay fixture's camera: the identity view, the
/// fixture's projection, the sun `sun` (above the horizon when `above`).
fn sun_boundary(sun: Option<[f32; 3]>, above: bool) -> PostBoundary {
    let a = FAR / (FAR - NEAR);
    PostBoundary {
        proj: [1.0, 1.0, a, -NEAR * a],
        sun_dir_eye: sun,
        sun_color: [1.0; 3],
        sun_above_horizon: above,
        view: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        ambient: [0.2; 3],
        frame: 0,
    }
}

/// No sun, or a sun below the horizon, renders no cascade; a sun up renders
/// one pass per cascade ahead of the world. The boundary keeps its
/// constants for the scene passes although no post effect runs.
#[test]
fn no_sun_records_no_cascades() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, sun(), (64, 64), 1) else {
        return;
    };
    for (sun, above) in [
        (None, false),
        (Some(OVERHEAD), false),
        (Some([0.0, 0.0, -1.0]), false),
    ] {
        world_draw(&mut enc);
        let boundary = sun_boundary(sun, above);
        enc.post_boundary(&boundary);
        assert_eq!(
            enc.scene.boundary(),
            Some(&boundary),
            "kept with no post effect"
        );
        assemble(&mut enc, false);
        assert_eq!(enc.pass_state.passes().len(), 1, "only the world: {sun:?}");
        assert!(enc.scene_cascades().is_none());
        begin(&mut enc, (64, 64), 1, false);
    }
    world_draw(&mut enc);
    enc.post_boundary(&sun_boundary(Some(OVERHEAD), true));
    assemble(&mut enc, false);
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 4, "three cascades, then the world");
    assert!(passes[..3].iter().all(Pass::is_scene));
    assert_eq!(passes[3].color_texture().raw(), BACK_BUFFER);
    assert_eq!(enc.scene_cascades().map(|set| set.count), Some(3));
    // The next submission starts with no cascades of its own.
    begin(&mut enc, (64, 64), 1, false);
    assert!(enc.scene_cascades().is_none());
    enc.shutdown_cleanup();
}

/// A flush that cut the frame before any world draw (a readback or a query)
/// leaves the world whole: the continuation is not split and casts.
#[test]
fn flush_before_the_world_is_no_split() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, sun(), (64, 64), 1) else {
        return;
    };
    // The flushed submission: an upload pass on the back buffer, no world.
    enc.pass_state
        .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
    enc.pass_state.end_current_pass("test");
    assemble(&mut enc, true);
    assert!(!enc.scene_split());

    // Its continuation holds the whole world and the boundary.
    begin(&mut enc, (64, 64), 1, true);
    world_draw(&mut enc);
    enc.post_boundary(&sun_boundary(Some(OVERHEAD), true));
    assemble(&mut enc, false);
    assert!(!enc.scene_split(), "the world is whole");
    assert_eq!(
        enc.pass_state.passes().len(),
        4,
        "three cascades, then the world"
    );

    // Two flushes, the world in the first part: the third part is split.
    begin(&mut enc, (64, 64), 1, false);
    world_draw(&mut enc);
    assemble(&mut enc, true);
    assert!(enc.scene_split());
    begin(&mut enc, (64, 64), 1, true);
    assemble(&mut enc, true);
    begin(&mut enc, (64, 64), 1, true);
    world_draw(&mut enc);
    enc.post_boundary(&sun_boundary(Some(OVERHEAD), true));
    assemble(&mut enc, false);
    assert!(
        enc.scene_split(),
        "the world's first part died two submissions ago"
    );
    assert!(
        enc.scene_cascades().is_none(),
        "a split world casts nothing"
    );
    enc.shutdown_cleanup();
}

/// An encoder with `config` on a fresh [`SIZE`] back buffer, its scene textures ready, and a GPU to run passes on.
fn gpu_encoder(
    coherent: &AtomicU64,
    retained: &AtomicU64,
    config: Mtld3dConfig,
) -> Option<(Gpu, FrameEncoder)> {
    let (device, mut enc) = encoder_with_config(coherent, retained, config)?;
    begin(&mut enc, (SIZE, SIZE), 1, false);
    Some((Gpu::new(device), enc))
}

/// Capture one fixture draw per vertex buffer in an open back-buffer pass,
/// end the pass, and wait for every caster build, as earlier frames would.
fn capture_world(
    enc: &mut FrameEncoder,
    fixture: &Fixture,
    buffers: &[Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>],
) -> Vec<CapturedDraw> {
    let snapshot = snapshot(0, 0);
    enc.pass_state
        .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
    let draws: Vec<CapturedDraw> = buffers
        .iter()
        .map(|buffer| {
            fixture.capture(
                enc,
                &snapshot,
                (0x91, Retained::as_ptr(buffer) as u64),
                None,
            )
        })
        .collect();
    enc.pass_state.end_current_pass("test");
    for d in &draws {
        replayed(enc, d, neutral_pos_fixup());
    }
    draws
}

/// One flat fixture quad at eye depth `z`, `half` wide each way.
fn flat_quad(z: f32, half: f32) -> Vec<u8> {
    quad([
        [-half, -half, z, 0.0, 0.0],
        [half, -half, z, 1.0, 0.0],
        [-half, half, z, 0.0, 1.0],
        [half, half, z, 1.0, 1.0],
    ])
}

/// Every captured caster is replayed exactly once into every cascade, each
/// pass clearing and storing its own slice of the cascade array under its
/// own re-projection (the set's matrix with the constant bias folded in) and
/// the slope bias.
#[test]
fn cascade_passes_replay_every_caster_once_per_cascade() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((gpu, mut enc)) = gpu_encoder(&coherent, &retained, sun()) else {
        return;
    };
    let fixture = Fixture::new();
    let buffers = [2.0, 4.0, 6.0].map(|z| gpu.buffer(&flat_quad(z, 1.0)));
    let draws = capture_world(&mut enc, &fixture, &buffers);
    assert_eq!(draws.len(), 3);
    let boundary = sun_boundary(Some(OVERHEAD), true);
    enc.post_boundary(&boundary);
    assemble(&mut enc, false);

    let set = *enc.scene_cascades().expect("the sun is up");
    assert_eq!(set.count, 3);
    let cascades = enc
        .scene_textures()
        .and_then(|textures| textures.cascades)
        .expect("cascades under shadows.sun");
    let resolution = sun().shadows.resolution;
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 4);
    let mut matrices = Vec::new();
    for (slice, pass) in passes[..3].iter().enumerate() {
        assert!(pass.is_scene());
        assert!(pass.color_texture().is_null());
        assert_eq!(pass.depth_texture(), cascades);
        assert_eq!(pass.depth_slice() as usize, slice);
        assert_eq!(
            pass.depth_load(),
            DepthLoad::Clear {
                value: 1.0f32.to_bits()
            }
        );
        assert_eq!(pass.depth_store(), StoreAction::Store);
        let of = |kind: CommandType| {
            pass.commands()
                .iter()
                .filter(move |command| command.cmd == kind as u32)
        };
        assert_eq!(
            of(CommandType::DrawPrimitives).count(),
            draws.len(),
            "cascade {slice}: every caster once"
        );
        let reprojects: Vec<_> = of(CommandType::SetVertexBytesAt)
            .filter(|command| command.param_a == VS_REPROJECT_SLOT)
            .map(|command| (command.param_b, command.param_c))
            .collect();
        assert_eq!(reprojects.len(), draws.len());
        assert!(
            reprojects.windows(2).all(|w| w[0] == w[1]),
            "one copy per pass"
        );
        let (ptr, len) = reprojects[0];
        assert_eq!(len, 64);
        // SAFETY: a copy in the live encoder's scratch arena.
        let bound = unsafe { core::slice::from_raw_parts(ptr as *const f32, 16) }.to_vec();
        let light_vp = set.matrices[slice];
        let biased = with_depth_offset(&light_vp, caster_depth_bias(&light_vp, resolution));
        let expected = reproject_for(&biased, &projection(boundary.proj));
        assert_eq!(bound, expected.as_flattened(), "cascade {slice}");
        matrices.push(bound);
        for bias in of(CommandType::SetDepthBias) {
            assert_eq!(
                (
                    f32::from_bits(bias.param_a),
                    f32::from_bits(u32::try_from(bias.param_b).expect("f32 bits"))
                ),
                (0.0, SLOPE_BIAS)
            );
        }
    }
    assert!(matrices[0] != matrices[1] && matrices[1] != matrices[2]);
    assert_eq!(passes[3].color_texture().raw(), BACK_BUFFER);
    assert_eq!(enc.scene_caster_skips(), 0);
    enc.shutdown_cleanup();
}

/// A caster far toward the sun, outside every cascade's bounding sphere,
/// still writes depth into every slice: run on the GPU, each cascade holds
/// the caster's biased light depth wherever it covers, and the cascades
/// differ only in how much of the caster they see.
#[test]
fn cascade_caster_toward_the_sun_writes_depth() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let config = Mtld3dConfig {
        shadows: ShadowConfig {
            resolution: 1024,
            ..sun().shadows
        },
        ..sun()
    };
    let Some((gpu, mut enc)) = gpu_encoder(&coherent, &retained, config) else {
        return;
    };
    let fixture = Fixture::new();
    // 3000 units above the camera, which looks straight down: between the
    // overhead sun and every slice, wider than any of them.
    let buffers = [gpu.buffer(&flat_quad(-3000.0, 40.0))];
    capture_world(&mut enc, &fixture, &buffers);
    let boundary = sun_boundary(Some(OVERHEAD), true);
    enc.post_boundary(&boundary);
    assemble(&mut enc, false);
    let set = *enc.scene_cascades().expect("the sun is up");
    let cascades = enc
        .scene_textures()
        .and_then(|textures| textures.cascades)
        .expect("cascades");
    let descriptors: Vec<_> = enc.pass_state.passes()[..usize::from(set.count)]
        .iter()
        .map(|pass| crate::encoder::pass_to_descriptor(pass, MetalHandle::NULL))
        .collect();
    gpu.run_descriptors(&descriptors);
    let texture = cascades.into_retained().expect("the cascade array");
    for slice in 0..usize::from(set.count) {
        let depths = gpu.depths_at(&texture, 1024, slice);
        let light_vp = set.matrices[slice];
        let caster = [0.0, 0.0, -3000.0, 1.0];
        let z: f32 = (0..4).map(|r| caster[r] * light_vp[r][2]).sum::<f32>()
            + caster_depth_bias(&light_vp, 1024);
        assert!((0.0..1.0).contains(&z), "cascade {slice}: {z}");
        let centre = f32::from_bits(depths[512 * 1024 + 512]);
        // The re-projection recovers the eye position's `w` from camera clip
        // `z`, a cancellation worth ~1e-4 of the depth range 3000 units
        // behind a near plane of 1.
        assert!(
            (centre - z).abs() < 1e-3,
            "cascade {slice}: {centre} vs {z}"
        );
        let written = depths.iter().filter(|&&d| d != 1.0f32.to_bits()).count();
        assert_eq!(
            written,
            depths.len(),
            "cascade {slice}: the caster covers it"
        );
    }
    enc.shutdown_cleanup();
}
