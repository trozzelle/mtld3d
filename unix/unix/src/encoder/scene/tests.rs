//! Scene passes: the scene textures' cache and what a submission splices in ahead of the world.

use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    config::Mtld3dConfig,
    dxso::FfVsFlags,
    encoder_data::PostBoundary,
    passes::{ColorLoad, DepthLoad, Pass, PassColorAttachment, StoreAction},
    rt::RtConfig,
    shadows::{
        ShadowConfig, ShadowDebug,
        cascades::{SLOPE_BIAS, caster_depth_bias, projection, reproject_for, with_depth_offset},
    },
};
use mtld3d_shared::{
    BlitCommandType, Command, CommandType, MetalHandle,
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
            IDENTITY, neutral_pos_fixup,
            tests::{FAR, Fixture, Gpu, NEAR, SIZE, quad, replayed, snapshot},
        },
        scene_capture::{
            CapturedDraw,
            tests::{Bytes, Declaration, Draw, buffer_binds, ff_ps, ff_vs, run},
        },
        tests::{encoder_with_config, post_boundary_op, post_reset_frame},
    },
    metal::{handle::IntoRetained, scene_vis::Kernel},
};

pub(super) const BACK_BUFFER: u64 = 0xB000;

pub(super) fn sun() -> Mtld3dConfig {
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
pub(super) fn begin(enc: &mut FrameEncoder, size: (u32, u32), samples: u8, continues_frame: bool) {
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

    // The next whole frame: no sun at its boundary, so full sun, cleared.
    begin(&mut enc, (64, 64), 1, false);
    assert!(!enc.scene_split());
    world_draw(&mut enc);
    enc.scene.note_boundary(&post_boundary_op());
    assemble(&mut enc, false);
    assert!(!enc.scene_split());
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 2, "the visibility clear, then the world");
    assert_eq!(clears(&passes[..1]), [(vec![textures.vis.raw()], 1.0)]);
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
    let vis = enc.scene_textures().expect("scene textures").vis.raw();
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 2, "the visibility clear, then the world");
    assert_eq!(clears(&passes[..1]), [(vec![vis], 1.0)]);

    // The continuation holds only the UI: no world draw, no split; its
    // visibility is written all the same.
    begin(&mut enc, (64, 64), 1, true);
    enc.pass_state
        .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
    enc.pass_state.end_current_pass("test");
    assemble(&mut enc, false);
    assert!(!enc.scene_split());
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 2);
    assert_eq!(clears(&passes[..1]), [(vec![vis], 1.0)]);
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
pub(super) const OVERHEAD: [f32; 3] = [0.0, 0.0, 1.0];

/// The boundary of the replay fixture's camera: the identity view, the
/// fixture's projection, the sun `sun` (above the horizon when `above`).
pub(super) fn sun_boundary(sun: Option<[f32; 3]>, above: bool) -> PostBoundary {
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

/// No sun, or a sun below the horizon, renders no cascade and clears the
/// visibility to full sun; a sun up renders one pass per cascade and the
/// depth pre-pass ahead of the world, and the world's first pass leads with
/// the visibility kernel. The boundary keeps its constants for the scene
/// passes although no post effect runs.
#[test]
fn no_sun_records_no_cascades() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, sun(), (64, 64), 1) else {
        return;
    };
    let textures = *enc.scene_textures().expect("scene textures");
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
        let passes = enc.pass_state.passes();
        assert_eq!(passes.len(), 2, "full sun, then the world: {sun:?}");
        assert_eq!(clears(&passes[..1]), [(vec![textures.vis.raw()], 1.0)]);
        assert!(passes[1].leading_blits().is_empty(), "no kernel: {sun:?}");
        assert!(enc.scene_cascades().is_none());
        begin(&mut enc, (64, 64), 1, false);
    }
    world_draw(&mut enc);
    enc.post_boundary(&sun_boundary(Some(OVERHEAD), true));
    assemble(&mut enc, false);
    let passes = enc.pass_state.passes();
    assert_eq!(
        passes.len(),
        5,
        "three cascades, the pre-pass, then the world"
    );
    assert!(passes[..4].iter().all(Pass::is_scene));
    assert_eq!(passes[3].depth_texture(), textures.predepth);
    assert_eq!(passes[4].color_texture().raw(), BACK_BUFFER);
    let kernel = vis_kernel(&passes[4]);
    assert_eq!(
        [
            kernel.src_handle,
            kernel.dst_handle,
            kernel.dst_offset,
            kernel.bytes_per_row
        ],
        [
            textures.predepth.raw(),
            textures.vis.raw(),
            textures.cascades.expect("cascades").raw(),
            0
        ]
    );
    assert_eq!((kernel.region_w, kernel.region_h), (64, 64));
    assert_eq!(enc.scene_cascades().map(|set| set.count), Some(3));
    // The next submission starts with no cascades of its own.
    begin(&mut enc, (64, 64), 1, false);
    assert!(enc.scene_cascades().is_none());
    enc.shutdown_cleanup();
}

/// The world pass's one leading blit: the sun visibility kernel, its constants in the scratch arena.
fn vis_kernel(world: &Pass) -> mtld3d_shared::BlitCommand {
    let [kernel] = world.leading_blits() else {
        panic!("the world leads with the visibility kernel alone");
    };
    assert_eq!(kernel.cmd, BlitCommandType::SceneKernel as u32);
    assert_eq!(kernel.mip_level, Kernel::SunVisCascades as u32);
    assert_ne!(kernel.src_offset, 0, "a pipeline");
    assert_ne!(kernel.byte_size, 0, "constants");
    *kernel
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
        5,
        "three cascades, the pre-pass, then the world"
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
    assert_eq!(passes.len(), 5, "three cascades, the pre-pass, the world");
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
    assert_eq!(passes[4].color_texture().raw(), BACK_BUFFER);
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

/// The depth pre-pass replays every captured draw as the camera drew it:
/// into `predepth` (cleared to the far plane, stored), under the draw's own
/// viewport, through the identity re-projection, with the draw's own
/// `pos_fixup` and slope bias.
#[test]
fn depth_prepass_replays_the_camera_view() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((gpu, mut enc)) = gpu_encoder(&coherent, &retained, sun()) else {
        return;
    };
    let fixture = Fixture::new();
    let buffers = [2.0, 4.0, 6.0].map(|z| gpu.buffer(&flat_quad(z, 1.0)));
    let draws = capture_world(&mut enc, &fixture, &buffers);
    enc.post_boundary(&sun_boundary(Some(OVERHEAD), true));
    assemble(&mut enc, false);
    let predepth = enc.scene_textures().expect("scene textures").predepth;
    let passes = enc.pass_state.passes();
    let prepass = &passes[3];
    assert!(prepass.is_scene());
    assert!(prepass.color_texture().is_null());
    assert_eq!(prepass.depth_texture(), predepth);
    assert_eq!(prepass.depth_slice(), 0);
    assert_eq!(
        prepass.depth_load(),
        DepthLoad::Clear {
            value: 1.0f32.to_bits()
        }
    );
    assert_eq!(prepass.depth_store(), StoreAction::Store);
    let of = |kind: CommandType| {
        prepass
            .commands()
            .iter()
            .filter(move |command| command.cmd == kind as u32)
    };
    assert_eq!(of(CommandType::DrawPrimitives).count(), draws.len());
    let viewports: Vec<_> = of(CommandType::SetViewport).collect();
    assert_eq!(viewports.len(), 1, "one viewport for draws that share it");
    let fields = |c: &Command| (c.cmd, c.param_a, c.param_b, c.param_c, c.param_d);
    assert_eq!(fields(viewports[0]), fields(&draws[0].viewport.command()));
    assert_eq!(draws[0].viewport.rect, (0, 0, SIZE, SIZE));
    let bytes_at = |slot: u32| -> Vec<(u64, u64)> {
        of(CommandType::SetVertexBytesAt)
            .filter(|command| command.param_a == slot)
            .map(|command| (command.param_b, command.param_c))
            .collect()
    };
    let reprojects = bytes_at(VS_REPROJECT_SLOT);
    assert_eq!(reprojects.len(), draws.len());
    for &(ptr, len) in &reprojects {
        assert_eq!(len, 64);
        // SAFETY: a copy in the live encoder's scratch arena.
        let bound = unsafe { core::slice::from_raw_parts(ptr as *const f32, 16) };
        assert_eq!(bound, IDENTITY.as_flattened());
    }
    let fixups = bytes_at(mtld3d_shared::mtl::VS_POS_FIXUP_SLOT);
    let own: Vec<(u64, u64)> = draws
        .iter()
        .map(|d| (d.pos_fixup.ptr, u64::from(d.pos_fixup.len)))
        .collect();
    assert_eq!(fixups, own, "each draw's own camera pos_fixup");
    for bias in of(CommandType::SetDepthBias) {
        assert_eq!(
            (
                f32::from_bits(bias.param_a),
                f32::from_bits(u32::try_from(bias.param_b).expect("f32 bits"))
            ),
            (0.0, draws[0].slope_scale)
        );
    }
    enc.shutdown_cleanup();
}

/// DAoC's near and far planes, and the projection terms of a 90° camera over them.
pub(super) const DAOC_NEAR: f32 = 10.0;
pub(super) const DAOC_FAR: f32 = 16_000.0;

pub(super) fn daoc_proj() -> [f32; 4] {
    let a = DAOC_FAR / (DAOC_FAR - DAOC_NEAR);
    [1.0, 1.0, a, -DAOC_NEAR * a]
}

/// The side of the visibility scenes' raster.
pub(super) const VIS_RASTER: u32 = 128;
/// How high the camera of the visibility scenes is over the floor.
pub(super) const CAMERA_HEIGHT: f64 = 150.0;

pub(super) type V3 = [f64; 3];

fn dot(a: V3, b: V3) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}

fn add(a: V3, b: V3, s: f64) -> V3 {
    [
        s.mul_add(b[0], a[0]),
        s.mul_add(b[1], a[1]),
        s.mul_add(b[2], a[2]),
    ]
}

/// The camera of the visibility scenes in DAoC's world (−Z up, so the floor
/// lies at `z = CAMERA_HEIGHT`): at the origin, looking north (+Y) 45° below
/// the horizon, its view mirrored as the game's is. Its eye axes in world
/// space: right, above, forward.
pub(super) struct Camera {
    pub(super) axes: [V3; 3],
}

impl Camera {
    pub(super) fn new() -> Self {
        let pitch = core::f64::consts::FRAC_PI_4;
        let forward = [0.0, pitch.cos(), pitch.sin()];
        // East is perpendicular to the forward axis: the axes are orthonormal.
        let right = [1.0, 0.0, 0.0];
        let above = cross(forward, right);
        Self {
            axes: [right.map(|v| -v), above, forward],
        }
    }

    /// `D3DTS_VIEW`: world to eye, row-vector.
    pub(super) fn view(&self) -> [[f32; 4]; 4] {
        let mut m = [[0.0f32; 4]; 4];
        for (col, axis) in self.axes.iter().enumerate() {
            for (row, &v) in axis.iter().enumerate() {
                m[row][col] = v as f32;
            }
        }
        m[3][3] = 1.0;
        m
    }

    pub(super) fn to_eye(&self, w: V3) -> V3 {
        self.axes.map(|axis| dot(w, axis))
    }

    /// The world ray through raster pixel `(x, y)`: the D3D9 screen position
    /// `(x, y)`, which the half-pixel shift puts at the pixel's centre.
    fn ray(&self, x: u32, y: u32) -> V3 {
        let side = f64::from(VIS_RASTER);
        let ndc = [
            2.0 * f64::from(x) / side - 1.0,
            1.0 - 2.0 * f64::from(y) / side,
        ];
        let [right, above, forward] = self.axes;
        add(add(forward, right, ndc[0]), above, ndc[1])
    }
}

/// A parallelogram `origin + a·u + b·v`, `a` and `b` in `[0, 1]`, in world space.
#[derive(Clone, Copy)]
pub(super) struct Quad {
    pub(super) origin: V3,
    pub(super) u: V3,
    pub(super) v: V3,
}

impl Quad {
    /// The horizontal rectangle `x0..x1` × `y0..y1` at world height `z`.
    pub(super) fn level(x: (f64, f64), y: (f64, f64), z: f64) -> Self {
        Self {
            origin: [x.0, y.0, z],
            u: [x.1 - x.0, 0.0, 0.0],
            v: [0.0, y.1 - y.0, 0.0],
        }
    }

    /// The floor, out to where the sky begins.
    pub(super) fn floor() -> Self {
        Self::level((-4000.0, 4000.0), (-50.0, 2000.0), CAMERA_HEIGHT)
    }

    /// The fixture vertices of the quad in `camera`'s eye space.
    pub(super) fn vertices(&self, camera: &Camera) -> Vec<u8> {
        let corner = |a: f64, b: f64| {
            let e = camera.to_eye(add(add(self.origin, self.u, a), self.v, b));
            [e[0] as f32, e[1] as f32, e[2] as f32, a as f32, b as f32]
        };
        quad([
            corner(0.0, 0.0),
            corner(1.0, 0.0),
            corner(0.0, 1.0),
            corner(1.0, 1.0),
        ])
    }

    /// Where the ray `p + t·w` (t > 0) meets the quad grown by `grow` world
    /// units on every side (shrunk when negative), as `t`.
    fn hit(&self, p: V3, w: V3, grow: f64) -> Option<f64> {
        let n = cross(self.u, self.v);
        let facing = dot(w, n);
        if facing.abs() < 1e-12 {
            return None;
        }
        let t = dot(add(self.origin, p, -1.0), n) / facing;
        if t <= 1e-6 {
            return None;
        }
        let q = add(add(p, w, t), self.origin, -1.0);
        let (uu, vv) = (dot(self.u, self.u), dot(self.v, self.v));
        let (a, b) = (dot(q, self.u) / uu, dot(q, self.v) / vv);
        let (ga, gb) = (grow / uu.sqrt(), grow / vv.sqrt());
        ((-ga..=1.0 + ga).contains(&a) && (-gb..=1.0 + gb).contains(&b)).then_some(t)
    }
}

/// What the camera sees at a pixel, by ray casting the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Seen {
    /// Nothing: the far plane.
    Sky,
    /// Quad `index`, its point in full sun even past `margin` of every occluder's edge.
    Lit(usize),
    /// Quad `index`, its point in shadow even `margin` inside an occluder's edge.
    Shadowed(usize),
    /// Quad `index` within `margin` of a shadow edge: the penumbra.
    Edge(usize),
}

impl Seen {
    pub(super) const fn quad(self) -> Option<usize> {
        match self {
            Self::Sky => None,
            Self::Lit(i) | Self::Shadowed(i) | Self::Edge(i) => Some(i),
        }
    }
}

/// A scene of quads under a sun, rendered through the scene passes.
pub(super) struct VisScene {
    pub(super) camera: Camera,
    pub(super) quads: Vec<Quad>,
    /// The sun's world travel direction.
    pub(super) sun: V3,
}

impl VisScene {
    /// The sun's travel direction from its elevation (degrees) and the azimuth (degrees from north, toward east) it shines from.
    pub(super) fn sun(elevation: f64, from_azimuth: f64) -> V3 {
        let (e, a) = (elevation.to_radians(), from_azimuth.to_radians());
        let toward = [a.sin() * e.cos(), a.cos() * e.cos(), -e.sin()];
        toward.map(|v| -v)
    }

    /// What pixel `(x, y)` sees, shadows judged `margin` world units clear of every edge.
    pub(super) fn seen_at(&self, x: u32, y: u32, margin: f64) -> Seen {
        let w = self.camera.ray(x, y);
        let hit = self
            .quads
            .iter()
            .enumerate()
            .filter_map(|(i, q)| q.hit([0.0; 3], w, 0.0).map(|t| (t, i)))
            .filter(|&(t, _)| t * dot(w, self.camera.axes[2]) < f64::from(DAOC_FAR))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((t, index)) = hit else {
            return Seen::Sky;
        };
        let point = add([0.0; 3], w, t);
        let toward = self.sun.map(|v| -v);
        let occluders = || {
            self.quads
                .iter()
                .enumerate()
                .filter(move |&(i, _)| i != index)
        };
        if occluders().any(|(_, q)| q.hit(point, toward, -margin).is_some()) {
            Seen::Shadowed(index)
        } else if occluders().all(|(_, q)| q.hit(point, toward, margin).is_none()) {
            Seen::Lit(index)
        } else {
            Seen::Edge(index)
        }
    }

    /// What every pixel sees, `None` where a neighbour sees another quad or
    /// another shadow class (the raster and the ray cast may part there).
    pub(super) fn seen(&self, margin: f64) -> Vec<Option<Seen>> {
        let side = VIS_RASTER;
        let at = |x: u32, y: u32| self.seen_at(x, y, margin);
        (0..side * side)
            .map(|i| {
                let (x, y) = (i % side, i / side);
                let centre = at(x, y);
                let steady = [(-1, 0), (1, 0), (0, -1), (0, 1)].iter().all(|&(dx, dy)| {
                    let (nx, ny) = (x.saturating_add_signed(dx), y.saturating_add_signed(dy));
                    nx >= side || ny >= side || at(nx, ny) == centre
                });
                steady.then_some(centre)
            })
            .collect()
    }

    /// The sun visibility the scene passes write for this scene under `config`; `None` without a Metal device.
    fn run(&self, config: Mtld3dConfig) -> Option<Vec<f32>> {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let (device, mut enc) = encoder_with_config(&coherent, &retained, config)?;
        begin(&mut enc, (VIS_RASTER, VIS_RASTER), 1, false);
        let gpu = Gpu::new(device);
        let fixture = Fixture::with_camera(DAOC_NEAR, DAOC_FAR, VIS_RASTER);
        let buffers: Vec<_> = self
            .quads
            .iter()
            .map(|q| gpu.buffer(&q.vertices(&self.camera)))
            .collect();
        capture_world(&mut enc, &fixture, &buffers);
        let sun = self.camera.to_eye(self.sun).map(|v| v as f32);
        enc.post_boundary(&PostBoundary {
            proj: daoc_proj(),
            sun_dir_eye: Some(sun),
            sun_color: [1.0; 3],
            sun_above_horizon: true,
            view: self.camera.view(),
            ambient: [0.2; 3],
            frame: 0,
        });
        assemble(&mut enc, false);
        let passes = enc.pass_state.passes();
        let (world, scene) = passes.split_last().expect("the world pass");
        assert!(scene.iter().all(Pass::is_scene));
        let mut descriptors: Vec<_> = scene
            .iter()
            .map(|pass| crate::encoder::pass_to_descriptor(pass, MetalHandle::NULL))
            .collect();
        vis_kernel(world);
        descriptors.push(crate::encoder::trailing_blit_descriptor(
            world.leading_blits(),
        ));
        gpu.run_descriptors(&descriptors);
        assert_eq!(enc.scene_caster_skips(), 0);
        let vis = enc
            .scene_textures()
            .expect("scene textures")
            .vis
            .into_retained()
            .expect("the visibility texture");
        let values = gpu.halves(&vis, VIS_RASTER);
        drop(vis);
        enc.shutdown_cleanup();
        Some(values)
    }
}

/// `shadows.sun` with `resolution` and `softness_pct`.
fn sun_with(resolution: u16, softness_pct: u16) -> Mtld3dConfig {
    Mtld3dConfig {
        shadows: ShadowConfig {
            resolution,
            softness_pct,
            ..sun().shadows
        },
        ..sun()
    }
}

/// A plate 60 units square floating 40 units over the floor, under the
/// middle of the view but off its axis (so a mirrored lookup misses the
/// shadow), with an overhead sun.
pub(super) fn plate_scene() -> VisScene {
    VisScene {
        camera: Camera::new(),
        quads: vec![
            Quad::floor(),
            Quad::level((-10.0, 50.0), (120.0, 180.0), CAMERA_HEIGHT - 40.0),
        ],
        sun: [0.0, 0.0, 1.0],
    }
}

/// The visibility of the pixels whose ray cast satisfies `class`.
pub(super) fn where_seen(
    vis: &[f32],
    seen: &[Option<Seen>],
    class: impl Fn(Seen) -> bool,
) -> Vec<f32> {
    vis.iter()
        .zip(seen)
        .filter(|(_, s)| s.is_some_and(&class))
        .map(|(&v, _)| v)
        .collect()
}

/// With the sun straight down, the floor under a floating plate is in
/// shadow (visibility ≈ 0) and the rest of the floor and the plate's top
/// are in full sun.
#[test]
fn vis_occluder_shadows_floor() {
    let scene = plate_scene();
    let Some(vis) = scene.run(sun()) else {
        return;
    };
    let seen = scene.seen(6.0);
    let shadowed = where_seen(&vis, &seen, |s| s == Seen::Shadowed(0));
    assert!(shadowed.len() >= 20, "{} shadowed pixels", shadowed.len());
    let darkest = shadowed.iter().copied().fold(0.0f32, f32::max);
    assert!(darkest <= 0.05, "under the plate: up to {darkest}");
    let lit = where_seen(&vis, &seen, |s| matches!(s, Seen::Lit(_)));
    assert!(lit.len() >= 2000, "{} lit pixels", lit.len());
    let dimmest = lit.iter().copied().fold(1.0f32, f32::min);
    assert!(dimmest >= 0.99, "in the sun: down to {dimmest}");
    let plate = where_seen(&vis, &seen, |s| s == Seen::Lit(1));
    assert!(plate.len() >= 20, "{} plate pixels", plate.len());
}

/// Where the camera sees nothing (the far plane), the sun is fully visible.
#[test]
fn vis_sky_is_one() {
    let scene = plate_scene();
    let Some(vis) = scene.run(sun()) else {
        return;
    };
    let seen = scene.seen(6.0);
    let sky = where_seen(&vis, &seen, |s| s == Seen::Sky);
    assert!(sky.len() >= 500, "{} sky pixels", sky.len());
    assert!(sky.iter().all(|&v| v == 1.0), "the sky is in full sun");
}

/// A flat floor under a sun 30° up, from the south (the camera's back) and
/// from the east (across the view), shadows nothing: no acne. Nor does a low
/// sun at 12° and 7°, where the depth slope is steep and only the normal
/// offset keeps the floor off its own depth.
#[test]
fn vis_no_acne_on_lit_plane() {
    for (elevation, from) in [
        (30.0, 180.0),
        (30.0, 90.0),
        (12.0, 180.0),
        (12.0, 90.0),
        (7.0, 135.0),
    ] {
        let scene = VisScene {
            camera: Camera::new(),
            quads: vec![Quad::floor()],
            sun: VisScene::sun(elevation, from),
        };
        let Some(vis) = scene.run(sun()) else {
            return;
        };
        let seen = scene.seen(0.0);
        let floor = where_seen(&vis, &seen, |s| s == Seen::Lit(0));
        assert!(floor.len() >= 4000, "{} floor pixels", floor.len());
        let acne = floor.iter().filter(|&&v| v < 0.99).count();
        let dimmest = floor.iter().copied().fold(1.0f32, f32::min);
        assert_eq!(
            acne, 0,
            "{elevation}° from {from}°: {acne} pixels down to {dimmest}"
        );
    }
}

/// A larger `softness` widens the penumbra: more floor pixels are partly lit.
#[test]
fn vis_softness_widens_penumbra() {
    let scene = plate_scene();
    // Every pixel that sees the floor, edges included: the penumbra is where
    // the shadow class changes.
    let floor: Vec<bool> = (0..VIS_RASTER * VIS_RASTER)
        .map(|i| scene.seen_at(i % VIS_RASTER, i / VIS_RASTER, 0.0).quad() == Some(0))
        .collect();
    let partial = |softness_pct| {
        scene.run(sun_with(1024, softness_pct)).map(|vis| {
            vis.iter()
                .zip(&floor)
                .filter(|&(&v, &on_floor)| on_floor && v > 0.1 && v < 0.9)
                .count()
        })
    };
    let (Some(hard), Some(soft)) = (partial(0), partial(400)) else {
        return;
    };
    assert!(hard > 0, "even the hardest edge is filtered");
    assert!(
        soft >= 2 * hard,
        "softness 4: {soft} penumbra pixels, 0: {hard}"
    );
}

/// With the sun 3° up, half way through the horizon fade, the long shadow
/// of a wall is half dark: never below 0.5, and about 0.5 inside it.
#[test]
fn vis_horizon_fade() {
    // A wall 60 wide and 40 tall standing on the floor 100 units north,
    // facing the sun in the south.
    let wall = Quad {
        origin: [-30.0, 100.0, CAMERA_HEIGHT],
        u: [60.0, 0.0, 0.0],
        v: [0.0, 0.0, -40.0],
    };
    let scene = VisScene {
        camera: Camera::new(),
        quads: vec![Quad::floor(), wall],
        sun: VisScene::sun(3.0, 180.0),
    };
    let Some(vis) = scene.run(sun()) else {
        return;
    };
    let seen = scene.seen(6.0);
    let surfaces = where_seen(&vis, &seen, |s| s.quad().is_some());
    let darkest = surfaces.iter().copied().fold(1.0f32, f32::min);
    assert!(darkest >= 0.5, "faded shadows stay at half: {darkest}");
    let shadowed = where_seen(&vis, &seen, |s| s == Seen::Shadowed(0));
    assert!(shadowed.len() >= 20, "{} shadowed pixels", shadowed.len());
    let lightest = shadowed.iter().copied().fold(0.0f32, f32::max);
    assert!(lightest <= 0.55, "the shadow is still there: {lightest}");
}

/// `shadows.debug` queues its kernel and the copy back at the boundary,
/// reading the back buffer, `predepth` and (for `visibility`) `vis`; with the
/// key off nothing is queued.
#[test]
fn debug_views_run_only_under_the_key() {
    for (debug, kernel) in [
        (ShadowDebug::Off, None),
        (ShadowDebug::Visibility, Some(Kernel::DebugVisibility)),
        (ShadowDebug::Cascades, Some(Kernel::DebugCascades)),
    ] {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let config = Mtld3dConfig {
            shadows: ShadowConfig {
                debug,
                ..sun().shadows
            },
            ..sun()
        };
        let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, config, (64, 64), 1)
        else {
            return;
        };
        let textures = *enc.scene_textures().expect("scene textures");
        world_draw(&mut enc);
        enc.post_boundary(&sun_boundary(Some(OVERHEAD), true));
        let blits = enc.pass_state.take_pending_leading_blits();
        let Some(kernel) = kernel else {
            assert!(blits.is_empty(), "nothing with the key off");
            enc.shutdown_cleanup();
            continue;
        };
        assert_eq!(blits.len(), 2, "{debug:?}: the kernel and the copy back");
        let (view, copy) = (blits[0], blits[1]);
        assert_eq!(view.cmd, BlitCommandType::SceneKernel as u32);
        assert_eq!(view.mip_level, kernel as u32);
        let scratch = enc
            .post
            .scratch_handle(crate::metal::post::Slot::ColourA)
            .expect("the colour scratch")
            .raw();
        let vis = if kernel == Kernel::DebugVisibility {
            textures.vis.raw()
        } else {
            0
        };
        assert_eq!(
            [
                view.src_handle,
                view.dst_handle,
                view.dst_offset,
                view.bytes_per_row
            ],
            [BACK_BUFFER, scratch, textures.predepth.raw(), vis]
        );
        assert_eq!((view.region_w, view.region_h), (64, 64));
        assert_ne!(view.byte_size, 0, "constants");
        assert_eq!(copy.cmd, BlitCommandType::CopyTextureToTexture as u32);
        assert_eq!((copy.src_handle, copy.dst_handle), (scratch, BACK_BUFFER));
        enc.shutdown_cleanup();
    }
}

/// Replays still waiting on a build and replays no caster can ever draw are
/// counted apart (the second is said once per world pipeline, at warn); a
/// drawn replay counts as neither.
#[test]
fn replay_skips_count_pending_and_failed_apart() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = scene_encoder(&coherent, &retained, sun(), (64, 64), 1) else {
        return;
    };
    world_draw(&mut enc);
    let d = enc.scene_draws()[0].clone();
    let mut skips = super::ReplaySkips::default();
    let ticket = mtld3d_core::async_compile::TicketSource::new().issue();
    skips.note(crate::encoder::replay::Replayed::Drawn, &d);
    skips.note(crate::encoder::replay::Replayed::Pending(ticket), &d);
    skips.note(crate::encoder::replay::Replayed::Failed, &d);
    skips.note(crate::encoder::replay::Replayed::Failed, &d);
    assert_eq!((skips.pending, skips.failed, skips.total()), (1, 2, 3));
    enc.shutdown_cleanup();
}
