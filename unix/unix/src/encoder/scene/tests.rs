//! Scene passes: the scene textures' cache and what a submission splices in ahead of the world.

use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    config::Mtld3dConfig,
    dxso::FfVsFlags,
    passes::{ColorLoad, Pass, PassColorAttachment, StoreAction},
    rt::RtConfig,
    shadows::ShadowConfig,
};
use mtld3d_shared::{Command, MetalHandle, mtl::PrimitiveType, mtl_handle::MTLTextureKind};
use objc2_metal::{MTLPixelFormat, MTLTexture, MTLTextureType};

use super::{SceneTextures, assemble};
use crate::{
    encoder::{
        FrameEncoder,
        scene_capture::tests::{Bytes, Declaration, Draw, buffer_binds, ff_ps, ff_vs, run},
        tests::{encoder_with_config, post_reset_frame},
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
    enc.scene.note_boundary();
    assemble(&mut enc, false);
    assert!(enc.scene_split(), "the continuation is split too");
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 3);
    assert_eq!(clears(&passes[..2]), expected);

    // The next whole frame.
    begin(&mut enc, (64, 64), 1, false);
    assert!(!enc.scene_split());
    world_draw(&mut enc);
    enc.scene.note_boundary();
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
    enc.scene.note_boundary();
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
    assemble(&mut enc, false);
    assert_eq!(enc.pass_state.passes().len(), 1);
    assert!(!enc.scene_split());
    assert!(enc.scene_targets.held.is_none());
    assert_eq!(enc.pending_resource_retention.len(), retained_before);
    enc.shutdown_cleanup();
}
