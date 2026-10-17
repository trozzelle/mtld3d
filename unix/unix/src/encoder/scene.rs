//! The scene passes: the frame's scene textures and the passes spliced in ahead of the world.
//!
//! While a key that replays the world is on
//! ([`mtld3d_core::shadows::scene_capture_wanted`]), every submission has a
//! set of [`SceneTextures`]: native textures with no D3D9 resource behind
//! them, at the back buffer's raster size, that the scene passes write and
//! the world's receivers sample. [`assemble`] decides, once per submission
//! after the pass rules, what runs ahead of the world and splices it in
//! before the first pass that renders into the back buffer.
//!
//! Under `shadows.sun` the sun's cascades come first: when the boundary of
//! this submission saw the sun above the horizon, one depth pass per
//! cascade replays every captured world draw from the sun
//! ([`mtld3d_core::shadows::cascades`]) into its slice of the cascade array.
//!
//! A submission holding only part of the world (a mid-frame flush cut the
//! frame, see [`super::scene_capture::SceneCapture::world_split`]) replays
//! nothing: the captured records of the other part died with their
//! submission. Its receivers read cleared buffers instead: full sun
//! visibility, no occlusion, no bounce light, unshadowed point lights.
//!
//! A multisampled back buffer steps aside the way the post chain does: the
//! world draws into its companion, and the scene passes have no textures.

use mtld3d_core::{
    passes::{Pass, SceneColorTarget},
    shadows::cascades::{
        CascadeSet, SLOPE_BIAS, cascade_set, caster_depth_bias, projection, reproject_for,
        with_depth_offset,
    },
};
use mtld3d_shared::{
    BlitCommand, Command, MetalHandle,
    mtl::{DestroyKind, PixelFormat},
    mtl_handle::MTLTextureKind,
};

use super::{
    FrameEncoder, LOG_TARGET, PendingResourceRetention,
    replay::{Replayed, neutral_pos_fixup, replay_draw, reproject_bytes},
    scene_capture::WorldSplit,
};
use crate::metal::handle::IntoRetained;

#[cfg(test)]
mod tests;

/// What a split world's receivers read for sun visibility, ambient
/// occlusion and point-light visibility: fully lit, unoccluded, every light
/// unshadowed.
const UNSHADOWED: f32 = 1.0;
/// What a split world's receivers read for bounce light: none.
const NO_BOUNCE: f32 = 0.0;
/// The depth a cascade clears to: nothing casts.
const NO_CASTER: f32 = 1.0;

/// The textures the scene passes write and the world's receivers read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneTextures {
    /// Sun visibility, `R16Float`, raster size.
    pub vis: MetalHandle<MTLTextureKind>,
    /// Ambient occlusion, `R16Float`, raster size.
    pub ao: MetalHandle<MTLTextureKind>,
    /// One-bounce sunlight, `RGBA16Float`, raster size.
    pub gi: MetalHandle<MTLTextureKind>,
    /// The visibility of four point lights, one per channel, `RGBA16Float`, raster size.
    pub points: MetalHandle<MTLTextureKind>,
    /// The camera depth pre-pass, `Depth32Float`, raster size.
    pub predepth: MetalHandle<MTLTextureKind>,
    /// The sun's cascades, `Depth32Float` 2D array of `shadows.cascades`
    /// slices `shadows.resolution` square; `None` unless `shadows.sun` is on
    /// and `rt.shadows`, which supersedes them, is off.
    pub cascades: Option<MetalHandle<MTLTextureKind>>,
}

impl SceneTextures {
    /// Every texture of the set.
    pub fn handles(self) -> impl Iterator<Item = MetalHandle<MTLTextureKind>> {
        [self.vis, self.ao, self.gi, self.points, self.predepth]
            .into_iter()
            .chain(self.cascades)
    }
}

/// What a set of scene textures is made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SceneKey {
    width: u32,
    height: u32,
    samples: u8,
    /// Cascade side in texels, 0 without cascades.
    resolution: u16,
    /// Cascade count, 0 without cascades.
    cascades: u8,
}

/// The encoder's scene textures: the set of the current key, and whether this submission has it.
#[derive(Default)]
pub struct SceneTargets {
    held: Option<(SceneKey, SceneTextures)>,
    /// `held` serves this submission: scene capture is on, the back buffer
    /// is single-sampled and every texture was made.
    live: bool,
    /// The cascades this submission's caster passes rendered.
    cascades: Option<CascadeSet>,
    /// Caster replays skipped this submission, one per draw and cascade:
    /// a pipeline still building or a draw no caster can draw.
    caster_skips: u32,
}

impl SceneTargets {
    /// Empty the set, handing its textures to the caller (shutdown).
    pub fn take_all(&mut self) -> impl Iterator<Item = MetalHandle<MTLTextureKind>> {
        self.live = false;
        self.held
            .take()
            .into_iter()
            .flat_map(|(_, textures)| textures.handles())
    }
}

impl FrameEncoder {
    /// This submission's scene textures, `None` while the scene passes step aside.
    #[must_use]
    pub fn scene_textures(&self) -> Option<&SceneTextures> {
        if self.scene_targets.live {
            self.scene_targets
                .held
                .as_ref()
                .map(|(_, textures)| textures)
        } else {
            None
        }
    }

    /// The cascades this submission's caster passes rendered, `None` when none ran.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the sun-visibility kernel reads them")
    )]
    #[must_use]
    pub const fn scene_cascades(&self) -> Option<&CascadeSet> {
        self.scene_targets.cascades.as_ref()
    }

    /// Caster replays skipped this submission, one per draw and cascade.
    #[cfg(test)]
    #[must_use]
    pub const fn scene_caster_skips(&self) -> u32 {
        self.scene_targets.caster_skips
    }

    /// Ready this submission's scene textures for a back buffer rasterized at `raster` with `samples`.
    ///
    /// Nothing at all unless scene capture is on. A multisampled back buffer
    /// steps aside. A set of another key (a resize, a `Reset`) retires behind
    /// the GPU and a new one is made; when one cannot be, the scene passes
    /// step aside this submission.
    pub(super) fn prepare_scene_textures(&mut self, raster: (u32, u32), samples: u8) {
        self.scene_targets.live = false;
        self.scene_targets.cascades = None;
        self.scene_targets.caster_skips = 0;
        if !self.scene.wanted() {
            return;
        }
        if samples > 1 {
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "scene: the back buffer is multisampled; the scene passes step aside"
            );
            return;
        }
        let cascades = self.config.shadows.sun && !self.config.rt.shadows;
        let key = SceneKey {
            width: raster.0,
            height: raster.1,
            samples,
            resolution: if cascades {
                self.config.shadows.resolution
            } else {
                0
            },
            cascades: if cascades {
                self.config.shadows.cascades
            } else {
                0
            },
        };
        if let Some((held, textures)) = self.scene_targets.held {
            if held == key {
                self.scene_targets.live = true;
                return;
            }
            self.scene_targets.held = None;
            for texture in textures.handles() {
                self.retire_scene_texture(texture);
            }
        }
        if let Some(textures) = self.create_scene_textures(&key) {
            self.scene_targets.held = Some((key, textures));
            self.scene_targets.live = true;
        } else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "scene: the {}x{} scene textures could not be made; the scene passes step aside",
                key.width,
                key.height
            );
        }
    }

    /// Every texture of `key`, or `None` (retiring the ones made) when one cannot be made.
    fn create_scene_textures(&mut self, key: &SceneKey) -> Option<SceneTextures> {
        let device = self.device_handle.into_retained()?;
        let (width, height) = (key.width, key.height);
        let raster = [
            PixelFormat::R16Float,
            PixelFormat::R16Float,
            PixelFormat::Rgba16Float,
            PixelFormat::Rgba16Float,
            PixelFormat::Depth32Float,
        ];
        let mut created = Vec::with_capacity(raster.len() + 1);
        let mut create = |width: u32, height: u32, layers: u32, format: PixelFormat| {
            let texture =
                crate::metal::create_scene_texture(&device, width, height, layers, format)?;
            created.push(texture);
            Some(texture)
        };
        let textures = (|| {
            let [vis, ao, gi, points, predepth] =
                raster.map(|format| create(width, height, 1, format));
            let cascades = if key.cascades == 0 {
                None
            } else {
                let side = u32::from(key.resolution);
                Some(create(
                    side,
                    side,
                    u32::from(key.cascades),
                    PixelFormat::Depth32Float,
                )?)
            };
            Some(SceneTextures {
                vis: vis?,
                ao: ao?,
                gi: gi?,
                points: points?,
                predepth: predepth?,
                cascades,
            })
        })();
        if textures.is_none() {
            for texture in created {
                self.retire_scene_texture(texture);
            }
        }
        textures
    }

    /// Destroy `texture` once the GPU is past this submission.
    fn retire_scene_texture(&mut self, texture: MetalHandle<MTLTextureKind>) {
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Texture,
                handle: texture.raw(),
                page_box: None,
                staging_arc: None,
                seq: self.current_submit_seq,
                from_texture: true,
            });
    }
}

/// What one submission runs ahead of its world.
#[derive(Default)]
struct SceneStages {
    /// Passes spliced in before the world, in order.
    before_world: Vec<Pass>,
    /// Blits and dispatches leading the world's first pass, after `before_world`.
    world_leading: Vec<BlitCommand>,
}

impl SceneStages {
    /// Clear what the receivers read to the values of a world with no scene passes.
    ///
    /// Two clear passes, since a render pass carries one clear colour: the
    /// visibility, occlusion and point-light targets to 1, the bounce light to 0.
    fn clear_scene_textures_for_split(&mut self, textures: &SceneTextures, raster: (u32, u32)) {
        let target = |texture, format| SceneColorTarget {
            texture,
            size: raster,
            format,
        };
        self.before_world.push(Pass::scene_clear(
            &[
                target(textures.vis, PixelFormat::R16Float),
                target(textures.ao, PixelFormat::R16Float),
                target(textures.points, PixelFormat::Rgba16Float),
            ],
            [UNSHADOWED; 4],
        ));
        self.before_world.push(Pass::scene_clear(
            &[target(textures.gi, PixelFormat::Rgba16Float)],
            [NO_BOUNCE; 4],
        ));
    }

    /// One caster pass per cascade into slice `i` of `cascades`, when this submission's boundary saw the sun up.
    ///
    /// Each pass clears its slice to [`NO_CASTER`] and replays every
    /// captured draw through the cascade's re-projection, with the constant
    /// bias folded into the matrix and the slope bias set. Casters outside a
    /// cascade are replayed all the same. A caster whose pipeline is still
    /// building, or which no caster can draw, is skipped this frame and
    /// counted; nothing waits. Records the set for the receivers.
    fn cast_cascades(&mut self, enc: &mut FrameEncoder, cascades: MetalHandle<MTLTextureKind>) {
        let Some(boundary) = enc.scene.boundary().copied() else {
            return;
        };
        let Some(sun) = boundary.sun_dir_eye.filter(|_| boundary.sun_above_horizon) else {
            return;
        };
        let shadows = enc.config.shadows;
        let Some(set) = cascade_set(
            &boundary.view,
            boundary.proj,
            sun,
            (
                f32::from(shadows.distance),
                shadows.cascades,
                shadows.resolution,
            ),
        ) else {
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "scene: the world's projection or view admits no cascades; the sun casts nothing"
            );
            return;
        };
        let proj = projection(boundary.proj);
        let side = u32::from(shadows.resolution);
        let draws = enc.scene.take_draws();
        let mut skips = 0u32;
        for (slice, light_vp) in (0u32..).zip(&set.matrices[..usize::from(set.count)]) {
            let biased =
                with_depth_offset(light_vp, caster_depth_bias(light_vp, shadows.resolution));
            let reproject = reproject_bytes(enc, &reproject_for(&biased, &proj));
            let mut commands = Vec::with_capacity(1 + draws.len() * 16);
            commands.push(Command::set_viewport(0, 0, side, side, 0.0, 1.0));
            for d in &draws {
                let replayed = replay_draw(
                    enc,
                    &mut commands,
                    d,
                    reproject,
                    neutral_pos_fixup(),
                    (0.0, SLOPE_BIAS),
                );
                if replayed != Replayed::Drawn {
                    skips += 1;
                }
            }
            self.before_world.push(Pass::scene_depth(
                cascades,
                (side, side),
                slice,
                NO_CASTER,
                commands,
            ));
        }
        enc.scene.put_draws(draws);
        if skips > 0 {
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "scene: {skips} caster replays skipped this frame (pipelines building or no \
                 caster for the draw); later frames retry"
            );
        }
        enc.scene_targets.cascades = Some(set);
        enc.scene_targets.caster_skips = skips;
    }
}

/// Splice this submission's scene stages in ahead of its world.
///
/// The one place that decides what runs before the world. Called by
/// `finalize_submit` after the pass rules and their debug checks, so no rule
/// reasons over a scene pass and every pass the scene passes carry keeps the
/// load and store actions it was built with. Nothing at all while the scene
/// passes step aside ([`FrameEncoder::scene_textures`]). A split world
/// (`no_present` is this submission's) clears the receivers' textures and
/// replays nothing, said once per cause.
pub(super) fn assemble(enc: &mut FrameEncoder, no_present: bool) {
    let Some(&textures) = enc.scene_textures() else {
        return;
    };
    let raster = enc
        .scene_targets
        .held
        .map(|(key, _)| (key.width, key.height))
        .expect("live scene textures are held");
    let mut stages = SceneStages::default();
    if let Some(split) = enc.scene.world_split(no_present) {
        match split {
            WorldSplit::Flushed => mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "scene: a mid-frame flush cut the world before the 3D/UI boundary; its receivers \
                 read cleared scene buffers"
            ),
            WorldSplit::Continued => mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "scene: the world continues after a mid-frame flush; its receivers read cleared \
                 scene buffers"
            ),
        }
        enc.scene.mark_split();
        stages.clear_scene_textures_for_split(&textures, raster);
    } else if let Some(cascades) = textures.cascades {
        stages.cast_cascades(enc, cascades);
    }
    if stages.before_world.is_empty() && stages.world_leading.is_empty() {
        return;
    }
    if !enc
        .pass_state
        .insert_scene_passes(stages.before_world, stages.world_leading)
    {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "scene: no pass of the submission renders into the back buffer; the scene passes step \
             aside"
        );
    }
}
