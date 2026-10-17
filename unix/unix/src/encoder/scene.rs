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
//! The camera depth pre-pass then replays them again as the camera drew
//! them, into `predepth`, and one compute dispatch leading the world's
//! first pass turns both into the sun's visibility, `vis`
//! ([`crate::metal::scene_vis`]). A submission with no cascades (no sun, a
//! sun below the horizon, `shadows.sun` off) clears `vis` to full sun, so a
//! receiver never reads another frame's. `shadows.debug` shows `vis`, or the
//! cascade of each pixel, over the world at the 3D/UI boundary
//! ([`FrameEncoder::scene_debug_view`]).
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
    encoder_data::PostBoundary,
    passes::{Pass, SceneColorTarget},
    shadows::{
        ShadowConfig, ShadowDebug,
        cascades::{
            CascadeSet, SLOPE_BIAS, cascade_set, caster_depth_bias, projection, reproject_for,
            with_depth_offset,
        },
    },
};
use mtld3d_shared::{
    BlitCommand, Command, MetalHandle, SceneKernelInfo,
    mtl::{DestroyKind, PixelFormat},
    mtl_handle::MTLTextureKind,
};

use super::{
    FrameEncoder, LOG_TARGET, PendingResourceRetention,
    replay::{IDENTITY, Replayed, neutral_pos_fixup, replay_draw, reproject_bytes},
    scene_capture::{CapturedDraw, WorldSplit},
};
use crate::metal::{
    handle::IntoRetained,
    post::Slot,
    scene_vis::{Kernel, SceneConstants, SceneKernels},
};

#[cfg(test)]
mod tests;

/// What a split world's receivers read for sun visibility, ambient
/// occlusion and point-light visibility: fully lit, unoccluded, every light
/// unshadowed. A world with no cascades reads it for sun visibility too.
const UNSHADOWED: f32 = 1.0;
/// What a split world's receivers read for bounce light: none.
const NO_BOUNCE: f32 = 0.0;
/// The depth a cascade clears to: nothing casts.
const NO_CASTER: f32 = 1.0;
/// The depth the pre-pass clears to: the far plane, where nothing was drawn.
const FAR_PLANE: f32 = 1.0;

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
    /// Replays skipped this submission's cascades, one per draw and
    /// cascade: a pipeline still building or a draw no caster can draw.
    caster_skips: u32,
    /// The scene kernels' pipelines, built on first use.
    kernels: SceneKernels,
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

    /// Release the scene kernels' pipelines (shutdown).
    pub fn destroy_kernels(&mut self) {
        self.kernels.destroy();
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
    /// step aside this submission. The receivers start the submission with
    /// no uniform and no bindings.
    pub(super) fn prepare_scene_textures(&mut self, raster: (u32, u32), samples: u8) {
        self.scene_targets.live = false;
        self.scene_targets.cascades = None;
        self.scene_targets.caster_skips = 0;
        self.receivers.begin_submission();
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
        let mut create = |width: u32, height: u32, layers: Option<u32>, format: PixelFormat| {
            let texture =
                crate::metal::create_scene_texture(&device, width, height, layers, format)?;
            created.push(texture);
            Some(texture)
        };
        let textures = (|| {
            let [vis, ao, gi, points, predepth] =
                raster.map(|format| create(width, height, None, format));
            let cascades = if key.cascades == 0 {
                None
            } else {
                let side = u32::from(key.resolution);
                Some(create(
                    side,
                    side,
                    Some(u32::from(key.cascades)),
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

    /// The raster the live scene textures were made for.
    fn scene_raster(&self) -> (u32, u32) {
        self.scene_targets
            .held
            .map(|(key, _)| (key.width, key.height))
            .expect("live scene textures are held")
    }

    /// The viewport the world drew through, `[x, y, width, height]` in raster pixels: the first captured draw's, the whole raster without one.
    fn world_viewport(&self, raster: (u32, u32)) -> [f32; 4] {
        let rect = self
            .scene_draws()
            .first()
            .map_or((0, 0, raster.0, raster.1), |d| d.viewport.rect);
        // Raster dimensions fit u16 (as `emit_draw` converts them).
        <[u32; 4]>::from(rect).map(|v| f32::from(u16::try_from(v).unwrap_or(u16::MAX)))
    }

    /// `shadows.debug`: the sun visibility or the cascades over the world, at the 3D/UI boundary.
    ///
    /// Queued, like a post effect, after every pass and clear recorded so
    /// far: one debug kernel reads the back buffer into the post chain's
    /// first colour scratch, which is copied back. Nothing unless the key is
    /// on and the scene textures are live; steps aside, said once, when the
    /// boundary draw does not target the back buffer at the scene raster or
    /// the kernel or the scratch cannot be made. The cascades are the ones
    /// the boundary's sun casts; without a sun up nothing is tinted.
    pub(super) fn scene_debug_view(&mut self, boundary: &PostBoundary) {
        let kernel = match self.config.shadows.debug {
            ShadowDebug::Off => return,
            ShadowDebug::Visibility => Kernel::DebugVisibility,
            ShadowDebug::Cascades => Kernel::DebugCascades,
        };
        let Some(&textures) = self.scene_textures() else {
            return;
        };
        let raster = self.scene_raster();
        let back_buffer = self.pass_state.backbuffer_texture();
        if back_buffer.is_null()
            || self.pass_state.current_color_texture() != back_buffer
            || self.pass_state.current_color_size() != raster
        {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "scene: the boundary draw does not target the back buffer at the scene raster; \
                 the shadow debug view steps aside"
            );
            return;
        }
        let format = self.pass_state.current_color_base_format();
        let (Some(pipeline), Some(output)) = (
            self.scene_targets
                .kernels
                .pipeline(self.device_handle, kernel),
            self.post_scratch(Slot::ColourA, raster, format),
        ) else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "scene: the {kernel:?} kernel or its scratch could not be made; the shadow debug \
                 view steps aside"
            );
            return;
        };
        let set = textures
            .cascades
            .and_then(|_| frame_cascades(boundary, &self.config.shadows));
        let constants = SceneConstants::new(
            boundary,
            set.as_ref(),
            &self.config.shadows,
            raster,
            self.world_viewport(raster),
        );
        let constants = self.alloc_scratch(constants.as_bytes());
        let vis = if kernel == Kernel::DebugVisibility {
            textures.vis.raw()
        } else {
            0
        };
        self.pass_state.push_leading_blit_after_clears(
            BlitCommand::scene_kernel(&SceneKernelInfo {
                kernel: kernel as u32,
                pipeline: pipeline.raw(),
                textures: [
                    back_buffer.raw(),
                    output.raw(),
                    textures.predepth.raw(),
                    vis,
                ],
                constants,
                width: raster.0,
                height: raster.1,
            }),
            "scene_debug",
        );
        self.pass_state
            .push_pending_leading_blit(BlitCommand::copy_texture_to_texture_full_mip(
                output.raw(),
                back_buffer.raw(),
                0,
                raster.0,
                raster.1,
            ));
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

    /// Full sun for every receiver this frame: one clear pass of `vis` to [`UNSHADOWED`].
    fn clear_visibility(&mut self, textures: &SceneTextures, raster: (u32, u32)) {
        self.before_world.push(Pass::scene_clear(
            &[SceneColorTarget {
                texture: textures.vis,
                size: raster,
                format: PixelFormat::R16Float,
            }],
            [UNSHADOWED; 4],
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
        let Some(set) = frame_cascades(&boundary, &enc.config.shadows) else {
            return;
        };
        let shadows = enc.config.shadows;
        let proj = projection(boundary.proj);
        let side = u32::from(shadows.resolution);
        let draws = enc.scene.take_draws();
        let mut skips = ReplaySkips::default();
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
                skips.note(replayed, d);
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
        skips.say(ScenePass::Cascades);
        enc.scene_targets.cascades = Some(set);
        enc.scene_targets.caster_skips = skips.total();
    }

    /// The camera depth pre-pass: every captured draw again into `predepth`, as the world drew it.
    ///
    /// Each replay binds the identity re-projection, the draw's own camera
    /// `pos_fixup` and slope bias, under the viewport it drew through, so
    /// `predepth` holds the depth the world rasterized for every captured
    /// draw, alpha-tested holes included. A replay that cannot draw yet is
    /// skipped, as a caster's is.
    fn depth_prepass(
        &mut self,
        enc: &mut FrameEncoder,
        predepth: MetalHandle<MTLTextureKind>,
        raster: (u32, u32),
    ) {
        let draws = enc.scene.take_draws();
        let identity = reproject_bytes(enc, &IDENTITY);
        let mut commands = Vec::with_capacity(1 + draws.len() * 16);
        let mut viewport = None;
        let mut skips = ReplaySkips::default();
        for d in &draws {
            if viewport != Some(d.viewport) {
                commands.push(d.viewport.command());
                viewport = Some(d.viewport);
            }
            let replayed = replay_draw(
                enc,
                &mut commands,
                d,
                identity,
                d.pos_fixup,
                (0.0, d.slope_scale),
            );
            skips.note(replayed, d);
        }
        enc.scene.put_draws(draws);
        skips.say(ScenePass::DepthPrepass);
        self.before_world
            .push(Pass::scene_depth(predepth, raster, 0, FAR_PLANE, commands));
    }

    /// The sun's visibility from this submission's cascades, ahead of the world.
    ///
    /// The depth pre-pass, then one [`Kernel::SunVisCascades`] dispatch over
    /// the raster leading the world's first pass. False, queuing nothing,
    /// when the kernel cannot be built.
    fn sun_visibility(
        &mut self,
        enc: &mut FrameEncoder,
        textures: &SceneTextures,
        (cascades, set): (MetalHandle<MTLTextureKind>, &CascadeSet),
        raster: (u32, u32),
    ) -> bool {
        let Some(boundary) = enc.scene.boundary().copied() else {
            return false;
        };
        let Some(pipeline) = enc
            .scene_targets
            .kernels
            .pipeline(enc.device_handle, Kernel::SunVisCascades)
        else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "scene: the sun visibility kernel could not be built; the world reads full sun"
            );
            return false;
        };
        self.depth_prepass(enc, textures.predepth, raster);
        let constants = SceneConstants::new(
            &boundary,
            Some(set),
            &enc.config.shadows,
            raster,
            enc.world_viewport(raster),
        );
        let constants = enc.alloc_scratch(constants.as_bytes());
        self.world_leading
            .push(BlitCommand::scene_kernel(&SceneKernelInfo {
                kernel: Kernel::SunVisCascades as u32,
                pipeline: pipeline.raw(),
                textures: [
                    textures.predepth.raw(),
                    textures.vis.raw(),
                    cascades.raw(),
                    0,
                ],
                constants,
                width: raster.0,
                height: raster.1,
            }));
        true
    }
}

/// The replays of one scene pass that drew nothing.
#[derive(Default)]
struct ReplaySkips {
    /// Still building: a later frame draws them.
    pending: u32,
    /// No caster can draw them, ever.
    failed: u32,
}

impl ReplaySkips {
    /// Count `replayed` unless it drew; a draw that cannot be replayed is said once per world pipeline.
    fn note(&mut self, replayed: Replayed, d: &CapturedDraw) {
        match replayed {
            Replayed::Drawn => {}
            Replayed::Pending(_) => self.pending += 1,
            Replayed::Failed => {
                self.failed += 1;
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: d.pipeline,
                    "scene: a world draw (pipeline {:#x}) cannot be replayed; it casts no \
                     shadow and its depth is missing from the pre-pass",
                    d.pipeline
                );
            }
        }
    }

    /// Say once (per call site) that replays of `pass` waited on builds.
    fn say(&self, pass: ScenePass) {
        if self.pending == 0 {
            return;
        }
        match pass {
            ScenePass::Cascades => mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "scene: {} caster replays skipped this frame while their pipelines build; later \
                 frames draw them",
                self.pending
            ),
            ScenePass::DepthPrepass => mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "scene: {} depth pre-pass replays skipped this frame while their pipelines \
                 build; later frames draw them",
                self.pending
            ),
        }
    }

    const fn total(&self) -> u32 {
        self.pending + self.failed
    }
}

/// The scene passes that replay the world.
#[derive(Clone, Copy)]
enum ScenePass {
    Cascades,
    DepthPrepass,
}

/// The cascades of the frame whose boundary is `boundary`, `None` without a sun above the horizon.
fn frame_cascades(boundary: &PostBoundary, shadows: &ShadowConfig) -> Option<CascadeSet> {
    let sun = boundary
        .sun_dir_eye
        .filter(|_| boundary.sun_above_horizon)?;
    let set = cascade_set(
        &boundary.view,
        boundary.proj,
        sun,
        (
            f32::from(shadows.distance),
            shadows.cascades,
            shadows.resolution,
        ),
    );
    if set.is_none() {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "scene: the world's projection or view admits no cascades; the sun casts nothing"
        );
    }
    set
}

/// Splice this submission's scene stages in ahead of its world.
///
/// The one place that decides what runs before the world. Called by
/// `finalize_submit` after the pass rules and their debug checks, so no rule
/// reasons over a scene pass and every pass the scene passes carry keeps the
/// load and store actions it was built with. Nothing at all while the scene
/// passes step aside ([`FrameEncoder::scene_textures`]). A split world
/// (`no_present` is this submission's) clears the receivers' textures and
/// replays nothing, said once per cause. Otherwise `vis` is written every
/// submission: from the cascades when the sun cast them, full sun (a clear)
/// when it did not.
pub(super) fn assemble(enc: &mut FrameEncoder, no_present: bool) {
    let Some(&textures) = enc.scene_textures() else {
        return;
    };
    let raster = enc.scene_raster();
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
    } else {
        if let Some(cascades) = textures.cascades {
            stages.cast_cascades(enc, cascades);
        }
        let sun = textures.cascades.zip(enc.scene_cascades().copied());
        let lit = sun.is_some_and(|(cascades, set)| {
            stages.sun_visibility(enc, &textures, (cascades, &set), raster)
        });
        if !lit {
            stages.clear_visibility(&textures, raster);
        }
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
