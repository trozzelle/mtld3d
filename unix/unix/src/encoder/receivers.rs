//! Scene receivers: the world's lit draws read the scene textures.
//!
//! While `shadows.sun` or `rt.shadows` is on
//! ([`mtld3d_core::shadows::sun_receivers_wanted`]), every lit draw with a
//! sun term into the back buffer recorded before this frame's 3D/UI
//! boundary reads the sun's visibility (`vis`) at its own pixel: its pixel
//! stage variant carries [`SCENE_VIS_SUN`] in `VariantKey::scene_vis`. A
//! draw lit per pixel scales the sun's own terms; any other lit draw, which
//! has only its vertex-lit colour, scales that by `mix(k, 1, vis)` with `k`
//! from the receiver uniform ([`SceneReceiver`]). Unlit and pre-transformed
//! draws, every draw into another target or after the boundary, and every
//! draw while the scene passes step aside read nothing.
//!
//! The ray-traced terms follow the same rule, each under its key:
//! - `rt.ao` ([`SCENE_VIS_AO`]): every such lit draw. A per-pixel
//!   fixed-function draw scales the ambient part its vertex stage writes
//!   to `texcoord8` (its VS keyed with `FfVsKey::SCENE_AMBIENT`), a
//!   per-pixel figure its ambient row; every other one, which cannot tell
//!   its ambient apart, scales its vertex-lit colour by
//!   `mix(1, ao, rt.ao · k)`, `k` the ambient's share (the sun's fallback).
//! - `rt.gi` ([`SCENE_VIS_GI`]): the per-pixel draws (fixed-function and
//!   figures), whose lit sum takes the bounce light.
//! - `rt.pointShadows` ([`SCENE_VIS_POINTS`]): the per-pixel
//!   fixed-function draws that light a point light per pixel (the key
//!   forces every point light per pixel, `mtld3d_core::shadows::ppl_force`);
//!   figures keep their point lights unshadowed.
//!
//! While draws read the visibility, the game's own Classic blob shadows
//! into the back buffer before the boundary are left out
//! ([`FrameEncoder::classic_blob_skipped`]): the sun's visibility shadows
//! the figures itself.
//!
//! The scene textures sit at fragment textures [`PS_SCENE_VIS_TEXTURE`] and
//! up and the uniform at fragment buffer [`PS_SCENE_RECEIVER_SLOT`], outside
//! the slots `LastBoundCache` tracks, so the first receiver of each pass
//! binds the textures of the keys on and the uniform, and the rest of the
//! pass reuses them. The uniform's values are the boundary's, known only
//! once the world is recorded: the draws bind one copy in this submission's
//! scratch arena, and [`FrameEncoder::fill_scene_receiver`] writes it at
//! finalize, before anything encodes.

use mtld3d_core::{
    config::Mtld3dConfig,
    draw_data::{FixedVsSource, VsSourceView},
    dxso::{SCENE_VIS_AO, SCENE_VIS_GI, SCENE_VIS_POINTS, SCENE_VIS_SUN, VariantKey},
    pipeline_state::PipelineRsBits,
    shadows::{BlobFacts, SceneReceiver, is_classic_blob},
};
use mtld3d_shared::{
    Command,
    mtl::{
        PS_SCENE_AO_TEXTURE, PS_SCENE_GI_TEXTURE, PS_SCENE_POINTS_TEXTURE, PS_SCENE_RECEIVER_SLOT,
        PS_SCENE_VIS_TEXTURE,
    },
};

use super::FrameEncoder;

#[cfg(test)]
mod tests;

/// The encoder's receiver state, see the module docs.
#[derive(Default)]
pub struct SceneReceivers {
    /// The `scene_vis` bits this encoder's configuration produces:
    /// [`SCENE_VIS_SUN`] under [`mtld3d_core::shadows::sun_receivers_wanted`],
    /// [`SCENE_VIS_AO`] / [`SCENE_VIS_GI`] / [`SCENE_VIS_POINTS`] under
    /// `rt.ao` / `rt.gi` / `rt.pointShadows`.
    keys: u8,
    /// This submission's receiver uniform in the scratch arena; 0 until a receiver binds it.
    uniform: u64,
    /// The open pass has the receiver bindings.
    bound: bool,
}

impl SceneReceivers {
    /// The receivers of `config` (whose `rt.*` keys the device can trace).
    #[must_use]
    pub const fn new(config: &Mtld3dConfig) -> Self {
        let mut keys = 0;
        if mtld3d_core::shadows::sun_receivers_wanted(config) {
            keys |= SCENE_VIS_SUN;
        }
        if config.rt.ao_pct > 0 {
            keys |= SCENE_VIS_AO;
        }
        if config.rt.gi_pct > 0 {
            keys |= SCENE_VIS_GI;
        }
        if config.rt.point_shadows {
            keys |= SCENE_VIS_POINTS;
        }
        Self {
            keys,
            uniform: 0,
            bound: false,
        }
    }

    /// Start a submission: the last one's uniform died with its arena.
    pub const fn begin_submission(&mut self) {
        self.uniform = 0;
        self.bound = false;
    }

    /// A pass opened: it holds no receiver bindings yet.
    pub const fn pass_opened(&mut self) {
        self.bound = false;
    }
}

impl FrameEncoder {
    /// The `VariantKey::scene_vis` bits of the draw being emitted, whose vertex stage is `vs` and pixel variant `variant`.
    ///
    /// Only for a lit draw into the back buffer before the boundary, while
    /// the scene textures are live, and only bits of the keys on:
    /// - a fixed-function VS lit with a normal (not pre-transformed) —
    ///   with a fixed-function or a programmable pixel stage alike:
    ///   [`SCENE_VIS_SUN`] with an enabled directional light (per pixel
    ///   when `variant` lights its directionals per pixel, vertex-lit
    ///   otherwise, a `ps_3_0` pairing); [`SCENE_VIS_AO`] either way;
    ///   [`SCENE_VIS_GI`] per pixel; [`SCENE_VIS_POINTS`] when `variant`
    ///   lights a point light per pixel;
    /// - a figure VS: sun and occlusion, and bounce light with the figure
    ///   terms (`variant.ppl_figure`); never point visibility;
    /// - the `SpeedTree` leaf VS, vertex-lit: sun and occlusion.
    ///
    /// Zero for everything else: unlit or pre-transformed fixed-function
    /// draws, lit ones without a normal (the VS zeroes every N·L term, so
    /// their colour is ambient and emissive only) and every other
    /// programmable VS.
    #[must_use]
    pub fn scene_receiver_bits(&self, vs: VsSourceView<'_>, variant: &VariantKey) -> u8 {
        let keys = self.receivers.keys;
        if keys == 0 || !self.scene_capture_armed() || self.scene_textures().is_none() {
            return 0;
        }
        let back_buffer = self.pass_state.backbuffer_texture();
        if back_buffer.is_null() || self.pass_state.current_color_texture() != back_buffer {
            return 0;
        }
        let bits = match vs {
            VsSourceView::FixedFunction(FixedVsSource { key, .. }) => {
                if !key.lighting_enabled() || !key.has_normal() || key.has_rhw() {
                    return 0;
                }
                let per_pixel = variant.ppl_mask != 0;
                let sun = if per_pixel {
                    variant.ppl_directional_mask != 0
                } else {
                    key.light_active_mask & key.light_directional_mask != 0
                };
                let points = per_pixel && variant.ppl_mask & !variant.ppl_directional_mask != 0;
                let mut bits = SCENE_VIS_AO;
                if sun {
                    bits |= SCENE_VIS_SUN;
                }
                if per_pixel {
                    bits |= SCENE_VIS_GI;
                }
                if points {
                    bits |= SCENE_VIS_POINTS;
                }
                bits
            }
            VsSourceView::Programmable(source) => {
                let id = source.vs_id.raw();
                if variant.ppl_figure != 0 {
                    SCENE_VIS_SUN | SCENE_VIS_AO | SCENE_VIS_GI
                } else if mtld3d_core::ppl::figures::figure_vs(id).is_some()
                    || mtld3d_core::ppl::figures::speedtree_vs(id)
                {
                    SCENE_VIS_SUN | SCENE_VIS_AO
                } else {
                    0
                }
            }
        };
        bits & keys
    }

    /// Whether the draw being emitted is one of the game's Classic blob shadows to leave out.
    ///
    /// While the sun's visibility darkens the world (`shadows.sun` or
    /// `rt.shadows`, [`mtld3d_core::shadows::sun_receivers_wanted`]) and the
    /// scene textures are live, a draw into the back buffer before the
    /// boundary that [`is_classic_blob`] recognises would darken it
    /// twice. With the scene passes stepped aside (a multisampled back
    /// buffer) or only `rt.ao`/`rt.gi`/`rt.pointShadows` on, blobs draw.
    /// `vs` is the draw's vertex stage, `pipeline_rs` its blend state,
    /// `depth_write` whether it writes depth and `vertices` how many it
    /// assembles (see [`BlobFacts::vertices`]).
    #[must_use]
    pub fn classic_blob_skipped(
        &self,
        vs: VsSourceView<'_>,
        pipeline_rs: &PipelineRsBits,
        depth_write: bool,
        vertices: u32,
    ) -> bool {
        if self.receivers.keys & SCENE_VIS_SUN == 0
            || !self.scene_capture_armed()
            || self.scene_textures().is_none()
        {
            return false;
        }
        let back_buffer = self.pass_state.backbuffer_texture();
        if back_buffer.is_null() || self.pass_state.current_color_texture() != back_buffer {
            return false;
        }
        let lit = match vs {
            VsSourceView::FixedFunction(FixedVsSource { key, .. }) => key.lighting_enabled(),
            VsSourceView::Programmable(_) => true,
        };
        is_classic_blob(&BlobFacts {
            pipeline_rs: *pipeline_rs,
            depth_write,
            lit,
            vertices,
        })
    }

    /// Bind what a receiver reads, once per pass: the scene texture of each
    /// key on (the sun's visibility, the occlusion, the bounce light, the
    /// point-light visibility) and the receiver uniform.
    ///
    /// Only a draw [`Self::scene_receiver_bits`] flagged calls it, so the
    /// scene textures are live.
    pub fn bind_scene_receivers(&mut self) {
        if self.receivers.bound {
            return;
        }
        let textures = *self
            .scene_textures()
            .expect("a receiver draws while the scene textures are live");
        if self.receivers.uniform == 0 {
            self.receivers.uniform = self.alloc_scratch(&SceneReceiver::UNSHADOWED.to_bytes());
        }
        let keys = self.receivers.keys;
        for (bit, texture, slot) in [
            (SCENE_VIS_SUN, textures.vis, PS_SCENE_VIS_TEXTURE),
            (SCENE_VIS_AO, textures.ao, PS_SCENE_AO_TEXTURE),
            (SCENE_VIS_GI, textures.gi, PS_SCENE_GI_TEXTURE),
            (SCENE_VIS_POINTS, textures.points, PS_SCENE_POINTS_TEXTURE),
        ] {
            if keys & bit != 0 {
                self.emit_command(Command::set_fragment_texture(texture.raw(), slot));
            }
        }
        self.emit_command(Command::set_fragment_bytes_at(
            self.receivers.uniform,
            u32::try_from(size_of::<SceneReceiver>()).expect("the receiver uniform fits u32"),
            PS_SCENE_RECEIVER_SLOT,
        ));
        self.receivers.bound = true;
    }

    /// Write this submission's receiver uniform from its boundary, before anything encodes.
    ///
    /// [`SceneReceiver::new`] of the boundary when it fired in this
    /// submission; full sun (`k = 1`, no darkening) and no point light
    /// matched when it did not, as on the flushed part of a split frame.
    /// Nothing when no receiver drew.
    pub fn fill_scene_receiver(&mut self) {
        if self.receivers.uniform == 0 {
            return;
        }
        let bytes = SceneReceiver::new(self.scene.boundary(), &self.config.rt).to_bytes();
        let uniform = core::ptr::with_exposed_provenance_mut::<u8>(
            usize::try_from(self.receivers.uniform).expect("a host address fits usize"),
        );
        // SAFETY: `uniform` is this submission's `alloc_scratch` copy of
        // `bytes.len()` bytes; the arena keeps it until the payload retires,
        // and nothing reads it before the submission encodes.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), uniform, bytes.len()) };
    }
}
