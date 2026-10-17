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
//! While draws read the visibility, the game's own Classic blob shadows
//! into the back buffer before the boundary are left out
//! ([`FrameEncoder::classic_blob_skipped`]): the sun's visibility shadows
//! the figures itself.
//!
//! `vis` sits at fragment texture [`PS_SCENE_VIS_TEXTURE`] and the uniform
//! at fragment buffer [`PS_SCENE_RECEIVER_SLOT`], outside the slots
//! `LastBoundCache` tracks, so the first receiver of each pass binds both and
//! the rest of the pass reuses them. The uniform's values are the boundary's,
//! known only once the world is recorded: the draws bind one copy in this
//! submission's scratch arena, and [`FrameEncoder::fill_scene_receiver`]
//! writes it at finalize, before anything encodes.

use mtld3d_core::{
    draw_data::{FixedVsSource, VsSourceView},
    dxso::{SCENE_VIS_SUN, VariantKey},
    pipeline_state::PipelineRsBits,
    shadows::{BlobFacts, SceneReceiver, is_classic_blob},
};
use mtld3d_shared::{
    Command,
    mtl::{PS_SCENE_RECEIVER_SLOT, PS_SCENE_VIS_TEXTURE},
};

use super::FrameEncoder;

#[cfg(test)]
mod tests;

/// The encoder's receiver state, see the module docs.
#[derive(Default)]
pub struct SceneReceivers {
    /// [`mtld3d_core::shadows::sun_receivers_wanted`] for this encoder's configuration.
    sun: bool,
    /// This submission's receiver uniform in the scratch arena; 0 until a receiver binds it.
    uniform: u64,
    /// The open pass has the receiver bindings.
    bound: bool,
}

impl SceneReceivers {
    #[must_use]
    pub const fn new(sun: bool) -> Self {
        Self {
            sun,
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
    /// [`SCENE_VIS_SUN`] for a lit draw with a sun term into the back
    /// buffer before the boundary, while the scene textures are live:
    /// - a fixed-function VS — with a fixed-function or a programmable
    ///   pixel stage alike — lit with a normal and an enabled directional
    ///   light: per pixel when `variant` lights its directionals per pixel,
    ///   vertex-lit otherwise (a `ps_3_0` pairing). One that lights per
    ///   pixel without a directional light has no sun and reads nothing;
    /// - a figure VS, per pixel with the figure terms and vertex-lit without;
    /// - the `SpeedTree` leaf VS, vertex-lit.
    ///
    /// Zero for everything else: unlit or pre-transformed fixed-function
    /// draws, lit ones without a normal (the VS zeroes every N·L term, so
    /// their colour is ambient and emissive only) and every other
    /// programmable VS.
    #[must_use]
    pub fn scene_receiver_bits(&self, vs: VsSourceView<'_>, variant: &VariantKey) -> u8 {
        if !self.receivers.sun || !self.scene_capture_armed() || self.scene_textures().is_none() {
            return 0;
        }
        let back_buffer = self.pass_state.backbuffer_texture();
        if back_buffer.is_null() || self.pass_state.current_color_texture() != back_buffer {
            return 0;
        }
        let sun = match vs {
            VsSourceView::FixedFunction(FixedVsSource { key, .. }) => {
                key.lighting_enabled()
                    && key.has_normal()
                    && !key.has_rhw()
                    && if variant.ppl_mask == 0 {
                        key.light_active_mask & key.light_directional_mask != 0
                    } else {
                        variant.ppl_directional_mask != 0
                    }
            }
            VsSourceView::Programmable(source) => {
                let id = source.vs_id.raw();
                variant.ppl_figure != 0
                    || mtld3d_core::ppl::figures::figure_vs(id).is_some()
                    || mtld3d_core::ppl::figures::speedtree_vs(id)
            }
        };
        if sun { SCENE_VIS_SUN } else { 0 }
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
        if !self.receivers.sun || !self.scene_capture_armed() || self.scene_textures().is_none() {
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

    /// Bind what a receiver reads, once per pass: `vis` and the receiver uniform.
    ///
    /// Only a draw [`Self::scene_receiver_bits`] flagged calls it, so the
    /// scene textures are live.
    pub fn bind_scene_receivers(&mut self) {
        if self.receivers.bound {
            return;
        }
        let vis = self
            .scene_textures()
            .expect("a receiver draws while the scene textures are live")
            .vis;
        if self.receivers.uniform == 0 {
            self.receivers.uniform = self.alloc_scratch(&SceneReceiver::UNSHADOWED.to_bytes());
        }
        self.emit_command(Command::set_fragment_texture(
            vis.raw(),
            PS_SCENE_VIS_TEXTURE,
        ));
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
    /// submission; full sun (`k = 1`, no darkening) when it did not, as on
    /// the flushed part of a split frame. Nothing when no receiver drew.
    pub fn fill_scene_receiver(&mut self) {
        if self.receivers.uniform == 0 {
            return;
        }
        let bytes = SceneReceiver::new(self.scene.boundary()).to_bytes();
        let uniform = core::ptr::with_exposed_provenance_mut::<u8>(
            usize::try_from(self.receivers.uniform).expect("a host address fits usize"),
        );
        // SAFETY: `uniform` is this submission's `alloc_scratch` copy of
        // `bytes.len()` bytes; the arena keeps it until the payload retires,
        // and nothing reads it before the submission encodes.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), uniform, bytes.len()) };
    }
}
