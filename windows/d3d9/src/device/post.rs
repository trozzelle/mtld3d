//! The 3D/UI boundary hook, see `mtld3d_core::post`.

use mtld3d_core::{encoder_data::RtBinding, post::DrawSite, render_scale::RenderScale};

use super::DeviceInner;

/// Whether render target 0 is the implicit back buffer; no binding yet means it is.
pub(super) fn rt0_is_back_buffer(binding: Option<&(RtBinding, RenderScale)>) -> bool {
    binding.is_none_or(|(binding, _)| matches!(binding, RtBinding::Backbuffer { .. }))
}

impl DeviceInner {
    /// Step the boundary rule for the draw about to be recorded; record the op when it fires.
    ///
    /// Runs after the draw's state deltas and before the draw, from both exits
    /// of `emit_snapshot_deltas`. With every post key off it tests one flag
    /// and touches nothing else.
    #[inline]
    pub(super) fn post_boundary_draw(&mut self) {
        let Self {
            post,
            config,
            ff_state,
            last_color_rt_binding,
            cached_ff_vs_layout,
            ..
        } = self;
        let op = post.note_draw(&config.post, || DrawSite {
            rt0_is_backbuffer: rt0_is_back_buffer(last_color_rt_binding.as_ref()),
            ortho_or_rhw: ff_state.projection_is_ortho() || cached_ff_vs_layout.has_rhw(),
            ff: ff_state,
        });
        if let Some(op) = op {
            self.push_control(op);
        }
    }
}
