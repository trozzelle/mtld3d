//! The post chain at the 3D/UI boundary (`PostBoundary` op), see `metal::post`.

use mtld3d_core::encoder_data::PostBoundary;
use mtld3d_shared::{
    BlitCommand, MetalHandle, PostEffectInfo,
    mtl::{DestroyKind, PixelFormat},
    mtl_handle::{MTLComputePipelineStateKind, MTLTextureKind},
};

use super::{FrameEncoder, LOG_TARGET, PendingResourceRetention};
use crate::metal::{
    handle::IntoRetained,
    post::{
        Effect, Input, Output, POST_KERNEL_COUNT, POST_SLOT_COUNT, PostConstants, Rekey, Slot,
        SunSkip, edr_headroom, enabled_effects,
    },
};

impl FrameEncoder {
    /// Run the enabled post effects on the back buffer ahead of the boundary draw.
    ///
    /// Queues, after every pass and clear recorded so far, one `PostEffect`
    /// per pass of every effect and the copy of the last colour back into the
    /// back buffer; they lead the boundary draw's pass. The first effect reads
    /// the back buffer as its colour, each later one the colour slot the one
    /// before wrote. Steps aside, leaving the frame as it was, when render
    /// target 0 is not the back buffer, when the back buffer is multisampled
    /// (the world was drawn into its companion, so a chain run on the resolve
    /// target would be overwritten), or when a pipeline or a scratch texture
    /// cannot be made. The effects that read depth step aside alone when no
    /// depth snapshot can be taken, and god rays when the boundary has no
    /// visible sun (`metal::post::sun_screen`).
    ///
    /// Under `color.sceneFloat` the tone map runs last, into the headroom of
    /// the frame's view when its layer is in EDR and 1 otherwise; once its
    /// pass is queued the frame presents as scene-linear.
    pub fn post_boundary(&mut self, boundary: &PostBoundary) {
        let (chain, sun_skip) =
            enabled_effects(self.config.post, self.config.scene_float).for_sun(boundary);
        if let Some(skip) = sun_skip {
            log_god_rays_skip(skip);
        }
        if chain.as_slice().is_empty() {
            return;
        }
        let back_buffer = self.pass_state.backbuffer_texture();
        if back_buffer.is_null() || self.pass_state.current_color_texture() != back_buffer {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "post: the boundary draw does not target the back buffer; the post chain steps aside"
            );
            return;
        }
        if self.pass_state.current_color_sample_count() > 1 {
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "post: the back buffer is multisampled; the post chain steps aside"
            );
            return;
        }
        let raster = self.pass_state.current_color_size();
        let format = self.pass_state.current_color_base_format();
        let depth = if chain.reads_depth() {
            self.depth_snapshot_for_sampling()
        } else {
            0
        };
        let chain = if depth == 0 && chain.reads_depth() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "post: the boundary draw has no depth attachment to snapshot; the effects that \
                 read depth step aside"
            );
            chain.without_depth()
        } else {
            chain
        };
        let effects = chain.as_slice();
        if effects.is_empty() {
            return;
        }
        // Every pipeline and scratch texture first, so a failure queues nothing.
        let mut pipelines = [MetalHandle::<MTLComputePipelineStateKind>::NULL; POST_KERNEL_COUNT];
        let mut scratch = [MetalHandle::<MTLTextureKind>::NULL; POST_SLOT_COUNT];
        for (index, &effect) in effects.iter().enumerate() {
            for pass in effect.passes() {
                let kernel = pass.kernel;
                if pipelines[kernel as usize].is_null() {
                    let Some(pipeline) = self.post.pipeline(self.device_handle, kernel) else {
                        mtld3d_shared::log_once_warn!(
                            target: LOG_TARGET,
                            "post: the {kernel:?} pipeline could not be built; the post chain \
                             steps aside"
                        );
                        return;
                    };
                    pipelines[kernel as usize] = pipeline;
                }
                let slots = [
                    Some(pass.input),
                    pass.aux,
                    Some(match pass.output {
                        Output::Colour => Input::Slot(Slot::colour(index)),
                        Output::Slot(slot) => Input::Slot(slot),
                    }),
                ];
                for slot in slots.into_iter().flatten().filter_map(|input| match input {
                    Input::Colour => None,
                    Input::Slot(slot) => Some(slot),
                }) {
                    if scratch[slot as usize].is_null() {
                        let Some(texture) = self.post_scratch(slot, raster, format) else {
                            let (width, height, slot_format) = slot.key(raster, format);
                            mtld3d_shared::log_once_warn!(
                                target: LOG_TARGET,
                                "post: no {width}x{height} {slot_format:?} {slot:?} scratch \
                                 texture; the post chain steps aside"
                            );
                            return;
                        };
                        scratch[slot as usize] = texture;
                    }
                }
            }
        }
        let row = self.post_constants.len() as u64;
        self.post_constants.push(PostConstants::new(
            boundary,
            self.config.post,
            raster.0,
            raster.1,
            depth != 0,
            if effects.contains(&Effect::ToneMap) {
                edr_headroom(self.frame_view)
            } else {
                1.0
            },
        ));
        let mut colour = back_buffer;
        for (index, &effect) in effects.iter().enumerate() {
            let next = scratch[Slot::colour(index) as usize];
            let texture = |input: Input| match input {
                Input::Colour => colour,
                Input::Slot(slot) => scratch[slot as usize],
            };
            for pass in effect.passes() {
                let (width, height) = pass.extent().of(raster);
                let command = BlitCommand::post_effect(&PostEffectInfo {
                    kernel: pass.kernel as u32,
                    pipeline: pipelines[pass.kernel as usize].raw(),
                    input: texture(pass.input).raw(),
                    output: match pass.output {
                        Output::Colour => next,
                        Output::Slot(slot) => scratch[slot as usize],
                    }
                    .raw(),
                    depth: if pass.kernel.reads_depth() { depth } else { 0 },
                    aux: pass.aux.map_or(0, |aux| texture(aux).raw()),
                    constants_row: row,
                    width,
                    height,
                });
                self.pass_state
                    .push_leading_blit_after_clears(command, "post_effect");
            }
            colour = next;
        }
        self.pass_state
            .push_pending_leading_blit(BlitCommand::copy_texture_to_texture_full_mip(
                colour.raw(),
                back_buffer.raw(),
                0,
                raster.0,
                raster.1,
            ));
        if effects.contains(&Effect::ToneMap) {
            self.post.tonemapped = true;
        }
    }

    /// Whether this submit's present shows scene-linear values: a present-
    /// bearing submit takes the tone map's flag, so a mid-frame flush
    /// (`no_present`) leaves it for the submit that presents.
    pub(super) fn take_present_scene_linear(&mut self, no_present: bool) -> bool {
        !no_present && core::mem::take(&mut self.post.tonemapped)
    }

    /// The `VariantKey::color_style` of a draw's pixel stage. Under
    /// `color.sceneFloat` the back buffer is half float: until the boundary
    /// queues the tone map, every pixel stage floors its colour instead of
    /// saturating it and ends with the clamp the 8-bit store used to apply.
    /// From the tone map to the present that takes its flag, the draws (the
    /// UI) keep the stock clamps and saturate their final colour, so they
    /// store [0, 1] over the tone-mapped scene and keep their SDR brightness.
    pub(crate) fn ps_color_style(&self) -> u8 {
        use mtld3d_core::dxso::{COLOR_STYLE_SCENE_FLOAT, COLOR_STYLE_STORE_CLAMP};
        match (self.config().scene_float, self.post.tonemapped) {
            (false, _) => 0,
            (true, false) => COLOR_STYLE_SCENE_FLOAT,
            (true, true) => COLOR_STYLE_STORE_CLAMP,
        }
    }

    /// `slot`'s scratch texture for the raster, made on a miss; a texture of
    /// another key retires behind the GPU.
    fn post_scratch(
        &mut self,
        slot: Slot,
        raster: (u32, u32),
        colour: PixelFormat,
    ) -> Option<MetalHandle<MTLTextureKind>> {
        let key = slot.key(raster, colour);
        match self.post.scratch.rekey(slot, key) {
            Rekey::Hit(texture) => return Some(texture),
            Rekey::Miss { stale } => {
                if let Some(texture) = stale {
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
        }
        let device = self.device_handle.into_retained()?;
        let (width, height, format) = key;
        let texture = crate::metal::create_post_scratch(&device, width, height, format)?;
        self.post.scratch.install(slot, key, texture);
        Some(texture)
    }
}

/// Say once per reason why god rays stepped aside at a boundary.
fn log_god_rays_skip(skip: SunSkip) {
    match skip {
        SunSkip::NoSun => {
            mtld3d_shared::log_once_info!(target: LOG_TARGET, "post: god rays skipped: no sun");
        }
        SunSkip::BelowHorizon => mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "post: god rays skipped: sun below the horizon"
        ),
        SunSkip::BehindCamera => mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "post: god rays skipped: sun behind the camera"
        ),
        SunSkip::OffScreen { ndc: [x, y] } => mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "post: god rays skipped: sun off screen (ndc {x:.2}, {y:.2})"
        ),
    }
}
