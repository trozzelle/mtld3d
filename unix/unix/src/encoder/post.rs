//! The post chain at the 3D/UI boundary (`PostBoundary` op), see `metal::post`.

use mtld3d_core::encoder_data::PostBoundary;
use mtld3d_shared::{
    BlitCommand, BlitCommandType, MetalHandle,
    mtl::DestroyKind,
    mtl_handle::{MTLComputePipelineStateKind, MTLTextureKind},
};

use super::{FrameEncoder, LOG_TARGET, PendingResourceRetention};
use crate::metal::{
    handle::IntoRetained,
    post::{POST_EFFECT_COUNT, PostConstants, Rekey, ScratchKey, enabled_effects},
};

impl FrameEncoder {
    /// Run the enabled post effects on the back buffer ahead of the boundary draw.
    ///
    /// Queues, after every pass and clear recorded so far, one `PostEffect`
    /// per effect and the copy of the last result back into the back buffer;
    /// they lead the boundary draw's pass. The first effect reads the back
    /// buffer, each later one the scratch the one before wrote. Steps aside,
    /// leaving the frame as it was, when render target 0 is not the back
    /// buffer, when the back buffer is multisampled (the world was drawn into
    /// its companion, so a chain run on the resolve target would be
    /// overwritten), or when a pipeline or the scratch pair cannot be made.
    pub fn post_boundary(&mut self, boundary: &PostBoundary) {
        let chain = enabled_effects(self.config.post);
        let effects = chain.as_slice();
        if effects.is_empty() {
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
        let (width, height) = self.pass_state.current_color_size();
        let format = self.pass_state.current_color_base_format();
        // Every pipeline first, so a failure queues nothing.
        let mut pipelines = [MetalHandle::<MTLComputePipelineStateKind>::NULL; POST_EFFECT_COUNT];
        for (slot, &effect) in pipelines.iter_mut().zip(effects) {
            let Some(pipeline) = self.post.pipeline(self.device_handle, effect) else {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "post: the {effect:?} pipeline could not be built; the post chain steps aside"
                );
                return;
            };
            *slot = pipeline;
        }
        let Some([first, second]) = self.post_scratch_pair((width, height, format)) else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "post: no {width}x{height} {format:?} scratch textures; the post chain steps aside"
            );
            return;
        };
        let depth = if effects.iter().any(|e| e.reads_depth()) {
            self.depth_snapshot_for_sampling()
        } else {
            0
        };
        let row = self.post_constants.len() as u64;
        self.post_constants
            .push(PostConstants::new(boundary, width, height, depth != 0));
        let mut read = back_buffer;
        for (i, (&effect, pipeline)) in effects.iter().zip(pipelines).enumerate() {
            let write = if i % 2 == 0 { first } else { second };
            let mut command = BlitCommand::copy_texture_to_texture_full_mip(
                read.raw(),
                write.raw(),
                0,
                width,
                height,
            );
            command.cmd = BlitCommandType::PostEffect as u32;
            command.mip_level = effect as u32;
            command.src_offset = pipeline.raw();
            command.dst_offset = depth;
            command.byte_size = row;
            self.pass_state
                .push_leading_blit_after_clears(command, "post_effect");
            read = write;
        }
        self.pass_state
            .push_pending_leading_blit(BlitCommand::copy_texture_to_texture_full_mip(
                read.raw(),
                back_buffer.raw(),
                0,
                width,
                height,
            ));
    }

    /// The scratch pair for `key`, made on a miss; a pair of another key retires behind the GPU.
    fn post_scratch_pair(&mut self, key: ScratchKey) -> Option<[MetalHandle<MTLTextureKind>; 2]> {
        match self.post.scratch.rekey(key) {
            Rekey::Hit(pair) => return Some(pair),
            Rekey::Miss { stale } => {
                for handle in stale.into_iter().flatten() {
                    self.pending_resource_retention
                        .push_back(PendingResourceRetention {
                            kind: DestroyKind::Texture,
                            handle: handle.raw(),
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
        let first = crate::metal::create_post_scratch(&device, width, height, format)?;
        let Some(second) = crate::metal::create_post_scratch(&device, width, height, format) else {
            // Never queued: nothing on the GPU names it yet.
            crate::metal::destroy_texture(first.raw());
            return None;
        };
        self.post.scratch.install(key, [first, second]);
        Some([first, second])
    }
}
