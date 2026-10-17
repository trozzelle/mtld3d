//! The post chain run on the back buffer at the 3D/UI boundary.
//!
//! `FrameEncoder::post_boundary` queues one [`BlitCommandType::PostEffect`]
//! per enabled [`Effect`], in [`Effect::CHAIN`] order, ahead of the boundary
//! draw's pass, then a copy of the last result back into the back buffer. Each
//! effect is a compute dispatch over the back buffer's raster extent, encoded
//! in the blit stream like a depth transfer. The first effect reads the back
//! buffer itself (it has `ShaderRead`); the effects then ping-pong between two
//! scratch textures with `ShaderRead | ShaderWrite`, which the back buffer
//! lacks: the write usage would cost it lossless compression on Apple GPUs.
//!
//! An effect is an id, a kernel in [`kernel_source`], and whether it reads the
//! depth snapshot. Its constants come from the submission's table of
//! [`PostConstants`], one row per boundary, which the command names by index.
//!
//! [`BlitCommandType::PostEffect`]: mtld3d_shared::BlitCommandType::PostEffect

use mtld3d_core::{encoder_data::PostBoundary, post::PostConfig};
use mtld3d_shared::{
    BlitCommand, MetalHandle,
    mtl::PixelFormat,
    mtl_handle::{MTLComputePipelineStateKind, MTLDeviceKind, MTLTextureKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCompileOptions, MTLComputeCommandEncoder, MTLDevice,
    MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLSize, MTLTexture,
};

use super::handle::{IntoRetained, ReleaseRetain};
use crate::LOG_TARGET;

#[cfg(test)]
mod tests;

/// One post effect. The discriminant is its id on the wire and its pipeline slot.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    /// `post.debug`: rgb → 1 − rgb, alpha kept.
    Invert = 0,
}

/// Number of effects, the length of the pipeline table.
pub const POST_EFFECT_COUNT: usize = Effect::CHAIN.len();

impl Effect {
    /// Every effect, in the order the chain runs them.
    pub const CHAIN: [Self; 1] = [Self::Invert];

    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Invert),
            _ => None,
        }
    }

    const fn enabled(self, cfg: PostConfig) -> bool {
        match self {
            Self::Invert => cfg.debug,
        }
    }

    /// Whether the kernel samples the depth snapshot at `texture(2)`.
    #[must_use]
    pub const fn reads_depth(self) -> bool {
        match self {
            Self::Invert => false,
        }
    }

    const fn entry(self) -> &'static str {
        match self {
            Self::Invert => "post_invert",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Invert => "mtld3d-post-invert",
        }
    }
}

/// The enabled effects of a config, in chain order, without allocating.
#[derive(Clone, Copy)]
pub struct EffectChain {
    effects: [Effect; POST_EFFECT_COUNT],
    len: usize,
}

impl EffectChain {
    #[must_use]
    pub fn as_slice(&self) -> &[Effect] {
        &self.effects[..self.len]
    }
}

#[must_use]
pub fn enabled_effects(cfg: PostConfig) -> EffectChain {
    let mut chain = EffectChain {
        effects: Effect::CHAIN,
        len: 0,
    };
    for effect in Effect::CHAIN {
        if effect.enabled(cfg) {
            chain.effects[chain.len] = effect;
            chain.len += 1;
        }
    }
    chain
}

/// One row of a submission's post constants table, as the kernels read it at `buffer(0)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostConstants {
    /// The world projection's `_11`, `_22`, `_33`, `_43`.
    pub proj: [f32; 4],
    /// The sun's eye-space travel direction; `w` is 1 with a sun, 0 without.
    pub sun_dir: [f32; 4],
    /// The sun's diffuse colour; `w` unused.
    pub sun_color: [f32; 4],
    /// Raster width and height, 1 when a depth snapshot is bound, and 0.
    pub extent: [u32; 4],
}

impl PostConstants {
    #[must_use]
    pub fn new(boundary: &PostBoundary, width: u32, height: u32, has_depth: bool) -> Self {
        let [r, g, b] = boundary.sun_color;
        let sun_dir = boundary
            .sun_dir_eye
            .map_or([0.0; 4], |[x, y, z]| [x, y, z, 1.0]);
        Self {
            proj: boundary.proj,
            sun_dir,
            sun_color: [r, g, b, 0.0],
            extent: [width, height, u32::from(has_depth), 0],
        }
    }
}

/// The MSL every effect compiles with: the constants row and its kernel.
#[must_use]
pub fn kernel_source(effect: Effect) -> String {
    let body = match effect {
        Effect::Invert => {
            r"
kernel void post_invert(texture2d<float, access::read> src [[texture(0)]],
                        texture2d<float, access::write> dst [[texture(1)]],
                        constant PostConstants& k [[buffer(0)]],
                        uint2 p [[thread_position_in_grid]]) {
    if (p.x >= k.extent.x || p.y >= k.extent.y) return;
    float4 c = src.read(p);
    dst.write(float4(1.0 - c.rgb, c.a), p);
}
"
        }
    };
    format!(
        r"
#include <metal_stdlib>
using namespace metal;
struct PostConstants {{ float4 proj; float4 sun_dir; float4 sun_color; uint4 extent; }};
{body}"
    )
}

/// Compile `effect`'s kernel into a compute pipeline owned by the caller.
pub fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    effect: Effect,
) -> Option<MetalHandle<MTLComputePipelineStateKind>> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Colour arithmetic on display values; fast math changes nothing visible.
    options.setMathMode(MTLMathMode::Fast);
    let library = device
        .newLibraryWithSource_options_error(
            &NSString::from_str(&kernel_source(effect)),
            Some(&options),
        )
        .map_err(|e| log::error!(target: LOG_TARGET, "post {effect:?}: library failed: {e}"))
        .ok()?;
    library.setLabel(Some(&NSString::from_str(effect.label())));
    let function = library.newFunctionWithName(&NSString::from_str(effect.entry()))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| log::error!(target: LOG_TARGET, "post {effect:?}: pipeline failed: {e}"))
        .ok()?;
    // SAFETY: the canonical retain is transferred to the caller's owner.
    Some(unsafe { MetalHandle::new(Retained::into_raw(pipeline) as u64) })
}

/// Release a pipeline's canonical retain after GPU retirement.
pub fn destroy_pipeline(handle: u64) {
    // SAFETY: the caller hands over a live canonical compute pipeline retain.
    let handle = unsafe { MetalHandle::<MTLComputePipelineStateKind>::new(handle) };
    // SAFETY: encoder teardown drains its queue before releasing pipeline ownership.
    unsafe { handle.release_retain() };
}

/// A scratch pair's key: raster width, height and the back buffer's format.
pub type ScratchKey = (u32, u32, PixelFormat);

/// What [`PostScratch::rekey`] found.
pub enum Rekey {
    /// The cached pair matches.
    Hit([MetalHandle<MTLTextureKind>; 2]),
    /// No pair for the key: create one. `stale` is a pair of another key the
    /// slot held, which the caller retires behind the GPU.
    Miss {
        stale: Option<[MetalHandle<MTLTextureKind>; 2]>,
    },
}

/// The one scratch pair, keyed by extent and format.
///
/// One slot rather than a map: the key changes at `Reset` or a resize and
/// never per frame, so a replacement retires the old pair instead of keeping
/// it.
#[derive(Default)]
pub struct PostScratch {
    key: Option<ScratchKey>,
    handles: Option<[MetalHandle<MTLTextureKind>; 2]>,
}

impl PostScratch {
    pub fn rekey(&mut self, key: ScratchKey) -> Rekey {
        match self.handles {
            Some(pair) if self.key == Some(key) => Rekey::Hit(pair),
            _ => {
                self.key = None;
                Rekey::Miss {
                    stale: self.handles.take(),
                }
            }
        }
    }

    pub const fn install(&mut self, key: ScratchKey, handles: [MetalHandle<MTLTextureKind>; 2]) {
        self.key = Some(key);
        self.handles = Some(handles);
    }

    #[cfg(test)]
    #[must_use]
    pub const fn handles(&self) -> Option<[MetalHandle<MTLTextureKind>; 2]> {
        self.handles
    }

    /// Empty the slot, handing its pair to the caller (shutdown).
    pub const fn take(&mut self) -> Option<[MetalHandle<MTLTextureKind>; 2]> {
        self.key = None;
        self.handles.take()
    }
}

/// The encoder's post resources: one lazily compiled pipeline per effect and the scratch pair.
#[derive(Default)]
pub struct PostState {
    pipelines: [MetalHandle<MTLComputePipelineStateKind>; POST_EFFECT_COUNT],
    pub scratch: PostScratch,
}

impl PostState {
    /// `effect`'s pipeline, compiled on first use.
    pub fn pipeline(
        &mut self,
        device: MetalHandle<MTLDeviceKind>,
        effect: Effect,
    ) -> Option<MetalHandle<MTLComputePipelineStateKind>> {
        let slot = &mut self.pipelines[effect as usize];
        if slot.is_null() {
            let device = device.into_retained()?;
            *slot = create_pipeline(&device, effect)?;
        }
        Some(*slot)
    }

    /// The scratch pair, if one was created.
    #[cfg(test)]
    #[must_use]
    pub const fn scratch_handles(&self) -> Option<[MetalHandle<MTLTextureKind>; 2]> {
        self.scratch.handles()
    }

    /// Release the pipelines; the caller destroys the scratch pair with its textures.
    pub fn destroy(&mut self) {
        for slot in &mut self.pipelines {
            if !slot.is_null() {
                destroy_pipeline(slot.raw());
            }
            *slot = MetalHandle::NULL;
        }
    }
}

/// Encode one effect on its own compute encoder.
///
/// The command buffer retains every bound texture and the pipeline until it
/// completes, so nothing here outlives the call.
pub fn encode(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    command: &BlitCommand,
    constants: &PostConstants,
) -> bool {
    let Some(effect) = Effect::from_id(command.mip_level) else {
        log::error!(target: LOG_TARGET, "post: unknown effect {}", command.mip_level);
        return false;
    };
    // SAFETY: a PostEffect command carries canonical texture handles the frame retains.
    let source = unsafe { MetalHandle::<MTLTextureKind>::new(command.src_handle) };
    // SAFETY: as above; the scratch texture is the encoder's and retires behind the GPU.
    let destination = unsafe { MetalHandle::<MTLTextureKind>::new(command.dst_handle) };
    // SAFETY: the encoder's post state owns this compute pipeline retain.
    let pipeline = unsafe { MetalHandle::<MTLComputePipelineStateKind>::new(command.src_offset) };
    let (Some(source), Some(destination), Some(pipeline)) = (
        source.into_retained(),
        destination.into_retained(),
        pipeline.into_retained(),
    ) else {
        log::error!(target: LOG_TARGET, "post {effect:?}: missing texture or pipeline");
        return false;
    };
    let depth = if command.dst_offset == 0 {
        None
    } else {
        // SAFETY: a non-zero depth field is the encoder's depth snapshot texture.
        let handle = unsafe { MetalHandle::<MTLTextureKind>::new(command.dst_offset) };
        let Some(depth) = handle.into_retained() else {
            log::error!(target: LOG_TARGET, "post {effect:?}: missing depth snapshot");
            return false;
        };
        Some(depth)
    };
    if !cb.retainedReferences() {
        log::error!(target: LOG_TARGET, "post: command buffer does not retain resources");
        return false;
    }
    let (width, height) = (command.region_w as usize, command.region_h as usize);
    if [constants.extent[0], constants.extent[1]] != [command.region_w, command.region_h]
        || source.width() < width
        || source.height() < height
        || destination.width() < width
        || destination.height() < height
    {
        log::error!(
            target: LOG_TARGET,
            "post {effect:?}: {width}x{height} does not fit {}x{} into {}x{}",
            source.width(),
            source.height(),
            destination.width(),
            destination.height(),
        );
        return false;
    }
    let Some(compute) = cb.computeCommandEncoder() else {
        log::error!(target: LOG_TARGET, "post {effect:?}: compute encoder failed");
        return false;
    };
    compute.setLabel(Some(&NSString::from_str(effect.label())));
    compute.setComputePipelineState(&pipeline);
    // SAFETY: the kernel reads slot 0, the retained source.
    unsafe { compute.setTexture_atIndex(Some(&source), 0) };
    // SAFETY: the kernel writes slot 1, the retained scratch, within the checked extent.
    unsafe { compute.setTexture_atIndex(Some(&destination), 1) };
    if let Some(depth) = depth.as_deref() {
        // SAFETY: a depth-reading kernel reads slot 2, the retained snapshot.
        unsafe { compute.setTexture_atIndex(Some(depth), 2) };
    }
    // SAFETY: the encoder copies this 64-byte row before returning.
    unsafe {
        compute.setBytes_length_atIndex(
            core::ptr::NonNull::from(constants).cast(),
            size_of::<PostConstants>(),
            0,
        );
    }
    let threadgroup = MTLSize {
        width: 8,
        height: 8,
        depth: 1,
    };
    let grid = MTLSize {
        width: width.div_ceil(8),
        height: height.div_ceil(8),
        depth: 1,
    };
    compute.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
    compute.endEncoding();
    true
}
