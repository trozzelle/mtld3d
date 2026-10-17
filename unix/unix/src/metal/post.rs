//! The post chain run on the back buffer at the 3D/UI boundary.
//!
//! `FrameEncoder::post_boundary` queues the enabled [`Effect`]s, in
//! [`Effect::CHAIN`] order, ahead of the boundary draw's pass, then a copy of
//! the last colour back into the back buffer. An effect is a short list of
//! [`Pass`]es; each pass is one compute dispatch, a
//! [`BlitCommandType::PostEffect`] encoded in the blit stream like a depth
//! transfer.
//!
//! A pass runs one [`Kernel`] over its output's extent (the raster or a
//! fraction of it, [`Extent`]). It reads an [`Input`] at `texture(0)`,
//! optionally a second one at `texture(3)`, and the depth snapshot at
//! `texture(2)` when its kernel reads depth, and writes an [`Output`] at
//! `texture(1)`. Inputs are the colour as the effect found it or one of the
//! effect's intermediate [`Slot`]s; only an effect's last pass writes the new
//! colour, so every pass of an effect sees the colour from before it.
//!
//! The colour starts as the back buffer itself (it has `ShaderRead`); each
//! effect writes its result into one of two colour slots, alternating, which
//! the back buffer cannot be: it lacks `ShaderWrite`, which would cost it
//! lossless compression on Apple GPUs. Every slot is a scratch texture with
//! `ShaderRead | ShaderWrite`, cached by (width, height, format) in
//! [`PostScratch`].
//!
//! The constants come from the submission's table of [`PostConstants`], one
//! row per boundary, which every pass names by index.
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

/// One post effect: the passes it runs and the key that enables it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    /// `post.debug`: rgb → 1 − rgb, alpha kept.
    Invert,
}

/// Number of effects, the capacity of an [`EffectChain`].
pub const POST_EFFECT_COUNT: usize = Effect::CHAIN.len();

impl Effect {
    /// Every effect, in the order the chain runs them.
    pub const CHAIN: [Self; 1] = [Self::Invert];

    const fn enabled(self, cfg: PostConfig) -> bool {
        match self {
            Self::Invert => cfg.debug,
        }
    }

    /// The effect's passes, in order; the last one writes the new colour.
    #[must_use]
    pub const fn passes(self) -> &'static [Pass] {
        match self {
            Self::Invert => &[Pass {
                kernel: Kernel::Invert,
                input: Input::Colour,
                aux: None,
                output: Output::Colour,
            }],
        }
    }

    /// Whether any pass samples the depth snapshot.
    #[must_use]
    pub fn reads_depth(self) -> bool {
        self.passes().iter().any(|pass| pass.kernel.reads_depth())
    }
}

/// One compute kernel. The discriminant is its id on the wire and its pipeline slot.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// rgb → 1 − rgb, alpha kept.
    Invert = 0,
}

/// Number of kernels, the length of the pipeline table.
pub const POST_KERNEL_COUNT: usize = Kernel::ALL.len();

impl Kernel {
    /// Every kernel, in id order.
    pub const ALL: [Self; 1] = [Self::Invert];

    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Invert),
            _ => None,
        }
    }

    /// Whether the kernel reads the depth snapshot at `texture(2)`.
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

/// A pass's output extent: the raster divided by `2^shift`, rounded up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    shift: u32,
}

impl Extent {
    /// The raster itself.
    pub const FULL: Self = Self { shift: 0 };
    /// Half the raster on each axis.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no pass runs at half extent until ambient occlusion"
        )
    )]
    pub const HALF: Self = Self { shift: 1 };

    /// This extent of a raster of `width` × `height`; never zero.
    #[must_use]
    pub const fn of(self, (width, height): (u32, u32)) -> (u32, u32) {
        let div = 1 << self.shift;
        let w = width.div_ceil(div);
        let h = height.div_ceil(div);
        (if w == 0 { 1 } else { w }, if h == 0 { 1 } else { h })
    }
}

/// A named scratch texture of the chain.
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// The colour an even-numbered effect of the chain writes.
    ColourA,
    /// The colour an odd-numbered effect writes.
    ColourB,
}

/// Number of slots, the length of the scratch table.
pub const POST_SLOT_COUNT: usize = Slot::ALL.len();

impl Slot {
    /// Every slot, in index order.
    pub const ALL: [Self; 2] = [Self::ColourA, Self::ColourB];

    /// The slot an effect at `index` in the chain writes its colour to.
    #[must_use]
    pub const fn colour(index: usize) -> Self {
        if index.is_multiple_of(2) {
            Self::ColourA
        } else {
            Self::ColourB
        }
    }

    /// Whether this is one of the two colour slots.
    #[cfg(test)]
    #[must_use]
    pub const fn is_colour(self) -> bool {
        matches!(self, Self::ColourA | Self::ColourB)
    }

    #[must_use]
    pub const fn extent(self) -> Extent {
        match self {
            Self::ColourA | Self::ColourB => Extent::FULL,
        }
    }

    /// The slot's pixel format; a colour slot takes the back buffer's `colour`.
    #[must_use]
    pub const fn format(self, colour: PixelFormat) -> PixelFormat {
        match self {
            Self::ColourA | Self::ColourB => colour,
        }
    }

    /// The scratch key of this slot for a raster and its colour format.
    #[must_use]
    pub const fn key(self, raster: (u32, u32), colour: PixelFormat) -> ScratchKey {
        let (width, height) = self.extent().of(raster);
        (width, height, self.format(colour))
    }
}

/// A texture a pass reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// The colour as the effect found it: the back buffer, or the colour slot
    /// the effect before wrote.
    Colour,
    /// One of the effect's intermediate textures.
    Slot(Slot),
}

/// The texture a pass writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// The effect's new colour, at full extent; only an effect's last pass.
    Colour,
    /// One of the effect's intermediate textures.
    #[expect(
        dead_code,
        reason = "no effect has intermediates until ambient occlusion"
    )]
    Slot(Slot),
}

/// One dispatch of an effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub kernel: Kernel,
    /// Read at `texture(0)`.
    pub input: Input,
    /// Read at `texture(3)`, if the kernel takes a second input.
    pub aux: Option<Input>,
    /// Written at `texture(1)`; its extent is the dispatch's.
    pub output: Output,
}

impl Pass {
    /// The extent the pass dispatches over: its output's.
    #[must_use]
    pub const fn extent(&self) -> Extent {
        match self.output {
            Output::Colour => Extent::FULL,
            Output::Slot(slot) => slot.extent(),
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

    /// Whether any effect of the chain reads the depth snapshot.
    #[must_use]
    pub fn reads_depth(self) -> bool {
        self.as_slice().iter().any(|effect| effect.reads_depth())
    }

    /// The chain without the effects that read depth.
    #[must_use]
    pub fn without_depth(self) -> Self {
        let mut chain = Self {
            effects: self.effects,
            len: 0,
        };
        for &effect in self.as_slice() {
            if !effect.reads_depth() {
                chain.effects[chain.len] = effect;
                chain.len += 1;
            }
        }
        chain
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

/// The MSL a kernel compiles with: the constants row and its function.
#[must_use]
pub fn kernel_source(kernel: Kernel) -> String {
    let body = match kernel {
        Kernel::Invert => {
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

/// Compile `kernel` into a compute pipeline owned by the caller.
pub fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: Kernel,
) -> Option<MetalHandle<MTLComputePipelineStateKind>> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Colour arithmetic on display values; fast math changes nothing visible.
    options.setMathMode(MTLMathMode::Fast);
    let library = device
        .newLibraryWithSource_options_error(
            &NSString::from_str(&kernel_source(kernel)),
            Some(&options),
        )
        .map_err(|e| log::error!(target: LOG_TARGET, "post {kernel:?}: library failed: {e}"))
        .ok()?;
    library.setLabel(Some(&NSString::from_str(kernel.label())));
    let function = library.newFunctionWithName(&NSString::from_str(kernel.entry()))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| log::error!(target: LOG_TARGET, "post {kernel:?}: pipeline failed: {e}"))
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

/// A scratch texture's key: its width, height and format.
pub type ScratchKey = (u32, u32, PixelFormat);

/// What [`PostScratch::rekey`] found for a slot.
pub enum Rekey {
    /// The cached texture matches.
    Hit(MetalHandle<MTLTextureKind>),
    /// No texture for the key: create one. `stale` is a texture of another
    /// key the slot held, which the caller retires behind the GPU.
    Miss {
        stale: Option<MetalHandle<MTLTextureKind>>,
    },
}

/// The chain's scratch textures, one per [`Slot`], each keyed by extent and format.
///
/// One texture per slot rather than a map: a key changes at `Reset` or a
/// resize and never per frame, so a replacement retires the old texture
/// instead of keeping it.
#[derive(Default)]
pub struct PostScratch {
    slots: [Option<(ScratchKey, MetalHandle<MTLTextureKind>)>; POST_SLOT_COUNT],
}

impl PostScratch {
    pub fn rekey(&mut self, slot: Slot, key: ScratchKey) -> Rekey {
        let entry = &mut self.slots[slot as usize];
        match *entry {
            Some((held, handle)) if held == key => Rekey::Hit(handle),
            _ => Rekey::Miss {
                stale: entry.take().map(|(_, handle)| handle),
            },
        }
    }

    pub const fn install(
        &mut self,
        slot: Slot,
        key: ScratchKey,
        handle: MetalHandle<MTLTextureKind>,
    ) {
        self.slots[slot as usize] = Some((key, handle));
    }

    #[cfg(test)]
    #[must_use]
    pub const fn handle(&self, slot: Slot) -> Option<MetalHandle<MTLTextureKind>> {
        match self.slots[slot as usize] {
            Some((_, handle)) => Some(handle),
            None => None,
        }
    }

    /// Empty every slot, handing the textures to the caller (shutdown).
    pub fn take_all(&mut self) -> impl Iterator<Item = MetalHandle<MTLTextureKind>> + '_ {
        self.slots
            .iter_mut()
            .filter_map(|entry| entry.take().map(|(_, handle)| handle))
    }
}

/// The encoder's post resources: one lazily compiled pipeline per kernel and the scratch slots.
#[derive(Default)]
pub struct PostState {
    pipelines: [MetalHandle<MTLComputePipelineStateKind>; POST_KERNEL_COUNT],
    pub scratch: PostScratch,
}

impl PostState {
    /// `kernel`'s pipeline, compiled on first use.
    pub fn pipeline(
        &mut self,
        device: MetalHandle<MTLDeviceKind>,
        kernel: Kernel,
    ) -> Option<MetalHandle<MTLComputePipelineStateKind>> {
        let slot = &mut self.pipelines[kernel as usize];
        if slot.is_null() {
            let device = device.into_retained()?;
            *slot = create_pipeline(&device, kernel)?;
        }
        Some(*slot)
    }

    /// A slot's scratch texture, if one was created.
    #[cfg(test)]
    #[must_use]
    pub const fn scratch_handle(&self, slot: Slot) -> Option<MetalHandle<MTLTextureKind>> {
        self.scratch.handle(slot)
    }

    /// Release the pipelines; the caller destroys the scratch textures with its textures.
    pub fn destroy(&mut self) {
        for slot in &mut self.pipelines {
            if !slot.is_null() {
                destroy_pipeline(slot.raw());
            }
            *slot = MetalHandle::NULL;
        }
    }
}

/// A texture handle of a pass, `None` for 0.
fn bound_texture(raw: u64) -> Option<MetalHandle<MTLTextureKind>> {
    // SAFETY: a PostEffect command carries canonical texture handles the frame
    // retains (the back buffer, the depth snapshot) or the encoder's scratch
    // textures, which retire behind the GPU.
    (raw != 0).then(|| unsafe { MetalHandle::<MTLTextureKind>::new(raw) })
}

/// Encode one pass on its own compute encoder.
///
/// The command buffer retains every bound texture and the pipeline until it
/// completes, so nothing here outlives the call. A kernel that reads depth
/// is never dispatched without the snapshot bound.
pub fn encode(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    command: &BlitCommand,
    constants: &PostConstants,
) -> bool {
    let Some(kernel) = Kernel::from_id(command.mip_level) else {
        log::error!(target: LOG_TARGET, "post: unknown kernel {}", command.mip_level);
        return false;
    };
    // SAFETY: the encoder's post state owns this compute pipeline retain.
    let pipeline = unsafe { MetalHandle::<MTLComputePipelineStateKind>::new(command.src_offset) };
    let (Some(source), Some(destination), Some(pipeline)) = (
        bound_texture(command.src_handle).and_then(IntoRetained::into_retained),
        bound_texture(command.dst_handle).and_then(IntoRetained::into_retained),
        pipeline.into_retained(),
    ) else {
        log::error!(target: LOG_TARGET, "post {kernel:?}: missing texture or pipeline");
        return false;
    };
    let depth = match (kernel.reads_depth(), bound_texture(command.dst_offset)) {
        (false, _) => None,
        (true, Some(handle)) => {
            let Some(depth) = handle.into_retained() else {
                log::error!(target: LOG_TARGET, "post {kernel:?}: missing depth snapshot");
                return false;
            };
            Some(depth)
        }
        (true, None) => {
            log::error!(target: LOG_TARGET, "post {kernel:?}: no depth snapshot bound");
            return false;
        }
    };
    let aux = match bound_texture(command.bytes_per_row) {
        None => None,
        Some(handle) => {
            let Some(aux) = handle.into_retained() else {
                log::error!(target: LOG_TARGET, "post {kernel:?}: missing second input");
                return false;
            };
            Some(aux)
        }
    };
    if !cb.retainedReferences() {
        log::error!(target: LOG_TARGET, "post: command buffer does not retain resources");
        return false;
    }
    let (width, height) = (command.region_w as usize, command.region_h as usize);
    if width == 0
        || height == 0
        || command.region_w > constants.extent[0]
        || command.region_h > constants.extent[1]
        || destination.width() < width
        || destination.height() < height
    {
        log::error!(
            target: LOG_TARGET,
            "post {kernel:?}: {width}x{height} does not fit the {}x{} raster or the {}x{} output",
            constants.extent[0],
            constants.extent[1],
            destination.width(),
            destination.height(),
        );
        return false;
    }
    let Some(compute) = cb.computeCommandEncoder() else {
        log::error!(target: LOG_TARGET, "post {kernel:?}: compute encoder failed");
        return false;
    };
    compute.setLabel(Some(&NSString::from_str(kernel.label())));
    compute.setComputePipelineState(&pipeline);
    // SAFETY: the kernel reads slot 0, the retained input.
    unsafe { compute.setTexture_atIndex(Some(&source), 0) };
    // SAFETY: the kernel writes slot 1, the retained scratch, within the checked extent.
    unsafe { compute.setTexture_atIndex(Some(&destination), 1) };
    if let Some(depth) = depth.as_deref() {
        // SAFETY: a depth-reading kernel reads slot 2, the retained snapshot.
        unsafe { compute.setTexture_atIndex(Some(depth), 2) };
    }
    if let Some(aux) = aux.as_deref() {
        // SAFETY: a two-input kernel reads slot 3, the retained second input.
        unsafe { compute.setTexture_atIndex(Some(aux), 3) };
    }
    // SAFETY: the encoder copies this row before returning.
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
