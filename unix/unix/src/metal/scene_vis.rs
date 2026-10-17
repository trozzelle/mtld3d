//! The scene kernels: the sun's visibility from its cascades, and the shadow debug views.
//!
//! Under `shadows.sun`, `encoder::scene` replays the world into the camera
//! depth pre-pass (`predepth`) and the sun's cascades, then runs
//! [`Kernel::SunVisCascades`] once over the raster, ahead of the world, to
//! fill `vis`: per pixel, the share of the sun that reaches the surface the
//! world draws there. The receivers read it. `shadows.debug` shows it, or
//! which cascade serves each pixel, over the world at the 3D/UI boundary.
//!
//! Each kernel is one compute dispatch, a [`BlitCommandType::SceneKernel`]
//! encoded in the blit stream like a post effect, with its
//! [`SceneConstants`] in the submission's scratch arena. The kernels are
//! compiled at run time on first use, like the post chain's, and never
//! enter the shader cache.
//!
//! The visibility of one pixel ([`Kernel::SunVisCascades`]):
//! 1. its eye position from `predepth`, through the world's projection and
//!    viewport;
//! 2. its normal from the depth derivatives, never across a sky neighbour
//!    (as `post`'s ambient occlusion does);
//! 3. the position moved [`NORMAL_OFFSET_TEXELS`] texels of the chosen
//!    cascade along the normal;
//! 4. the cascade by eye depth, blending into the next over the last
//!    [`CASCADE_BLEND`] of each (the last one fades to full sun);
//! 5. a 16-tap Poisson PCF of comparison samples, its radius `softness`
//!    cascade-0 texels in world units, so the penumbra keeps its width across
//!    a cascade seam ([`PCF_MIN_TEXELS`] at least);
//! 6. full sun on the far plane (depth ≥ 1);
//! 7. shadows fade out as the sun's elevation drops from
//!    [`HORIZON_FADE_DEGREES`] to 0.
//!
//! [`BlitCommandType::SceneKernel`]: mtld3d_shared::BlitCommandType::SceneKernel

use mtld3d_core::{
    encoder_data::PostBoundary,
    shadows::{
        ShadowConfig,
        cascades::{CascadeSet, MAX_CASCADES, Mat4},
    },
};
use mtld3d_shared::{
    BlitCommand, MetalHandle,
    mtl_handle::{MTLComputePipelineStateKind, MTLDeviceKind, MTLTextureKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCompileOptions, MTLComputeCommandEncoder, MTLDevice,
    MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLSize, MTLTexture,
};

use super::handle::IntoRetained;
use crate::LOG_TARGET;

#[cfg(test)]
mod tests;

/// The sun elevation, in degrees, at and above which shadows are at full strength; they fade out linearly to 0°.
pub const HORIZON_FADE_DEGREES: f32 = 6.0;
/// The share of each cascade, at its far end, over which it blends into the next.
pub const CASCADE_BLEND: f32 = 0.1;
/// How far a receiver moves along its normal before the lookup, in texels of the chosen cascade.
pub const NORMAL_OFFSET_TEXELS: f32 = 1.0;
/// The smallest PCF radius, in texels of the chosen cascade.
pub const PCF_MIN_TEXELS: f32 = 1.0;

/// One scene kernel. The discriminant is its id on the wire and its pipeline slot.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// The sun's visibility from its cascades: `predepth` at `texture(0)`,
    /// `vis` written at `texture(1)`, the cascades at `texture(2)`.
    SunVisCascades = 0,
    /// `shadows.debug = visibility`: `vis` (`texture(3)`) as grey over the
    /// world, the colour (`texture(0)`) kept where `predepth` (`texture(2)`)
    /// is the far plane, into `texture(1)`.
    DebugVisibility = 1,
    /// `shadows.debug = cascades`: the colour (`texture(0)`) tinted by the
    /// cascade that serves each pixel of `predepth` (`texture(2)`), into
    /// `texture(1)`.
    DebugCascades = 2,
}

/// Number of kernels, the length of the pipeline table.
pub const SCENE_KERNEL_COUNT: usize = Kernel::ALL.len();

impl Kernel {
    /// Every kernel, in id order.
    pub const ALL: [Self; 3] = [
        Self::SunVisCascades,
        Self::DebugVisibility,
        Self::DebugCascades,
    ];

    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::SunVisCascades),
            1 => Some(Self::DebugVisibility),
            2 => Some(Self::DebugCascades),
            _ => None,
        }
    }

    /// Which of `texture(0)` to `texture(3)` the kernel reads or writes; each must be bound.
    #[must_use]
    pub const fn binds(self) -> [bool; 4] {
        match self {
            Self::SunVisCascades | Self::DebugCascades => [true, true, true, false],
            Self::DebugVisibility => [true; 4],
        }
    }

    const fn entry(self) -> &'static str {
        match self {
            Self::SunVisCascades => "scene_sun_vis_cascades",
            Self::DebugVisibility => "scene_debug_visibility",
            Self::DebugCascades => "scene_debug_cascades",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::SunVisCascades => "mtld3d-scene-sun-vis-cascades",
            Self::DebugVisibility => "mtld3d-scene-debug-visibility",
            Self::DebugCascades => "mtld3d-scene-debug-cascades",
        }
    }
}

/// What a scene kernel reads at `buffer(0)`; mirrored by the MSL `SceneConstants`.
///
/// 384 bytes, no padding: every member is 16 bytes or a multiple of them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneConstants {
    /// The world projection's `_11`, `_22`, `_33`, `_43`.
    pub proj: [f32; 4],
    /// The world's viewport on the raster, in raster pixels: x, y, width, height.
    pub viewport: [f32; 4],
    /// Raster width and height, the live cascade count (0 without cascades)
    /// and the cascade side in texels.
    pub extent: [u32; 4],
    /// The sun's eye-space travel direction; `w` is the shadow strength the
    /// horizon fade leaves ([`horizon_fade`]), 0 without a sun.
    pub sun: [f32; 4],
    /// The eye-space distance where each cascade ends.
    pub split_far: [f32; 4],
    /// The eye-space distance where each cascade starts blending into the next.
    pub blend_start: [f32; 4],
    /// One texel of each cascade, in world units.
    pub texel: [f32; 4],
    /// The PCF radius in each cascade's texels.
    pub pcf_radius: [f32; 4],
    /// Eye space to each cascade's clip space, unbiased; row-vector rows,
    /// which MSL reads as the columns of the same transform.
    pub cascades: [Mat4; MAX_CASCADES],
}

impl SceneConstants {
    /// The constants of a frame whose boundary is `boundary`, rasterized at
    /// `raster` through the world viewport `viewport`, with the cascades
    /// `set` the caster passes rendered (`None`: none, no shadow).
    #[must_use]
    pub fn new(
        boundary: &PostBoundary,
        set: Option<&CascadeSet>,
        shadows: &ShadowConfig,
        raster: (u32, u32),
        viewport: [f32; 4],
    ) -> Self {
        let mut constants = Self {
            proj: boundary.proj,
            viewport,
            extent: [raster.0, raster.1, 0, u32::from(shadows.resolution)],
            sun: boundary
                .sun_dir_eye
                .map_or([0.0; 4], |[x, y, z]| [x, y, z, horizon_fade(boundary)]),
            split_far: [0.0; 4],
            blend_start: [0.0; 4],
            texel: [0.0; 4],
            pcf_radius: [0.0; 4],
            cascades: [[[0.0; 4]; 4]; MAX_CASCADES],
        };
        let Some(set) = set else {
            return constants;
        };
        let count = usize::from(set.count).min(MAX_CASCADES);
        let texels: [f64; MAX_CASCADES] =
            core::array::from_fn(|i| texel_world(&set.matrices[i], shadows.resolution));
        let radius = f64::from(shadows.softness_pct) / 100.0 * texels[0];
        for (i, &texel) in texels.iter().enumerate().take(count) {
            let (near, far) = (set.splits[i], set.splits[i + 1]);
            constants.split_far[i] = far;
            constants.blend_start[i] = CASCADE_BLEND.mul_add(near - far, far);
            constants.texel[i] = narrow(texel);
            constants.pcf_radius[i] = if texel > 0.0 {
                narrow(radius / texel).max(PCF_MIN_TEXELS)
            } else {
                PCF_MIN_TEXELS
            };
            constants.cascades[i] = set.matrices[i];
        }
        constants.extent[2] = u32::try_from(count).expect("at most four cascades");
        constants
    }

    /// The constants as the bytes a kernel binds.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8] {
        // SAFETY: `#[repr(C)]` of 4-byte scalars in 16-byte rows: no padding,
        // every byte initialized, and the slice borrows `self`.
        unsafe {
            core::slice::from_raw_parts(core::ptr::from_ref(self).cast::<u8>(), size_of::<Self>())
        }
    }
}

const _: () = assert!(size_of::<SceneConstants>() == 384);

/// One texel of the cascade `light_vp` at `res`, in world (eye) units.
///
/// The cascade's x axis spans `2 / |column 0|` units over `res` texels; an
/// orthonormal view leaves the column at its world length.
fn texel_world(light_vp: &Mat4, res: u16) -> f64 {
    let per_unit = (0..3)
        .map(|r| f64::from(light_vp[r][0]).powi(2))
        .sum::<f64>()
        .sqrt();
    if per_unit > 0.0 {
        2.0 / (f64::from(res) * per_unit)
    } else {
        0.0
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a texel size or radius narrowed to the f32 a kernel reads"
)]
const fn narrow(v: f64) -> f32 {
    v as f32
}

/// The shadow strength the sun's elevation leaves: 1 at [`HORIZON_FADE_DEGREES`] and above, falling linearly to 0 at the horizon.
///
/// The elevation is the angle between the direction toward the sun
/// (against its travel) and the world's up in eye space
/// ([`mtld3d_core::ppl::hemi_up_eye`] of the boundary's view), as
/// `PostBoundary::sun_above_horizon` measures it. 0 without a sun.
#[must_use]
pub fn horizon_fade(boundary: &PostBoundary) -> f32 {
    let Some(dir) = boundary.sun_dir_eye else {
        return 0.0;
    };
    let mut view = [0.0f32; 16];
    view.copy_from_slice(boundary.view.as_flattened());
    let up = mtld3d_core::ppl::hemi_up_eye(&mtld3d_types::D3DMATRIX { m: view });
    let length = dir.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length.is_nan() || length <= 0.0 {
        return 0.0;
    }
    let height = -dir.iter().zip(&up).map(|(d, u)| d * u).sum::<f32>() / length;
    let degrees = height.clamp(-1.0, 1.0).asin().to_degrees();
    (degrees / HORIZON_FADE_DEGREES).clamp(0.0, 1.0)
}

/// The MSL mirror of [`SceneConstants`].
pub const SCENE_CONSTANTS_MSL: &str = "struct SceneConstants { float4 proj; float4 viewport; uint4 extent; float4 sun; float4 split_far; float4 blend_start; float4 texel; float4 pcf_radius; float4x4 cascades[4]; };\n";

/// What every scene kernel shares: the eye-space position and normal of a
/// raster pixel, and the cascade that covers an eye depth.
///
/// The world's depth is standard D3D depth, `d = _33 + _43 / z` for eye
/// depth `z`. The world drew with D3D9's half-pixel shift (`pos_fixup`
/// moves every vertex half a pixel right and down), so the centre of raster
/// pixel `p` shows the D3D9 screen position `p` of the viewport, not
/// `p + 0.5`. The normal takes, on each axis, the neighbour with the smaller
/// depth change, never one on the far plane (sky), as `post`'s ambient
/// occlusion does.
const GEOMETRY: &str = r"
static float eye_depth(float d, constant SceneConstants& k) {
    return k.proj.w / (d - k.proj.z);
}

static float3 eye_position(int2 p, float d, constant SceneConstants& k) {
    float z = eye_depth(d, k);
    float2 ndc = float2(2.0 * (float(p.x) - k.viewport.x) / k.viewport.z - 1.0,
                        1.0 - 2.0 * (float(p.y) - k.viewport.y) / k.viewport.w);
    return float3(ndc.x * z / k.proj.x, ndc.y * z / k.proj.y, z);
}

static float3 surface_step(depth2d<float, access::read> depth, int2 p, int2 axis, float3 c,
                           constant SceneConstants& k) {
    int2 size = int2(k.extent.xy);
    int2 a = p + axis;
    int2 b = p - axis;
    float da = all(a < size) ? depth_at(depth, a) : 1.0;
    float db = all(b >= int2(0)) ? depth_at(depth, b) : 1.0;
    bool has_a = da < 1.0;
    bool has_b = db < 1.0;
    float3 to_a = has_a ? eye_position(a, da, k) - c : float3(0.0);
    float3 from_b = has_b ? c - eye_position(b, db, k) : float3(0.0);
    if (!has_a) return from_b;
    if (!has_b) return to_a;
    return abs(to_a.z) <= abs(from_b.z) ? to_a : from_b;
}

static float3 surface_normal(depth2d<float, access::read> depth, int2 p, float3 c,
                             constant SceneConstants& k) {
    float3 n = cross(surface_step(depth, p, int2(1, 0), c, k),
                     surface_step(depth, p, int2(0, 1), c, k));
    if (dot(n, n) < 1.0e-12) return normalize(-c);
    n = normalize(n);
    return dot(n, c) > 0.0 ? -n : n;
}

// The cascade that covers eye depth z; the cascade count past the last one.
static uint cascade_of(float z, constant SceneConstants& k) {
    uint i = 0;
    while (i < k.extent.z && z > k.split_far[i]) {
        ++i;
    }
    return i;
}
";

/// The sun's visibility from its cascades, one raster pixel per thread.
const SUN_VIS_CASCADES: &str = r"
constant int PCF_TAPS = 16;
constant float2 POISSON[16] = {
    float2(-0.94201624, -0.39906216), float2(0.94558609, -0.76890725),
    float2(-0.09418410, -0.92938870), float2(0.34495938, 0.29387760),
    float2(-0.91588581, 0.45771432), float2(-0.81544232, -0.87912464),
    float2(-0.38277543, 0.27676845), float2(0.97484398, 0.75648379),
    float2(0.44323325, -0.97511554), float2(0.53742981, -0.47373420),
    float2(-0.26496911, -0.41893023), float2(0.79197514, 0.19090188),
    float2(-0.24188840, 0.99706507), float2(-0.81409955, 0.91437590),
    float2(0.19984126, 0.78641367), float2(0.14383161, -0.14100790)
};
constexpr sampler SHADOW(coord::normalized, filter::linear, address::clamp_to_edge,
                         compare_func::less_equal);

// The share of the sun cascade i lets through to eye position c with normal
// n: c moved along n, into the cascade's clip space (an orthographic view,
// w = 1, v down the texture), compared at 16 Poisson taps over the cascade's
// PCF radius. A reference past the far side of the cascade is clamped to it,
// where nothing is behind the cleared depth.
static float cascade_visibility(depth2d_array<float> cascades, uint i, float3 c, float3 n,
                                constant SceneConstants& k) {
    float3 at = c + n * (k.texel[i] * NORMAL_OFFSET_TEXELS);
    float4 lc = k.cascades[i] * float4(at, 1.0);
    float2 uv = float2(0.5 + 0.5 * lc.x, 0.5 - 0.5 * lc.y);
    float reference = min(lc.z, 1.0);
    float2 radius = float2(k.pcf_radius[i] / float(k.extent.w));
    float lit = 0.0;
    for (int t = 0; t < PCF_TAPS; ++t) {
        lit += cascades.sample_compare(SHADOW, uv + POISSON[t] * radius, i, reference);
    }
    return lit / float(PCF_TAPS);
}

kernel void scene_sun_vis_cascades(depth2d<float, access::read> depth [[texture(0)]],
                                   texture2d<float, access::write> vis [[texture(1)]],
                                   depth2d_array<float> cascades [[texture(2)]],
                                   constant SceneConstants& k [[buffer(0)]],
                                   uint2 q [[thread_position_in_grid]]) {
    if (q.x >= k.extent.x || q.y >= k.extent.y) return;
    int2 p = int2(q);
    float d = depth_at(depth, p);
    if (d >= 1.0) { vis.write(float4(1.0), q); return; }
    float3 c = eye_position(p, d, k);
    uint i = cascade_of(c.z, k);
    if (!(c.z > 0.0) || i >= k.extent.z) { vis.write(float4(1.0), q); return; }
    float3 n = surface_normal(depth, p, c, k);
    float v = cascade_visibility(cascades, i, c, n, k);
    float blend = saturate((c.z - k.blend_start[i]) / (k.split_far[i] - k.blend_start[i]));
    if (blend > 0.0) {
        float next = i + 1 < k.extent.z ? cascade_visibility(cascades, i + 1, c, n, k) : 1.0;
        v = mix(v, next, blend);
    }
    vis.write(float4(mix(1.0, v, k.sun.w)), q);
}
";

/// `shadows.debug = visibility`.
const DEBUG_VISIBILITY: &str = r"
kernel void scene_debug_visibility(texture2d<float, access::read> colour [[texture(0)]],
                                   texture2d<float, access::write> out [[texture(1)]],
                                   depth2d<float, access::read> depth [[texture(2)]],
                                   texture2d<float, access::read> vis [[texture(3)]],
                                   constant SceneConstants& k [[buffer(0)]],
                                   uint2 q [[thread_position_in_grid]]) {
    if (q.x >= k.extent.x || q.y >= k.extent.y) return;
    float4 c = colour.read(q);
    if (depth_at(depth, int2(q)) >= 1.0) { out.write(c, q); return; }
    out.write(float4(float3(vis.read(q).r), c.a), q);
}
";

/// `shadows.debug = cascades`: red, green, blue and yellow, half over the colour.
const DEBUG_CASCADES: &str = r"
constant float3 CASCADE_TINTS[4] = {
    float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0), float3(1.0, 1.0, 0.0)
};

kernel void scene_debug_cascades(texture2d<float, access::read> colour [[texture(0)]],
                                 texture2d<float, access::write> out [[texture(1)]],
                                 depth2d<float, access::read> depth [[texture(2)]],
                                 constant SceneConstants& k [[buffer(0)]],
                                 uint2 q [[thread_position_in_grid]]) {
    if (q.x >= k.extent.x || q.y >= k.extent.y) return;
    float4 c = colour.read(q);
    float d = depth_at(depth, int2(q));
    uint i = d < 1.0 ? cascade_of(eye_depth(d, k), k) : k.extent.z;
    if (i >= k.extent.z) { out.write(c, q); return; }
    out.write(float4(mix(c.rgb, CASCADE_TINTS[i], 0.5), c.a), q);
}
";

/// The MSL a kernel compiles with: the constants, depth reads, the shared geometry and its function.
#[must_use]
pub fn kernel_source(kernel: Kernel) -> String {
    let body = match kernel {
        Kernel::SunVisCascades => SUN_VIS_CASCADES,
        Kernel::DebugVisibility => DEBUG_VISIBILITY,
        Kernel::DebugCascades => DEBUG_CASCADES,
    };
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{SCENE_CONSTANTS_MSL}constant float \
         NORMAL_OFFSET_TEXELS = {NORMAL_OFFSET_TEXELS:?};\n{}{GEOMETRY}{body}",
        super::post::DEPTH_READ
    )
}

/// Compile `kernel` into a compute pipeline owned by the caller.
pub fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: Kernel,
) -> Option<MetalHandle<MTLComputePipelineStateKind>> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Precise maths: the eye position comes back from depth through a
    // division that fast math may reorder.
    options.setMathMode(MTLMathMode::Safe);
    let library = device
        .newLibraryWithSource_options_error(
            &NSString::from_str(&kernel_source(kernel)),
            Some(&options),
        )
        .map_err(|e| log::error!(target: LOG_TARGET, "scene {kernel:?}: library failed: {e}"))
        .ok()?;
    library.setLabel(Some(&NSString::from_str(kernel.label())));
    let function = library.newFunctionWithName(&NSString::from_str(kernel.entry()))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| log::error!(target: LOG_TARGET, "scene {kernel:?}: pipeline failed: {e}"))
        .ok()?;
    // SAFETY: the canonical retain is transferred to the caller's owner.
    Some(unsafe { MetalHandle::new(Retained::into_raw(pipeline) as u64) })
}

/// The encoder's scene kernels: one pipeline per kernel, compiled on first use and kept until teardown.
#[derive(Default)]
pub struct SceneKernels {
    pipelines: [MetalHandle<MTLComputePipelineStateKind>; SCENE_KERNEL_COUNT],
}

impl SceneKernels {
    /// `kernel`'s pipeline, compiled on first use; `None` when it cannot be (tried again next time).
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

    /// Release the pipelines.
    pub fn destroy(&mut self) {
        for slot in &mut self.pipelines {
            if !slot.is_null() {
                super::post::destroy_pipeline(slot.raw());
            }
            *slot = MetalHandle::NULL;
        }
    }
}

/// A texture handle of a dispatch, `None` for 0.
fn bound_texture(raw: u64) -> Option<MetalHandle<MTLTextureKind>> {
    // SAFETY: a SceneKernel command carries canonical texture handles the
    // frame retains (the back buffer) or the encoder's scene and post
    // scratch textures, which retire behind the GPU.
    (raw != 0).then(|| unsafe { MetalHandle::<MTLTextureKind>::new(raw) })
}

/// Encode one scene kernel on its own compute encoder.
///
/// The command buffer retains every bound texture and the pipeline until it
/// completes; `setBytes` copies the constants. A kernel is never dispatched
/// with a texture it reads unbound.
pub fn encode(cb: &ProtocolObject<dyn MTLCommandBuffer>, command: &BlitCommand) -> bool {
    let Some(kernel) = Kernel::from_id(command.mip_level) else {
        log::error!(target: LOG_TARGET, "scene: unknown kernel {}", command.mip_level);
        return false;
    };
    if command.byte_size == 0 {
        log::error!(target: LOG_TARGET, "scene {kernel:?}: no constants");
        return false;
    }
    // SAFETY: `byte_size` is the address of a `SceneConstants` copied into
    // the submission's scratch arena (`BlitCommand::scene_kernel`), which
    // outlives this encode; the read copies it out unaligned.
    let constants =
        unsafe { core::ptr::read_unaligned(command.byte_size as *const SceneConstants) };
    // SAFETY: the encoder's scene kernels own this compute pipeline retain.
    let pipeline = unsafe { MetalHandle::<MTLComputePipelineStateKind>::new(command.src_offset) };
    let Some(pipeline) = pipeline.into_retained() else {
        log::error!(target: LOG_TARGET, "scene {kernel:?}: missing pipeline");
        return false;
    };
    let raw = [
        command.src_handle,
        command.dst_handle,
        command.dst_offset,
        command.bytes_per_row,
    ];
    let mut textures: [Option<Retained<ProtocolObject<dyn MTLTexture>>>; 4] = Default::default();
    for (slot, (&wanted, &raw)) in kernel.binds().iter().zip(&raw).enumerate() {
        if !wanted {
            continue;
        }
        let Some(texture) = bound_texture(raw).and_then(IntoRetained::into_retained) else {
            log::error!(target: LOG_TARGET, "scene {kernel:?}: texture({slot}) is not bound");
            return false;
        };
        textures[slot] = Some(texture);
    }
    if !cb.retainedReferences() {
        log::error!(target: LOG_TARGET, "scene: command buffer does not retain resources");
        return false;
    }
    let (width, height) = (command.region_w as usize, command.region_h as usize);
    let output = textures[1]
        .as_deref()
        .expect("every kernel writes texture(1)");
    if width == 0
        || height == 0
        || command.region_w > constants.extent[0]
        || command.region_h > constants.extent[1]
        || output.width() < width
        || output.height() < height
    {
        log::error!(
            target: LOG_TARGET,
            "scene {kernel:?}: {width}x{height} does not fit the {}x{} raster or the {}x{} output",
            constants.extent[0],
            constants.extent[1],
            output.width(),
            output.height(),
        );
        return false;
    }
    let Some(compute) = cb.computeCommandEncoder() else {
        log::error!(target: LOG_TARGET, "scene {kernel:?}: compute encoder failed");
        return false;
    };
    compute.setLabel(Some(&NSString::from_str(kernel.label())));
    compute.setComputePipelineState(&pipeline);
    for (slot, texture) in textures.iter().enumerate() {
        if let Some(texture) = texture.as_deref() {
            // SAFETY: the kernel uses this slot (`Kernel::binds`); the texture is retained.
            unsafe { compute.setTexture_atIndex(Some(texture), slot) };
        }
    }
    let bytes = constants.as_bytes();
    // SAFETY: the encoder copies the constants before returning.
    unsafe {
        compute.setBytes_length_atIndex(core::ptr::NonNull::from(bytes).cast(), bytes.len(), 0);
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
