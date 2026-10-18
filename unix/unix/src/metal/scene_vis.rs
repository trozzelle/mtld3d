//! The scene kernels: the sun's visibility from its cascades or ray traced, the ray-traced occlusion, bounce light and point-light visibility, and the shadow debug views.
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
//! Under `rt.shadows` the sun's visibility is ray traced instead, against
//! the submission's TLAS (`encoder::rt_scene`): [`Kernel::Geometry`] keeps
//! each pixel's eye normal and depth (this frame's and the previous
//! frame's), [`Kernel::SunVisRays`] traces one or two rays per pixel
//! (`rt.quality`; half resolution at `low`) from the surface,
//! [`RAY_OFFSET`] of its eye depth along its normal, toward a point of the
//! sun's disc (`rt.sunSize` across) picked by a low-discrepancy sequence
//! over the frames, offset per pixel by interleaved gradient noise; a ray
//! that reaches the far plane unblocked lets the sun through. The horizon
//! fade applies as above; with no sun up no ray is traced. The raw result
//! is denoised ([`super::denoise`]) into `vis`. These kernels read an
//! [`RtConstants`] and up to eight textures from a [`SceneDispatch`]
//! description, and the ray kernels the frame's scene ([`SceneTrace`]).
//!
//! The other ray-traced terms trace from the same surfaces and denoise the
//! same way, each into what its receivers read:
//! - [`Kernel::AoRays`] (`rt.ao`): cosine-weighted rays over the
//!   hemisphere, `rt.aoRadius` long; a hit at distance `t` occludes by
//!   `1 − (t / r)²`. Into `ao`.
//! - [`Kernel::GiRays`] (`rt.gi`): one cosine-weighted ray to the closest
//!   surface, which reflects `albedo × (sun × V × max(N·L, 0) + ambient)`
//!   (`V` a second ray toward the sun, `N` the hit triangle's normal); a
//!   miss brings `ambient × sky` (the fog colour). Into `gi`.
//! - [`Kernel::PointRays`] (`rt.pointShadows`): for each of the four scene
//!   point lights whose range reaches the surface, rays toward a point of a
//!   `mtld3d_core::rt::POINT_LIGHT_RADIUS`-unit sphere around it. Into
//!   `points`, one channel per light.
//!
//! The shape of each (resolution, rays) is
//! [`DenoiseKind::trace_shape`](super::denoise::DenoiseKind::trace_shape).
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
    perf::GpuStage,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCompileOptions, MTLComputeCommandEncoder, MTLDevice,
    MTLDispatchType, MTLIntersectionFunctionTable, MTLLanguageVersion, MTLLibrary, MTLMathMode,
    MTLSize, MTLTexture,
};

use super::{
    denoise::{self, DenoiseKind},
    handle::IntoRetained,
    rt_kernels::{self, RtPipeline, SceneBinding},
    stage_time::{StageTimer, compute_encoder},
};
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
/// How far a ray starts off its surface along the normal, as a share of the surface's eye depth.
pub const RAY_OFFSET: f32 = 1.0e-3;
/// The largest eye depth [`Kernel::Geometry`] keeps (an `RGBA16Float` channel's largest value).
pub const GEOMETRY_MAX_DEPTH: f32 = 65504.0;
/// The side, in pixels, of the threadgroup of the kernels that share their
/// reads through threadgroup memory: the geometry and the denoiser's.
pub const TILE: usize = 16;

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
    /// Each pixel's eye-space normal and eye depth (`RGBA16Float`, all zero
    /// on the far plane) from `predepth` (`texture(0)`) into `texture(1)`.
    Geometry = 3,
    /// The sun's raw visibility, ray traced: `predepth` at `texture(0)`, the
    /// raw buffer written at `texture(1)` (its top-left quarter at half
    /// resolution), the geometry at `texture(2)`; the scene at buffers 0..2.
    SunVisRays = 4,
    /// The temporal pass of each [`DenoiseKind`]: see [`denoise::passes`].
    TemporalSun = 5,
    TemporalAo = 6,
    TemporalGi = 7,
    TemporalPoints = 8,
    /// One à-trous pass of each [`DenoiseKind`]: `texture(0)` filtered into
    /// `texture(1)` on the geometry at `texture(2)`, by the noise estimate
    /// beside the history's length at `texture(3)`.
    AtrousSun = 9,
    AtrousAo = 10,
    AtrousGi = 11,
    AtrousPoints = 12,
    /// The raw ambient occlusion, ray traced, bound as [`Self::SunVisRays`]:
    /// cosine-weighted rays of `rt.aoRadius` over the surface's hemisphere.
    AoRays = 13,
    /// The raw one-bounce sunlight, bound as [`Self::SunVisRays`]: one
    /// cosine-weighted ray, the radiance the surface it reaches reflects.
    GiRays = 14,
    /// The raw visibility of the four scene point lights, one per channel,
    /// bound as [`Self::SunVisRays`].
    PointRays = 15,
}

/// Number of kernels, the length of the pipeline table.
pub const SCENE_KERNEL_COUNT: usize = Kernel::ALL.len();

/// Number of ray-traced kernels, the length of their pipeline table.
pub const RAY_KERNEL_COUNT: usize = 4;

impl Kernel {
    /// Every kernel, in id order.
    pub const ALL: [Self; 16] = [
        Self::SunVisCascades,
        Self::DebugVisibility,
        Self::DebugCascades,
        Self::Geometry,
        Self::SunVisRays,
        Self::TemporalSun,
        Self::TemporalAo,
        Self::TemporalGi,
        Self::TemporalPoints,
        Self::AtrousSun,
        Self::AtrousAo,
        Self::AtrousGi,
        Self::AtrousPoints,
        Self::AoRays,
        Self::GiRays,
        Self::PointRays,
    ];

    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        if (id as usize) < Self::ALL.len() {
            Some(Self::ALL[id as usize])
        } else {
            None
        }
    }

    /// Which of `texture(0)` to `texture(7)` the kernel reads or writes; each must be bound.
    #[must_use]
    pub const fn binds(self) -> [bool; 8] {
        const fn first(n: usize) -> [bool; 8] {
            let mut slots = [false; 8];
            let mut i = 0;
            while i < n {
                slots[i] = true;
                i += 1;
            }
            slots
        }
        match self {
            Self::Geometry => first(2),
            Self::SunVisCascades
            | Self::DebugCascades
            | Self::SunVisRays
            | Self::AoRays
            | Self::GiRays
            | Self::PointRays => first(3),
            Self::DebugVisibility
            | Self::AtrousSun
            | Self::AtrousAo
            | Self::AtrousGi
            | Self::AtrousPoints => first(4),
            Self::TemporalSun | Self::TemporalAo | Self::TemporalGi | Self::TemporalPoints => {
                first(7)
            }
        }
    }

    /// Whether its command's constants address is a [`SceneDispatch`]
    /// (the ray-traced kernels and the denoiser) rather than a bare
    /// [`SceneConstants`] (the cascades' kernels and the debug views).
    #[must_use]
    pub const fn described(self) -> bool {
        match self {
            Self::SunVisCascades | Self::DebugVisibility | Self::DebugCascades => false,
            Self::Geometry
            | Self::SunVisRays
            | Self::TemporalSun
            | Self::TemporalAo
            | Self::TemporalGi
            | Self::TemporalPoints
            | Self::AtrousSun
            | Self::AtrousAo
            | Self::AtrousGi
            | Self::AtrousPoints
            | Self::AoRays
            | Self::GiRays
            | Self::PointRays => true,
        }
    }

    /// Its slot in the ray-traced kernels' pipeline table, `None` for a
    /// kernel that traces nothing.
    #[must_use]
    pub const fn ray_slot(self) -> Option<usize> {
        match self {
            Self::SunVisRays => Some(0),
            Self::AoRays => Some(1),
            Self::GiRays => Some(2),
            Self::PointRays => Some(3),
            _ => None,
        }
    }

    /// Whether it traces the frame's scene ([`SceneDispatch::trace`]).
    #[must_use]
    pub const fn traces(self) -> bool {
        self.ray_slot().is_some()
    }

    /// The ray-traced scene stage its time counts under: the sun's rays,
    /// the sun's denoiser with the geometry it reads, and each other kind's
    /// rays and denoiser together; `None` for the cascades' kernel and the
    /// debug views.
    #[must_use]
    pub const fn stage(self) -> Option<GpuStage> {
        match self {
            Self::SunVisCascades | Self::DebugVisibility | Self::DebugCascades => None,
            Self::SunVisRays => Some(GpuStage::RtRays),
            Self::Geometry | Self::TemporalSun | Self::AtrousSun => Some(GpuStage::RtDenoise),
            Self::AoRays | Self::TemporalAo | Self::AtrousAo => Some(GpuStage::RtAo),
            Self::GiRays | Self::TemporalGi | Self::AtrousGi => Some(GpuStage::RtGi),
            Self::PointRays | Self::TemporalPoints | Self::AtrousPoints => Some(GpuStage::RtPoints),
        }
    }

    /// The buffer slot of its constants: past the scene's three for a ray-traced kernel.
    #[must_use]
    pub const fn constants_slot(self) -> usize {
        if self.traces() { 3 } else { 0 }
    }

    /// The buffer a denoise kernel smooths.
    #[must_use]
    pub const fn denoise_kind(self) -> Option<DenoiseKind> {
        match self {
            Self::TemporalSun | Self::AtrousSun => Some(DenoiseKind::Sun),
            Self::TemporalAo | Self::AtrousAo => Some(DenoiseKind::Ao),
            Self::TemporalGi | Self::AtrousGi => Some(DenoiseKind::Gi),
            Self::TemporalPoints | Self::AtrousPoints => Some(DenoiseKind::Points),
            _ => None,
        }
    }

    /// The threadgroup a dispatch of it takes, width and height: a
    /// [`TILE`]-square tile for the kernels that read their neighbourhoods
    /// through threadgroup memory, 8 × 8 for the others.
    #[must_use]
    pub const fn threadgroup(self) -> (usize, usize) {
        match self {
            Self::SunVisCascades
            | Self::DebugVisibility
            | Self::DebugCascades
            | Self::SunVisRays
            | Self::AoRays
            | Self::GiRays
            | Self::PointRays => (8, 8),
            _ => (TILE, TILE),
        }
    }

    const fn entry(self) -> &'static str {
        match self {
            Self::SunVisCascades => "scene_sun_vis_cascades",
            Self::DebugVisibility => "scene_debug_visibility",
            Self::DebugCascades => "scene_debug_cascades",
            Self::Geometry => "scene_geometry",
            Self::SunVisRays => "scene_sun_vis_rays",
            Self::TemporalSun | Self::TemporalAo | Self::TemporalGi | Self::TemporalPoints => {
                "scene_temporal"
            }
            Self::AtrousSun | Self::AtrousAo | Self::AtrousGi | Self::AtrousPoints => {
                "scene_atrous"
            }
            Self::AoRays => "scene_ao_rays",
            Self::GiRays => "scene_gi_rays",
            Self::PointRays => "scene_point_rays",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::SunVisCascades => "mtld3d-scene-sun-vis-cascades",
            Self::DebugVisibility => "mtld3d-scene-debug-visibility",
            Self::DebugCascades => "mtld3d-scene-debug-cascades",
            Self::Geometry => "mtld3d-scene-geometry",
            Self::SunVisRays => "mtld3d-scene-sun-vis-rays",
            Self::TemporalSun => "mtld3d-scene-temporal-sun",
            Self::TemporalAo => "mtld3d-scene-temporal-ao",
            Self::TemporalGi => "mtld3d-scene-temporal-gi",
            Self::TemporalPoints => "mtld3d-scene-temporal-points",
            Self::AtrousSun => "mtld3d-scene-atrous-sun",
            Self::AtrousAo => "mtld3d-scene-atrous-ao",
            Self::AtrousGi => "mtld3d-scene-atrous-gi",
            Self::AtrousPoints => "mtld3d-scene-atrous-points",
            Self::AoRays => "mtld3d-scene-ao-rays",
            Self::GiRays => "mtld3d-scene-gi-rays",
            Self::PointRays => "mtld3d-scene-point-rays",
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

/// What the ray-traced and denoise kernels read: the frame's
/// [`SceneConstants`], the temporal state and the lights the ray-traced
/// terms shade with; mirrored by the MSL `RtConstants`. 624 bytes, no
/// padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtConstants {
    pub scene: SceneConstants,
    /// The frame's index in the rays' sample sequence, the rays per pixel,
    /// the raw buffer's downscale (1 full, 2 half resolution) and the
    /// history's cap in frames.
    pub trace: [u32; 4],
    /// The tangent of the sun's angular radius, the rays' reach (the far
    /// plane), 1 when the history may be merged (0 otherwise), the à-trous
    /// step in pixels.
    pub params: [f32; 4],
    /// The previous frame's projection terms.
    pub prev_proj: [f32; 4],
    /// The previous frame's viewport on the raster.
    pub prev_viewport: [f32; 4],
    /// This frame's eye space to the previous frame's; row-vector rows,
    /// which MSL reads as the columns of the same transform.
    pub reproject: Mat4,
    /// The sun's colour (black without a sun up, the horizon fade in
    /// `scene.sun.w` applied by the kernel), and in `w` the occlusion rays'
    /// reach (`rt.aoRadius`).
    pub light: [f32; 4],
    /// The ambient colour (`D3DRS_AMBIENT`); `w` unused.
    pub ambient: [f32; 4],
    /// The sky's tint (the fog colour); `w` unused.
    pub sky: [f32; 4],
    /// The scene point lights: eye position, and the range in `w` (0 for an
    /// empty entry).
    pub points: [[f32; 4]; mtld3d_core::rt::MAX_SCENE_POINTS],
}

const _: () = assert!(size_of::<RtConstants>() == 624);

impl RtConstants {
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

/// The frame's scene as a ray-traced kernel binds it ([`rt_kernels::bind_scene`]):
/// raw canonical handles, the residency lists in the submission's scratch arena.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneTrace {
    /// The pipeline's intersection function table.
    pub table: u64,
    pub tlas: u64,
    /// The Shared buffer of the instance table.
    pub instances: u64,
    /// `blas_count` BLAS handles.
    pub blas: u64,
    /// `texture_count` texture handles the instance table names.
    pub textures: u64,
    /// `buffer_count` buffer handles whose addresses the instance table holds.
    pub buffers: u64,
    pub blas_count: u32,
    pub texture_count: u32,
    pub buffer_count: u32,
    pub pad: u32,
}

/// What a described kernel's command names ([`Kernel::described`]), in the
/// submission's scratch arena: its constants, `texture(4)` to `texture(7)`
/// (0 unbound) and, for a ray-traced kernel, the scene it traces.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SceneDispatch {
    pub constants: RtConstants,
    pub textures: [u64; 4],
    pub trace: SceneTrace,
}

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

/// The MSL mirror of [`RtConstants`], after [`SCENE_CONSTANTS_MSL`].
pub const RT_CONSTANTS_MSL: &str = "struct RtConstants { SceneConstants scene; uint4 trace; float4 params; float4 prev_proj; float4 prev_viewport; float4x4 reproject; float4 light; float4 ambient; float4 sky; float4 points[4]; };\n";

/// The eye position of raster pixel `p` at eye depth `z`, as `eye_position`
/// finds it from a depth.
const RT_GEOMETRY: &str = r"
static float3 eye_at(int2 p, float z, constant SceneConstants& k) {
    float2 ndc = float2(2.0 * (float(p.x) - k.viewport.x) / k.viewport.z - 1.0,
                        1.0 - 2.0 * (float(p.y) - k.viewport.y) / k.viewport.w);
    return float3(ndc.x * z / k.proj.x, ndc.y * z / k.proj.y, z);
}
";

/// Each pixel's eye normal and depth, kept for the denoiser and the rays.
///
/// The threadgroup finds the eye position of every pixel of its tile and
/// one around it once, into threadgroup memory, and each pixel's normal
/// steps to its neighbours' there: what `surface_normal` finds from the
/// depth, with about a quarter of the eye positions.
const GEOMETRY_KERNEL: &str = r"
constant int GEOMETRY_SIDE = TILE + 2;

// The step to a neighbour along `axis` (`surface_step` over the tile).
static float3 tile_step(threadgroup const float4* tile, int at, int axis, float3 c) {
    float4 a = tile[at + axis];
    float4 b = tile[at - axis];
    bool has_a = a.w < 1.0;
    bool has_b = b.w < 1.0;
    float3 to_a = has_a ? a.xyz - c : float3(0.0);
    float3 from_b = has_b ? c - b.xyz : float3(0.0);
    if (!has_a) return from_b;
    if (!has_b) return to_a;
    return abs(to_a.z) <= abs(from_b.z) ? to_a : from_b;
}

kernel void scene_geometry(depth2d<float, access::read> depth [[texture(0)]],
                           texture2d<float, access::write> geometry [[texture(1)]],
                           constant RtConstants& k [[buffer(0)]],
                           uint2 q [[thread_position_in_grid]],
                           uint2 l [[thread_position_in_threadgroup]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    // Each pixel's eye position and depth; a depth of 1 where there is no
    // surface (the far plane, past the raster).
    threadgroup float4 tile[GEOMETRY_SIDE * GEOMETRY_SIDE];
    int2 size = int2(k.scene.extent.xy);
    int2 origin = int2(group) * TILE - 1;
    for (int i = int(l.y) * TILE + int(l.x); i < GEOMETRY_SIDE * GEOMETRY_SIDE; i += TILE * TILE) {
        int2 t = origin + int2(i % GEOMETRY_SIDE, i / GEOMETRY_SIDE);
        float4 e = float4(0.0, 0.0, 0.0, 1.0);
        if (all(t >= int2(0)) && all(t < size)) {
            float d = depth_at(depth, t);
            if (d < 1.0) e = float4(eye_position(t, d, k.scene), d);
        }
        tile[i] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (q.x >= k.scene.extent.x || q.y >= k.scene.extent.y) return;
    int at = (int(l.y) + 1) * GEOMETRY_SIDE + int(l.x) + 1;
    float4 e = tile[at];
    float3 c = e.xyz;
    if (e.w >= 1.0 || !(c.z > 0.0)) { geometry.write(float4(0.0), q); return; }
    float3 n = cross(tile_step(tile, at, 1, c), tile_step(tile, at, GEOMETRY_SIDE, c));
    if (dot(n, n) < 1.0e-12) {
        n = normalize(-c);
    } else {
        n = normalize(n);
        n = dot(n, c) > 0.0 ? -n : n;
    }
    geometry.write(float4(n, min(c.z, GEOMETRY_MAX_DEPTH)), q);
}
";

/// What every ray-traced kernel shares: the sample sequence and the
/// surface a thread starts from.
///
/// A thread's surface is raster pixel `thread × scale` (the raw buffer's
/// downscale): its eye position from `predepth` and its normal from the
/// geometry, `false` on the far plane. The sample `rt_sample(p, index)` of
/// a pixel's sequence is the R2 sequence over the frame's sample indices,
/// offset per pixel by interleaved gradient noise, in 32-bit fixed point so
/// it stays exact however many frames run. Every ray starts [`RAY_OFFSET`]
/// of the eye depth off its surface along the normal.
const RAY_COMMON: &str = r"
static float2 rt_sample(int2 p, uint index) {
    float2 f = float2(p);
    float2 noise = fract(52.9829189 * fract(float2(dot(f, float2(0.06711056, 0.00583715)),
                                                    dot(f.yx + 37.0, float2(0.06711056, 0.00583715)))));
    uint2 r2 = uint2(index * 3242174889u, index * 2447445414u);
    return fract(noise + float2(r2) * (1.0 / 4294967296.0));
}

// A direction over the hemisphere around `n`, cosine-weighted by `u`.
static float3 rt_cosine(float3 n, float2 u) {
    float3 t = normalize(cross(abs(n.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), n));
    float3 b = cross(n, t);
    float r = sqrt(u.x);
    float a = 6.28318530718 * u.y;
    return normalize(t * (r * cos(a)) + b * (r * sin(a)) + n * sqrt(max(1.0 - u.x, 0.0)));
}

// The surface of thread `t`; false past the raw buffer or on the far plane.
static bool rt_surface(uint2 t, constant RtConstants& k, depth2d<float, access::read> depth,
                       texture2d<float, access::read> geometry, thread int2 &p,
                       thread float3 &c, thread float3 &n, thread bool &inside) {
    uint scale = max(k.trace.z, 1u);
    uint2 size = (k.scene.extent.xy + scale - 1u) / scale;
    inside = t.x < size.x && t.y < size.y;
    if (!inside) return false;
    p = int2(t * scale);
    float d = depth_at(depth, p);
    c = eye_position(p, d, k.scene);
    if (d >= 1.0 || !(c.z > 0.0)) return false;
    n = geometry.read(uint2(p)).xyz;
    return true;
}
";

/// The raw ambient occlusion, one ray-traced sample point per thread.
///
/// Each ray leaves the surface over its hemisphere, cosine-weighted, and
/// reaches `rt.aoRadius` (`light.w`). An unblocked ray counts fully open;
/// one that hits at distance `t` counts `(t / r)²`, the occlusion falling
/// off as `1 − (t / r)²`. The closest hit counts, through the alpha test.
const AO_RAYS: &str = r"
kernel void scene_ao_rays(instance_acceleration_structure accel [[buffer(0)]],
                          intersection_function_table<triangle_data, instancing> table [[buffer(1)]],
                          device const RtInstanceInfo *instances [[buffer(2)]],
                          constant RtConstants& k [[buffer(3)]],
                          depth2d<float, access::read> depth [[texture(0)]],
                          texture2d<float, access::write> raw [[texture(1)]],
                          texture2d<float, access::read> geometry [[texture(2)]],
                          uint2 t [[thread_position_in_grid]]) {
    int2 p;
    float3 c, n;
    bool inside;
    if (!rt_surface(t, k, depth, geometry, p, c, n, inside)) {
        if (inside) raw.write(float4(1.0), t);
        return;
    }
    float radius = max(k.light.w, 1.0);
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    RtPayload payload;
    payload.instances = instances;
    payload.alpha_calls = 0u;
    uint rays = max(k.trace.y, 1u);
    float open = 0.0;
    for (uint s = 0u; s < rays; ++s) {
        ray r;
        r.origin = c + n * (RAY_OFFSET * c.z);
        r.direction = rt_cosine(n, rt_sample(p, k.trace.x * rays + s));
        r.min_distance = 0.0;
        r.max_distance = radius;
        intersection_result<triangle_data, instancing> hit = isect.intersect(r, accel, 0xFFu, table, payload);
        float f = saturate(hit.distance / radius);
        open += hit.type == intersection_type::none ? 1.0 : f * f;
    }
    raw.write(float4(open / float(rays)), t);
}
";

/// The raw one-bounce sunlight, one ray-traced sample point per thread.
///
/// One cosine-weighted ray per sample to the closest surface it reaches
/// (to the far plane). That surface reflects its albedo (texture times
/// material diffuse, `rt_hit_albedo`) times the sun's light on it
/// (`sun × V × max(N·L, 0)`, `V` from a second ray toward the sun, `N`
/// the hit triangle's normal) plus the ambient; a ray that reaches nothing
/// brings the ambient tinted by the sky (the fog colour). The mean over
/// the cosine-weighted rays is the light a white diffuse surface gets back,
/// which the receivers multiply by their material diffuse.
const GI_RAYS: &str = r"
kernel void scene_gi_rays(instance_acceleration_structure accel [[buffer(0)]],
                          intersection_function_table<triangle_data, instancing> table [[buffer(1)]],
                          device const RtInstanceInfo *instances [[buffer(2)]],
                          constant RtConstants& k [[buffer(3)]],
                          depth2d<float, access::read> depth [[texture(0)]],
                          texture2d<float, access::write> raw [[texture(1)]],
                          texture2d<float, access::read> geometry [[texture(2)]],
                          uint2 t [[thread_position_in_grid]]) {
    int2 p;
    float3 c, n;
    bool inside;
    if (!rt_surface(t, k, depth, geometry, p, c, n, inside)) {
        if (inside) raw.write(float4(0.0), t);
        return;
    }
    bool sun_up = k.scene.sun.w > 0.0;
    float3 toward = sun_up ? -normalize(k.scene.sun.xyz) : float3(0.0);
    float3 sun = sun_up ? k.light.rgb * k.scene.sun.w : float3(0.0);
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    intersector<triangle_data, instancing> shadow;
    shadow.accept_any_intersection(true);
    shadow.assume_geometry_type(geometry_type::triangle);
    RtPayload payload;
    payload.instances = instances;
    payload.alpha_calls = 0u;
    uint rays = max(k.trace.y, 1u);
    float3 sum = float3(0.0);
    for (uint s = 0u; s < rays; ++s) {
        ray r;
        r.origin = c + n * (RAY_OFFSET * c.z);
        r.direction = rt_cosine(n, rt_sample(p, k.trace.x * rays + s));
        r.min_distance = 0.0;
        r.max_distance = k.params.y;
        intersection_result<triangle_data, instancing> hit = isect.intersect(r, accel, 0xFFu, table, payload);
        if (hit.type == intersection_type::none) {
            sum += k.ambient.rgb * k.sky.rgb;
            continue;
        }
        device const RtInstanceInfo &info = instances[hit.instance_id];
        float3 albedo = rt_hit_albedo(info, hit.primitive_id, hit.triangle_barycentric_coord);
        float3 nh = rt_hit_normal(info, hit.primitive_id, r.direction);
        float3 x = r.origin + r.direction * hit.distance;
        float3 light = k.ambient.rgb;
        float ndl = dot(nh, toward);
        if (sun_up && ndl > 0.0) {
            ray sr;
            sr.origin = x + nh * (RAY_OFFSET * max(x.z, 1.0));
            sr.direction = toward;
            sr.min_distance = 0.0;
            sr.max_distance = k.params.y;
            intersection_result<triangle_data, instancing> blocked = shadow.intersect(sr, accel, 0xFFu, table, payload);
            if (blocked.type == intersection_type::none) light += sun * ndl;
        }
        sum += albedo * light;
    }
    raw.write(float4(sum / float(rays), 1.0), t);
}
";

/// The raw visibility of the four scene point lights, one per channel, one
/// ray-traced sample point per thread.
///
/// For each scene point light whose range reaches the surface, each ray
/// heads for a point of a [`POINT_LIGHT_RADIUS`]-unit sphere around the
/// light (uniform over the sphere, a sequence of its own per light) and
/// lets the light through when nothing blocks it on the way; a light out
/// of range, or an empty entry, leaves its channel at 1.
const POINT_RAYS: &str = r"
kernel void scene_point_rays(instance_acceleration_structure accel [[buffer(0)]],
                             intersection_function_table<triangle_data, instancing> table [[buffer(1)]],
                             device const RtInstanceInfo *instances [[buffer(2)]],
                             constant RtConstants& k [[buffer(3)]],
                             depth2d<float, access::read> depth [[texture(0)]],
                             texture2d<float, access::write> raw [[texture(1)]],
                             texture2d<float, access::read> geometry [[texture(2)]],
                             uint2 t [[thread_position_in_grid]]) {
    int2 p;
    float3 c, n;
    bool inside;
    if (!rt_surface(t, k, depth, geometry, p, c, n, inside)) {
        if (inside) raw.write(float4(1.0), t);
        return;
    }
    intersector<triangle_data, instancing> isect;
    isect.accept_any_intersection(true);
    isect.assume_geometry_type(geometry_type::triangle);
    RtPayload payload;
    payload.instances = instances;
    payload.alpha_calls = 0u;
    uint rays = max(k.trace.y, 1u);
    float3 origin = c + n * (RAY_OFFSET * c.z);
    float4 vis = float4(1.0);
    for (uint i = 0u; i < 4u; ++i) {
        float4 light = k.points[i];
        if (!(light.w > 0.0) || distance(light.xyz, c) > light.w) continue;
        float lit = 0.0;
        for (uint s = 0u; s < rays; ++s) {
            float2 u = rt_sample(p, (k.trace.x * rays + s) * 4u + i);
            float z = 1.0 - 2.0 * u.x;
            float a = 6.28318530718 * u.y;
            float rr = sqrt(max(1.0 - z * z, 0.0));
            float3 target = light.xyz + float3(rr * cos(a), rr * sin(a), z) * POINT_LIGHT_RADIUS;
            float3 to = target - origin;
            float len = length(to);
            if (!(len > 1.0e-3)) { lit += 1.0; continue; }
            ray r;
            r.origin = origin;
            r.direction = to / len;
            r.min_distance = 0.0;
            r.max_distance = len;
            intersection_result<triangle_data, instancing> hit = isect.intersect(r, accel, 0xFFu, table, payload);
            lit += hit.type == intersection_type::none ? 1.0 : 0.0;
        }
        vis[i] = lit / float(rays);
    }
    raw.write(vis, t);
}
";

/// The sun's raw visibility, one ray-traced sample point per thread.
///
/// The point is raster pixel `thread × scale`. Each ray starts
/// [`RAY_OFFSET`] of the eye depth off the surface along its normal and
/// heads for a point of the sun's disc: the sample `u` of the pixel's
/// sequence (`RAY_COMMON`), uniform over the disc of radius
/// `tan(sunSize / 2)` around the direction toward the sun. Any hit before
/// the far plane blocks it (the alpha test through the scene's
/// intersection function); the mask leaves no instance out.
const SUN_VIS_RAYS: &str = r"
kernel void scene_sun_vis_rays(instance_acceleration_structure accel [[buffer(0)]],
                               intersection_function_table<triangle_data, instancing> table [[buffer(1)]],
                               device const RtInstanceInfo *instances [[buffer(2)]],
                               constant RtConstants& k [[buffer(3)]],
                               depth2d<float, access::read> depth [[texture(0)]],
                               texture2d<float, access::write> raw [[texture(1)]],
                               texture2d<float, access::read> geometry [[texture(2)]],
                               uint2 t [[thread_position_in_grid]]) {
    uint scale = max(k.trace.z, 1u);
    uint2 size = (k.scene.extent.xy + scale - 1u) / scale;
    if (t.x >= size.x || t.y >= size.y) return;
    int2 p = int2(t * scale);
    float d = depth_at(depth, p);
    float3 c = eye_position(p, d, k.scene);
    if (d >= 1.0 || !(c.z > 0.0) || !(k.scene.sun.w > 0.0)) { raw.write(float4(1.0), t); return; }
    float3 n = geometry.read(uint2(p)).xyz;
    float3 toward = -normalize(k.scene.sun.xyz);
    float3 side = normalize(cross(abs(toward.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), toward));
    float3 up = cross(toward, side);
    intersector<triangle_data, instancing> isect;
    isect.accept_any_intersection(true);
    // Every BLAS is triangles. Opaque instances (and their geometry) never
    // call the alpha test: only alpha-tested ones are non-opaque.
    isect.assume_geometry_type(geometry_type::triangle);
    RtPayload payload;
    payload.instances = instances;
    payload.alpha_calls = 0u;
    uint rays = max(k.trace.y, 1u);
    float lit = 0.0;
    for (uint s = 0u; s < rays; ++s) {
        float2 u = rt_sample(p, k.trace.x * rays + s);
        float radius = sqrt(u.x) * k.params.x;
        float angle = 6.28318530718 * u.y;
        ray r;
        r.origin = c + n * (RAY_OFFSET * c.z);
        r.direction = normalize(toward + radius * (cos(angle) * side + sin(angle) * up));
        r.min_distance = 0.0;
        r.max_distance = k.params.y;
        intersection_result<triangle_data, instancing> hit = isect.intersect(r, accel, 0xFFu, table, payload);
        lit += hit.type == intersection_type::none ? 1.0 : 0.0;
    }
    raw.write(float4(mix(1.0, lit / float(rays), k.scene.sun.w)), t);
}
";

/// The MSL a kernel compiles with: the constants, depth reads, the shared geometry and its function.
///
/// A ray-traced kernel's ([`Kernel::traces`]) goes after
/// [`rt_kernels::RT_COMMON_MSL`], which opens it ([`create_rt_kernel`]).
#[must_use]
pub fn kernel_source(kernel: Kernel) -> String {
    const HEADER: &str = "#include <metal_stdlib>\nusing namespace metal;\n";
    let depth = super::post::DEPTH_READ;
    let rt = format!(
        "{SCENE_CONSTANTS_MSL}{RT_CONSTANTS_MSL}constant float RAY_OFFSET = \
         {RAY_OFFSET:?};\nconstant float GEOMETRY_MAX_DEPTH = \
         {GEOMETRY_MAX_DEPTH:?};\nconstant int TILE = {TILE};\n{depth}{GEOMETRY}{RT_GEOMETRY}"
    );
    let cascades = |body: &str| {
        format!(
            "{HEADER}{SCENE_CONSTANTS_MSL}constant float NORMAL_OFFSET_TEXELS = \
             {NORMAL_OFFSET_TEXELS:?};\n{depth}{GEOMETRY}{body}"
        )
    };
    let rays = |body: &str| {
        format!(
            "{rt}constant float POINT_LIGHT_RADIUS = {:?};\n{RAY_COMMON}{body}",
            mtld3d_core::rt::POINT_LIGHT_RADIUS
        )
    };
    match kernel {
        Kernel::SunVisCascades => cascades(SUN_VIS_CASCADES),
        Kernel::DebugVisibility => cascades(DEBUG_VISIBILITY),
        Kernel::DebugCascades => cascades(DEBUG_CASCADES),
        Kernel::Geometry => format!("{HEADER}{rt}{GEOMETRY_KERNEL}"),
        Kernel::SunVisRays => rays(SUN_VIS_RAYS),
        Kernel::AoRays => rays(AO_RAYS),
        Kernel::GiRays => rays(GI_RAYS),
        Kernel::PointRays => rays(POINT_RAYS),
        Kernel::TemporalSun
        | Kernel::TemporalAo
        | Kernel::TemporalGi
        | Kernel::TemporalPoints
        | Kernel::AtrousSun
        | Kernel::AtrousAo
        | Kernel::AtrousGi
        | Kernel::AtrousPoints => {
            let kind = kernel.denoise_kind().expect("a denoise kernel");
            format!("{HEADER}{rt}{}", denoise::kernel_msl(kind))
        }
    }
}

/// Compile `kernel` into a compute pipeline owned by the caller; `None`
/// for a ray-traced kernel, which [`create_rt_kernel`] compiles.
pub fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: Kernel,
) -> Option<MetalHandle<MTLComputePipelineStateKind>> {
    if kernel.traces() {
        return None;
    }
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Precise maths where the eye position comes back from depth through a
    // division that fast math may reorder; the denoiser reads the depth the
    // geometry kept and only weighs with it.
    if kernel.denoise_kind().is_none() {
        options.setMathMode(MTLMathMode::Safe);
    }
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

/// Compile the ray-traced `kernel` with the shared MSL and its alpha test; `None` for another kernel.
#[must_use]
pub fn create_rt_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: Kernel,
) -> Option<RtPipeline> {
    if !kernel.traces() {
        return None;
    }
    rt_kernels::create_rt_pipeline(device, &kernel_source(kernel), kernel.entry())
}

/// The encoder's scene kernels: one pipeline per kernel, compiled on first use and kept until teardown.
#[derive(Default)]
pub struct SceneKernels {
    pipelines: [MetalHandle<MTLComputePipelineStateKind>; SCENE_KERNEL_COUNT],
    /// The ray-traced kernels and their intersection function tables, by [`Kernel::ray_slot`].
    rays: [Option<RtPipeline>; RAY_KERNEL_COUNT],
}

impl SceneKernels {
    /// `kernel`'s pipeline, compiled on first use; `None` when it cannot be
    /// (tried again next time) and for a ray-traced kernel
    /// ([`Self::rt_pipeline`]).
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

    /// The ray-traced `kernel`'s pipeline and intersection function table,
    /// raw, compiled on first use; `None` when it cannot be (tried again next
    /// time) and for another kernel. The scene kernels keep both alive.
    pub fn rt_pipeline(
        &mut self,
        device: MetalHandle<MTLDeviceKind>,
        kernel: Kernel,
    ) -> Option<(u64, u64)> {
        let slot = &mut self.rays[kernel.ray_slot()?];
        if slot.is_none() {
            *slot = create_rt_kernel(&*device.into_retained()?, kernel);
        }
        slot.as_ref().map(|rt| {
            (
                Retained::as_ptr(&rt.pipeline) as u64,
                Retained::as_ptr(&rt.table) as u64,
            )
        })
    }

    /// Release the pipelines.
    pub fn destroy(&mut self) {
        for slot in &mut self.pipelines {
            if !slot.is_null() {
                super::post::destroy_pipeline(slot.raw());
            }
            *slot = MetalHandle::NULL;
        }
        self.rays = Default::default();
    }
}

/// A texture handle of a dispatch, `None` for 0.
fn bound_texture(raw: u64) -> Option<MetalHandle<MTLTextureKind>> {
    // SAFETY: a SceneKernel command carries canonical texture handles the
    // frame retains (the back buffer) or the encoder's scene and post
    // scratch textures, which retire behind the GPU.
    (raw != 0).then(|| unsafe { MetalHandle::<MTLTextureKind>::new(raw) })
}

/// `count` raw handles at `address` in the submission's scratch arena.
fn handles<'a>(address: u64, count: u32) -> &'a [u64] {
    if address == 0 || count == 0 {
        return &[];
    }
    // SAFETY: the encoder copied `count` handles to `address` in the
    // submission's scratch arena, which outlives this encode.
    unsafe { core::slice::from_raw_parts(address as *const u64, count as usize) }
}

/// What a command's constants address holds: a kernel's bare constants, or a described kernel's dispatch.
enum Described<'a> {
    Bare(&'a SceneConstants),
    Dispatch(&'a SceneDispatch),
}

impl Described<'_> {
    const fn scene(&self) -> &SceneConstants {
        match self {
            Self::Bare(constants) => constants,
            Self::Dispatch(dispatch) => &dispatch.constants.scene,
        }
    }

    const fn bytes(&self) -> &[u8] {
        match self {
            Self::Bare(constants) => constants.as_bytes(),
            Self::Dispatch(dispatch) => dispatch.constants.as_bytes(),
        }
    }
}

/// Encode one scene kernel on its own compute encoder, timed under its
/// stage ([`Kernel::stage`]) when `timer` is given.
///
/// The command buffer retains every bound texture and the pipeline until it
/// completes; `setBytes` copies the constants. A kernel is never dispatched
/// with a texture it reads unbound. A ray-traced kernel binds its scene and
/// makes every object the scene reaches resident ([`rt_kernels::bind_scene`]).
pub fn encode(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    command: &BlitCommand,
    timer: Option<&mut StageTimer>,
) -> bool {
    let Some(kernel) = Kernel::from_id(command.mip_level) else {
        log::error!(target: LOG_TARGET, "scene: unknown kernel {}", command.mip_level);
        return false;
    };
    if command.byte_size == 0 {
        log::error!(target: LOG_TARGET, "scene {kernel:?}: no constants");
        return false;
    }
    // The description is copied out unaligned onto the stack, no heap.
    let (dispatch, bare);
    // SAFETY: `byte_size` is the address of a `SceneDispatch` (a described
    // kernel) or a `SceneConstants` copied into the submission's scratch
    // arena (`BlitCommand::scene_kernel`), which outlives this encode; the
    // read copies it out unaligned.
    let described = unsafe {
        if kernel.described() {
            dispatch = core::ptr::read_unaligned(command.byte_size as *const SceneDispatch);
            Described::Dispatch(&dispatch)
        } else {
            bare = core::ptr::read_unaligned(command.byte_size as *const SceneConstants);
            Described::Bare(&bare)
        }
    };
    let constants = described.scene();
    // SAFETY: the encoder's scene kernels own this compute pipeline retain.
    let pipeline = unsafe { MetalHandle::<MTLComputePipelineStateKind>::new(command.src_offset) };
    let Some(pipeline) = pipeline.into_retained() else {
        log::error!(target: LOG_TARGET, "scene {kernel:?}: missing pipeline");
        return false;
    };
    let extra = match &described {
        Described::Dispatch(dispatch) => dispatch.textures,
        Described::Bare(_) => [0; 4],
    };
    let raw = [
        command.src_handle,
        command.dst_handle,
        command.dst_offset,
        command.bytes_per_row,
        extra[0],
        extra[1],
        extra[2],
        extra[3],
    ];
    let mut textures: [Option<Retained<ProtocolObject<dyn MTLTexture>>>; 8] = Default::default();
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
    let encoder = match kernel.stage() {
        Some(stage) => compute_encoder(cb, timer, stage, MTLDispatchType::Serial),
        None => cb.computeCommandEncoder(),
    };
    let Some(compute) = encoder else {
        log::error!(target: LOG_TARGET, "scene {kernel:?}: compute encoder failed");
        return false;
    };
    compute.setLabel(Some(&NSString::from_str(kernel.label())));
    match &described {
        Described::Dispatch(dispatch) if kernel.traces() => {
            let trace = &dispatch.trace;
            // SAFETY: the scene kernels own the table, which outlives the
            // pipeline that made it; the retain is the command buffer's.
            let table = unsafe {
                Retained::retain(
                    trace.table as *mut ProtocolObject<dyn MTLIntersectionFunctionTable>,
                )
            };
            let Some(table) = table else {
                log::error!(target: LOG_TARGET, "scene {kernel:?}: missing intersection function table");
                compute.endEncoding();
                return false;
            };
            let rt = RtPipeline { pipeline, table };
            let scene = SceneBinding {
                tlas: trace.tlas,
                instances: trace.instances,
                blas: handles(trace.blas, trace.blas_count),
                textures: handles(trace.textures, trace.texture_count),
                buffers: handles(trace.buffers, trace.buffer_count),
            };
            if !rt_kernels::bind_scene(&compute, &rt, &scene) {
                compute.endEncoding();
                return false;
            }
        }
        _ => compute.setComputePipelineState(&pipeline),
    }
    for (slot, texture) in textures.iter().enumerate() {
        if let Some(texture) = texture.as_deref() {
            // SAFETY: the kernel uses this slot (`Kernel::binds`); the texture is retained.
            unsafe { compute.setTexture_atIndex(Some(texture), slot) };
        }
    }
    let bytes = described.bytes();
    // SAFETY: the encoder copies the constants before returning.
    unsafe {
        compute.setBytes_length_atIndex(
            core::ptr::NonNull::from(bytes).cast(),
            bytes.len(),
            kernel.constants_slot(),
        );
    }
    let (tw, th) = kernel.threadgroup();
    let threadgroup = MTLSize {
        width: tw,
        height: th,
        depth: 1,
    };
    let grid = MTLSize {
        width: width.div_ceil(tw),
        height: height.div_ceil(th),
        depth: 1,
    };
    compute.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
    compute.endEncoding();
    true
}
