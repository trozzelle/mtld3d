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
    /// `post.ao`: horizon-based ambient occlusion at half extent, blurred and
    /// multiplied into the colour.
    Ao,
    /// `post.godRays`: light shafts from a visible sun, a radial blur of the
    /// sky toward it at half extent, added to the colour in the sun's colour.
    GodRays,
    /// `post.bloom`: the colour above `post.bloomThreshold`, blurred through
    /// five halving levels and added back.
    Bloom,
    /// `post.debug`: rgb → 1 − rgb, alpha kept.
    Invert,
}

/// Number of effects, the capacity of an [`EffectChain`].
pub const POST_EFFECT_COUNT: usize = Effect::CHAIN.len();

impl Effect {
    /// Every effect, in the order the chain runs them.
    pub const CHAIN: [Self; 4] = [Self::Ao, Self::GodRays, Self::Bloom, Self::Invert];

    const fn enabled(self, cfg: PostConfig) -> bool {
        match self {
            Self::Ao => cfg.ao_pct > 0,
            Self::GodRays => cfg.god_rays_pct > 0,
            Self::Bloom => cfg.bloom_pct > 0,
            Self::Invert => cfg.debug,
        }
    }

    /// The effect's passes, in order; the last one writes the new colour.
    #[must_use]
    pub const fn passes(self) -> &'static [Pass] {
        match self {
            Self::Ao => &[
                Pass {
                    kernel: Kernel::AoHorizon,
                    input: Input::Colour,
                    aux: None,
                    output: Output::Slot(Slot::AoRaw),
                },
                Pass {
                    kernel: Kernel::AoBlur,
                    input: Input::Slot(Slot::AoRaw),
                    aux: None,
                    output: Output::Slot(Slot::AoBlurred),
                },
                Pass {
                    kernel: Kernel::AoApply,
                    input: Input::Colour,
                    aux: Some(Input::Slot(Slot::AoBlurred)),
                    output: Output::Colour,
                },
            ],
            Self::GodRays => &[
                Pass {
                    kernel: Kernel::GodRaysMask,
                    input: Input::Colour,
                    aux: None,
                    output: Output::Slot(Slot::GodRaysMask),
                },
                Pass {
                    kernel: Kernel::GodRaysBlur,
                    input: Input::Slot(Slot::GodRaysMask),
                    aux: None,
                    output: Output::Slot(Slot::GodRays),
                },
                Pass {
                    kernel: Kernel::GodRaysApply,
                    input: Input::Colour,
                    aux: Some(Input::Slot(Slot::GodRays)),
                    output: Output::Colour,
                },
            ],
            Self::Bloom => &BLOOM_PASSES,
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

/// Bloom's passes: the threshold into level 0, four downsamples to level 4,
/// four upsamples back to level 0 each adding the coarser level, the apply.
const BLOOM_PASSES: [Pass; 10] = [
    Pass {
        kernel: Kernel::BloomThreshold,
        input: Input::Colour,
        aux: None,
        output: Output::Slot(Slot::BloomDown0),
    },
    bloom_down(Slot::BloomDown0, Slot::BloomDown1),
    bloom_down(Slot::BloomDown1, Slot::BloomDown2),
    bloom_down(Slot::BloomDown2, Slot::BloomDown3),
    bloom_down(Slot::BloomDown3, Slot::BloomDown4),
    bloom_up(Slot::BloomDown3, Slot::BloomDown4, Slot::BloomUp3),
    bloom_up(Slot::BloomDown2, Slot::BloomUp3, Slot::BloomUp2),
    bloom_up(Slot::BloomDown1, Slot::BloomUp2, Slot::BloomUp1),
    bloom_up(Slot::BloomDown0, Slot::BloomUp1, Slot::BloomUp0),
    Pass {
        kernel: Kernel::BloomApply,
        input: Input::Colour,
        aux: Some(Input::Slot(Slot::BloomUp0)),
        output: Output::Colour,
    },
];

/// Bloom's downsample from one level to the next, half its extent.
const fn bloom_down(from: Slot, to: Slot) -> Pass {
    Pass {
        kernel: Kernel::BloomDownsample,
        input: Input::Slot(from),
        aux: None,
        output: Output::Slot(to),
    }
}

/// Bloom's upsample: the coarser level `coarse`, tent-filtered up to `level`'s
/// extent and added to it, into `to`.
const fn bloom_up(level: Slot, coarse: Slot, to: Slot) -> Pass {
    Pass {
        kernel: Kernel::BloomUpsample,
        input: Input::Slot(level),
        aux: Some(Input::Slot(coarse)),
        output: Output::Slot(to),
    }
}

/// One compute kernel. The discriminant is its id on the wire and its pipeline slot.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// rgb → 1 − rgb, alpha kept.
    Invert = 0,
    /// AO's horizon search: occlusion per half-extent texel, from depth alone.
    AoHorizon = 1,
    /// AO's 4-tap bilateral blur at half extent.
    AoBlur = 2,
    /// The colour darkened by the blurred occlusion, at full extent.
    AoApply = 3,
    /// God rays' mask: the sky's colour per half-extent texel, black elsewhere.
    GodRaysMask = 4,
    /// God rays' 32-sample radial blur of the mask toward the sun.
    GodRaysBlur = 5,
    /// The rays, in the sun's colour, added to the colour at full extent.
    GodRaysApply = 6,
    /// Bloom's soft-knee threshold, averaged into the first half-extent level.
    BloomThreshold = 7,
    /// Bloom's 5-tap downsample from one level to the next.
    BloomDownsample = 8,
    /// Bloom's 9-tap tent upsample of a coarser level, added to a finer one.
    BloomUpsample = 9,
    /// The blurred levels, scaled by the strength, added to the colour.
    BloomApply = 10,
}

/// Number of kernels, the length of the pipeline table.
pub const POST_KERNEL_COUNT: usize = Kernel::ALL.len();

impl Kernel {
    /// Every kernel, in id order.
    pub const ALL: [Self; 11] = [
        Self::Invert,
        Self::AoHorizon,
        Self::AoBlur,
        Self::AoApply,
        Self::GodRaysMask,
        Self::GodRaysBlur,
        Self::GodRaysApply,
        Self::BloomThreshold,
        Self::BloomDownsample,
        Self::BloomUpsample,
        Self::BloomApply,
    ];

    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Invert),
            1 => Some(Self::AoHorizon),
            2 => Some(Self::AoBlur),
            3 => Some(Self::AoApply),
            4 => Some(Self::GodRaysMask),
            5 => Some(Self::GodRaysBlur),
            6 => Some(Self::GodRaysApply),
            7 => Some(Self::BloomThreshold),
            8 => Some(Self::BloomDownsample),
            9 => Some(Self::BloomUpsample),
            10 => Some(Self::BloomApply),
            _ => None,
        }
    }

    /// Whether the kernel reads the depth snapshot at `texture(2)`.
    #[must_use]
    pub const fn reads_depth(self) -> bool {
        match self {
            Self::Invert
            | Self::GodRaysBlur
            | Self::GodRaysApply
            | Self::BloomThreshold
            | Self::BloomDownsample
            | Self::BloomUpsample
            | Self::BloomApply => false,
            Self::AoHorizon | Self::AoBlur | Self::AoApply | Self::GodRaysMask => true,
        }
    }

    /// Whether the kernel reads a second input at `texture(3)`.
    #[must_use]
    pub const fn reads_aux(self) -> bool {
        matches!(
            self,
            Self::AoApply | Self::GodRaysApply | Self::BloomUpsample | Self::BloomApply
        )
    }

    const fn entry(self) -> &'static str {
        match self {
            Self::Invert => "post_invert",
            Self::AoHorizon => "post_ao_horizon",
            Self::AoBlur => "post_ao_blur",
            Self::AoApply => "post_ao_apply",
            Self::GodRaysMask => "post_god_rays_mask",
            Self::GodRaysBlur => "post_god_rays_blur",
            Self::GodRaysApply => "post_god_rays_apply",
            Self::BloomThreshold => "post_bloom_threshold",
            Self::BloomDownsample => "post_bloom_downsample",
            Self::BloomUpsample => "post_bloom_upsample",
            Self::BloomApply => "post_bloom_apply",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Invert => "mtld3d-post-invert",
            Self::AoHorizon => "mtld3d-post-ao-horizon",
            Self::AoBlur => "mtld3d-post-ao-blur",
            Self::AoApply => "mtld3d-post-ao-apply",
            Self::GodRaysMask => "mtld3d-post-god-rays-mask",
            Self::GodRaysBlur => "mtld3d-post-god-rays-blur",
            Self::GodRaysApply => "mtld3d-post-god-rays-apply",
            Self::BloomThreshold => "mtld3d-post-bloom-threshold",
            Self::BloomDownsample => "mtld3d-post-bloom-downsample",
            Self::BloomUpsample => "mtld3d-post-bloom-upsample",
            Self::BloomApply => "mtld3d-post-bloom-apply",
        }
    }
}

/// A pass's output extent: the raster divided by `2^shift`, rounded up and
/// never below 1 px.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    shift: u32,
}

impl Extent {
    /// The raster itself.
    pub const FULL: Self = Self { shift: 0 };
    /// Half the raster on each axis.
    pub const HALF: Self = Self { shift: 1 };

    /// The raster divided by `2^shift` on each axis.
    const fn down(shift: u32) -> Self {
        Self { shift }
    }

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
    /// AO's occlusion straight from the horizon search.
    AoRaw,
    /// AO's occlusion after the bilateral blur.
    AoBlurred,
    /// God rays' far-depth mask: the sky's colour, black elsewhere.
    GodRaysMask,
    /// God rays after the radial blur.
    GodRays,
    /// Bloom's thresholded colour, half extent: the first level.
    BloomDown0,
    /// Bloom's levels 1–4, each half the one before (1/4 to 1/32 extent).
    BloomDown1,
    BloomDown2,
    BloomDown3,
    BloomDown4,
    /// Bloom's level `n` with every coarser level added (`BloomUp0` is the
    /// whole glow, half extent).
    BloomUp0,
    BloomUp1,
    BloomUp2,
    BloomUp3,
}

/// Number of slots, the length of the scratch table.
pub const POST_SLOT_COUNT: usize = Slot::ALL.len();

impl Slot {
    /// Every slot, in index order.
    pub const ALL: [Self; 15] = [
        Self::ColourA,
        Self::ColourB,
        Self::AoRaw,
        Self::AoBlurred,
        Self::GodRaysMask,
        Self::GodRays,
        Self::BloomDown0,
        Self::BloomDown1,
        Self::BloomDown2,
        Self::BloomDown3,
        Self::BloomDown4,
        Self::BloomUp0,
        Self::BloomUp1,
        Self::BloomUp2,
        Self::BloomUp3,
    ];

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
            Self::AoRaw
            | Self::AoBlurred
            | Self::GodRaysMask
            | Self::GodRays
            | Self::BloomDown0
            | Self::BloomUp0 => Extent::HALF,
            Self::BloomDown1 | Self::BloomUp1 => Extent::down(2),
            Self::BloomDown2 | Self::BloomUp2 => Extent::down(3),
            Self::BloomDown3 | Self::BloomUp3 => Extent::down(4),
            Self::BloomDown4 => Extent::down(5),
        }
    }

    /// The slot's pixel format; a colour slot takes the back buffer's `colour`.
    #[must_use]
    pub const fn format(self, colour: PixelFormat) -> PixelFormat {
        match self {
            Self::ColourA | Self::ColourB => colour,
            // One occlusion factor in [0, 1]; half floats keep the blur smooth.
            Self::AoRaw | Self::AoBlurred => PixelFormat::R16Float,
            // Light in [0, 1] per channel, summed smoothly by the blur.
            Self::GodRaysMask | Self::GodRays => PixelFormat::Rgba16Float,
            // Bloom's levels: a faint glow survives nine stores, and a float
            // scene's values above 1 pass through.
            Self::BloomDown0
            | Self::BloomDown1
            | Self::BloomDown2
            | Self::BloomDown3
            | Self::BloomDown4
            | Self::BloomUp0
            | Self::BloomUp1
            | Self::BloomUp2
            | Self::BloomUp3 => PixelFormat::Rgba16Float,
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

    /// The chain for this boundary's sun: without god rays when the sun is
    /// not visible ([`sun_screen`]), so none of their passes is queued, and
    /// then why. The reason is `None` when the chain had no god rays to drop.
    #[must_use]
    pub fn for_sun(self, boundary: &PostBoundary) -> (Self, Option<SunSkip>) {
        if !self.as_slice().contains(&Effect::GodRays) {
            return (self, None);
        }
        let Err(skip) = sun_screen(boundary) else {
            return (self, None);
        };
        let mut chain = Self {
            effects: self.effects,
            len: 0,
        };
        for &effect in self.as_slice() {
            if effect != Effect::GodRays {
                chain.effects[chain.len] = effect;
                chain.len += 1;
            }
        }
        (chain, Some(skip))
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

/// How far off screen the sun may project and still cast rays: `|NDC| ≤ 1.5`
/// on both axes, the screen grown 1.5× about its centre.
const SUN_NDC_MARGIN: f32 = 1.5;

/// Why a boundary's sun casts no rays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SunSkip {
    /// No directional light is enabled.
    NoSun,
    /// The sun has set: it is not above the horizon.
    BelowHorizon,
    /// The sun is behind the eye or in the eye plane.
    BehindCamera,
    /// The sun projects outside the margin, at this NDC.
    OffScreen { ndc: [f32; 2] },
}

/// The sun's position on the raster, `[u, v]` in `[0, 1]` from the top left
/// (beyond it inside the margin), or why it casts no rays: there is no sun,
/// it is below the horizon (`PostBoundary::sun_above_horizon`), it is behind
/// the eye or in the eye plane, or it projects outside the margin.
///
/// The sun lies along −direction: a D3D light's direction is its light's
/// travel, from the sun toward the scene. It projects through the world
/// projection's `_11` and `_22`: `ndc = (x·_11 / z, y·_22 / z)`.
///
/// # Errors
/// The [`SunSkip`] reason when the sun is not visible.
pub fn sun_screen(boundary: &PostBoundary) -> Result<[f32; 2], SunSkip> {
    let [dx, dy, dz] = boundary.sun_dir_eye.ok_or(SunSkip::NoSun)?;
    if !boundary.sun_above_horizon {
        return Err(SunSkip::BelowHorizon);
    }
    let (x, y, z) = (-dx, -dy, -dz);
    if z.is_nan() || z <= 0.0 {
        return Err(SunSkip::BehindCamera);
    }
    let ndc_x = x * boundary.proj[0] / z;
    let ndc_y = y * boundary.proj[1] / z;
    if !(ndc_x.abs() <= SUN_NDC_MARGIN && ndc_y.abs() <= SUN_NDC_MARGIN) {
        return Err(SunSkip::OffScreen {
            ndc: [ndc_x, ndc_y],
        });
    }
    Ok([0.5_f32.mul_add(ndc_x, 0.5), 0.5_f32.mul_add(-ndc_y, 0.5)])
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
    /// Effect strengths: `x` is `post.ao`, `y` `post.godRays`, `z` `post.bloom`;
    /// `w` is `post.bloomThreshold` with bloom on, 0 otherwise.
    pub strength: [f32; 4],
    /// The sun on the raster ([`sun_screen`]): `u`, `v`, 1, 0; all 0 without a visible sun.
    pub sun_screen: [f32; 4],
}

impl PostConstants {
    #[must_use]
    pub fn new(
        boundary: &PostBoundary,
        cfg: PostConfig,
        width: u32,
        height: u32,
        has_depth: bool,
    ) -> Self {
        let [r, g, b] = boundary.sun_color;
        let sun_dir = boundary
            .sun_dir_eye
            .map_or([0.0; 4], |[x, y, z]| [x, y, z, 1.0]);
        let sun_screen = sun_screen(boundary).map_or([0.0; 4], |[u, v]| [u, v, 1.0, 0.0]);
        Self {
            proj: boundary.proj,
            sun_dir,
            sun_color: [r, g, b, 0.0],
            extent: [width, height, u32::from(has_depth), 0],
            strength: [
                f32::from(cfg.ao_pct) / 100.0,
                f32::from(cfg.god_rays_pct) / 100.0,
                f32::from(cfg.bloom_pct) / 100.0,
                if cfg.bloom_pct > 0 {
                    f32::from(cfg.bloom_threshold_pct) / 100.0
                } else {
                    0.0
                },
            ],
            sun_screen,
        }
    }
}

/// The depth snapshot at a raster pixel, clamped to its edges.
const DEPTH_READ: &str = r"
static float depth_at(depth2d<float, access::read> depth, int2 p) {
    int2 last = int2(depth.get_width(), depth.get_height()) - 1;
    return depth.read(uint2(clamp(p, int2(0), last)));
}
";

/// What every AO kernel shares: its constants, depth reads, and the eye-space
/// position of a raster pixel.
///
/// The game renders with standard (not reversed) D3D depth into a `D3DFMT_D32`
/// buffer (Metal `Depth32Float`), with a left-handed perspective projection
/// (near 10, far 16000). There `d = _33 + _43 / z` for eye depth `z`, with
/// `_33 = f / (f − n)` and `_43 = −n·f / (f − n)`, so `z = _43 / (d − _33)`.
/// Eye `x` and `y` come back through `_11` and `_22`. A depth of 1 or more is
/// the far plane or sky, which occludes nothing and is never darkened.
const AO_COMMON: &str = r"
// The search radius, in world units.
constant float AO_RADIUS = 64.0;
constant int AO_DIRECTIONS = 8;
constant int AO_STEPS = 4;
// Elevations under this sine count as the surface itself, not an occluder.
constant float AO_ANGLE_BIAS = 0.1;
// The search radius in pixels stays under this share of the raster height, so
// a surface at the near plane does not search across the screen.
constant float AO_MAX_RADIUS_SHARE = 0.125;
// The blur's depth tolerance, relative to the centre's eye depth.
constant float AO_BLUR_TOLERANCE = 0.1;

static float eye_depth(float d, constant PostConstants& k) {
    return k.proj.w / (d - k.proj.z);
}

// The eye-space position of raster pixel p's centre at depth d.
static float3 eye_position(int2 p, float d, constant PostConstants& k) {
    float z = eye_depth(d, k);
    float2 ndc = float2((float(p.x) + 0.5) / float(k.extent.x) * 2.0 - 1.0,
                        1.0 - (float(p.y) + 0.5) / float(k.extent.y) * 2.0);
    return float3(ndc.x * z / k.proj.x, ndc.y * z / k.proj.y, z);
}

// The step from c to its neighbour along `axis`, on the side with the smaller
// depth change, so a silhouette does not bend the normal. A neighbour on the
// far plane is sky, not surface, and is never taken.
static float3 surface_step(depth2d<float, access::read> depth, int2 p, int2 axis, float3 c,
                           constant PostConstants& k) {
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

// The surface normal at c from the depth derivatives, facing the eye.
static float3 surface_normal(depth2d<float, access::read> depth, int2 p, float3 c,
                             constant PostConstants& k) {
    float3 n = cross(surface_step(depth, p, int2(1, 0), c, k),
                     surface_step(depth, p, int2(0, 1), c, k));
    if (dot(n, n) < 1.0e-12) return normalize(-c);
    n = normalize(n);
    return dot(n, c) > 0.0 ? -n : n;
}
";

/// AO's horizon search, one half-extent texel per thread, read at the top-left
/// raster pixel of its 2×2 block. Each of 8 directions walks 4 steps out to
/// the world radius projected to pixels and keeps the highest elevation seen
/// above the tangent plane; each rise adds its sine, weighted by distance
/// falloff. A 2×2 interleave rotates the directions and offsets the steps,
/// which the blur's 2×2 footprint averages out.
const AO_HORIZON: &str = r"
kernel void post_ao_horizon(texture2d<float, access::write> dst [[texture(1)]],
                            depth2d<float, access::read> depth [[texture(2)]],
                            constant PostConstants& k [[buffer(0)]],
                            uint2 q [[thread_position_in_grid]]) {
    if (q.x >= dst.get_width() || q.y >= dst.get_height()) return;
    int2 p = min(int2(q) * 2, int2(k.extent.xy) - 1);
    float d = depth_at(depth, p);
    if (d >= 1.0) { dst.write(float4(1.0), q); return; }
    float3 c = eye_position(p, d, k);
    if (!(c.z > 0.0)) { dst.write(float4(1.0), q); return; }
    float radius_px = min(AO_RADIUS * k.proj.x * 0.5 * float(k.extent.x) / c.z,
                          AO_MAX_RADIUS_SHARE * float(k.extent.y));
    if (radius_px < 1.0) { dst.write(float4(1.0), q); return; }
    float3 n = surface_normal(depth, p, c, k);
    uint cell = (q.x & 1u) | ((q.y & 1u) << 1);
    float rotation = float(cell) * 0.25;
    float jitter = float((cell * 3u + 1u) & 3u) * 0.25 + 0.25;
    float step_px = radius_px / float(AO_STEPS);
    float r2 = AO_RADIUS * AO_RADIUS;
    float ao = 0.0;
    for (int i = 0; i < AO_DIRECTIONS; ++i) {
        float angle = (float(i) + rotation) * (2.0 * M_PI_F / float(AO_DIRECTIONS));
        float2 dir = float2(cos(angle), sin(angle));
        float horizon = AO_ANGLE_BIAS;
        for (int j = 0; j < AO_STEPS; ++j) {
            float t = max((float(j) + jitter) * step_px, 1.0);
            int2 s = p + int2(round(dir * t));
            if (any(s < int2(0)) || any(s >= int2(k.extent.xy))) break;
            float ds = depth_at(depth, s);
            if (ds >= 1.0) continue;
            float3 v = eye_position(s, ds, k) - c;
            float l2 = dot(v, v);
            if (l2 >= r2 || l2 < 1.0e-4) continue;
            float elevation = dot(v, n) * rsqrt(l2);
            if (elevation > horizon) {
                ao += (elevation - horizon) * (1.0 - l2 / r2);
                horizon = elevation;
            }
        }
    }
    float occlusion = 1.0 - ao / (float(AO_DIRECTIONS) * (1.0 - AO_ANGLE_BIAS));
    dst.write(float4(saturate(occlusion)), q);
}
";

/// AO's blur: the 2×2 block of half-extent texels from this one, each weighted
/// by how close its eye depth is to the centre's, so occlusion does not bleed
/// across a silhouette. The far plane stays 1.
const AO_BLUR: &str = r"
kernel void post_ao_blur(texture2d<float, access::read> src [[texture(0)]],
                         texture2d<float, access::write> dst [[texture(1)]],
                         depth2d<float, access::read> depth [[texture(2)]],
                         constant PostConstants& k [[buffer(0)]],
                         uint2 q [[thread_position_in_grid]]) {
    if (q.x >= dst.get_width() || q.y >= dst.get_height()) return;
    int2 last_pixel = int2(k.extent.xy) - 1;
    float dc = depth_at(depth, min(int2(q) * 2, last_pixel));
    if (dc >= 1.0) { dst.write(float4(1.0), q); return; }
    float zc = eye_depth(dc, k);
    uint2 last = uint2(src.get_width(), src.get_height()) - 1u;
    float sum = 0.0;
    float weight = 0.0;
    for (uint i = 0; i < 4u; ++i) {
        uint2 t = min(q + uint2(i & 1u, i >> 1), last);
        float w = 1.0;
        if (i != 0u) {
            float dt = depth_at(depth, min(int2(t) * 2, last_pixel));
            float r = dt >= 1.0 ? 1.0e4 : (eye_depth(dt, k) - zc) / (AO_BLUR_TOLERANCE * zc);
            w = exp2(-r * r);
        }
        sum += src.read(t).r * w;
        weight += w;
    }
    dst.write(float4(sum / weight), q);
}
";

/// AO into the colour: the blurred occlusion, sampled bilinearly at this
/// pixel, scales rgb by `1 − strength·(1 − occlusion)` (never below 0); the
/// far plane and alpha are kept.
const AO_APPLY: &str = r"
kernel void post_ao_apply(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          depth2d<float, access::read> depth [[texture(2)]],
                          texture2d<float, access::sample> ao [[texture(3)]],
                          constant PostConstants& k [[buffer(0)]],
                          uint2 p [[thread_position_in_grid]]) {
    if (p.x >= k.extent.x || p.y >= k.extent.y) return;
    float4 c = src.read(p);
    float occlusion = 1.0;
    if (depth_at(depth, int2(p)) < 1.0) {
        constexpr sampler bilinear(coord::normalized, filter::linear, address::clamp_to_edge);
        occlusion = ao.sample(bilinear, (float2(p) + 0.5) / float2(k.extent.xy)).r;
    }
    float shade = max(1.0 - k.strength.x * (1.0 - occlusion), 0.0);
    dst.write(float4(c.rgb * shade, c.a), p);
}
";

/// God rays' mask, one half-extent texel per thread: the mean of its 2×2
/// raster block's colour, where only pixels at the far plane (the sky, the
/// one light source) count and geometry adds black.
const GOD_RAYS_MASK: &str = r"
kernel void post_god_rays_mask(texture2d<float, access::read> src [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]],
                               depth2d<float, access::read> depth [[texture(2)]],
                               constant PostConstants& k [[buffer(0)]],
                               uint2 q [[thread_position_in_grid]]) {
    if (q.x >= dst.get_width() || q.y >= dst.get_height()) return;
    uint2 last = k.extent.xy - 1u;
    float3 sky = float3(0.0);
    for (uint i = 0; i < 4u; ++i) {
        uint2 p = min(q * 2u + uint2(i & 1u, i >> 1), last);
        if (depth_at(depth, int2(p)) >= 1.0) sky += src.read(p).rgb;
    }
    dst.write(float4(sky * 0.25, 1.0), q);
}
";

/// God rays' radial blur: 32 bilinear samples of the mask on the way from
/// this texel toward the sun, each weaker than the one before. Samples off
/// the raster read black, so the screen's edge does not smear into shafts.
const GOD_RAYS_BLUR: &str = r"
constant int GOD_RAYS_SAMPLES = 32;
// The share of the way to the sun the samples cover.
constant float GOD_RAYS_REACH = 0.9;
// Each sample's weight relative to the one before.
constant float GOD_RAYS_DECAY = 0.96;

kernel void post_god_rays_blur(texture2d<float, access::sample> src [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]],
                               constant PostConstants& k [[buffer(0)]],
                               uint2 q [[thread_position_in_grid]]) {
    if (q.x >= dst.get_width() || q.y >= dst.get_height()) return;
    constexpr sampler bilinear(coord::normalized, filter::linear, address::clamp_to_zero);
    float2 uv = (float2(q) + 0.5) / float2(dst.get_width(), dst.get_height());
    float2 step = (k.sun_screen.xy - uv) * (GOD_RAYS_REACH / float(GOD_RAYS_SAMPLES));
    float3 sum = float3(0.0);
    float weight = 1.0;
    for (int i = 0; i < GOD_RAYS_SAMPLES; ++i) {
        sum += src.sample(bilinear, uv).rgb * weight;
        weight *= GOD_RAYS_DECAY;
        uv += step;
    }
    dst.write(float4(sum / float(GOD_RAYS_SAMPLES), 1.0), q);
}
";

/// God rays into the colour: the rays, sampled bilinearly at this pixel and
/// tinted by the sun's colour, add `strength` of themselves to rgb; alpha is kept.
const GOD_RAYS_APPLY: &str = r"
kernel void post_god_rays_apply(texture2d<float, access::read> src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                texture2d<float, access::sample> rays [[texture(3)]],
                                constant PostConstants& k [[buffer(0)]],
                                uint2 p [[thread_position_in_grid]]) {
    if (p.x >= k.extent.x || p.y >= k.extent.y) return;
    constexpr sampler bilinear(coord::normalized, filter::linear, address::clamp_to_edge);
    float4 c = src.read(p);
    float3 light = rays.sample(bilinear, (float2(p) + 0.5) / float2(k.extent.xy)).rgb;
    dst.write(float4(c.rgb + light * k.sun_color.rgb * k.strength.y, c.a), p);
}
";

/// Bloom's threshold, one half-extent texel per thread: each pixel of its
/// 2×2 raster block keeps `colour · w`, then the four are averaged.
///
/// `w` is a quadratic soft knee on the brightest channel `b` that starts at
/// the threshold `t` and spans `2·BLOOM_KNEE` above it: with
/// `r = clamp(b − t, 0, 2K)`, the kept brightness is
/// `max(r² / 4K, b − t − K)`, which is 0 at and below `t`, eases in
/// quadratically, and meets the line `b − t − K` with slope 1 at `t + 2K`.
/// Starting the knee at the threshold (rather than centring it there) keeps
/// everything at or below the threshold out of the glow. Nothing is clamped
/// above: a float scene's values above 1 pass through.
const BLOOM_THRESHOLD: &str = r"
// Half the knee's width above the threshold.
constant float BLOOM_KNEE = 0.1;

kernel void post_bloom_threshold(texture2d<float, access::read> src [[texture(0)]],
                                 texture2d<float, access::write> dst [[texture(1)]],
                                 constant PostConstants& k [[buffer(0)]],
                                 uint2 q [[thread_position_in_grid]]) {
    if (q.x >= dst.get_width() || q.y >= dst.get_height()) return;
    uint2 last = k.extent.xy - 1u;
    float t = k.strength.w;
    float3 sum = float3(0.0);
    for (uint i = 0; i < 4u; ++i) {
        float3 c = src.read(min(q * 2u + uint2(i & 1u, i >> 1), last)).rgb;
        float b = max(c.r, max(c.g, c.b));
        float r = clamp(b - t, 0.0, 2.0 * BLOOM_KNEE);
        float kept = max(r * r / (4.0 * BLOOM_KNEE), b - t - BLOOM_KNEE);
        sum += c * (kept / max(b, 1.0e-4));
    }
    dst.write(float4(sum * 0.25, 1.0), q);
}
";

/// Bloom's downsample to the next level: a bilinear sample at the texel's
/// centre (its 2×2 source block) weighted ½, and four at the diagonal corners
/// one source texel out weighted ⅛ each, a smooth 4×4 footprint.
const BLOOM_DOWNSAMPLE: &str = r"
kernel void post_bloom_downsample(texture2d<float, access::sample> src [[texture(0)]],
                                  texture2d<float, access::write> dst [[texture(1)]],
                                  constant PostConstants& k [[buffer(0)]],
                                  uint2 q [[thread_position_in_grid]]) {
    if (q.x >= dst.get_width() || q.y >= dst.get_height()) return;
    constexpr sampler bilinear(coord::normalized, filter::linear, address::clamp_to_edge);
    float2 uv = (float2(q) + 0.5) / float2(dst.get_width(), dst.get_height());
    float2 o = 1.0 / float2(src.get_width(), src.get_height());
    float3 c = src.sample(bilinear, uv).rgb * 0.5;
    c += src.sample(bilinear, uv + float2(-o.x, -o.y)).rgb * 0.125;
    c += src.sample(bilinear, uv + float2(o.x, -o.y)).rgb * 0.125;
    c += src.sample(bilinear, uv + float2(-o.x, o.y)).rgb * 0.125;
    c += src.sample(bilinear, uv + float2(o.x, o.y)).rgb * 0.125;
    dst.write(float4(c, 1.0), q);
}
";

/// Bloom's upsample: this level plus the coarser one at `texture(3)`, read
/// through a 3×3 tent (weights 1 2 1 / 2 4 2 / 1 2 1 over 16) of bilinear
/// samples one texel of this level apart.
const BLOOM_UPSAMPLE: &str = r"
kernel void post_bloom_upsample(texture2d<float, access::read> src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                texture2d<float, access::sample> coarse [[texture(3)]],
                                constant PostConstants& k [[buffer(0)]],
                                uint2 q [[thread_position_in_grid]]) {
    if (q.x >= dst.get_width() || q.y >= dst.get_height()) return;
    constexpr sampler bilinear(coord::normalized, filter::linear, address::clamp_to_edge);
    float2 size = float2(dst.get_width(), dst.get_height());
    float2 uv = (float2(q) + 0.5) / size;
    float2 o = 1.0 / size;
    float3 c = float3(0.0);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float w = float((2 - abs(x)) * (2 - abs(y)));
            c += coarse.sample(bilinear, uv + float2(x, y) * o).rgb * w;
        }
    }
    dst.write(float4(src.read(q).rgb + c * (1.0 / 16.0), 1.0), q);
}
";

/// Bloom into the colour: the summed levels, sampled bilinearly at this
/// pixel and scaled by `strength`, add to rgb (the five levels summed, not
/// averaged: the knee keeps little of a display-range colour); alpha is
/// kept. No clamp: a UNORM target clamps on store, a float one keeps values
/// above 1.
const BLOOM_APPLY: &str = r"
kernel void post_bloom_apply(texture2d<float, access::read> src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             texture2d<float, access::sample> glow [[texture(3)]],
                             constant PostConstants& k [[buffer(0)]],
                             uint2 p [[thread_position_in_grid]]) {
    if (p.x >= k.extent.x || p.y >= k.extent.y) return;
    constexpr sampler bilinear(coord::normalized, filter::linear, address::clamp_to_edge);
    float4 c = src.read(p);
    float3 g = glow.sample(bilinear, (float2(p) + 0.5) / float2(k.extent.xy)).rgb;
    dst.write(float4(c.rgb + g * k.strength.z, c.a), p);
}
";

/// The MSL a kernel compiles with: the constants row and its function.
#[must_use]
pub fn kernel_source(kernel: Kernel) -> String {
    let (depth, common, body) = match kernel {
        Kernel::Invert => (
            "",
            "",
            r"
kernel void post_invert(texture2d<float, access::read> src [[texture(0)]],
                        texture2d<float, access::write> dst [[texture(1)]],
                        constant PostConstants& k [[buffer(0)]],
                        uint2 p [[thread_position_in_grid]]) {
    if (p.x >= k.extent.x || p.y >= k.extent.y) return;
    float4 c = src.read(p);
    dst.write(float4(1.0 - c.rgb, c.a), p);
}
",
        ),
        Kernel::AoHorizon => (DEPTH_READ, AO_COMMON, AO_HORIZON),
        Kernel::AoBlur => (DEPTH_READ, AO_COMMON, AO_BLUR),
        Kernel::AoApply => (DEPTH_READ, AO_COMMON, AO_APPLY),
        Kernel::GodRaysMask => (DEPTH_READ, "", GOD_RAYS_MASK),
        Kernel::GodRaysBlur => ("", "", GOD_RAYS_BLUR),
        Kernel::GodRaysApply => ("", "", GOD_RAYS_APPLY),
        Kernel::BloomThreshold => ("", "", BLOOM_THRESHOLD),
        Kernel::BloomDownsample => ("", "", BLOOM_DOWNSAMPLE),
        Kernel::BloomUpsample => ("", "", BLOOM_UPSAMPLE),
        Kernel::BloomApply => ("", "", BLOOM_APPLY),
    };
    format!(
        r"
#include <metal_stdlib>
using namespace metal;
struct PostConstants {{ float4 proj; float4 sun_dir; float4 sun_color; uint4 extent; float4 strength; float4 sun_screen; }};
{depth}{common}{body}"
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
    let aux = match (kernel.reads_aux(), bound_texture(command.bytes_per_row)) {
        (false, _) => None,
        (true, Some(handle)) => {
            let Some(aux) = handle.into_retained() else {
                log::error!(target: LOG_TARGET, "post {kernel:?}: missing second input");
                return false;
            };
            Some(aux)
        }
        (true, None) => {
            log::error!(target: LOG_TARGET, "post {kernel:?}: no second input bound");
            return false;
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
