//! Fog overrides for DAoC draws whose stock fog is broken.
//!
//! DAoC's skinned figure VS compute their fog from `$fogParams`
//! (`c[10 + layout]`, holding `(start, end, end - start, 0.6 / end)`) as
//! exponential fog `exp(-0.6 d / end)`: a haze from distance 0 that terrain
//! (linear fog from `start`) does not have. `fog.figures` replaces it with
//! the world's linear fog on the clip-space `w` (see [`crate::ppl::figures::fog_params_register`]).
//!
//! `fog.curve` and `fog.desaturate` reshape every fogged draw's end-of-PS
//! blend. They reach the pixel shader as the [`VariantKey::fog_style`] byte,
//! which the device sets on the API side right after
//! [`crate::ff_state::FfState::variant_key`] (see [`fog_style`]); the
//! desaturation amount travels in a third `fog_data` row
//! ([`crate::ff_state::build_fog_color_bytes`]).

use core::fmt::Write as _;

use crate::{
    dxso::{VariantKey, fog_blend_active},
    ppl::figures,
};

/// Fog overrides. Default: all off. File keys: `fog.figures` (`off` | `on`),
/// `fog.curve` (`linear` | `exp` | `exp2`) and `fog.desaturate` (`off` | a
/// decimal in `[0, 1]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FogConfig {
    /// Linear fog on DAoC's figure VS (`fog.figures`); independent of the
    /// `lighting.perPixel*` keys.
    pub figures: bool,
    /// The remap of the linear fog factor (`fog.curve`).
    pub curve: FogCurve,
    /// How far fully fogged pixels move toward their luma (`fog.desaturate`),
    /// in hundredths (`0..=100`); zero is off.
    pub desaturate_pct: u8,
}

/// The shape `fog.curve` gives the fog factor `f` (1 = clear, 0 = fogged).
///
/// With `t = 1 - f` and `k = FOG_CURVE_K`: `exp` maps
/// `a = (1 - e^(-k t)) / (1 - e^(-k))`, `exp2` maps
/// `a = (1 - e^(-(k t)^2)) / (1 - e^(-k^2))`, and the remapped factor is
/// `1 - a`. Both keep `f = 0` and `f = 1` fixed and fog nearer pixels harder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FogCurve {
    /// The stock factor.
    #[default]
    Linear,
    /// Exponential ramp.
    Exp,
    /// Squared exponential ramp.
    Exp2,
}

/// `fog_style` bits 0-1: the [`FogCurve`] id (0 linear, 1 exp, 2 exp2).
pub const FOG_STYLE_CURVE_MASK: u8 = 0b011;
/// `fog_style` bit 2: desaturate toward luma by `fog_data[2].x`.
pub const FOG_STYLE_DESATURATE: u8 = 0b100;
/// The curve steepness `k`.
pub const FOG_CURVE_K: f32 = 3.0;
/// Rec. 709 luma weights, as the water glint uses.
const LUMA_MSL: &str = "float3(0.2126, 0.7152, 0.0722)";

impl FogCurve {
    /// The curve's id in `fog_style` bits 0-1.
    #[must_use]
    pub const fn style_bits(self) -> u8 {
        match self {
            Self::Linear => 0,
            Self::Exp => 1,
            Self::Exp2 => 2,
        }
    }

    /// The curve a `fog_style` byte selects; id 3 is unused and reads linear.
    #[must_use]
    pub const fn from_style(style: u8) -> Self {
        match style & FOG_STYLE_CURVE_MASK {
            1 => Self::Exp,
            2 => Self::Exp2,
            _ => Self::Linear,
        }
    }

    /// The `fog.curve` spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
        }
    }

    /// The normaliser `1 / (1 - e^(-k))` (exp) or `1 / (1 - e^(-k^2))`
    /// (exp2) that maps `t = 1` to `a = 1`; `None` for linear.
    fn normaliser(self) -> Option<f32> {
        let k = FOG_CURVE_K;
        match self {
            Self::Linear => None,
            Self::Exp => Some(1.0 / (1.0 - (-k).exp())),
            Self::Exp2 => Some(1.0 / (1.0 - (-k * k).exp())),
        }
    }
}

/// The `fog_style` byte for a draw: zero unless fog is active on `variant`
/// and a key is on, so stock draws keep every key, hash and record.
#[must_use]
pub fn fog_style(cfg: &FogConfig, variant: VariantKey) -> u8 {
    if !fog_blend_active(variant) {
        return 0;
    }
    let desaturate = if cfg.desaturate_pct == 0 {
        0
    } else {
        FOG_STYLE_DESATURATE
    };
    cfg.curve.style_bits() | desaturate
}

/// The `fog_data` length `variant` reads: 0 without fog, 32 bytes (colour and
/// params rows), 48 with the desaturation row.
#[must_use]
pub const fn fog_data_len(variant: VariantKey) -> usize {
    if !fog_blend_active(variant) {
        0
    } else if variant.fog_style & FOG_STYLE_DESATURATE != 0 {
        48
    } else {
        32
    }
}

/// The remapped fog factor, evaluated on the host exactly as
/// [`fog_remap_msl`] emits it (same `k` and normaliser).
#[must_use]
pub fn fog_remap(curve: FogCurve, f: f32) -> f32 {
    let Some(norm) = curve.normaliser() else {
        return f;
    };
    let kt = FOG_CURVE_K * (1.0 - f);
    let e = if curve == FogCurve::Exp2 {
        (-kt * kt).exp()
    } else {
        (-kt).exp()
    };
    (1.0 - (1.0 - e) * norm).clamp(0.0, 1.0)
}

/// Emit the in-place remap of the MSL local `fog_f`; nothing for linear.
///
/// `precise::exp` keeps the endpoints exact under the fast-math mode.
pub fn fog_remap_msl(out: &mut String, curve: FogCurve) {
    let Some(norm) = curve.normaliser() else {
        return;
    };
    let _ = writeln!(
        out,
        "    float fog_kt = {k:?} * (1.0 - fog_f);",
        k = FOG_CURVE_K
    );
    let exponent = if curve == FogCurve::Exp2 {
        "-fog_kt * fog_kt"
    } else {
        "-fog_kt"
    };
    let _ = writeln!(
        out,
        "    fog_f = saturate(1.0 - (1.0 - precise::exp({exponent})) * {norm:?});"
    );
}

/// Emit the fog mix of `target` toward `fog_data[0]` by the local `fog_f`,
/// remapped by the style's curve, and the optional desaturation toward luma
/// by `fog_data[2].x * (1 - fog_f)`.
pub(crate) fn write_styled_fog_mix(out: &mut String, style: u8, target: &str) {
    fog_remap_msl(out, FogCurve::from_style(style));
    let _ = writeln!(
        out,
        "    {target} = float4(mix(fog_data[0].rgb, {target}.rgb, fog_f), {target}.a);"
    );
    if style & FOG_STYLE_DESATURATE != 0 {
        let _ = writeln!(out, "    float fog_luma = dot({target}.rgb, {LUMA_MSL});");
        let _ = writeln!(
            out,
            "    {target} = float4(mix({target}.rgb, float3(fog_luma), fog_data[2].x * (1.0 - fog_f)), {target}.a);"
        );
    }
}

/// The draw's programmable VS is one of the figure shaders and `fog.figures`
/// opted them in.
#[must_use]
pub fn figure_fog_flag(cfg: &FogConfig, vs_id: u64) -> bool {
    cfg.figures && figures::figure_vs(vs_id).is_some()
}
