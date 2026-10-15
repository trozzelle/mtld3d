//! Per-pixel fixed-function lighting (experiment).
//!
//! Picks, per draw, which active lights move from the FF vertex shader's
//! light loop to the pixel stage. Everything else stays per vertex.

use crate::dxso::FfVsKey;

/// `lighting.perPixel`: which light types are evaluated per pixel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PplMode {
    #[default]
    Off,
    /// POINT lights only.
    Points,
    /// POINT and DIRECTIONAL lights.
    All,
}

/// `lighting.perPixelModel`: the per-pixel N·L term.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PplModel {
    #[default]
    D3d9,
    HalfLambert,
}

/// `lighting.perPixelFalloff`: the range cut of a point light.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PplFalloff {
    #[default]
    D3d9,
    Smooth,
}

/// The `lighting.perPixel*` keys of `mtld3d.conf`; all off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PplConfig {
    pub mode: PplMode,
    /// Also light FF-VS draws paired with a programmable `ps_1_x`/`ps_2_x` (`lighting.perPixelMixed`).
    pub mixed: bool,
    pub model: PplModel,
    pub falloff: PplFalloff,
    /// Output only the per-pixel light sum (`lighting.perPixelDebug = term`).
    pub debug_term: bool,
}

/// The pixel stage a draw's FF vertex shader is paired with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PsPairing {
    FixedFunction,
    Programmable { major: u8 },
}

/// Light slots evaluated per pixel for one draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PplMasks {
    /// Lights the pixel stage adds; the VS leaves their N·L diffuse out.
    pub lights: u8,
    /// The DIRECTIONAL subset of `lights`.
    pub directional: u8,
}

pub const PPL_MODEL_HALF_LAMBERT: u8 = 1 << 0;
pub const PPL_MODEL_SMOOTH_FALLOFF: u8 = 1 << 1;
pub const PPL_MODEL_DEBUG_TERM: u8 = 1 << 2;

/// Lights evaluated per pixel for this draw; zero means the stock per-vertex path.
#[must_use]
pub const fn ppl_masks(cfg: &PplConfig, vs: &FfVsKey, ps: PsPairing) -> PplMasks {
    if matches!(cfg.mode, PplMode::Off)
        || !vs.lighting_enabled()
        || !vs.has_normal()
        || vs.has_rhw()
    {
        return PplMasks {
            lights: 0,
            directional: 0,
        };
    }
    if let PsPairing::Programmable { major } = ps
        && (!cfg.mixed || major >= 3)
    {
        return PplMasks {
            lights: 0,
            directional: 0,
        };
    }
    let candidates = vs.light_active_mask & !vs.light_spot_mask;
    let lights = match cfg.mode {
        PplMode::Points => candidates & !vs.light_directional_mask,
        _ => candidates,
    };
    PplMasks {
        lights,
        directional: lights & vs.light_directional_mask,
    }
}

/// The `PPL_MODEL_*` bits for `cfg`.
#[must_use]
pub const fn model_bits(cfg: &PplConfig) -> u8 {
    let mut bits = 0;
    if matches!(cfg.model, PplModel::HalfLambert) {
        bits |= PPL_MODEL_HALF_LAMBERT;
    }
    if matches!(cfg.falloff, PplFalloff::Smooth) {
        bits |= PPL_MODEL_SMOOTH_FALLOFF;
    }
    if cfg.debug_term {
        bits |= PPL_MODEL_DEBUG_TERM;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::{
        PplConfig, PplFalloff, PplMasks, PplMode, PplModel, PsPairing, model_bits, ppl_masks,
    };
    use crate::dxso::{FfVsFlags, FfVsKey};

    fn key(flags: FfVsFlags) -> FfVsKey {
        FfVsKey {
            reserved: 0,
            ppl_mask: 0,
            ppl_model: 0,
            flags,
            input_tex_coord_count: 0,
            tex_coord_count: 0,
            light_active_mask: 0b1111,
            light_directional_mask: 0b0001,
            light_spot_mask: 0b1000,
            diffuse_source: 0,
            ambient_source: 0,
            specular_source: 0,
            emissive_source: 0,
            fog_mode: 0,
            tci: [0; 8],
            passthrough: [0; 8],
            tex_coord_dims: [0; 8],
            tt_flags: [0; 8],
            vertex_blend_count: 0,
            declared_weights_count: 0,
            clip_plane_count: 0,
        }
    }

    fn lit() -> FfVsKey {
        key(FfVsFlags::HAS_NORMAL | FfVsFlags::COLOR_VERTEX | FfVsFlags::LIGHTING_ENABLED)
    }

    fn cfg(mode: PplMode, mixed: bool) -> PplConfig {
        PplConfig {
            mode,
            mixed,
            ..PplConfig::default()
        }
    }

    #[test]
    fn off_gives_zero() {
        assert_eq!(
            ppl_masks(&PplConfig::default(), &lit(), PsPairing::FixedFunction),
            PplMasks::default()
        );
    }

    #[test]
    fn points_excludes_sun_and_spots() {
        let m = ppl_masks(
            &cfg(PplMode::Points, false),
            &lit(),
            PsPairing::FixedFunction,
        );
        assert_eq!(
            m,
            PplMasks {
                lights: 0b0110,
                directional: 0
            }
        );
    }

    #[test]
    fn all_includes_sun() {
        let m = ppl_masks(&cfg(PplMode::All, false), &lit(), PsPairing::FixedFunction);
        assert_eq!(
            m,
            PplMasks {
                lights: 0b0111,
                directional: 0b0001
            }
        );
    }

    #[test]
    fn unlit_normal_less_or_rhw_gives_zero() {
        let c = cfg(PplMode::All, true);
        for flags in [
            FfVsFlags::HAS_NORMAL | FfVsFlags::COLOR_VERTEX,
            FfVsFlags::COLOR_VERTEX | FfVsFlags::LIGHTING_ENABLED,
            FfVsFlags::HAS_NORMAL | FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_RHW,
        ] {
            assert_eq!(
                ppl_masks(&c, &key(flags), PsPairing::FixedFunction),
                PplMasks::default(),
                "{flags:?}"
            );
        }
    }

    #[test]
    fn mixed_requires_flag() {
        let ps = PsPairing::Programmable { major: 2 };
        assert_eq!(
            ppl_masks(&cfg(PplMode::Points, false), &lit(), ps),
            PplMasks::default()
        );
        assert_eq!(
            ppl_masks(&cfg(PplMode::Points, true), &lit(), ps).lights,
            0b0110
        );
    }

    #[test]
    fn sm3_ps_never() {
        let ps = PsPairing::Programmable { major: 3 };
        assert_eq!(
            ppl_masks(&cfg(PplMode::All, true), &lit(), ps),
            PplMasks::default()
        );
    }

    #[test]
    fn model_bits_pack() {
        let c = PplConfig {
            model: PplModel::HalfLambert,
            falloff: PplFalloff::Smooth,
            debug_term: true,
            ..PplConfig::default()
        };
        assert_eq!(model_bits(&c), 0b111);
        assert_eq!(model_bits(&PplConfig::default()), 0);
    }
}
