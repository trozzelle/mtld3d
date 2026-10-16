//! Per-pixel fixed-function lighting (experiment).
//!
//! Picks, per draw, which active lights move from the FF vertex shader's
//! light loop to the pixel stage. Everything else stays per vertex.

use mtld3d_types::D3DMATRIX;

use crate::dxso::{FfVsFlags, FfVsKey};

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
/// `lighting.hemiAmbient`: the hemispherical ambient term, `off` by default.
///
/// `SKY,GROUND` are decimals in `[0, 2]`, stored as hundredths so the wire
/// and the shader constants carry exact `u16`s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HemiAmbient {
    /// Sky colour weight, in hundredths (`1.25` → `125`).
    pub sky_pct: u16,
    /// Ground colour weight, in hundredths (`0.6` → `60`).
    pub ground_pct: u16,
}

/// World +Z expressed in the eye space the FF VS normal lives in.
///
/// `D3DMATRIX` is row-vector convention (`v_eye = v_world · View`), so world
/// +Z maps through the view's third row `(m[8], m[9], m[10])`; normalized,
/// or `(0, 0, 1)` when that row is zero (a singular view keeps the default
/// up axis).
#[must_use]
pub fn hemi_up_eye(view: &D3DMATRIX) -> [f32; 3] {
    let row = [view.m[8], view.m[9], view.m[10]];
    let len = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
    if len == 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        [row[0] / len, row[1] / len, row[2] / len]
    }
}

/// Whether a draw's FF vertex shader applies the hemispherical ambient term.
///
/// Only lit draws with a vertex normal and an untransformed position have an
/// eye-space normal to weight the term with; everything else keeps the stock
/// global ambient. Independent of `lighting.perPixel`: the term lives in the
/// fixed-function vertex shader's ambient line either way.
#[must_use]
pub const fn hemi_flag(cfg: Option<HemiAmbient>, vs: &FfVsKey) -> bool {
    cfg.is_some()
        && vs.flags.contains(FfVsFlags::LIGHTING_ENABLED)
        && vs.flags.contains(FfVsFlags::HAS_NORMAL)
        && !vs.flags.contains(FfVsFlags::HAS_RHW)
}

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
        D3DMATRIX, HemiAmbient, PplConfig, PplFalloff, PplMasks, PplMode, PplModel, PsPairing,
        hemi_flag, hemi_up_eye, model_bits, ppl_masks,
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

    #[test]
    fn hemi_up_eye_identity() {
        assert_eq!(hemi_up_eye(&D3DMATRIX::IDENTITY), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn hemi_up_eye_level_camera() {
        // A view whose third row maps world Z to eye Y: the up axis follows.
        let mut view = D3DMATRIX::IDENTITY;
        view.m[8] = 0.0;
        view.m[9] = 1.0;
        view.m[10] = 0.0;
        assert_eq!(hemi_up_eye(&view), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn hemi_flag_only_on_lit_normal_draws() {
        let hemi = Some(HemiAmbient {
            sky_pct: 125,
            ground_pct: 60,
        });
        assert!(hemi_flag(hemi, &lit()));
        assert!(!hemi_flag(None, &lit()));
        for flags in [
            FfVsFlags::HAS_NORMAL | FfVsFlags::COLOR_VERTEX,
            FfVsFlags::COLOR_VERTEX | FfVsFlags::LIGHTING_ENABLED,
            FfVsFlags::HAS_NORMAL | FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_RHW,
        ] {
            assert!(!hemi_flag(hemi, &key(flags)), "{flags:?}");
        }
    }
}
