//! Per-pixel fixed-function lighting (experiment).
//!
//! Picks, per draw, which active lights move from the FF vertex shader's
//! light loop to the pixel stage. Everything else stays per vertex.

pub mod figures;

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

/// The `lighting.perPixel*` and `lighting.specular*` keys of `mtld3d.conf`;
/// all off (or neutral) by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PplConfig {
    pub mode: PplMode,
    /// Also light FF-VS draws paired with a programmable `ps_1_x`/`ps_2_x` (`lighting.perPixelMixed`).
    pub mixed: bool,
    pub model: PplModel,
    pub falloff: PplFalloff,
    /// Output only the per-pixel light sum (`lighting.perPixelDebug = term`).
    pub debug_term: bool,
    /// Soft shoulder on the per-pixel sum, in hundredths (`0.8` → `80`;
    /// `lighting.perPixelShoulder`). `0` = off: the stock `saturate` clamp.
    pub shoulder_pct: u8,
    /// Also light DAoC's skinned figure VS per pixel (`lighting.perPixelFigures`);
    /// has an effect only when `mode` is `points` or `all`.
    pub figures: bool,
    /// Per-pixel Blinn-Phong specular term (`lighting.perPixelSpecular`).
    pub specular: bool,
    /// Specular term scale, in hundredths (`lighting.specularScale`; `1.0` → `100`).
    pub specular_scale_pct: u16,
    /// Specular power override (`lighting.specularPower`); `None` = `auto`.
    pub specular_power: Option<u8>,
    /// Fallback material specular RGB, in hundredths, used when the material's
    /// is zero (`lighting.specularDefault`); `None` = unset.
    pub specular_default_pct: Option<[u16; 3]>,
}

impl Default for PplConfig {
    fn default() -> Self {
        Self {
            mode: PplMode::Off,
            mixed: false,
            model: PplModel::D3d9,
            falloff: PplFalloff::D3d9,
            debug_term: false,
            shoulder_pct: 0,
            figures: false,
            specular: false,
            specular_scale_pct: 100,
            specular_power: None,
            specular_default_pct: None,
        }
    }
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
pub const PPL_MODEL_SPECULAR: u8 = 1 << 3;
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

/// World −Z ("up" in DAoC) expressed in the eye space the FF VS normal lives
/// in.
///
/// DAoC's D3D world is Z-down: the noon sun's world `Direction` carries +Z
/// (light travels toward +Z, so the sky sits at −Z), and the camera `VIEW`
/// (determinant −1) maps world +Z to screen-down. `D3DMATRIX` is row-vector
/// convention (`v_eye = v_world · View`), so world −Z maps through the
/// negated view third row `-(m[8], m[9], m[10])`; normalized, or `(0, 0, -1)`
/// when that row is zero (a singular view keeps the default up axis).
#[must_use]
pub fn hemi_up_eye(view: &D3DMATRIX) -> [f32; 3] {
    let row = [0.0 - view.m[8], 0.0 - view.m[9], 0.0 - view.m[10]];
    let len = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
    if len == 0.0 {
        [0.0, 0.0, -1.0]
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

/// Per-pixel lighting for DAoC's skinned figure VS: the draw's programmable
/// vertex shader is one of the figure shaders in [`figures::figure_vs`] and
/// `lighting.perPixelFigures` opted them in. Requires a non-`Off` mode — the
/// figure terms replace the stock per-vertex lights, so they only make sense
/// on a per-pixel draw.
#[must_use]
pub fn figure_flag(cfg: &PplConfig, vs_id: u64) -> bool {
    !matches!(cfg.mode, PplMode::Off) && cfg.figures && figures::figure_vs(vs_id).is_some()
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
    if cfg.specular {
        bits |= PPL_MODEL_SPECULAR;
    }
    bits
}

/// The per-pixel fields of a fixed-function VS key: the light mask, the
/// `PPL_MODEL_*` bits and the shoulder knee, in hundredths.
///
/// The knee only shapes the per-pixel sum, so it is non-zero only when the
/// mask is; a stock per-vertex draw keeps the key's identity.
#[must_use]
pub const fn ppl_key_fields(cfg: &PplConfig, masks: PplMasks) -> (u8, u8, u8) {
    let mask = masks.lights;
    if mask == 0 {
        return (0, 0, 0);
    }
    (mask, model_bits(cfg), cfg.shoulder_pct)
}

/// Size of the per-pixel specular uniform in bytes: two `float4` rows.
pub const PPL_PARAMS_BYTES: usize = 32;

/// The per-pixel specular uniform (`sp`) a specular pixel stage reads at
/// `PS_PPL_PARAMS_SLOT`; built once per config.
///
/// `sp[0] = (scale, power, 0, 0)`: `lighting.specularScale` and the
/// `lighting.specularPower` number, `0` for `auto` (the material power).
/// `sp[1] = (r, g, b, 1)`: `lighting.specularDefault`, the colour used for a
/// material whose specular is black; all zero when unset.
#[must_use]
pub fn build_ppl_params_bytes(cfg: &PplConfig) -> [u8; PPL_PARAMS_BYTES] {
    let pct = |v: u16| f32::from(v) / 100.0;
    let power = cfg.specular_power.map_or(0.0, f32::from);
    let [r, g, b, use_default] = cfg
        .specular_default_pct
        .map_or([0.0; 4], |[r, g, b]| [pct(r), pct(g), pct(b), 1.0]);
    let lanes = [
        pct(cfg.specular_scale_pct),
        power,
        0.0,
        0.0,
        r,
        g,
        b,
        use_default,
    ];
    let mut out = [0u8; PPL_PARAMS_BYTES];
    for (chunk, lane) in out.as_chunks_mut::<4>().0.iter_mut().zip(lanes) {
        *chunk = lane.to_le_bytes();
    }
    out
}

/// Size of a figure draw's specular uniform in bytes: the two config rows
/// plus the eye row.
pub const PPL_FIGURE_PARAMS_BYTES: usize = PPL_PARAMS_BYTES + 16;

/// The specular uniform of a figure draw (`fig_light`): `params` (the
/// config's two rows from [`build_ppl_params_bytes`]) followed by
/// `sp[2]`, [`figures::figure_eye_row`] of the draw's VS float constants.
///
/// `vs_constants` is the draw's bound VS float-constant bytes; rows past its
/// end read as zero (no camera, so the fallback direction).
#[must_use]
pub fn build_figure_ppl_params_bytes(
    params: &[u8; PPL_PARAMS_BYTES],
    vs_constants: &[u8],
) -> [u8; PPL_FIGURE_PARAMS_BYTES] {
    let mut rows = [[0.0f32; 4]; 7];
    for (lane, chunk) in rows
        .as_flattened_mut()
        .iter_mut()
        .zip(vs_constants.as_chunks::<4>().0)
    {
        *lane = f32::from_le_bytes(*chunk);
    }
    let mut out = [0u8; PPL_FIGURE_PARAMS_BYTES];
    out[..PPL_PARAMS_BYTES].copy_from_slice(params);
    for (chunk, lane) in out[PPL_PARAMS_BYTES..]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(figures::figure_eye_row(&rows))
    {
        *chunk = lane.to_le_bytes();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        D3DMATRIX, HemiAmbient, PPL_MODEL_SPECULAR, PplConfig, PplFalloff, PplMasks, PplMode,
        PplModel, PsPairing, hemi_flag, hemi_up_eye, model_bits, ppl_key_fields, ppl_masks,
    };
    use crate::dxso::{FfVsFlags, FfVsKey};

    fn key(flags: FfVsFlags) -> FfVsKey {
        FfVsKey {
            ppl_knee: 0,
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
        let s = PplConfig {
            specular: true,
            ..PplConfig::default()
        };
        assert_eq!(model_bits(&s), PPL_MODEL_SPECULAR);
        assert_eq!(
            model_bits(&PplConfig {
                specular: true,
                ..c
            }),
            0b1111
        );
    }
    #[test]
    fn hemi_up_eye_identity() {
        assert_eq!(hemi_up_eye(&D3DMATRIX::IDENTITY), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn ppl_knee_set_only_with_mask() {
        let cfg = PplConfig {
            mode: PplMode::All,
            shoulder_pct: 80,
            ..PplConfig::default()
        };
        let masks = ppl_masks(&cfg, &lit(), PsPairing::FixedFunction);
        assert_eq!(ppl_key_fields(&cfg, masks), (0b0111, 0, 80));
        // A stock per-vertex draw keeps the knee at zero whatever the config.
        assert_eq!(
            ppl_key_fields(&cfg, PplMasks::default()),
            (0, 0, 0),
            "the knee is only meaningful with per-pixel lighting on"
        );
    }

    #[test]
    fn hemi_up_eye_level_camera() {
        // A view whose third row maps world Z to eye Y: world −Z (DAoC's up)
        // lands on eye −Y.
        let mut view = D3DMATRIX::IDENTITY;
        view.m[8] = 0.0;
        view.m[9] = 1.0;
        view.m[10] = 0.0;
        assert_eq!(hemi_up_eye(&view), [0.0, -1.0, 0.0]);
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

    #[test]
    fn figure_flag_needs_mode_key_and_table() {
        let figure_id = crate::ppl::figures::FIGURE_VS[0].id;
        let on = PplConfig {
            mode: PplMode::All,
            figures: true,
            ..PplConfig::default()
        };
        assert!(super::figure_flag(&on, figure_id));
        // The mode gate: `Off` disables the figure path even with the key on.
        assert!(!super::figure_flag(
            &PplConfig {
                mode: PplMode::Off,
                ..on
            },
            figure_id
        ));
        // The key gate: without `lighting.perPixelFigures` no draw is flagged.
        assert!(!super::figure_flag(
            &PplConfig {
                figures: false,
                ..on
            },
            figure_id
        ));
        // The table gate is covered by `unknown_vs_gets_no_figure_fields`.
    }

    #[test]
    fn unknown_vs_gets_no_figure_fields() {
        let on = PplConfig {
            mode: PplMode::Points,
            figures: true,
            ..PplConfig::default()
        };
        assert!(!super::figure_flag(&on, 0));
        assert!(!super::figure_flag(&on, u64::from(u32::MAX) + 1));
        for entry in crate::ppl::figures::FIGURE_VS {
            assert!(super::figure_flag(&on, entry.id));
        }
    }

    fn lanes(bytes: [u8; super::PPL_PARAMS_BYTES]) -> [f32; 8] {
        let mut out = [0.0; 8];
        for (lane, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *lane = f32::from_le_bytes(*chunk);
        }
        out
    }

    #[test]
    fn ppl_params_bytes_follow_the_config() {
        use super::build_ppl_params_bytes;
        // Defaults: scale 1.00, power `auto` (0), no fallback colour.
        assert_eq!(
            lanes(build_ppl_params_bytes(&PplConfig::default())),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        let set = PplConfig {
            specular: true,
            specular_scale_pct: 150,
            specular_power: Some(24),
            specular_default_pct: Some([20, 25, 30]),
            ..PplConfig::default()
        };
        assert_eq!(
            lanes(build_ppl_params_bytes(&set)),
            [1.5, 24.0, 0.0, 0.0, 0.2, 0.25, 0.3, 1.0]
        );
    }

    /// A figure draw's slot-6 bytes are the config's two rows followed by the
    /// eye row recovered from the draw's VS constants `c0..c6`.
    #[test]
    fn figure_ppl_params_bytes_carry_the_eye_row() {
        use super::{
            PPL_FIGURE_PARAMS_BYTES, PPL_PARAMS_BYTES, build_figure_ppl_params_bytes,
            build_ppl_params_bytes, figures::figure_eye_row,
        };
        let cfg = PplConfig {
            specular: true,
            specular_power: Some(24),
            specular_default_pct: Some([20, 25, 30]),
            ..PplConfig::default()
        };
        let params = build_ppl_params_bytes(&cfg);
        // A camera at (2, 3, 5): clip x = X - 2, y = Y - 3, w = Z - 5 (its
        // null vector is the camera centre); light space = world + (10, 20, 30).
        let rows: [[f32; 4]; 7] = [
            [1.0, 0.0, 0.0, -2.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, -5.0],
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
        ];
        // The draw binds every row the VS reads; the builder reads c0..c6.
        let mut vs_constants = vec![0u8; 16 * 40];
        for (chunk, lane) in vs_constants
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(rows.iter().flatten())
        {
            *chunk = lane.to_le_bytes();
        }
        let bytes = build_figure_ppl_params_bytes(&params, &vs_constants);
        assert_eq!(bytes.len(), PPL_FIGURE_PARAMS_BYTES);
        assert_eq!(bytes[..PPL_PARAMS_BYTES], params);
        let mut eye = [0.0f32; 4];
        for (lane, chunk) in eye
            .iter_mut()
            .zip(bytes[PPL_PARAMS_BYTES..].as_chunks::<4>().0)
        {
            *lane = f32::from_le_bytes(*chunk);
        }
        assert_eq!(eye, figure_eye_row(&rows));
        assert_eq!(eye, [12.0, 23.0, 35.0, 1.0]);
        // A short constant buffer reads its missing rows as zero: no centre,
        // so the fallback direction.
        let short = build_figure_ppl_params_bytes(&params, &vs_constants[..16 * 3]);
        assert_eq!(short[..PPL_PARAMS_BYTES], params);
        let mut fallback = [0.0f32; 4];
        for (lane, chunk) in fallback
            .iter_mut()
            .zip(short[PPL_PARAMS_BYTES..].as_chunks::<4>().0)
        {
            *lane = f32::from_le_bytes(*chunk);
        }
        assert_eq!(fallback, super::figures::FIGURE_EYE_FALLBACK);
    }
}
