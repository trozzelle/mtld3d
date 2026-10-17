//! Fresnel opacity and a sun glint on DAoC's reflective water (`water.*`).
//!
//! The reflective water draw samples a projective reflection render target
//! through a bump-environment stage and blends it with a constant opacity.
//! `water.fresnel` replaces that constant with a Fresnel weight around it and
//! `water.glint` adds the sun's highlight off the rippled surface. Both run in
//! the fixed-function pixel stage after the texture cascade and before fog, on
//! the draws [`is_reflective_water`] matches. The vertex stage passes the eye
//! position and the eye-space images of world X, Y and up (world −Z, see
//! [`crate::ppl::hemi_up_eye`]); the pixel stage reads the sun and the
//! strengths from the [`build_water_bytes`] uniform.

use mtld3d_types::{
    D3DTA_CURRENT, D3DTA_DIFFUSE, D3DTA_TEXTURE, D3DTA_TFACTOR, D3DTOP_BUMPENVMAP, D3DTOP_MODULATE,
    D3DTTFF_COUNT3,
};

use crate::dxso::{FfPsKey, FfStage, FfVsKey, tt_count, tt_projected};

#[cfg(test)]
pub(crate) mod tests;

/// The `water.*` keys. Default: both off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaterConfig {
    /// `water.fresnel`: scale the water's opacity by a Fresnel weight.
    pub fresnel: bool,
    /// `water.glint`: the sun glint strength in hundredths (`0.5` → `50`),
    /// a decimal in `[0, 4]`; zero (`off`) adds no glint.
    pub glint_pct: u16,
}

impl WaterConfig {
    /// Whether either term is on, the gate on every water key bit and uniform.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.fresnel || self.glint_pct != 0
    }
}

/// `D3DTSS_TCI_CAMERASPACEPOSITION`, as the high nibble of an `FfVsKey::tci` entry.
const TCI_MODE_CAMERASPACEPOSITION: u8 = 2;

fn stage_is(stage: &FfStage, op: u32, arg1: u32, arg2: u32) -> bool {
    u32::from(stage.color_op) == op
        && u32::from(stage.color_arg1) == arg1
        && u32::from(stage.color_arg2) == arg2
}

/// Whether the pixel stages are the reflective water cascade.
///
/// Stage 0 `BUMPENVMAP(TEXTURE, DIFFUSE)` and stage 1 `MODULATE(TEXTURE,
/// CURRENT)`, both with a bound texture, then stage 2 `MODULATE(CURRENT,
/// TFACTOR)`. The pixel emitter checks this before it writes the water term,
/// since the term reads the stage-0 texel and the bump matrix the cascade
/// declares only in this shape.
#[must_use]
pub fn ps_stages_match(ps: &FfPsKey) -> bool {
    let [bump, reflection, opacity, ..] = &ps.stages;
    stage_is(bump, D3DTOP_BUMPENVMAP, D3DTA_TEXTURE, D3DTA_DIFFUSE)
        && bump.has_texture()
        && stage_is(reflection, D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_CURRENT)
        && reflection.has_texture()
        && stage_is(opacity, D3DTOP_MODULATE, D3DTA_CURRENT, D3DTA_TFACTOR)
}

/// Whether a fixed-function draw is DAoC's reflective water.
///
/// The state the keys can see: lighting off on an untransformed layout, the
/// [`ps_stages_match`] cascade, and stage 1 sampling the reflection target
/// through `TEXCOORDINDEX = CAMERASPACEPOSITION` with `TTFF = COUNT3 |
/// PROJECTED`. Lighting off also keeps the water varyings
/// (`texcoord12`..`15`) clear of the per-pixel lighting ones, which only lit
/// draws write.
///
/// Unchecked, because neither key carries them: the stage-0 map being
/// `D3DFMT_V8U8` and the draw alpha blending. Another unlit bump-mapped
/// projective draw with these stage states gets the water term too.
#[must_use]
pub fn is_reflective_water(vs: &FfVsKey, ps: &FfPsKey) -> bool {
    let tt = vs.tt_flags[1];
    !vs.lighting_enabled()
        && !vs.has_rhw()
        && vs.tci_mode(1) == TCI_MODE_CAMERASPACEPOSITION
        && u32::from(tt_count(tt)) == D3DTTFF_COUNT3
        && tt_projected(tt)
        && ps_stages_match(ps)
}

/// The `FfVsFlags::WATER` bit for a draw: the config is on and the draw is reflective water.
#[must_use]
pub fn water_flag(cfg: &WaterConfig, vs: &FfVsKey, ps: &FfPsKey) -> bool {
    cfg.enabled() && is_reflective_water(vs, ps)
}

/// The sun the glint reflects: the lowest-index enabled directional light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSun {
    /// The light's travel direction in eye space, unit length (zero for a
    /// zero direction), as the FF light rows pack it.
    pub dir_eye: [f32; 3],
    /// The light's diffuse colour.
    pub diffuse: [f32; 3],
}

/// Size of the water uniform in bytes: two `float4` rows.
pub const WATER_BYTES: usize = 32;

/// The water uniform the pixel stage reads at `PS_WATER_SLOT`.
///
/// Row 0 is `(sun direction in eye space, glint strength)` and row 1
/// `(sun diffuse rgb, 1.0 with the Fresnel term else 0.0)`. Without a sun
/// both sun fields are zero, so the glint vanishes.
#[must_use]
pub fn build_water_bytes(cfg: &WaterConfig, sun: Option<WaterSun>) -> [u8; WATER_BYTES] {
    let sun = sun.unwrap_or(WaterSun {
        dir_eye: [0.0; 3],
        diffuse: [0.0; 3],
    });
    let [dx, dy, dz] = sun.dir_eye;
    let [r, g, b] = sun.diffuse;
    let lanes = [
        dx,
        dy,
        dz,
        f32::from(cfg.glint_pct) / 100.0,
        r,
        g,
        b,
        if cfg.fresnel { 1.0 } else { 0.0 },
    ];
    let mut out = [0u8; WATER_BYTES];
    for (chunk, lane) in out.as_chunks_mut::<4>().0.iter_mut().zip(lanes) {
        *chunk = lane.to_le_bytes();
    }
    out
}
