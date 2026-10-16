//! Relight's light controls, transcribed from `dev/lighting/relight/relight_core.c`
//! (`rl_luma`, `rl_classify`, `rl_apply_light`, `rl_apply_ambient[_rgb]`,
//! `rl_ctab_register`, `rl_vs_map_from_ctab`, `rl_apply_vs_constants`).
//!
//! Settings arrive as native config keys; `relight`'s `enabled` flag has no
//! native equivalent — an identity [`LightControls`] is "off". Scales and tints
//! are stored as u16 hundredths and converted to f32 at use.
//!
//! Parity is the requirement: the math below is a transcription, not a new
//! model. Tests port every case from `dev/lighting/relight/test_core.c`.

use mtld3d_types::{D3DCOLORVALUE, D3DLIGHT_POINT, D3DLIGHT9};

/// `SUN_WEAK` in relight_core.c: sun diffuse luminance below this counts as
/// night for the ambient floor.
const SUN_WEAK: f32 = 0.2;

/// `CTAB_FOURCC` in relight_core.c: `"CTAB"`.
const CTAB_FOURCC: u32 = 0x4241_5443;

/// A scale/tint pair (`rl_config`'s `X_scale`/`X_tint`), in hundredths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scaled {
    pub scale_pct: u16,
    pub tint_pct: [u16; 3],
}

impl Scaled {
    const IDENTITY: Scaled = Scaled {
        scale_pct: 100,
        tint_pct: [100; 3],
    };

    #[inline]
    fn scale(&self) -> f32 {
        f32::from(self.scale_pct) / 100.0
    }

    #[inline]
    fn tint(&self) -> [f32; 3] {
        [
            f32::from(self.tint_pct[0]) / 100.0,
            f32::from(self.tint_pct[1]) / 100.0,
            f32::from(self.tint_pct[2]) / 100.0,
        ]
    }
}

/// Relight's `rl_config`, restricted to the light controls, in hundredths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightControls {
    pub sun: Scaled,
    pub ambient: Scaled,
    /// Minimum ambient luminance while the sun is weak (< 0.2); 0 = off.
    pub night_floor_pct: u16,
    pub torch: Scaled,
    /// Diffuse scale for placed lights.
    pub placed_scale_pct: u16,
    /// Range factor k: att1 /= k, att2 /= k*k, range *= k.
    pub placed_range_pct: u16,
    /// The per-figure head light.
    pub figure_scale_pct: u16,
}

impl Default for LightControls {
    fn default() -> Self {
        LightControls {
            sun: Scaled::IDENTITY,
            ambient: Scaled::IDENTITY,
            night_floor_pct: 0,
            torch: Scaled::IDENTITY,
            placed_scale_pct: 100,
            placed_range_pct: 100,
            figure_scale_pct: 100,
        }
    }
}

impl LightControls {
    /// Identity is "off": exactly the default, like relight's `enabled = 0`.
    pub fn is_identity(&self) -> bool {
        *self == LightControls::default()
    }
}

/// Relight's `rl_kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightClass {
    Sun,
    Torch,
    Placed,
    Figure,
    Other,
}

/// `rl_luma`: Rec.709 luminance of an RGB triple.
#[must_use]
pub fn luma(rgb: &[f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// `mul` from relight_core.c: `c *= s * t` componentwise.
fn mul_rgb(rgb: &mut [f32; 3], s: f32, t: &[f32; 3]) {
    rgb[0] *= s * t[0];
    rgb[1] *= s * t[1];
    rgb[2] *= s * t[2];
}

/// `mul` on the RGB part of a packed constant row (the `.w` stays).
fn mul_row(row: &mut [f32; 4], s: f32, t: &[f32; 3]) {
    let mut rgb = [row[0], row[1], row[2]];
    mul_rgb(&mut rgb, s, t);
    row[0] = rgb[0];
    row[1] = rgb[1];
    row[2] = rgb[2];
}

fn clamp01(v: f32) -> f32 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// `rl_classify`. Type 1 point, 2 spot, 3 directional (`D3DLIGHTTYPE` values);
/// point lights classify by their attenuation factors.
#[must_use]
pub fn classify(light: &D3DLIGHT9) -> LightClass {
    if light.type_ == 3 {
        return LightClass::Sun;
    }
    if light.type_ != 1 {
        return LightClass::Other;
    }
    let (att1, att2) = (light.attenuation1, light.attenuation2);
    if att1 > 0.0 && att2 > 0.0 {
        LightClass::Torch
    } else if att1 > 0.0 && att2 == 0.0 {
        LightClass::Figure
    } else if att1 == 0.0 && att2 > 0.0 {
        LightClass::Placed
    } else {
        LightClass::Other
    }
}

/// `rl_apply_light` on the six packed rows of one light in the `ff_state`
/// layout (`fill_lights_section`): row +2 is the diffuse RGB (its `.w` carries
/// the specular alpha the shader reads), row +4 is attenuation0/1/2 and range.
/// Only the diffuse RGB and the attenuation/range row are touched.
pub fn apply_light_rows(c: &LightControls, class: LightClass, rows: &mut [[f32; 4]; 6]) {
    if c.is_identity() {
        return; // relight's `!c->enabled`
    }
    match class {
        LightClass::Sun => {
            let (s, t) = (c.sun.scale(), c.sun.tint());
            mul_row(&mut rows[2], s, &t);
        }
        LightClass::Torch => {
            let (s, t) = (c.torch.scale(), c.torch.tint());
            mul_row(&mut rows[2], s, &t);
        }
        LightClass::Placed => {
            let k = f32::from(c.placed_range_pct) / 100.0;
            let s = f32::from(c.placed_scale_pct) / 100.0;
            mul_row(&mut rows[2], s, &[1.0, 1.0, 1.0]);
            if k > 0.0 && k != 1.0 {
                rows[4][1] /= k;
                rows[4][2] /= k * k;
                if rows[4][3] < 1e18 {
                    rows[4][3] *= k;
                }
            }
        }
        LightClass::Figure => {
            let s = f32::from(c.figure_scale_pct) / 100.0;
            mul_row(&mut rows[2], s, &[1.0; 3]);
        }
        LightClass::Other => {}
    }
}

/// `rl_apply_ambient_rgb`: ambient as a 0..1 colour. A black ambient (shadow
/// casters, sky) is never changed.
pub fn apply_ambient_rgb(c: &LightControls, rgb: &mut [f32; 3], scene_sun_luma: f32) {
    if c.is_identity() {
        return; // relight's `!c->enabled`: no clamp, no mutation
    }
    if rgb[0] == 0.0 && rgb[1] == 0.0 && rgb[2] == 0.0 {
        return;
    }
    let (s, t) = (c.ambient.scale(), c.ambient.tint());
    mul_rgb(rgb, s, &t);
    let floor = f32::from(c.night_floor_pct) / 100.0;
    if floor > 0.0 && scene_sun_luma < SUN_WEAK {
        let lum = luma(rgb);
        if lum > 0.0 && lum < floor {
            mul_rgb(rgb, floor / lum, &[1.0; 3]);
        }
    }
    rgb[0] = clamp01(rgb[0]);
    rgb[1] = clamp01(rgb[1]);
    rgb[2] = clamp01(rgb[2]);
}

/// `rl_apply_ambient`: a `D3DRS_AMBIENT` value 0xAARRGGBB; alpha kept.
#[must_use]
pub fn apply_ambient(c: &LightControls, ambient: u32, scene_sun_luma: f32) -> u32 {
    if c.is_identity() || ambient & 0x00ff_ffff == 0 {
        return ambient;
    }
    let mut a = [
        f32::from(((ambient >> 16) & 255) as u8) / 255.0,
        f32::from(((ambient >> 8) & 255) as u8) / 255.0,
        f32::from((ambient & 255) as u8) / 255.0,
    ];
    apply_ambient_rgb(c, &mut a, scene_sun_luma);
    (ambient & 0xff00_0000)
        | ((u32::from((a[0] * 255.0 + 0.5) as u8)) << 16)
        | ((u32::from((a[1] * 255.0 + 0.5) as u8)) << 8)
        | u32::from((a[2] * 255.0 + 0.5) as u8)
}

/// Register index of a named constant in a shader's CTAB comment, `None` if
/// absent (`rl_ctab_register`).
#[must_use]
pub fn ctab_register(tokens: &[u32], name: &str) -> Option<u16> {
    let words = tokens.len();
    // The CTAB comment precedes the first instruction.
    let mut i = 1usize;
    while i < words {
        let tok = tokens[i];
        if tok & 0xffff != 0xfffe {
            break;
        }
        let len = ((tok >> 16) & 0x7fff) as usize;
        if len >= 2 && i + 1 + len <= words && tokens[i + 1] == CTAB_FOURCC {
            // Table bytes: the `len - 1` words after the fourcc token.
            let tsize = (len - 1) * 4;
            if tsize < 28 {
                return None;
            }
            let bytes: Vec<u8> = tokens[i + 2..i + 1 + len]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let read_u32 = |off: usize| {
                u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
            };
            let n = read_u32(12) as usize;
            let info = read_u32(16) as usize;
            for k in 0..n {
                // Checked: a malformed CTAB must return None, not panic on
                // 32-bit usize (i686).
                let Some(end) = 20usize
                    .checked_mul(k + 1)
                    .and_then(|step| info.checked_add(step))
                else {
                    break;
                };
                if end > tsize {
                    break;
                }
                let off = end - 20;
                let nm = read_u32(off) as usize;
                let reg = u16::from_le_bytes([bytes[off + 6], bytes[off + 7]]);
                if nm < tsize
                    && bytes[nm..tsize].contains(&0)
                    && bytes[nm..tsize]
                        .iter()
                        .take_while(|&&b| b != 0)
                        .eq(name.as_bytes().iter())
                {
                    return Some(reg);
                }
            }
            return None;
        }
        i += 1 + len;
    }
    None
}

/// Where a skinned-figure vertex shader keeps its lights (`rl_vs_map`).
/// A register of [`ABSENT`] means the shader lacks that constant.
pub struct FigureLightMap {
    pub sun_diffuse: u16,
    pub sun_ambient: u16,
    pub plight_color: [u16; 3],
    pub plight_factors: [u16; 3],
}

/// Sentinel for "register absent" in a [`FigureLightMap`].
pub const ABSENT: u16 = u16::MAX;

/// Where a skinned-figure vertex shader keeps its lights; `None` unless the
/// shader has both sun constants (`rl_vs_map_from_ctab`).
#[must_use]
pub fn figure_map_from_ctab(tokens: &[u32]) -> Option<FigureLightMap> {
    let sun_diffuse = ctab_register(tokens, "$sunlight_Diffuse")?;
    let sun_ambient = ctab_register(tokens, "$sunlight_Ambient")?;
    let mut plight_color = [ABSENT; 3];
    let mut plight_factors = [ABSENT; 3];
    for i in 0..3 {
        plight_color[i] =
            ctab_register(tokens, &format!("$plight{}_Color", i + 1)).unwrap_or(ABSENT);
        plight_factors[i] =
            ctab_register(tokens, &format!("$plight{}_Factors", i + 1)).unwrap_or(ABSENT);
    }
    Some(FigureLightMap {
        sun_diffuse,
        sun_ambient,
        plight_color,
        plight_factors,
    })
}

/// `reg_ptr`: row index of register `r` within the `[start, start + n)`
/// window, or `None`.
fn reg_row(start: u16, n: usize, r: u16) -> Option<usize> {
    if r == ABSENT {
        return None;
    }
    let r = usize::from(r);
    let start = usize::from(start);
    if r >= start && r < start + n {
        Some(r - start)
    } else {
        None
    }
}

/// `rl_apply_vs_constants` on one `SetVertexShaderConstantF` write: apply the
/// settings to the sun diffuse/ambient rows and classify each point light from
/// its factors (attenuation 0/1/2), which must be in the same write. Rows
/// whose registers are absent or outside the write are skipped.
pub fn apply_figure_rows(
    c: &LightControls,
    map: &FigureLightMap,
    start_reg: u16,
    scene_sun_luma: f32,
    rows: &mut [[f32; 4]],
) {
    if c.is_identity() {
        return; // relight's `!c->enabled`
    }
    let n = rows.len();
    if let Some(row) = reg_row(start_reg, n, map.sun_diffuse) {
        let (s, t) = (c.sun.scale(), c.sun.tint());
        mul_row(&mut rows[row], s, &t);
    }
    if let Some(row) = reg_row(start_reg, n, map.sun_ambient) {
        let mut rgb = [rows[row][0], rows[row][1], rows[row][2]];
        apply_ambient_rgb(c, &mut rgb, scene_sun_luma);
        rows[row][0] = rgb[0];
        rows[row][1] = rgb[1];
        rows[row][2] = rgb[2];
    }
    for i in 0..3 {
        let (Some(col), Some(fac)) = (
            reg_row(start_reg, n, map.plight_color[i]),
            reg_row(start_reg, n, map.plight_factors[i]),
        ) else {
            continue;
        };
        // Classify by factors: a type-1 point with attenuation from
        // `$plightN_Factors` and the sentinel range 1e19.
        let light = D3DLIGHT9 {
            type_: D3DLIGHT_POINT,
            diffuse: D3DCOLORVALUE {
                r: rows[col][0],
                g: rows[col][1],
                b: rows[col][2],
                a: 0.0,
            },
            attenuation0: rows[fac][0],
            attenuation1: rows[fac][1],
            attenuation2: rows[fac][2],
            range: 1e19,
            ..D3DLIGHT9::default()
        };
        let class = classify(&light);
        let mut block = [
            [0.0; 4],
            [0.0; 4],
            [rows[col][0], rows[col][1], rows[col][2], 0.0],
            [0.0; 4],
            [rows[fac][0], rows[fac][1], rows[fac][2], 1e19],
            [0.0; 4],
        ];
        apply_light_rows(c, class, &mut block);
        rows[col][0] = block[2][0];
        rows[col][1] = block[2][1];
        rows[col][2] = block[2][2];
    }
}

/// The figure-constant step of `SetVertexShaderConstantF` (relight's
/// `dev_SetVSConstF`): patch the incoming game rows for a bound figure VS and
/// store them into the device's VS constant mirror with the stock redundant-set
/// semantics — the mirror window is compared bit-exact (NaN payloads and
/// signed zero included, `rows_differ`) against the patched rows and
/// overwritten only when they differ.
///
/// `mirror_window` is the mirror rows `[start_reg..start_reg + incoming.len()]`
/// (the caller validated the window against the register file). Returns `true`
/// when any mirror row changed, filling `patched` (at least `incoming.len()`
/// rows) with the patched rows for the caller's delta push. The game's rows are
/// never written: the mirror — and the `GetVertexShaderConstantF` that reads it
/// — holds the patched rows, and a repeated write of the same game values
/// dedups against the patched mirror.
///
/// # Panics
/// Panics when `mirror_window` and `incoming` differ in length or `patched` is
/// shorter than `incoming`.
pub fn write_figure_vs_constants(
    c: &LightControls,
    map: &FigureLightMap,
    start_reg: u16,
    scene_sun_luma: f32,
    incoming: &[[f32; 4]],
    mirror_window: &mut [[f32; 4]],
    patched: &mut [[f32; 4]],
) -> bool {
    assert_eq!(
        mirror_window.len(),
        incoming.len(),
        "mirror window must match the incoming rows"
    );
    assert!(
        patched.len() >= incoming.len(),
        "patched buffer shorter than the incoming rows"
    );
    for (dst, src) in patched[..incoming.len()].iter_mut().zip(incoming) {
        *dst = *src;
    }
    apply_figure_rows(
        c,
        map,
        start_reg,
        scene_sun_luma,
        &mut patched[..incoming.len()],
    );
    let changed = crate::shader_constants::rows_differ(mirror_window, &patched[..incoming.len()]);
    if changed {
        crate::shader_constants::store_window(mirror_window, &patched[..incoming.len()]);
    }
    changed
}

#[cfg(test)]
mod tests;
