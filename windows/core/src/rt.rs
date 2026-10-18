//! Metal ray-traced lighting (`rt.*`).

/// `rt.quality`: the ray budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtQuality {
    /// Half resolution, one sample per pixel.
    Low,
    /// Full resolution, one sample per pixel, one spatial denoise pass
    /// fewer than the others.
    Medium,
    /// Full resolution, two samples per pixel.
    High,
}

/// `rt.debug`: an internal view of one ray-traced buffer. Not documented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtDebug {
    Off,
    Visibility,
    Ao,
    Gi,
    Points,
}

/// The `rt.*` keys. Default: off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtConfig {
    /// `rt.shadows` (`off` | `on`): ray-traced sun shadows.
    pub shadows: bool,
    /// `rt.sunSize`: the sun's angular diameter in hundredths of a degree,
    /// `[25, 400]`.
    pub sun_size_pct: u16,
    /// `rt.ao`: ambient occlusion strength in hundredths, `[0, 200]` (`off`
    /// and `0` are zero, which runs nothing).
    pub ao_pct: u16,
    /// `rt.aoRadius`: occlusion ray length in world units, `[50, 500]`.
    pub ao_radius: u16,
    /// `rt.gi`: one-bounce sunlight strength in hundredths, `[0, 200]` (`off`
    /// and `0` are zero, which runs nothing).
    pub gi_pct: u16,
    /// `rt.pointShadows` (`off` | `on`): shadows from point lights.
    pub point_shadows: bool,
    /// `rt.quality`.
    pub quality: RtQuality,
    /// `rt.debug` (internal).
    pub debug: RtDebug,
}

/// Default `rt.sunSize`, in hundredths of a degree (`1.0`).
pub const SUN_SIZE_DEFAULT_PCT: u16 = 100;
/// Default `rt.aoRadius`, in world units.
pub const AO_RADIUS_DEFAULT: u16 = 160;

impl RtConfig {
    /// Whether a key that traces rays is on: `rt.shadows`, `rt.ao`, `rt.gi` or `rt.pointShadows`.
    #[must_use]
    pub const fn traces(&self) -> bool {
        self.shadows || self.ao_pct > 0 || self.gi_pct > 0 || self.point_shadows
    }
}

impl Default for RtConfig {
    fn default() -> Self {
        Self {
            shadows: false,
            sun_size_pct: SUN_SIZE_DEFAULT_PCT,
            ao_pct: 0,
            ao_radius: AO_RADIUS_DEFAULT,
            gi_pct: 0,
            point_shadows: false,
            quality: RtQuality::Medium,
            debug: RtDebug::Off,
        }
    }
}
