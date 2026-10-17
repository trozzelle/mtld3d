//! Cascaded sun shadow maps (`shadows.*`).
//!
//! The native encoder captures the world's back-buffer draws so it can
//! replay them from the sun; [`scene_capture_wanted`] names the keys that
//! need that capture, for both this module and ray tracing ([`crate::rt`]).

pub mod cascades;

use crate::config::Mtld3dConfig;

/// `shadows.debug`: an internal view of the shadow terms. Not documented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowDebug {
    Off,
    /// Tints the world by cascade index.
    Cascades,
    /// Shows the sun visibility over the world.
    Visibility,
}

/// The `shadows.*` keys. Default: off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowConfig {
    /// `shadows.sun` (`off` | `on`).
    pub sun: bool,
    /// `shadows.resolution`: texels per cascade side, `1024`, `2048` or `4096`.
    pub resolution: u16,
    /// `shadows.cascades`: cascade count, `[1, 4]`.
    pub cascades: u8,
    /// `shadows.distance`: shadow distance in world units, `[2000, 16000]`.
    pub distance: u16,
    /// `shadows.softness`: PCF radius in cascade-0 texels, in hundredths,
    /// `[0, 400]`.
    pub softness_pct: u16,
    /// `shadows.debug` (internal).
    pub debug: ShadowDebug,
}

/// Default `shadows.resolution`.
pub const RESOLUTION_DEFAULT: u16 = 2048;
/// Default `shadows.cascades`.
pub const CASCADES_DEFAULT: u8 = 3;
/// Default `shadows.distance`, in world units.
pub const DISTANCE_DEFAULT: u16 = 8000;
/// Default `shadows.softness`, in hundredths (`1.5`).
pub const SOFTNESS_DEFAULT_PCT: u16 = 150;

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            sun: false,
            resolution: RESOLUTION_DEFAULT,
            cascades: CASCADES_DEFAULT,
            distance: DISTANCE_DEFAULT,
            softness_pct: SOFTNESS_DEFAULT_PCT,
            debug: ShadowDebug::Off,
        }
    }
}

/// Whether the native encoder captures the world's draws for replay.
///
/// True iff a key that replays the world is on: `shadows.sun`, `rt.shadows`,
/// `rt.ao`, `rt.gi` or `rt.pointShadows`. With every `shadows.*` and `rt.*`
/// key absent nothing is captured.
#[must_use]
pub const fn scene_capture_wanted(cfg: &Mtld3dConfig) -> bool {
    cfg.shadows.sun
        || cfg.rt.shadows
        || cfg.rt.ao_pct > 0
        || cfg.rt.gi_pct > 0
        || cfg.rt.point_shadows
}
