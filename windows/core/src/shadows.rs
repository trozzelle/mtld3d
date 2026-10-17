//! Cascaded sun shadow maps (`shadows.*`).
//!
//! The native encoder captures the world's back-buffer draws so it can
//! replay them from the sun; [`scene_capture_wanted`] names the keys that
//! need that capture, for both this module and ray tracing ([`crate::rt`]).

pub mod cascades;

use crate::{config::Mtld3dConfig, encoder_data::PostBoundary};

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

/// Whether the world's lit draws read the sun's visibility (`scene_vis` bit
/// 0): `shadows.sun` or `rt.shadows`.
///
/// The same keys force the sun per pixel ([`crate::ppl::ppl_masks`]'s
/// `force_sun`, [`crate::ppl::figures_enabled`]), so the sun's own terms
/// take the visibility wherever a pixel stage can light them.
#[must_use]
pub const fn sun_receivers_wanted(cfg: &Mtld3dConfig) -> bool {
    cfg.shadows.sun || cfg.rt.shadows
}

/// The scene receiver uniform a pixel stage reads at fragment buffer
/// `mtld3d_shared::mtl::PS_SCENE_RECEIVER_SLOT` ([`SCENE_RECEIVER_MSL`]).
///
/// `k` is the share of a vertex-lit colour that stays in full shadow: the
/// ambient's luminance over the ambient's and the sun's,
/// `lum(amb) / (lum(amb) + lum(sun))`, in `[0, 1]`. A draw the pixel stage
/// cannot light per pixel scales its vertex colour by `mix(k, 1, vis)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct SceneReceiver {
    pub k: f32,
}

const _: () = assert!(core::mem::size_of::<SceneReceiver>() == 4);

/// The MSL mirror of [`SceneReceiver`].
pub const SCENE_RECEIVER_MSL: &str = "struct SceneReceiver {\n    float k;\n};\n\n";

impl SceneReceiver {
    /// No darkening: a shadowed pixel keeps its whole colour.
    pub const UNSHADOWED: Self = Self { k: 1.0 };

    /// The receiver values of a submission whose boundary is `boundary`.
    ///
    /// [`Self::UNSHADOWED`] when the boundary did not fire in the
    /// submission, or when neither the ambient nor the sun has any light.
    #[must_use]
    pub fn new(boundary: Option<&PostBoundary>) -> Self {
        let Some(boundary) = boundary else {
            return Self::UNSHADOWED;
        };
        let ambient = luminance(boundary.ambient);
        let sun = luminance(boundary.sun_color);
        let total = ambient + sun;
        if total > 0.0 {
            Self {
                k: (ambient / total).clamp(0.0, 1.0),
            }
        } else {
            Self::UNSHADOWED
        }
    }

    /// The bytes a pixel stage reads.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 4] {
        self.k.to_le_bytes()
    }
}

/// Rec. 709 luminance of a linear colour, never below 0.
fn luminance(rgb: [f32; 3]) -> f32 {
    // Array-sum, not a polynomial: no `fmaf` libcall on the i686 baseline.
    let y: f32 = [0.2126 * rgb[0], 0.7152 * rgb[1], 0.0722 * rgb[2]]
        .iter()
        .sum();
    y.max(0.0)
}

#[cfg(test)]
mod tests;
