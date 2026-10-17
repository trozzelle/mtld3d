//! Fog overrides for DAoC draws whose stock fog is broken.
//!
//! DAoC's skinned figure VS compute their fog from `$fogParams`
//! (`c[10 + layout]`, holding `(start, end, end - start, 0.6 / end)`) as
//! exponential fog `exp(-0.6 d / end)`: a haze from distance 0 that terrain
//! (linear fog from `start`) does not have. `fog.figures` replaces it with
//! the world's linear fog on the clip-space `w` (see [`crate::ppl::figures::fog_params_register`]).

use crate::ppl::figures;

/// Fog overrides. Default: all off. File key: `fog.figures` (`off` | `on`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FogConfig {
    /// Linear fog on DAoC's figure VS (`fog.figures`); independent of the
    /// `lighting.perPixel*` keys.
    pub figures: bool,
}

/// The draw's programmable VS is one of the figure shaders and `fog.figures`
/// opted them in.
#[must_use]
pub fn figure_fog_flag(cfg: &FogConfig, vs_id: u64) -> bool {
    cfg.figures && figures::figure_vs(vs_id).is_some()
}
