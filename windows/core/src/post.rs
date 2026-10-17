//! The 3D/UI boundary the native post chain runs at (`post.*`).
//!
//! Post effects belong to the world, not to the interface drawn over it. The
//! game draws the world in perspective onto the back buffer, then its UI with an
//! orthographic projection or pre-transformed (`XYZRHW`) vertices. The
//! boundary is the first draw of a frame that renders to the implicit
//! back buffer orthographically or pre-transformed, after at least one
//! perspective back-buffer draw in the same frame. It fires once per frame
//! and resets at `Present`; a frame without one (login, a loading screen) gets
//! no post chain.
//!
//! At the boundary the device records one [`PostBoundary`] op ahead of the
//! draw, and the native encoder runs the enabled effects on the back buffer
//! there. The op carries the world's projection terms and sun, captured at the
//! last perspective back-buffer draw: at the boundary the bound projection is
//! already the UI's.

use crate::{encoder_data::PostBoundary, ff_state::FfState};

#[cfg(test)]
mod tests;

/// The `post.*` keys. Default: all off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PostConfig {
    /// `post.debug`: an internal check of the boundary that inverts the
    /// world's colour (rgb → 1 − rgb, alpha kept). Not documented.
    pub debug: bool,
    /// `post.ao`: ambient occlusion strength in hundredths, `[0, 200]`
    /// (`off` and `0` are zero, which runs nothing).
    pub ao_pct: u16,
    /// `post.godRays`: god-ray strength in hundredths, `[0, 200]` (`off` and
    /// `0` are zero, which runs nothing).
    pub god_rays_pct: u16,
}

impl PostConfig {
    /// Whether any key needs the boundary op recorded.
    #[must_use]
    pub const fn boundary_wanted(&self) -> bool {
        self.debug || self.ao_pct > 0 || self.god_rays_pct > 0
    }
}

/// The per-frame boundary rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundaryState {
    /// A perspective back-buffer draw ran this frame.
    saw_world: bool,
    /// The boundary already fired this frame.
    fired: bool,
}

impl BoundaryState {
    /// Step the rule for one draw; true when this draw is the boundary.
    ///
    /// `rt0_is_backbuffer`: render target 0 is the implicit back buffer.
    /// `ortho_or_rhw`: the draw has an orthographic projection or a
    /// pre-transformed vertex layout.
    pub const fn on_draw(&mut self, rt0_is_backbuffer: bool, ortho_or_rhw: bool) -> bool {
        if self.fired || !rt0_is_backbuffer {
            return false;
        }
        if !ortho_or_rhw {
            self.saw_world = true;
            return false;
        }
        if !self.saw_world {
            return false;
        }
        self.fired = true;
        true
    }

    /// End the frame.
    pub const fn on_present(&mut self) {
        *self = Self {
            saw_world: false,
            fired: false,
        };
    }

    /// Whether the boundary fired this frame.
    #[must_use]
    pub const fn fired(&self) -> bool {
        self.fired
    }
}

/// What the boundary rule reads about one draw.
pub struct DrawSite<'a> {
    pub rt0_is_backbuffer: bool,
    pub ortho_or_rhw: bool,
    /// The stored fixed-function state, for the world constants.
    pub ff: &'a FfState,
}

/// The device's boundary state for the frame being recorded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PostFrame {
    boundary: BoundaryState,
    /// The constants of the last perspective back-buffer draw.
    world: PostBoundary,
}

impl PostFrame {
    /// The draw hook: the op to record ahead of this draw, if it is the boundary.
    ///
    /// With every post key off this is one flag test and `site` is never
    /// called, so the default path records nothing and reads no draw state.
    #[inline]
    pub fn note_draw<'a>(
        &mut self,
        cfg: &PostConfig,
        site: impl FnOnce() -> DrawSite<'a>,
    ) -> Option<PostBoundary> {
        if !cfg.boundary_wanted() {
            return None;
        }
        let site = site();
        if !self.boundary.fired() && site.rt0_is_backbuffer && !site.ortho_or_rhw {
            self.world = world_constants(site.ff);
        }
        self.boundary
            .on_draw(site.rt0_is_backbuffer, site.ortho_or_rhw)
            .then_some(self.world)
    }

    /// End the frame, at `Present` or a device reset.
    pub const fn on_present(&mut self) {
        self.boundary.on_present();
    }
}

/// The constants a world draw leaves for the post chain: its projection terms and the sun.
#[must_use]
pub fn world_constants(ff: &FfState) -> PostBoundary {
    let sun = ff.scene_sun_eye();
    PostBoundary {
        proj: ff.projection_terms(),
        sun_dir_eye: sun.map(|s| s.dir_eye),
        sun_color: sun.map_or([0.0; 3], |s| s.diffuse),
    }
}
