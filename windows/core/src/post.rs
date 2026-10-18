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
//! there. The op carries the world's projection terms, view, sun and ambient,
//! captured at the last perspective back-buffer draw: at the boundary the
//! bound projection is already the UI's.

use crate::{encoder_data::PostBoundary, ff_state::FfState};

#[cfg(test)]
mod tests;

/// The `post.*` keys. Default: all off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// `post.bloom`: bloom strength in hundredths, `[0, 200]` (`off` and `0`
    /// are zero, which runs nothing).
    pub bloom_pct: u16,
    /// `post.bloomThreshold`: the bloom threshold in hundredths, `[50, 100]`,
    /// default [`BLOOM_THRESHOLD_DEFAULT_PCT`].
    pub bloom_threshold_pct: u16,
}

/// Default `post.bloomThreshold`, in hundredths (`0.8`).
pub const BLOOM_THRESHOLD_DEFAULT_PCT: u16 = 80;

impl Default for PostConfig {
    fn default() -> Self {
        Self {
            debug: false,
            ao_pct: 0,
            god_rays_pct: 0,
            bloom_pct: 0,
            bloom_threshold_pct: BLOOM_THRESHOLD_DEFAULT_PCT,
        }
    }
}

impl PostConfig {
    /// Whether any key needs the boundary op recorded.
    #[must_use]
    pub const fn boundary_wanted(&self) -> bool {
        self.debug || self.ao_pct > 0 || self.god_rays_pct > 0 || self.bloom_pct > 0
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
    /// `D3DRS_AMBIENT`, which the device holds outside the fixed-function state.
    pub ambient: u32,
    /// `D3DRS_FOGCOLOR`, the sky tint of the ray-traced bounce light.
    pub fog_color: u32,
    /// The stored fixed-function state, for the world constants.
    pub ff: &'a FfState,
}

/// The device's boundary state for the frame being recorded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PostFrame {
    boundary: BoundaryState,
    /// The constants of the last perspective back-buffer draw.
    world: PostBoundary,
    /// The boundaries recorded so far, across frames.
    boundaries: u32,
}

impl PostFrame {
    /// The draw hook: the op to record ahead of this draw, if it is the boundary.
    ///
    /// `wanted` is [`crate::config::Mtld3dConfig::post_boundary_wanted`].
    /// Without it this is one flag test and `site` is never called, so the
    /// default path records nothing and reads no draw state.
    #[inline]
    pub fn note_draw<'a>(
        &mut self,
        wanted: bool,
        site: impl FnOnce() -> DrawSite<'a>,
    ) -> Option<PostBoundary> {
        if !wanted {
            return None;
        }
        let site = site();
        if !self.boundary.fired() && site.rt0_is_backbuffer && !site.ortho_or_rhw {
            self.world = world_constants(site.ff, site.ambient, site.fog_color);
        }
        if !self
            .boundary
            .on_draw(site.rt0_is_backbuffer, site.ortho_or_rhw)
        {
            return None;
        }
        let op = PostBoundary {
            frame: self.boundaries,
            ..self.world
        };
        self.boundaries = self.boundaries.wrapping_add(1);
        Some(op)
    }

    /// End the frame, at `Present` or a device reset.
    pub const fn on_present(&mut self) {
        self.boundary.on_present();
    }
}

/// The constants a world draw leaves for the post chain and the scene
/// passes: its projection terms, view, sun, whether the sun is above the
/// horizon, the ambient colour `ambient` (`D3DRS_AMBIENT`), the fog colour
/// `fog_color` (`D3DRS_FOGCOLOR`) and the point lights nearest the camera.
///
/// The sun lies against its light's travel direction, and the world's up in
/// eye space is [`crate::ppl::hemi_up_eye`] of the view; the sun is up when
/// `dot(−dir, up) > 0`. `frame` is left 0: [`PostFrame::note_draw`] counts.
#[must_use]
pub fn world_constants(ff: &FfState, ambient: u32, fog_color: u32) -> PostBoundary {
    let view = ff.transform(mtld3d_types::D3DTS_VIEW);
    let sun = ff.scene_sun_eye();
    let sun_above_horizon = sun.is_some_and(|s| {
        let up = view.map_or([0.0, 0.0, -1.0], crate::ppl::hemi_up_eye);
        // Array-sum, not a polynomial: no `fmaf` libcall on the i686 baseline.
        let height: f32 = [
            -s.dir_eye[0] * up[0],
            -s.dir_eye[1] * up[1],
            -s.dir_eye[2] * up[2],
        ]
        .iter()
        .sum();
        height > 0.0
    });
    let [r, g, b, _] = crate::convert::d3dcolor_to_rgba_f32(ambient);
    let [fr, fg, fb, _] = crate::convert::d3dcolor_to_rgba_f32(fog_color);
    PostBoundary {
        proj: ff.projection_terms(),
        sun_dir_eye: sun.map(|s| s.dir_eye),
        sun_color: sun.map_or([0.0; 3], |s| s.diffuse),
        sun_above_horizon,
        view: view.map_or(IDENTITY_ROWS, |m| {
            [
                [m.m[0], m.m[1], m.m[2], m.m[3]],
                [m.m[4], m.m[5], m.m[6], m.m[7]],
                [m.m[8], m.m[9], m.m[10], m.m[11]],
                [m.m[12], m.m[13], m.m[14], m.m[15]],
            ]
        }),
        ambient: [r, g, b],
        frame: 0,
        points: ff.scene_points_eye(crate::rt::MAX_SCENE_POINTS),
        fog_color: [fr, fg, fb],
    }
}

const IDENTITY_ROWS: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
