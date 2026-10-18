//! Cascaded sun shadow maps (`shadows.*`).
//!
//! The native encoder captures the world's back-buffer draws so it can
//! replay them from the sun; [`scene_capture_wanted`] names the keys that
//! need that capture, for both this module and ray tracing ([`crate::rt`]).
//!
//! While the sun's visibility darkens the world ([`sun_receivers_wanted`]),
//! the game's own Classic blob shadows (a dark quad under each figure,
//! `shadow.bmp`) would darken what the sun's
//! visibility already darkens, so the encoder leaves them out
//! ([`is_classic_blob`]). The blob is recognised by a heuristic render-state
//! signature, not by its texture: no capture of the game recorded the
//! upload of `shadow.bmp`, so its content hash is unknown.

pub mod cascades;

use mtld3d_types::{D3DBLEND_INVSRCCOLOR, D3DBLEND_ZERO};

use crate::{config::Mtld3dConfig, encoder_data::PostBoundary, pipeline_state::PipelineRsBits};

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
    cfg.shadows.sun || cfg.rt.traces()
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

/// What the scene keys force per pixel ([`crate::ppl::ppl_masks`]), whatever
/// `lighting.perPixel` says.
///
/// The directional lights and the figure terms
/// ([`crate::ppl::figures_enabled`]) under every key that captures the
/// world ([`scene_capture_wanted`]): the sun takes its visibility, the
/// ambient its occlusion and the surface its bounce light per pixel. The
/// point lights under `rt.pointShadows`, so each has a per-pixel
/// attenuation its visibility scales. Nothing with every key absent.
#[must_use]
pub const fn ppl_force(cfg: &Mtld3dConfig) -> crate::ppl::PplForce {
    crate::ppl::PplForce {
        directional: scene_capture_wanted(cfg),
        points: cfg.rt.point_shadows,
    }
}

/// What [`is_classic_blob`] reads of a draw.
#[derive(Clone, Copy, Debug)]
pub struct BlobFacts {
    /// The draw's blend state.
    pub pipeline_rs: PipelineRsBits,
    /// The draw writes depth.
    pub depth_write: bool,
    /// The vertex stage lights the draw (a fixed-function `D3DRS_LIGHTING`,
    /// or any programmable VS, which may light).
    pub lit: bool,
    /// The vertices the draw assembles: 4 for one quad, whether a 4-vertex
    /// strip or fan or a 6-index triangle list.
    pub vertices: u32,
}

/// Whether a draw looks like one of the game's Classic blob shadows.
///
/// A heuristic (see the module docs): an unlit single quad multiplied into
/// its target (blend `ZERO` / `INVSRCCOLOR`) with depth write off. Whether
/// it lands in the back buffer before the boundary is the caller's part.
#[must_use]
pub const fn is_classic_blob(snap: &BlobFacts) -> bool {
    snap.pipeline_rs.blend_enable()
        && snap.pipeline_rs.src_blend == D3DBLEND_ZERO as u8
        && snap.pipeline_rs.dst_blend == D3DBLEND_INVSRCCOLOR as u8
        && !snap.depth_write
        && !snap.lit
        && snap.vertices == 4
}

/// The scene receiver uniform a pixel stage reads at fragment buffer
/// `mtld3d_shared::mtl::PS_SCENE_RECEIVER_SLOT` ([`SCENE_RECEIVER_MSL`],
/// [`SCENE_RECEIVER_RT_MSL`]).
///
/// `k` is the share of a vertex-lit colour that stays in full shadow: the
/// ambient's luminance over the ambient's and the sun's,
/// `lum(amb) / (lum(amb) + lum(sun))`, in `[0, 1]`. A draw the pixel stage
/// cannot light per pixel scales its vertex colour by `mix(k, 1, vis)` for
/// the sun, and by `mix(1, ao, ao_strength · k)` for ambient occlusion (the
/// ambient's share is what occlusion darkens).
///
/// `ao` and `gi` are the `rt.ao` and `rt.gi` strengths. `points[k]` is
/// scene point `k`'s eye position (`w` 1 for a light, 0 for an empty
/// entry), which a draw's own point light matches within
/// [`crate::rt::SCENE_POINT_MATCH`] to take channel `k` of the point-light
/// visibility.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct SceneReceiver {
    pub k: f32,
    pub ao: f32,
    pub gi: f32,
    pub pad: f32,
    pub points: [[f32; 4]; crate::rt::MAX_SCENE_POINTS],
}

const _: () = assert!(core::mem::size_of::<SceneReceiver>() == 80);

/// The MSL mirror of [`SceneReceiver`]'s first member, all a sun-only receiver reads.
pub const SCENE_RECEIVER_MSL: &str = "struct SceneReceiver {\n    float k;\n};\n\n";

/// The MSL mirror of [`SceneReceiver`], for a receiver of the ray-traced
/// terms (`scene_vis` bits past the sun), and its point-light match: the
/// visibility channel of the scene point within
/// [`crate::rt::SCENE_POINT_MATCH`] of `p`, 1 (unshadowed) for a light
/// no scene point matches.
pub const SCENE_RECEIVER_RT_MSL: &str = "struct SceneReceiver {\n    float k;\n    float ao;\n    float gi;\n    float pad;\n    float4 points[4];\n};\n\nstatic inline float scene_point_vis(float3 p, float4 vis, constant SceneReceiver &rx) {\n    float v = 1.0;\n    for (int k = 0; k < 4; ++k) {\n        if (rx.points[k].w > 0.5 && distance(p, rx.points[k].xyz) <= 1.0) {\n            v = vis[k];\n        }\n    }\n    return v;\n}\n\n";

impl SceneReceiver {
    /// No darkening: a shadowed pixel keeps its whole colour; no occlusion,
    /// no bounce light, no point light matched.
    pub const UNSHADOWED: Self = Self {
        k: 1.0,
        ao: 0.0,
        gi: 0.0,
        pad: 0.0,
        points: [[0.0; 4]; crate::rt::MAX_SCENE_POINTS],
    };

    /// The receiver values of a submission whose boundary is `boundary`
    /// under the `rt.*` keys `rt`.
    ///
    /// `k` is 1 when the boundary did not fire in the submission or when
    /// neither the ambient nor the sun has any light, and no point light is
    /// matched without the boundary; the strengths are the keys'.
    #[must_use]
    pub fn new(boundary: Option<&PostBoundary>, rt: &crate::rt::RtConfig) -> Self {
        let mut out = Self {
            ao: f32::from(rt.ao_pct) / 100.0,
            gi: f32::from(rt.gi_pct) / 100.0,
            ..Self::UNSHADOWED
        };
        let Some(boundary) = boundary else {
            return out;
        };
        let ambient = luminance(boundary.ambient);
        let sun = luminance(boundary.sun_color);
        let total = ambient + sun;
        if total > 0.0 {
            out.k = (ambient / total).clamp(0.0, 1.0);
        }
        for (row, point) in out.points.iter_mut().zip(&boundary.points) {
            let [x, y, z] = point.position;
            *row = [x, y, z, if point.is_light() { 1.0 } else { 0.0 }];
        }
        out
    }

    /// The bytes a pixel stage reads.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 80] {
        let mut bytes = [0u8; 80];
        let floats = [self.k, self.ao, self.gi, self.pad]
            .into_iter()
            .chain(self.points.into_iter().flatten());
        for (chunk, v) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(floats) {
            *chunk = v.to_le_bytes();
        }
        bytes
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
