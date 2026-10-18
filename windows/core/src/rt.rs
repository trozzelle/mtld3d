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

/// The point lights the scene carries to the ray-traced kernels and the receivers.
pub const MAX_SCENE_POINTS: usize = 4;

/// One enabled point light of the world, in eye space
/// (`FfState::scene_points_eye`). A `range` of 0 is an empty entry.
///
/// The receivers match a light of their own draw to it by position: within
/// [`SCENE_POINT_MATCH`] eye units of `position`, its attenuation takes the
/// visibility `rt.pointShadows` traced toward this light.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScenePoint {
    /// Eye-space position.
    pub position: [f32; 3],
    /// `D3DLIGHT9::Range`, 0 for an empty entry.
    pub range: f32,
    /// `D3DLIGHT9::Diffuse` red, green and blue.
    pub diffuse: [f32; 3],
    /// `D3DLIGHT9::Attenuation0..2`.
    pub attenuation: [f32; 3],
}

impl ScenePoint {
    /// Whether the entry holds a light.
    #[must_use]
    pub fn is_light(&self) -> bool {
        self.range > 0.0
    }
}

/// How close, in eye units, a draw's light must be to a [`ScenePoint`] to take its visibility.
pub const SCENE_POINT_MATCH: f32 = 1.0;

/// The radius, in world units, of the sphere a point light's shadow rays aim at.
pub const POINT_LIGHT_RADIUS: f32 = 15.0;

/// How far short of a point light, in world units, its shadow rays stop
/// (ruling R54): the emitter's own fixture — the logs and stones of a fire
/// pit, a lantern's housing, the body of a figure carrying a torch — sits
/// within it and must not shadow the light it holds. A ray whose length is
/// at most this lets the light through.
pub const POINT_LIGHT_CLEARANCE: f32 = 32.0;

/// Three point lights and an empty entry, every float distinct.
#[cfg(test)]
pub(crate) fn test_points() -> [ScenePoint; MAX_SCENE_POINTS] {
    let point = |i: f32| ScenePoint {
        position: [i, i + 1.0, i + 2.0],
        range: 40.0 * i,
        diffuse: [0.1 * i, 0.2 * i, 0.3 * i],
        attenuation: [1.0, 0.01 * i, 0.001 * i],
    };
    [
        point(1.0),
        ScenePoint {
            position: [-5.0, 6.0, 7.0],
            ..point(2.0)
        },
        point(3.0),
        ScenePoint::default(),
    ]
}

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
