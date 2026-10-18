//! The denoiser of the ray-traced scene buffers: a temporal pass, then an edge-stopping à-trous filter.
//!
//! A ray-traced kernel writes a raw, noisy buffer each frame (one or two
//! rays per pixel). [`passes`] lays out the four dispatches that turn it
//! into what the receivers read, all on the frame's geometry
//! (`Kernel::Geometry`: each pixel's eye-space normal and depth, kept for
//! the previous frame too):
//!
//! 1. **Temporal** ([`DenoiseKind::temporal`]): each pixel's eye position
//!    goes back into the previous frame through `prev_view · inv(view)`
//!    ([`reprojection`]) and its projection. The history there is rejected
//!    when the previous frame saw another surface: an eye depth more than
//!    [`REJECT_DEPTH`] away, or a normal whose dot with this one (turned
//!    into the previous eye space) is under [`REJECT_NORMAL`]. Otherwise it
//!    is clamped to the 3×3 neighbourhood's minimum and maximum of this
//!    frame's raw values, widened by [`CLAMP_MARGIN`] (a changed scene the
//!    geometry cannot see, like a moving occluder's shadow, leaves no
//!    ghost) and blended with the raw value at `1 / (length + 1)`, where
//!    `length`, the frames merged, grows by one per frame up to the kind's
//!    cap ([`DenoiseKind::cap`]). A rejected pixel starts over at length 0,
//!    the raw value alone. At half resolution the raw value of a pixel is
//!    that of the nearest-depth raw texel around it. Beside the length it
//!    keeps the raw samples' second moment, merged as the value is, and so
//!    the noise left in the value: the samples' variance over the frames
//!    merged plus one (none on a pixel's first frame).
//! 2. **Spatial**: three à-trous passes of the 5×5 B3 spline at steps
//!    [`ATROUS_STEPS`] (1, 2, 4), each tap weighted by its distance from the
//!    pixel's tangent plane (relative to the depth, [`PLANE_SIGMA`]), its
//!    normal ([`NORMAL_POWER`]) and its value's difference (the luminance of
//!    bounce light, the largest channel difference of the point lights),
//!    against [`VALUE_SIGMA_FLOOR`] plus [`VALUE_SIGMA_SCALE`] standard
//!    deviations of the pixel's noise: a noisy penumbra averages, a hard
//!    edge stays. The temporal output feeds them and stays the history;
//!    they ping-pong through the raw buffer and end in the output.
//!
//! The history (value and length, two of each) lives with the scene
//! textures, so a resize or a rescale makes it anew and [`TemporalHistory`]
//! is reset. [`TemporalHistory`] also keeps the previous traced frame's
//! camera: a camera cut ([`camera_cut`]), a split world or a world traced
//! with no TLAS or no sun resets it, and every pixel starts over.

use mtld3d_core::shadows::cascades::Mat4;
use mtld3d_shared::{MetalHandle, mtl::PixelFormat, mtl_handle::MTLTextureKind};

use super::scene_vis::Kernel;
use crate::LOG_TARGET;

#[cfg(test)]
mod tests;

/// The relative eye-depth difference past which the previous frame's pixel is another surface.
pub const REJECT_DEPTH: f32 = 0.02;
/// The normal dot under which the previous frame's pixel is another surface.
pub const REJECT_NORMAL: f32 = 0.9;
/// The à-trous passes' steps, in pixels.
pub const ATROUS_STEPS: [u32; 3] = [1, 2, 4];
/// A tap's distance from the pixel's tangent plane, relative to its eye depth, at which its weight falls by `1/e`.
pub const PLANE_SIGMA: f32 = 0.02;
/// The power of the normals' dot that weighs a tap.
pub const NORMAL_POWER: f32 = 32.0;
/// A tap's value difference at which its weight falls by `1/e`, for a pixel with no noise.
pub const VALUE_SIGMA_FLOOR: f32 = 0.02;
/// How many standard deviations of a pixel's estimated noise widen that.
pub const VALUE_SIGMA_SCALE: f32 = 4.0;
/// How far past this frame's 3×3 raw minimum and maximum the history may
/// stay: one ray per pixel leaves a whole neighbourhood unlit (or lit) now
/// and then near the edge of a penumbra, and a clamp to it would make the
/// history jump there; a scene change moves it all the same.
pub const CLAMP_MARGIN: f32 = 0.05;
/// A camera that moves more than this many world units between two frames cuts.
pub const CUT_TRANSLATION: f64 = 500.0;
/// A camera that turns more than this many degrees between two frames cuts.
pub const CUT_ROTATION_DEGREES: f64 = 30.0;

/// The buffer a denoise pass smooths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenoiseKind {
    /// Sun visibility, `R16Float`.
    Sun,
    /// Ambient occlusion, `R16Float`.
    Ao,
    /// One-bounce sunlight, `RGBA16Float`.
    Gi,
    /// Four point lights' visibility, `RGBA16Float`.
    Points,
}

/// Number of kinds.
pub const DENOISE_KINDS: usize = DenoiseKind::ALL.len();

impl DenoiseKind {
    pub const ALL: [Self; 4] = [Self::Sun, Self::Ao, Self::Gi, Self::Points];

    /// The most frames its history merges.
    #[must_use]
    pub const fn cap(self) -> u32 {
        match self {
            Self::Sun | Self::Ao | Self::Points => 16,
            Self::Gi => 32,
        }
    }

    /// The format of its raw buffer, its history and its output.
    #[must_use]
    pub const fn format(self) -> PixelFormat {
        match self {
            Self::Sun | Self::Ao => PixelFormat::R16Float,
            Self::Gi | Self::Points => PixelFormat::Rgba16Float,
        }
    }

    /// What a pixel with no surface (the sky) reads: full visibility, no bounce light.
    #[must_use]
    pub const fn empty(self) -> f32 {
        match self {
            Self::Gi => 0.0,
            Self::Sun | Self::Ao | Self::Points => 1.0,
        }
    }

    /// Its temporal pass.
    #[must_use]
    pub const fn temporal(self) -> Kernel {
        match self {
            Self::Sun => Kernel::TemporalSun,
            Self::Ao => Kernel::TemporalAo,
            Self::Gi => Kernel::TemporalGi,
            Self::Points => Kernel::TemporalPoints,
        }
    }

    /// Its à-trous pass.
    #[must_use]
    pub const fn atrous(self) -> Kernel {
        match self {
            Self::Sun => Kernel::AtrousSun,
            Self::Ao => Kernel::AtrousAo,
            Self::Gi => Kernel::AtrousGi,
            Self::Points => Kernel::AtrousPoints,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// The history of one kind: its value and, per pixel (`RGBA16Float`), the
/// frames merged, the second moment of the raw samples merged and the noise
/// variance left in the value; this frame's and the previous by
/// [`HistoryFrame`] index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryTextures {
    pub value: [MetalHandle<MTLTextureKind>; 2],
    pub length: [MetalHandle<MTLTextureKind>; 2],
}

impl HistoryTextures {
    /// Every texture.
    pub fn handles(self) -> impl Iterator<Item = MetalHandle<MTLTextureKind>> {
        self.value.into_iter().chain(self.length)
    }
}

/// The camera of one traced frame: what the next frame reprojects into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameCamera {
    /// `D3DTS_VIEW`, row-vector.
    pub view: Mat4,
    /// The projection's `_11`, `_22`, `_33`, `_43`.
    pub proj: [f32; 4],
    /// The world's viewport on the raster: x, y, width, height.
    pub viewport: [f32; 4],
}

/// This frame's place in the history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryFrame {
    /// The index of this frame's history and geometry textures.
    pub current: usize,
    /// The index of the previous frame's.
    pub previous: usize,
    /// The frame's index in the rays' sample sequence.
    pub frame: u32,
    /// This frame's eye space to the previous frame's (row-vector); the
    /// identity without a previous frame.
    pub reproject: Mat4,
    /// The previous frame's camera; this frame's without one.
    pub previous_camera: FrameCamera,
}

/// The temporal state of the traced frames: the previous camera, which
/// history texture is this frame's, which kinds' histories hold.
#[derive(Clone, Debug, Default)]
pub struct TemporalHistory {
    /// The previous traced frame's camera, `None` after a reset.
    previous: Option<FrameCamera>,
    /// The index of this frame's textures.
    current: usize,
    /// Frames traced, the rays' sequence index.
    frame: u32,
    /// The kinds this frame's history serves.
    valid: [bool; DENOISE_KINDS],
    /// The kinds denoised this frame, whose history the next one may use.
    ran: [bool; DENOISE_KINDS],
}

impl TemporalHistory {
    /// Start a traced frame seen by `camera`: the textures swap, and each
    /// kind denoised last frame keeps its history unless the camera cut.
    pub fn begin(&mut self, camera: FrameCamera) -> HistoryFrame {
        let previous = self.previous.replace(camera);
        let continues = previous.is_some_and(|p| {
            let cut = camera_cut(&p.view, &camera.view);
            if cut {
                mtld3d_shared::log_once_info!(target: LOG_TARGET,
                    "rt: a camera cut starts the denoiser's history over");
            }
            !cut
        });
        self.valid = if continues {
            self.ran
        } else {
            [false; DENOISE_KINDS]
        };
        self.ran = [false; DENOISE_KINDS];
        self.current ^= 1;
        self.frame = self.frame.wrapping_add(1);
        let reproject = previous
            .and_then(|p| reprojection(&p.view, &camera.view))
            .unwrap_or(IDENTITY);
        HistoryFrame {
            current: self.current,
            previous: self.current ^ 1,
            frame: self.frame,
            reproject,
            previous_camera: previous.unwrap_or(camera),
        }
    }

    /// Whether this frame's `kind` merges its history.
    #[must_use]
    pub const fn valid(&self, kind: DenoiseKind) -> bool {
        self.valid[kind.index()]
    }

    /// `kind` was denoised this frame: the next frame may merge it.
    pub const fn ran(&mut self, kind: DenoiseKind) {
        self.ran[kind.index()] = true;
    }

    /// Forget every history and the previous camera: the next frame starts over.
    pub fn reset(&mut self) {
        *self = Self {
            frame: self.frame,
            current: self.current,
            ..Self::default()
        };
    }

    /// The index of this frame's textures.
    #[cfg(test)]
    #[must_use]
    pub const fn current(&self) -> usize {
        self.current
    }
}

const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

type M3 = [[f64; 3]; 3];

/// The upper 3×3 of `m`, widened.
fn linear(m: &Mat4) -> M3 {
    core::array::from_fn(|r| core::array::from_fn(|c| f64::from(m[r][c])))
}

/// The inverse of `m`, `None` when it is singular.
fn invert3(m: &M3) -> Option<M3> {
    let cof = |r0: usize, r1: usize, c0: usize, c1: usize| {
        m[r0][c0].mul_add(m[r1][c1], -(m[r0][c1] * m[r1][c0]))
    };
    let adj = [
        [cof(1, 2, 1, 2), -cof(0, 2, 1, 2), cof(0, 1, 1, 2)],
        [-cof(1, 2, 0, 2), cof(0, 2, 0, 2), -cof(0, 1, 0, 2)],
        [cof(1, 2, 0, 1), -cof(0, 2, 0, 1), cof(0, 1, 0, 1)],
    ];
    let det = m[0][0].mul_add(adj[0][0], m[0][1].mul_add(adj[1][0], m[0][2] * adj[2][0]));
    (det.abs() > 1e-12).then(|| adj.map(|row| row.map(|v| v / det)))
}

fn mul3(a: &M3, b: &M3) -> M3 {
    core::array::from_fn(|r| core::array::from_fn(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

/// The row vector `v · m`.
fn apply3(v: [f64; 3], m: &M3) -> [f64; 3] {
    core::array::from_fn(|c| (0..3).map(|k| v[k] * m[k][c]).sum())
}

fn translation(m: &Mat4) -> [f64; 3] {
    [m[3][0], m[3][1], m[3][2]].map(f64::from)
}

/// Where the camera of the row-vector view `view` is, in world space: `-t · R⁻¹`.
fn camera_position(view: &Mat4) -> Option<[f64; 3]> {
    let inverse = invert3(&linear(view))?;
    Some(apply3(translation(view), &inverse).map(|v| -v))
}

/// Whether the camera cut between the views `previous` and `view`: it moved
/// more than [`CUT_TRANSLATION`] world units or turned more than
/// [`CUT_ROTATION_DEGREES`]. A singular view always cuts.
#[must_use]
pub fn camera_cut(previous: &Mat4, view: &Mat4) -> bool {
    let (Some(a), Some(b), Some(inverse)) = (
        camera_position(previous),
        camera_position(view),
        invert3(&linear(previous)),
    ) else {
        return true;
    };
    let moved = (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
    // The turn from the previous orientation to this one; a mirrored view
    // stays mirrored, so the product is a proper rotation.
    let turn = mul3(&inverse, &linear(view));
    let trace = turn[0][0] + turn[1][1] + turn[2][2];
    let degrees = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
    moved > CUT_TRANSLATION || degrees > CUT_ROTATION_DEGREES
}

/// This frame's eye space to the previous frame's, `prev_view · inv(view)`
/// in column-vector terms: the row-vector `inv(view) · previous`. `None` for
/// a singular view.
#[must_use]
pub fn reprojection(previous: &Mat4, view: &Mat4) -> Option<Mat4> {
    let inverse = invert3(&linear(view))?;
    // inv([R 0; t 1]) = [R⁻¹ 0; -t·R⁻¹ 1].
    let back = apply3(translation(view), &inverse).map(|v| -v);
    let r_prev = linear(previous);
    let rotation = mul3(&inverse, &r_prev);
    let moved = apply3(back, &r_prev);
    let t_prev = translation(previous);
    let narrow = |v: f64| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a matrix element for a kernel"
        )]
        let v = v as f32;
        v
    };
    let mut m = IDENTITY;
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = narrow(rotation[r][c]);
        }
    }
    for c in 0..3 {
        m[3][c] = narrow(moved[c] + t_prev[c]);
    }
    Some(m)
}

/// One dispatch of the denoiser: its kernel, `texture(0)` to `texture(7)`
/// (0 unbound), its à-trous step (0 for the temporal pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenoisePass {
    pub kernel: Kernel,
    pub textures: [MetalHandle<MTLTextureKind>; 8],
    pub step: u32,
}

/// The dispatches that denoise `kind`'s `raw` buffer into `out`, through its
/// `history`, on the frame's `geometry` (by [`HistoryFrame`] index), in order.
///
/// The temporal pass reads `raw`, the previous history and geometry and
/// writes this frame's history; the à-trous passes read this frame's
/// geometry and ping-pong from the history through `out` and `raw` back
/// into `out`. `raw` is free by then: the temporal pass read it.
#[must_use]
pub fn passes(
    kind: DenoiseKind,
    raw: MetalHandle<MTLTextureKind>,
    history: &HistoryTextures,
    geometry: [MetalHandle<MTLTextureKind>; 2],
    out: MetalHandle<MTLTextureKind>,
    frame: &HistoryFrame,
) -> [DenoisePass; 4] {
    let (now, before) = (frame.current, frame.previous);
    let none = MetalHandle::NULL;
    let temporal = DenoisePass {
        kernel: kind.temporal(),
        textures: [
            raw,
            history.value[now],
            geometry[now],
            history.value[before],
            geometry[before],
            history.length[before],
            history.length[now],
            none,
        ],
        step: 0,
    };
    let atrous = |input, output, step| DenoisePass {
        kernel: kind.atrous(),
        textures: [
            input,
            output,
            geometry[now],
            history.length[now],
            none,
            none,
            none,
            none,
        ],
        step,
    };
    let [s0, s1, s2] = ATROUS_STEPS;
    [
        temporal,
        atrous(history.value[now], out, s0),
        atrous(out, raw, s1),
        atrous(raw, out, s2),
    ]
}

/// The MSL of `kind`'s temporal and à-trous kernels, after the constants,
/// the depth reads and the geometry helpers (`eye_at`).
#[must_use]
pub fn kernel_msl(kind: DenoiseKind) -> String {
    format!(
        "#define DENOISE_KIND {}\nconstant float DENOISE_EMPTY = {:?};\nconstant float \
         REJECT_DEPTH = {REJECT_DEPTH:?};\nconstant float REJECT_NORMAL = \
         {REJECT_NORMAL:?};\nconstant float PLANE_SIGMA = {PLANE_SIGMA:?};\nconstant float \
         NORMAL_POWER = {NORMAL_POWER:?};\nconstant float VALUE_SIGMA_FLOOR = \
         {VALUE_SIGMA_FLOOR:?};\nconstant float VALUE_SIGMA_SCALE = \
         {VALUE_SIGMA_SCALE:?};\nconstant float CLAMP_MARGIN = {CLAMP_MARGIN:?};\n{DENOISE_MSL}",
        kind.index(),
        kind.empty()
    )
}

/// The temporal and à-trous kernels; `DENOISE_KIND` picks the value type
/// (`float` for sun and occlusion, `float4` for bounce light and points) and
/// how two values differ.
const DENOISE_MSL: &str = r"
#if DENOISE_KIND < 2
typedef float Value;
static Value load(texture2d<float, access::read> t, uint2 q) { return t.read(q).r; }
static void store(texture2d<float, access::write> t, Value v, uint2 q) { t.write(float4(v), q); }
#else
typedef float4 Value;
static Value load(texture2d<float, access::read> t, uint2 q) { return t.read(q); }
static void store(texture2d<float, access::write> t, Value v, uint2 q) { t.write(v, q); }
#endif

static float value_distance(Value a, Value b) {
#if DENOISE_KIND == 2
    return abs(dot(a.rgb - b.rgb, float3(0.2126, 0.7152, 0.0722)));
#elif DENOISE_KIND == 3
    float4 d = abs(a - b);
    return max(max(d.x, d.y), max(d.z, d.w));
#else
    return abs(a - b);
#endif
}

// The scalar whose spread the variance estimate measures: the value, the
// luminance of bounce light, the largest point-light channel.
static float value_scalar(Value v) {
#if DENOISE_KIND == 2
    return dot(v.rgb, float3(0.2126, 0.7152, 0.0722));
#elif DENOISE_KIND == 3
    return max(max(v.x, v.y), max(v.z, v.w));
#else
    return v;
#endif
}

// The eye depth of a geometry texel, 0 for the sky.
static float geometry_depth(texture2d<float, access::read> geometry, int2 p, int2 size) {
    return geometry.read(uint2(clamp(p, int2(0), size - 1))).w;
}

kernel void scene_temporal(texture2d<float, access::read> raw [[texture(0)]],
                           texture2d<float, access::write> out [[texture(1)]],
                           texture2d<float, access::read> geometry [[texture(2)]],
                           texture2d<float, access::read> previous [[texture(3)]],
                           texture2d<float, access::read> previous_geometry [[texture(4)]],
                           texture2d<float, access::read> previous_length [[texture(5)]],
                           texture2d<float, access::write> merged_out [[texture(6)]],
                           constant RtConstants& k [[buffer(0)]],
                           uint2 q [[thread_position_in_grid]]) {
    int2 size = int2(k.scene.extent.xy);
    if (int(q.x) >= size.x || int(q.y) >= size.y) return;
    float4 g = geometry.read(q);
    if (!(g.w > 0.0)) {
        store(out, Value(DENOISE_EMPTY), q);
        merged_out.write(float4(0.0), q);
        return;
    }
    int scale = int(max(k.trace.z, 1u));
    int2 raw_size = (size + scale - 1) / scale;
    int2 p = int2(q);
    int2 base = min(p / scale, raw_size - 1);
    // The raw value of this pixel: at half resolution, the nearest-depth
    // raw texel of the four around it.
    Value current = load(raw, uint2(base));
    if (scale > 1) {
        float best = INFINITY;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                int2 t = min(base + int2(i, j), raw_size - 1);
                float z = geometry_depth(geometry, t * scale, size);
                if (z > 0.0 && abs(z - g.w) < best) {
                    best = abs(z - g.w);
                    current = load(raw, uint2(t));
                }
            }
        }
    }
    // This frame's raw neighbourhood, the sky left out.
    Value lo = current;
    Value hi = current;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 t = base + int2(i, j);
            if (any(t < int2(0)) || any(t >= raw_size)) continue;
            if (!(geometry_depth(geometry, t * scale, size) > 0.0)) continue;
            Value v = load(raw, uint2(t));
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    float merged = 0.0;
    Value history = current;
    float s = value_scalar(current);
    float squares = s * s;
    if (k.params.z > 0.5) {
        float3 c = eye_at(p, g.w, k.scene);
        float3 back = (k.reproject * float4(c, 1.0)).xyz;
        if (back.z > 0.0) {
            float2 pixel = float2(
                k.prev_viewport.x + (back.x * k.prev_proj.x / back.z + 1.0) * 0.5 * k.prev_viewport.z,
                k.prev_viewport.y + (1.0 - back.y * k.prev_proj.y / back.z) * 0.5 * k.prev_viewport.w);
            int2 pp = int2(floor(pixel + 0.5));
            if (all(pp >= int2(0)) && all(pp < size)) {
                float4 pg = previous_geometry.read(uint2(pp));
                float3 n = normalize((k.reproject * float4(g.xyz, 0.0)).xyz);
                if (pg.w > 0.0 && abs(pg.w - back.z) <= REJECT_DEPTH * back.z
                    && dot(n, pg.xyz) >= REJECT_NORMAL) {
                    float4 before = previous_length.read(uint2(pp));
                    history = load(previous, uint2(pp));
                    merged = min(before.r + 1.0, float(k.trace.w));
                    squares = before.g;
                }
            }
        }
    }
    float blend = 1.0 / (merged + 1.0);
    history = clamp(history, lo - CLAMP_MARGIN, hi + CLAMP_MARGIN);
    // The moment follows the clamp: a history that changed with the scene
    // carries no noise of the old one.
    float low = max(value_scalar(lo) - CLAMP_MARGIN, 0.0);
    float high = value_scalar(hi) + CLAMP_MARGIN;
    squares = clamp(squares, low * low, high * high);
    Value value = history + (current - history) * blend;
    store(out, value, q);
    // The merged frames, the second moment of the raw samples merged as
    // the value is, and the noise variance left in the value: the samples'
    // variance over the frames merged.
    squares += (s * s - squares) * blend;
    float m = value_scalar(value);
    merged_out.write(float4(merged, squares, max(squares - m * m, 0.0) * blend, 0.0), q);
}

constant float B3[5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };

kernel void scene_atrous(texture2d<float, access::read> in [[texture(0)]],
                         texture2d<float, access::write> out [[texture(1)]],
                         texture2d<float, access::read> geometry [[texture(2)]],
                         texture2d<float, access::read> merged [[texture(3)]],
                         constant RtConstants& k [[buffer(0)]],
                         uint2 q [[thread_position_in_grid]]) {
    int2 size = int2(k.scene.extent.xy);
    if (int(q.x) >= size.x || int(q.y) >= size.y) return;
    Value c = load(in, q);
    float4 g = geometry.read(q);
    if (!(g.w > 0.0)) { store(out, c, q); return; }
    int step = int(k.params.w);
    int2 p = int2(q);
    float3 at = eye_at(p, g.w, k.scene);
    float3 n = g.xyz;
    // A noisy pixel averages over values its noise could reach; a clean
    // one (a hard shadow's edge) keeps it. The noise is the largest of the
    // 3×3 around: a pixel whose own samples were steady for a while beside
    // noisy neighbours is no cleaner than they are.
    float variance = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            variance = max(variance, merged.read(uint2(clamp(p + int2(i, j), int2(0), size - 1))).b);
        }
    }
    float sigma = VALUE_SIGMA_FLOOR + VALUE_SIGMA_SCALE * sqrt(variance);
    Value sum = c * (B3[2] * B3[2]);
    float total = B3[2] * B3[2];
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            if (i == 0 && j == 0) continue;
            int2 s = p + int2(i, j) * step;
            if (any(s < int2(0)) || any(s >= size)) continue;
            float4 gs = geometry.read(uint2(s));
            if (!(gs.w > 0.0)) continue;
            Value v = load(in, uint2(s));
            float plane = abs(dot(n, eye_at(s, gs.w, k.scene) - at)) / (PLANE_SIGMA * g.w);
            float w = B3[i + 2] * B3[j + 2]
                    * exp(-plane - value_distance(v, c) / sigma)
                    * pow(saturate(dot(n, gs.xyz)), NORMAL_POWER);
            sum += v * w;
            total += w;
        }
    }
    store(out, sum / total, q);
}
";
