//! Cascade fitting for the sun's shadow maps.
//!
//! The camera frustum, out to `shadows.distance`, is cut into
//! `shadows.cascades` slices ([`cascade_splits`]); each slice gets one
//! orthographic view from the sun ([`cascade_matrix`]) that the caster passes
//! render the frame's world into, re-projected from the camera's clip space
//! ([`reproject_for`]).
//!
//! # Conventions
//!
//! Every [`Mat4`] is a D3D row-vector matrix, `m[row][col]`, applied as
//! `v' = v · M`, the way `D3DMATRIX` stores `_11.._44`. Eye space is the
//! D3D view space (left-handed, `+z` ahead). Clip space is D3D's and
//! Metal's: depth in `[0, w]`.
//!
//! The 64 bytes of such a matrix, row after row, are the same transform for
//! an MSL `float4x4` that multiplies a column vector: MSL reads `m[c]` as
//! column `c`, and `M_msl · v = (vᵀ · M)ᵀ`. So the caster passes bind
//! [`reproject_for`]'s result as it is, with no transpose.
//!
//! The maths runs in `f64`: the game's world coordinates are large, and the
//! eye to world to light chain cancels them.

#[cfg(test)]
mod tests;

/// A 3-vector.
pub type Vec3 = [f32; 3];
/// A 3x3 matrix whose rows are the light's axes in world space (see [`light_basis`]).
pub type Mat3 = [[f32; 3]; 3];
/// A D3D row-vector 4x4 matrix, see the module docs.
pub type Mat4 = [[f32; 4]; 4];

/// The most cascades a set holds (`shadows.cascades` is `[1, 4]`).
pub const MAX_CASCADES: usize = 4;
/// The share of the logarithmic split scheme in [`cascade_splits`].
pub const SPLIT_LAMBDA: f32 = 0.75;
/// The caster passes' slope-scaled depth bias (`setDepthBias` slope).
pub const SLOPE_BIAS: f32 = 2.0;
/// The caster passes' constant depth bias, in texels of each cascade.
pub const CONSTANT_BIAS_TEXELS: f32 = 1.5;
/// How far toward the sun, past a cascade's bounding sphere, casters still land in its depth range.
///
/// Trees and hills between the sun and the slice shadow it although they
/// lie outside the sphere. The fork has no depth clamp on the caster passes
/// (encoder-level clamp is not honoured by every Metal device), so the
/// light's near plane moves this far toward the sun instead. Above the
/// game's tallest trees (~2600 units) with room for the hills behind them.
pub const SUN_MARGIN: f32 = 8000.0;
/// How close to vertical the sun may travel before [`light_basis`] stops
/// using world up as its reference, in degrees.
pub const VERTICAL_SUN_DEGREES: f32 = 5.0;

/// The cascades of one frame, as the caster passes rendered them and the visibility kernel reads them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadeSet {
    /// Eye space to cascade clip space, per cascade ([`cascade_matrix`]); with no depth bias.
    pub matrices: [Mat4; MAX_CASCADES],
    /// Eye-space distances along the view axis: cascade `i` covers `splits[i]..splits[i + 1]`.
    pub splits: [f32; MAX_CASCADES + 1],
    /// How many cascades are live, `[1, 4]`.
    pub count: u8,
}

type M4 = [[f64; 4]; 4];
type V3 = [f64; 3];

/// The split distances of `n` cascades from `near` to `far`.
///
/// The practical split scheme: each inner split blends the logarithmic
/// split (`lambda`) with the uniform one. `splits[0]` is `near`, `splits[n]`
/// and every entry after it `far`. `n` is clamped to `[1, 4]`.
#[must_use]
pub fn cascade_splits(near: f32, far: f32, n: u8, lambda: f32) -> [f32; MAX_CASCADES + 1] {
    let n = n.clamp(1, 4);
    let mut splits = [far; MAX_CASCADES + 1];
    splits[0] = near;
    let (near, far, lambda) = (f64::from(near), f64::from(far), f64::from(lambda));
    for (i, split) in (1..n).zip(&mut splits[1..]) {
        let t = f64::from(i) / f64::from(n);
        let log = near * (far / near).powf(t);
        let uniform = (far - near).mul_add(t, near);
        *split = narrow(lambda.mul_add(log - uniform, uniform));
    }
    splits
}

/// The camera's near and far planes from its projection terms `[_11, _22, _33, _43]`.
///
/// `None` unless the projection is a D3D perspective with a positive near
/// plane. The far plane is infinite for `_33 <= 1`.
#[must_use]
pub fn camera_near_far(proj_terms: [f32; 4]) -> Option<(f32, f32)> {
    let [_, _, p33, p43] = proj_terms;
    if !(p33 > 0.0 && p43 < 0.0) {
        return None;
    }
    let near = -p43 / p33;
    let far = if p33 > 1.0 {
        p43 / (1.0 - p33)
    } else {
        f32::INFINITY
    };
    (near.is_finite() && near > 0.0 && far > near).then_some((near, far))
}

/// The camera projection of `proj_terms` `[_11, _22, _33, _43]`, as a row-vector matrix.
#[must_use]
pub const fn projection(proj_terms: [f32; 4]) -> Mat4 {
    let [p11, p22, p33, p43] = proj_terms;
    [
        [p11, 0.0, 0.0, 0.0],
        [0.0, p22, 0.0, 0.0],
        [0.0, 0.0, p33, 1.0],
        [0.0, 0.0, p43, 0.0],
    ]
}

/// The sun's axes in world space: rows `x`, `y` and the travel direction.
///
/// Built from the sun direction alone, so the texel grid does not turn with
/// the camera. `x` is horizontal, perpendicular to the sun's heading (world
/// up, −Z, crossed with the direction), so it turns smoothly as the sun
/// crosses the sky, low in the east or west included; only a sun within
/// [`VERTICAL_SUN_DEGREES`] of the zenith or nadir, whose heading is
/// undefined, takes world X as the reference instead. `y = dir × x`: a
/// right-handed basis.
#[must_use]
pub fn light_basis(sun_dir_world: Vec3) -> Mat3 {
    let d = normalize(sun_dir_world.map(f64::from));
    let reference = if d[2].abs() < f64::from(VERTICAL_SUN_DEGREES).to_radians().cos() {
        [0.0, 0.0, -1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let x = normalize(cross(reference, d));
    let y = cross(d, x);
    [x, y, d].map(|row| row.map(narrow))
}

/// The world-space travel direction of a sun whose eye-space direction is `sun_dir_eye` under `view`.
///
/// `None` for a singular view or a zero direction.
#[must_use]
pub fn sun_dir_world(view: &Mat4, sun_dir_eye: Vec3) -> Option<Vec3> {
    let inverse = invert(&widen(view))?;
    let d = sun_dir_eye.map(f64::from);
    let world: V3 = core::array::from_fn(|c| (0..3).map(|r| d[r] * inverse[r][c]).sum());
    let length = dot(world, world).sqrt();
    (length > 0.0).then(|| world.map(|v| narrow(v / length)))
}

/// The eye-space bounding sphere of the frustum slice `split` (`(near, far)` along the view axis).
///
/// `(centre z, radius)`: the centre lies on the view axis, equidistant
/// from the slice's near and far corners, or at the far plane when that is
/// closer. It depends on the projection alone, so neither the camera's
/// position nor its rotation changes its size.
#[must_use]
pub fn slice_sphere(proj_terms: [f32; 4], split: (f32, f32)) -> (f32, f32) {
    let (centre, radius) = sphere(proj_terms, split);
    (narrow(centre), narrow(radius))
}

fn sphere(proj_terms: [f32; 4], (near, far): (f32, f32)) -> (f64, f64) {
    let (near, far) = (f64::from(near), f64::from(far));
    let (tan_x, tan_y) = (
        1.0 / f64::from(proj_terms[0]),
        1.0 / f64::from(proj_terms[1]),
    );
    // A corner at depth z lies z·√k off the axis.
    let k = tan_x.mul_add(tan_x, tan_y * tan_y);
    let centre = (far + near) * (1.0 + k) / 2.0;
    if centre >= far {
        (far, far * k.sqrt())
    } else {
        let off_axis = far * far * k;
        (
            centre,
            (far - centre).mul_add(far - centre, off_axis).sqrt(),
        )
    }
}

/// The eye-space `LightVP` of the cascade over the frustum slice `split`.
///
/// Maps an eye-space point to the cascade's clip space: an orthographic
/// view along the sun over the slice's bounding sphere ([`slice_sphere`])
/// padded by one texel each way, whose centre is snapped to the cascade's
/// texel grid in light space, so a moving camera moves no texel and the
/// snap (half a texel at most) never pushes the sphere off the map. Depth
/// runs from [`SUN_MARGIN`] toward the sun past the sphere to its far side.
/// `basis` is [`light_basis`]'s; under a mirroring view (negative
/// determinant, as the game's) its `x` axis is flipped, so the light view
/// has the camera's handedness and every triangle keeps the winding the
/// draw's cull mode was set for.
#[must_use]
pub fn cascade_matrix(
    view: &Mat4,
    proj_terms: [f32; 4],
    split: (f32, f32),
    basis: &Mat3,
    res: u16,
) -> Mat4 {
    let view = widen(view);
    let Some(inverse) = invert(&view) else {
        return [[0.0; 4]; 4];
    };
    let (centre_z, radius) = sphere(proj_terms, split);
    let centre: V3 = core::array::from_fn(|c| centre_z.mul_add(inverse[2][c], inverse[3][c]));
    let mut axes = basis.map(|row| row.map(f64::from));
    if det3(&view) < 0.0 {
        axes[0] = axes[0].map(|v| -v);
    }
    // `res` texels span the sphere's diameter and one texel on each side.
    let texel = 2.0 * radius / (f64::from(res.max(4)) - 2.0);
    let half_extent = radius + texel;
    let snap = |v: f64| (v / texel).round() * texel;
    let origin = [
        snap(dot(centre, axes[0])),
        snap(dot(centre, axes[1])),
        dot(centre, axes[2]) - radius - f64::from(SUN_MARGIN),
    ];
    let scale = [
        half_extent,
        half_extent,
        2.0f64.mul_add(radius, f64::from(SUN_MARGIN)),
    ];
    // World to cascade clip: light coordinates relative to the snapped
    // origin, over the half extent in x and y and the depth range in z.
    let mut world_to_light = [[0.0; 4]; 4];
    for (axis, ((dir, &start), &extent)) in axes.iter().zip(&origin).zip(&scale).enumerate() {
        for (row, &component) in dir.iter().enumerate() {
            world_to_light[row][axis] = component / extent;
        }
        world_to_light[3][axis] = -start / extent;
    }
    world_to_light[3][3] = 1.0;
    narrow_m4(&mul(&inverse, &world_to_light))
}

/// The matrix a caster pass binds: camera clip space to the cascade's, `light_vp · P⁻¹`.
///
/// In row-vector form `P⁻¹ · light_vp`: a camera clip position `c = e · P`
/// maps to `c · P⁻¹ · light_vp = e · light_vp`. Its bytes are what the MSL
/// re-projection reads (see the module docs). The zero matrix for a
/// singular `proj`.
#[must_use]
pub fn reproject_for(light_vp: &Mat4, proj: &Mat4) -> Mat4 {
    invert(&widen(proj)).map_or([[0.0; 4]; 4], |inverse| {
        narrow_m4(&mul(&inverse, &widen(light_vp)))
    })
}

/// [`CONSTANT_BIAS_TEXELS`] of the cascade `light_vp` covers at `res`, in its depth units.
///
/// A texel's world size is twice the half extent over `res`; one world unit
/// along the sun is the depth scale. Both come from `light_vp`'s columns,
/// which an orthonormal view leaves at their world lengths.
#[must_use]
pub fn caster_depth_bias(light_vp: &Mat4, res: u16) -> f32 {
    let column = |c: usize| {
        let v: V3 = core::array::from_fn(|r| f64::from(light_vp[r][c]));
        dot(v, v).sqrt()
    };
    let (per_unit_x, per_unit_z) = (column(0), column(2));
    if per_unit_x == 0.0 {
        return 0.0;
    }
    let texel = 2.0 / (f64::from(res) * per_unit_x);
    narrow(texel * per_unit_z * f64::from(CONSTANT_BIAS_TEXELS))
}

/// `m` with every output depth moved by `offset` clip-space depth units.
#[must_use]
pub fn with_depth_offset(m: &Mat4, offset: f32) -> Mat4 {
    let mut out = *m;
    for row in &mut out {
        row[2] = offset.mul_add(row[3], row[2]);
    }
    out
}

/// The cascades of a frame: its view, projection terms, eye-space sun direction and the `shadows.*` keys.
///
/// The slices run from the camera's near plane to `distance` or the
/// camera's far plane, whichever is closer. `None` when the projection is
/// not a perspective one, the view is singular, the sun direction is zero
/// or the range is empty.
#[must_use]
pub fn cascade_set(
    view: &Mat4,
    proj_terms: [f32; 4],
    sun_dir_eye: Vec3,
    (distance, count, res): (f32, u8, u16),
) -> Option<CascadeSet> {
    let (near, camera_far) = camera_near_far(proj_terms)?;
    let far = distance.min(camera_far);
    if far <= near {
        return None;
    }
    let basis = light_basis(sun_dir_world(view, sun_dir_eye)?);
    let count = count.clamp(1, 4);
    let splits = cascade_splits(near, far, count, SPLIT_LAMBDA);
    let mut matrices = [[[0.0; 4]; 4]; MAX_CASCADES];
    for (i, matrix) in matrices.iter_mut().enumerate().take(usize::from(count)) {
        *matrix = cascade_matrix(view, proj_terms, (splits[i], splits[i + 1]), &basis, res);
    }
    Some(CascadeSet {
        matrices,
        splits,
        count,
    })
}

fn widen(m: &Mat4) -> M4 {
    m.map(|row| row.map(f64::from))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the result is an f32 the GPU reads"
)]
const fn narrow(v: f64) -> f32 {
    v as f32
}

fn narrow_m4(m: &M4) -> Mat4 {
    m.map(|row| row.map(narrow))
}

fn mul(a: &M4, b: &M4) -> M4 {
    core::array::from_fn(|r| core::array::from_fn(|c| (0..4).map(|k| a[r][k] * b[k][c]).sum()))
}

/// The inverse of `m` by Gauss-Jordan elimination with partial pivoting, `None` when it is singular.
fn invert(m: &M4) -> Option<M4> {
    let mut a = *m;
    let mut inverse: M4 = core::array::from_fn(|r| core::array::from_fn(|c| f64::from(r == c)));
    for col in 0..4 {
        let pivot = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-30 {
            return None;
        }
        a.swap(col, pivot);
        inverse.swap(col, pivot);
        let p = a[col][col];
        a[col] = a[col].map(|v| v / p);
        inverse[col] = inverse[col].map(|v| v / p);
        for row in (0..4).filter(|&row| row != col) {
            let factor = a[row][col];
            let (pivot_row, pivot_inverse) = (a[col], inverse[col]);
            for k in 0..4 {
                a[row][k] = factor.mul_add(-pivot_row[k], a[row][k]);
                inverse[row][k] = factor.mul_add(-pivot_inverse[k], inverse[row][k]);
            }
        }
    }
    Some(inverse)
}

/// The determinant of `m`'s upper-left 3x3.
fn det3(m: &M4) -> f64 {
    let rows: [V3; 3] = core::array::from_fn(|r| [m[r][0], m[r][1], m[r][2]]);
    dot(rows[0], cross(rows[1], rows[2]))
}

fn dot(a: V3, b: V3) -> f64 {
    a.iter().zip(&b).map(|(x, y)| x * y).sum()
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1].mul_add(b[2], -(a[2] * b[1])),
        a[2].mul_add(b[0], -(a[0] * b[2])),
        a[0].mul_add(b[1], -(a[1] * b[0])),
    ]
}

fn normalize(v: V3) -> V3 {
    let length = dot(v, v).sqrt();
    if length == 0.0 {
        v
    } else {
        v.map(|c| c / length)
    }
}
