//! Cascade fitting: splits, spheres, the light basis, snapping, re-projection and winding.

use super::{
    CONSTANT_BIAS_TEXELS, Mat4, SPLIT_LAMBDA, SUN_MARGIN, Vec3, camera_near_far, cascade_matrix,
    cascade_set, cascade_splits, caster_depth_bias, light_basis, projection, reproject_for,
    slice_sphere, sun_dir_world, with_depth_offset,
};

/// DAoC's projection: 1.3 by 1.73, near plane 10, far plane 16000.
fn daoc_terms() -> [f32; 4] {
    let (near, far) = (10.0_f32, 16_000.0_f32);
    let p33 = far / (far - near);
    [1.3, 1.73, p33, -near * p33]
}

/// A D3D view at `eye` looking along yaw `yaw` and pitch `pitch` (radians,
/// positive down) in DAoC's world, whose up is −Z; `mirror` flips the eye
/// `x` axis, as DAoC's own view does (determinant −1).
fn view(eye: [f64; 3], yaw: f64, pitch: f64, mirror: bool) -> Mat4 {
    let forward = [
        pitch.cos() * yaw.cos(),
        pitch.cos() * yaw.sin(),
        pitch.sin(),
    ];
    let up = [0.0, 0.0, -1.0];
    let mut right = normalized(cross(up, forward));
    let above = cross(forward, right);
    if mirror {
        right = right.map(|v| -v);
    }
    let axes = [right, above, forward];
    let mut m = [[0.0f32; 4]; 4];
    for (col, axis) in axes.iter().enumerate() {
        for (row, &v) in axis.iter().enumerate() {
            m[row][col] = v as f32;
        }
        m[3][col] = -dot(eye, *axis) as f32;
    }
    m[3][3] = 1.0;
    m
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(&b).map(|(x, y)| x * y).sum()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(v: [f64; 3]) -> [f64; 3] {
    let length = dot(v, v).sqrt();
    v.map(|c| c / length)
}

/// `v · m` in `f64`.
fn apply(v: [f64; 4], m: &Mat4) -> [f64; 4] {
    core::array::from_fn(|c| (0..4).map(|r| v[r] * f64::from(m[r][c])).sum())
}

/// What MSL computes for `float4x4 m * float4 v` over the 64 bytes of `m`:
/// the bytes hold four columns of four floats, and the product sums the
/// columns weighted by `v`.
fn msl_mul(bytes: &[f32; 16], v: [f64; 4]) -> [f64; 4] {
    let column = |c: usize| &bytes[4 * c..4 * c + 4];
    core::array::from_fn(|r| (0..4).map(|c| f64::from(column(c)[r]) * v[c]).sum())
}

/// The eye-space position of world point `w` under `view`.
fn to_eye(view: &Mat4, w: [f64; 3]) -> [f64; 4] {
    apply([w[0], w[1], w[2], 1.0], view)
}

/// A small deterministic generator of `[0, 1)` values.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A sun high in the south-east, travelling down (+Z) through DAoC's world.
const SUN_WORLD: Vec3 = [0.3, -0.4, 0.866];

#[test]
fn splits_cover_near_to_far() {
    for n in 1..=4u8 {
        let splits = cascade_splits(10.0, 8000.0, n, SPLIT_LAMBDA);
        let n = usize::from(n);
        assert_eq!(splits[0], 10.0, "{n}: starts at the near plane");
        assert_eq!(splits[n], 8000.0, "{n}: ends at the distance");
        assert!(
            splits[..=n].windows(2).all(|w| w[0] < w[1]),
            "{n}: increasing {splits:?}"
        );
        assert!(splits[n..].iter().all(|&s| s == 8000.0), "{n}: {splits:?}");
    }
    // The practical scheme at λ 0.75: three quarters logarithmic.
    let splits = cascade_splits(10.0, 8000.0, 3, SPLIT_LAMBDA);
    assert!((splits[1] - 737.957).abs() < 0.01, "{splits:?}");
    assert!((splits[2] - 1980.497).abs() < 0.01, "{splits:?}");

    // Through the set: the camera's near plane, and the distance or the
    // camera's far plane, whichever is closer.
    let terms = daoc_terms();
    // `_33` in f32 leaves the far plane good to a few parts in 10⁵.
    let (near, far) = camera_near_far(terms).expect("a perspective projection");
    assert!(
        (near - 10.0).abs() < 1e-4 && (far - 16_000.0).abs() < 1.0,
        "{near} {far}"
    );
    let v = view([1000.0, 2000.0, -300.0], 0.4, 0.2, true);
    let sun = [0.0, 0.5, 0.866];
    let set = cascade_set(&v, terms, sun, (8000.0, 3, 2048)).expect("a set");
    assert_eq!(set.count, 3);
    assert!((set.splits[0] - 10.0).abs() < 1e-3);
    assert_eq!(set.splits[3], 8000.0);
    let short = {
        let (near, far) = (10.0_f32, 5000.0_f32);
        let p33 = far / (far - near);
        [1.3, 1.73, p33, -near * p33]
    };
    let set = cascade_set(&v, short, sun, (8000.0, 4, 2048)).expect("a set");
    assert!((set.splits[4] - 5000.0).abs() < 1.0, "{:?}", set.splits);
    assert!(cascade_set(&v, [1.0, 1.0, 1.0, 0.0], sun, (8000.0, 3, 2048)).is_none());
}

#[test]
fn sphere_contains_slice_corners() {
    let terms = daoc_terms();
    let v = view([30_000.0, 20_000.0, -800.0], 2.1, 0.35, true);
    let basis = light_basis(SUN_WORLD);
    let splits = cascade_splits(10.0, 8000.0, 4, SPLIT_LAMBDA);
    for i in 0..4 {
        let (near, far) = (splits[i], splits[i + 1]);
        let (centre, radius) = slice_sphere(terms, (near, far));
        let corners: Vec<[f64; 3]> = [near, far]
            .into_iter()
            .flat_map(|z| {
                let z = f64::from(z);
                let (x, y) = (z / f64::from(terms[0]), z / f64::from(terms[1]));
                [(-x, -y), (x, -y), (-x, y), (x, y)].map(|(x, y)| [x, y, z])
            })
            .collect();
        let distances: Vec<f64> = corners
            .iter()
            .map(|c| {
                dot(
                    [c[0], c[1], c[2] - f64::from(centre)],
                    [c[0], c[1], c[2] - f64::from(centre)],
                )
                .sqrt()
            })
            .collect();
        let radius = f64::from(radius);
        let farthest = distances.iter().copied().fold(0.0, f64::max);
        assert!(
            farthest <= radius * (1.0 + 1e-5),
            "{i}: {farthest} > {radius}"
        );
        assert!(
            farthest >= radius * (1.0 - 1e-5),
            "{i}: the sphere is tight"
        );
        // Every corner lands inside the cascade's clip volume.
        let m = cascade_matrix(&v, terms, (near, far), &basis, 2048);
        for c in &corners {
            let clip = apply([c[0], c[1], c[2], 1.0], &m);
            assert!((clip[3] - 1.0).abs() < 1e-6, "orthographic");
            assert!(
                clip[0].abs() <= 1.0 + 1e-4 && clip[1].abs() <= 1.0 + 1e-4,
                "{i}: {clip:?}"
            );
            assert!((0.0..=1.0).contains(&clip[2]), "{i}: {clip:?}");
        }
    }
}

/// The texel coordinates of world point `w` in cascade `m` (eye space) under `view`, at `res`.
fn texel_of(view: &Mat4, m: &Mat4, w: [f64; 3], res: u16) -> [f64; 2] {
    let clip = apply(to_eye(view, w), m);
    let res = f64::from(res);
    [(clip[0] * 0.5 + 0.5) * res, (clip[1] * 0.5 + 0.5) * res]
}

#[test]
fn snapped_matrix_stable_under_subtexel_translation() {
    let terms = daoc_terms();
    let basis = light_basis(SUN_WORLD);
    let res = 2048;
    let split = (10.0, 737.957);
    let (_, radius) = slice_sphere(terms, split);
    let texel = 2.0 * f64::from(radius) / f64::from(res);
    let eye = [31_250.5, 18_400.25, -650.0];
    let before = view(eye, 0.7, 0.3, true);
    let m_before = cascade_matrix(&before, terms, split, &basis, res);
    let probe = [eye[0] + 120.0, eye[1] + 80.0, eye[2] + 300.0];
    // 0.3 texel along the light's x axis, then along an arbitrary direction.
    let light_x = basis[0].map(f64::from);
    for step in [light_x, normalized([0.6, -0.3, 0.2])] {
        let moved = [0, 1, 2].map(|i| eye[i] + 0.3 * texel * step[i]);
        let after = view(moved, 0.7, 0.3, true);
        let m_after = cascade_matrix(&after, terms, split, &basis, res);
        let (a, b) = (
            texel_of(&before, &m_before, probe, res),
            texel_of(&after, &m_after, probe, res),
        );
        for axis in 0..2 {
            let shift = b[axis] - a[axis];
            assert!(
                (shift - shift.round()).abs() < 2e-3,
                "axis {axis}: the grid moved by {shift} texels"
            );
        }
    }
}

#[test]
fn cascade_matrix_rotation_independent_of_camera_yaw() {
    let terms = daoc_terms();
    let basis = light_basis(SUN_WORLD);
    let res = 2048;
    let split = (737.957, 1980.497);
    let eye = [31_250.5, 18_400.25, -650.0];
    let world_to_light = |yaw: f64| {
        let v = view(eye, yaw, 0.3, true);
        let m = cascade_matrix(&v, terms, split, &basis, res);
        // World to cascade clip: the view, then the cascade.
        let rows: [[f64; 4]; 3] = core::array::from_fn(|axis| {
            let mut unit = [0.0; 4];
            unit[axis] = 1.0;
            let direction = apply(unit, &v);
            apply(direction, &m)
        });
        let near_point = [eye[0] + 500.0, eye[1] - 300.0, eye[2] + 200.0];
        let origin_texel = texel_of(&v, &m, near_point, res);
        (rows, origin_texel)
    };
    let (a, grid_a) = world_to_light(0.2);
    for yaw in [1.1, 2.9, -2.0] {
        let (b, grid_b) = world_to_light(yaw);
        for (ra, rb) in a.iter().zip(&b) {
            for (x, y) in ra.iter().zip(rb) {
                assert!(
                    (x - y).abs() <= 1e-5 * ra.iter().map(|v| v.abs()).fold(0.0, f64::max),
                    "yaw {yaw}: {a:?} vs {b:?}"
                );
            }
        }
        for axis in 0..2 {
            let shift = grid_b[axis] - grid_a[axis];
            assert!(
                (shift - shift.round()).abs() < 0.05,
                "yaw {yaw}: the grid's phase moved by {shift}"
            );
        }
    }
}

#[test]
fn reproject_maps_eye_point_like_light_vp() {
    let terms = daoc_terms();
    let p = projection(terms);
    let v = view([30_000.0, 20_000.0, -800.0], 2.1, 0.35, true);
    let sun_eye = {
        let e = apply([0.3, -0.4, 0.866, 0.0], &v);
        [e[0] as f32, e[1] as f32, e[2] as f32]
    };
    let set = cascade_set(&v, terms, sun_eye, (8000.0, 4, 2048)).expect("a set");
    let mut random = Lcg(7);
    for i in 0..usize::from(set.count) {
        let m = set.matrices[i];
        let r = reproject_for(&m, &p);
        for _ in 0..100 {
            let (near, far) = (f64::from(set.splits[i]), f64::from(set.splits[i + 1]));
            let z = near + (far - near) * random.next();
            let x = (random.next() * 2.0 - 1.0) * z / f64::from(terms[0]);
            let y = (random.next() * 2.0 - 1.0) * z / f64::from(terms[1]);
            let e = [x, y, z, 1.0];
            let clip = apply(e, &p).map(|c| f64::from(c as f32));
            let (via_camera, direct) = (apply(clip, &r), apply(e, &m));
            for (a, b) in via_camera.iter().zip(&direct) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "cascade {i}: {via_camera:?} vs {direct:?}"
                );
            }
        }
    }
}

/// The bytes a caster binds, read the way MSL reads a `float4x4` at the
/// re-projection slot, move a camera clip position as the row-vector matrix
/// does, for a matrix that is neither the identity nor symmetric.
#[test]
fn reproject_bytes_are_the_msl_columns() {
    let terms = daoc_terms();
    let p = projection(terms);
    let mut light_vp = [[0.0f32; 4]; 4];
    light_vp[0] = [0.5, 0.1, 0.02, 0.0];
    light_vp[1] = [-0.2, 0.4, 0.03, 0.0];
    light_vp[2] = [0.05, -0.1, 0.01, 0.0];
    light_vp[3] = [0.25, -0.75, 0.4, 1.0];
    let r = reproject_for(&light_vp, &p);
    let bytes: [f32; 16] = *r.as_flattened().as_array().expect("16 floats");
    let e = [12.0, -7.0, 350.0, 1.0];
    let clip = apply(e, &p);
    let msl = msl_mul(&bytes, clip);
    let expected = apply(e, &light_vp);
    for (a, b) in msl.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5, "{msl:?} vs {expected:?}");
    }
    // The transpose is not the same transform: the test tells them apart.
    let transposed: [f32; 16] = core::array::from_fn(|i| bytes[(i % 4) * 4 + i / 4]);
    let wrong = msl_mul(&transposed, clip);
    assert!(
        wrong
            .iter()
            .zip(&expected)
            .any(|(a, b)| (a - b).abs() > 0.1)
    );
}

/// The signed area of triangle `t` after the perspective divide.
fn ndc_area(t: [[f64; 4]; 3]) -> f64 {
    let p = t.map(|c| [c[0] / c[3], c[1] / c[3]]);
    (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])
}

/// Under a plain and a mirroring view, a triangle that faces both the camera
/// and the sun winds the same way on screen and in the cascade, so the
/// draw's own cull mode culls the same faces in the caster pass.
#[test]
fn light_view_keeps_the_camera_winding() {
    let terms = daoc_terms();
    let p = projection(terms);
    let eye = [30_000.0, 20_000.0, -800.0];
    let mut random = Lcg(11);
    for mirror in [false, true] {
        let v = view(eye, 2.1, 0.35, mirror);
        let sun_eye = {
            let e = apply([0.3, -0.4, 0.866, 0.0], &v);
            [e[0] as f32, e[1] as f32, e[2] as f32]
        };
        let set = cascade_set(&v, terms, sun_eye, (8000.0, 2, 2048)).expect("a set");
        let m = set.matrices[1];
        let sun = sun_dir_world(&v, sun_eye).expect("a sun").map(f64::from);
        let mut checked = 0;
        for _ in 0..200 {
            // A triangle ahead of the camera, inside the second slice.
            let along = 900.0 + 800.0 * random.next();
            let forward = apply([0.0, 0.0, 1.0, 0.0], &inverse_view_rows(&v));
            let centre: [f64; 3] = core::array::from_fn(|i| eye[i] + forward[i] * along);
            let corners: [[f64; 3]; 3] = core::array::from_fn(|_| {
                core::array::from_fn(|i| centre[i] + (random.next() - 0.5) * 200.0)
            });
            let normal = cross(
                [0, 1, 2].map(|i| corners[1][i] - corners[0][i]),
                [0, 1, 2].map(|i| corners[2][i] - corners[0][i]),
            );
            let to_camera = [0, 1, 2].map(|i| eye[i] - corners[0][i]);
            let facing_camera = dot(normal, to_camera);
            let facing_sun = -dot(normal, sun);
            if facing_camera.abs() < 1.0 || facing_sun.abs() < 1.0 {
                continue;
            }
            let camera = corners.map(|c| apply(to_eye(&v, c), &p));
            let light = corners.map(|c| apply(to_eye(&v, c), &m));
            let (on_screen, in_cascade) = (ndc_area(camera), ndc_area(light));
            assert_eq!(
                (on_screen * facing_camera).signum(),
                (in_cascade * facing_sun).signum(),
                "mirror {mirror}: a triangle winds differently from the sun"
            );
            checked += 1;
        }
        assert!(checked > 100, "{checked} triangles checked");
    }
}

/// The rows of the inverse of a rigid view: eye directions to world.
fn inverse_view_rows(v: &Mat4) -> Mat4 {
    let mut m = [[0.0f32; 4]; 4];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = v[c][r];
        }
    }
    m[3][3] = 1.0;
    m
}

/// A caster between the sun and the slice, outside the bounding sphere,
/// still lands in the cascade's depth range: tall trees and hills toward the
/// sun shadow the slice.
#[test]
fn caster_toward_the_sun_lands_in_depth_range() {
    let terms = daoc_terms();
    let v = view([30_000.0, 20_000.0, -800.0], 2.1, 0.35, true);
    let basis = light_basis(SUN_WORLD);
    let sun = basis[2].map(f64::from);
    let split = (10.0, 737.957);
    let (centre_z, radius) = slice_sphere(terms, split);
    let m = cascade_matrix(&v, terms, split, &basis, 2048);
    let centre_eye = [0.0, 0.0, f64::from(centre_z), 1.0];
    for beyond in [2600.0, 5000.0, f64::from(SUN_MARGIN) - 10.0] {
        let offset =
            apply([-sun[0], -sun[1], -sun[2], 0.0], &v).map(|c| c * (f64::from(radius) + beyond));
        let caster: [f64; 4] = core::array::from_fn(|i| centre_eye[i] + offset[i]);
        let clip = apply(caster, &m);
        assert!(
            (0.0..=1.0).contains(&clip[2]) && clip[0].abs() < 1.0 && clip[1].abs() < 1.0,
            "{beyond} units toward the sun: {clip:?}"
        );
    }
    // The constant bias is 1.5 texels of this cascade in its depth units.
    let bias = caster_depth_bias(&m, 2048);
    let texel = 2.0 * radius / 2048.0;
    let expected = CONSTANT_BIAS_TEXELS * texel / 2.0f32.mul_add(radius, SUN_MARGIN);
    assert!(
        (bias - expected).abs() < expected * 1e-3,
        "{bias} vs {expected}"
    );
    let biased = with_depth_offset(&m, bias);
    let clip = apply(centre_eye, &m);
    let moved = apply(centre_eye, &biased);
    assert!((moved[2] - clip[2] - f64::from(bias)).abs() < 1e-6);
    assert_eq!((moved[0], moved[1]), (clip[0], clip[1]));
}
