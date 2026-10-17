//! DAoC's hardware-skinned character ("figure") vertex shaders.
//!
//! DAoC ships a family of `vs_1_1` shaders for player characters and mobs
//! that skin by palette lookup (`dp4` against `c[a0.x+26..28]`) and then run
//! DAoC's own lighting: one directional sun plus three point lights, with
//! the light constants named in the CTAB (`$sunlight_Direction`,
//! `$sunlight_Diffuse`, `$plight1_Color`, …). Phase 4 moves those light
//! terms to the pixel stage; this table tells the emitter where the terms
//! live and where to capture the per-vertex inputs they need.
//!
//! Fixtures: `dxso/emit/fixtures/daoc_figure_vs_<id>.bin` (this table) and
//! `daoc_figure_ps_<id>.bin` (the `ps_1_x` shaders the game draws with them,
//! identified via the probe traces' per-draw VS+PS pairs).

/// One figure vertex shader, identified by its [`crate::ids::ProgramId`].
///
/// `capture_before` indexes `DxsoProgram::instructions` — the parsed body
/// list the VS emitter iterates (`for inst in &vs.instructions`), which
/// excludes the version token, the CTAB comment, the `dcl` declarations and
/// the `def` constants (those live in `declarations`/`def_constants`). Task 3
/// inserts the per-vertex capture right before
/// `instructions[capture_before]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FigureVs {
    /// `ProgramId::from_tokens(words).raw()` of the shader.
    pub id: u64,
    /// Constant layout: `1` = `$sunlight_Direction` at `c12` (diffuse `c13`,
    /// ambient `c14`, point lights `c15/16/17`, `c18/19/20`, `c21/22/23`);
    /// `2` = everything one register higher (direction `c13`, …, third point
    /// light `c22/23/24`).
    pub layout: u8,
    /// Temp register holding the skinned normal (xyz) the light terms dot
    /// with.
    pub normal_reg: u8,
    /// Temp register holding the light-space position (xyz) the point-light
    /// vectors subtract.
    pub pos_reg: u8,
    /// Index of the first body instruction that reads the sun-direction
    /// register; both `normal_reg` and `pos_reg` hold their final values
    /// here (verified by `capture_before_is_first_sun_read` and the census).
    pub capture_before: u16,
    /// What the VS multiplies the clamped light sum by when it writes `oD0`:
    /// the pixel stage must reproduce the oD0 write exactly, so the capture
    /// ships this factor and `fig_light` applies it after the per-pixel sum.
    pub od0_tint: Od0Tint,
}

/// The modulation the figure VS applies to the clamped light sum when it
/// writes `oD0` (`oD0.rgb = saturate(sum) * <factor>`).
///
/// The census found three shapes across the ten fixtures: a plain clamp, a
/// `$tintColor` constant (`c11` on every tinted fixture) and the D3D9
/// vertex-colour modulation (`D3DCOLORVERTEX` — the mesh's per-vertex
/// diffuse). The factor is per-vertex data the pixel stage cannot recompute
/// from the light math, so the capture ships it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Od0Tint {
    /// `oD0.rgb = saturate(sum)` — no modulation; the capture ships white.
    None,
    /// `oD0.rgb = saturate(sum) * c[N].rgb` — a float-constant tint.
    Const(u8),
    /// `oD0 = saturate(sum) * v[N]` — the vertex colour (rgb; the alpha lane
    /// reaches the PS through `oD0.a`, which the PS reads as `base.a`).
    Input(u8),
}

/// The ten lit figure VS from the `shots-ppl2.txt` census (village, lake,
/// dungeon), sorted by id. Every one skins with `$boneMatrices` rows
/// `c26..` and lights with the `$sunlight_*`/`$plight*` constants; they
/// differ in vertex format (blend weights, tangents, texture coords) and in
/// whether the compiler placed the light constants at `c12` (layout 1) or
/// `c13` (layout 2).
pub(crate) static FIGURE_VS: [FigureVs; 10] = [
    FigureVs {
        id: 0x15ecaea13070e5af,
        layout: 2,
        normal_reg: 4,
        pos_reg: 3,
        capture_before: 48,
        od0_tint: Od0Tint::Const(11),
    },
    FigureVs {
        id: 0x31a064b8fe30b4ed,
        layout: 2,
        normal_reg: 4,
        pos_reg: 3,
        capture_before: 48,
        od0_tint: Od0Tint::None,
    },
    FigureVs {
        id: 0x4c76b4b8b4466a51,
        layout: 1,
        normal_reg: 5,
        pos_reg: 1,
        capture_before: 45,
        od0_tint: Od0Tint::Input(5),
    },
    FigureVs {
        id: 0x8c436dc740ce3674,
        layout: 2,
        normal_reg: 4,
        pos_reg: 3,
        capture_before: 48,
        od0_tint: Od0Tint::Const(11),
    },
    FigureVs {
        id: 0xcbb5aac252c08202,
        layout: 1,
        normal_reg: 2,
        pos_reg: 3,
        capture_before: 43,
        od0_tint: Od0Tint::None,
    },
    FigureVs {
        id: 0xd0e250f335b00f24,
        layout: 1,
        normal_reg: 4,
        pos_reg: 1,
        capture_before: 47,
        od0_tint: Od0Tint::None,
    },
    FigureVs {
        id: 0xd3c4595fec6bd3e2,
        layout: 2,
        normal_reg: 5,
        pos_reg: 2,
        capture_before: 50,
        od0_tint: Od0Tint::Const(11),
    },
    FigureVs {
        id: 0xdcf0475f8ef49250,
        layout: 2,
        normal_reg: 5,
        pos_reg: 2,
        capture_before: 50,
        od0_tint: Od0Tint::None,
    },
    FigureVs {
        id: 0xe389281fb2b88a58,
        layout: 1,
        normal_reg: 4,
        pos_reg: 1,
        capture_before: 46,
        od0_tint: Od0Tint::None,
    },
    FigureVs {
        id: 0xe8b6913dccdf28ed,
        layout: 2,
        normal_reg: 5,
        pos_reg: 2,
        capture_before: 50,
        od0_tint: Od0Tint::Const(11),
    },
];

/// The figure VS table entry for `id`, or `None` for non-figure shaders
/// (terrain, speedtrees, the shadow caster, unlit figures, …).
#[must_use]
pub fn figure_vs(id: u64) -> Option<&'static FigureVs> {
    FIGURE_VS.iter().find(|vs| vs.id == id)
}

/// The `$fogParams` float register of a figure VS with constant `layout`:
/// `c11` for layout 1, `c12` for layout 2. It holds
/// `(start, end, end - start, density)`; `fog.figures` reads `.y` and `.z`.
#[must_use]
pub const fn fog_params_register(layout: u8) -> u8 {
    10 + layout
}

/// The eye row [`figure_eye_row`] yields when the draw's clip transform has
/// no usable camera: a direction (`w = 0`) along light-space −Z, which is
/// DAoC's world up when light space is the Z-down world — a highlight lit
/// from above rather than none or a NaN.
pub const FIGURE_EYE_FALLBACK: [f32; 4] = [0.0, 0.0, -1.0, 0.0];

/// Where the camera sits in the light space a figure VS lights in: the
/// third row (`sp[2]`) of a figure draw's specular uniform.
///
/// `c` is the draw's VS float constants `c0..c6`. Every figure VS (verified
/// on all ten fixtures by `figure_clip_and_light_rows_share_the_skinned_position`)
/// computes `oPos` as `dp4` of the skinned position `r` (with `r.w = 1`) and
/// `c0..c3`, and its light-space position as `dp4` of the same `r` and
/// `c4..c6`. The camera centre is the point every view ray meets, the null
/// vector `e` of the clip rows `c0`, `c1`, `c3` (clip `x = y = w = 0`):
/// - perspective (`e.w != 0`): the centre `e.xyz / e.w` mapped through
///   `c4..c6`, returned as `(eye, 1)`;
/// - orthographic (`e.w == 0`): no centre, `e.xyz` is the view axis; it is
///   turned toward the camera (clip depth `dot(e, c2)` decreasing, the D3D
///   near-to-far order), mapped through the linear part of `c4..c6` and
///   normalized, returned as `(dir, 0)`.
///
/// Rows without a single centre (zero or dependent rows, non-finite values,
/// an orthographic depth row blind to the view axis) yield
/// [`FIGURE_EYE_FALLBACK`]; the result is always finite.
#[must_use]
pub fn figure_eye_row(c: &[[f32; 4]; 7]) -> [f32; 4] {
    let row = |k: usize| c[k].map(f64::from);
    let det3 = |a: [f64; 3], b: [f64; 3], d: [f64; 3]| {
        a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0])
            + a[2] * (b[0] * d[1] - b[1] * d[0])
    };
    let pick = |v: [f64; 4], lanes: [usize; 3]| lanes.map(|l| v[l]);
    let (u, v, w) = (row(0), row(1), row(3));
    // The 4D cross product: `dot(e, t) = det(u, v, w, t)`, so `e` is
    // orthogonal to all three rows (the cofactor expansion along `t`).
    let e = [
        -det3(pick(u, [1, 2, 3]), pick(v, [1, 2, 3]), pick(w, [1, 2, 3])),
        det3(pick(u, [0, 2, 3]), pick(v, [0, 2, 3]), pick(w, [0, 2, 3])),
        -det3(pick(u, [0, 1, 3]), pick(v, [0, 1, 3]), pick(w, [0, 1, 3])),
        det3(pick(u, [0, 1, 2]), pick(v, [0, 1, 2]), pick(w, [0, 1, 2])),
    ];
    let norm = e.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return FIGURE_EYE_FALLBACK;
    }
    let light = |k: usize, p: [f64; 3], w: f64| {
        let r = row(4 + k);
        r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3] * w
    };
    let out = if e[3].abs() > 1e-9 * norm {
        let eye = [e[0] / e[3], e[1] / e[3], e[2] / e[3]];
        let [x, y, z] = [0, 1, 2].map(|k| light(k, eye, 1.0) as f32);
        [x, y, z, 1.0]
    } else {
        let depth = row(2);
        let dz = e[0] * depth[0] + e[1] * depth[1] + e[2] * depth[2];
        if !dz.is_finite() || dz == 0.0 {
            return FIGURE_EYE_FALLBACK;
        }
        let sign = if dz > 0.0 { -1.0 } else { 1.0 };
        let axis = [sign * e[0], sign * e[1], sign * e[2]];
        let dir = [0, 1, 2].map(|k| light(k, axis, 0.0));
        let len = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !len.is_finite() || len == 0.0 {
            return FIGURE_EYE_FALLBACK;
        }
        let [x, y, z] = dir.map(|x| (x / len) as f32);
        [x, y, z, 0.0]
    };
    if out.iter().all(|x| x.is_finite()) {
        out
    } else {
        FIGURE_EYE_FALLBACK
    }
}

#[cfg(test)]
mod tests {
    use super::{FIGURE_EYE_FALLBACK, FIGURE_VS, Od0Tint, figure_eye_row, figure_vs};
    use crate::{
        dxso::{RegKind, parse},
        ids::ProgramId,
    };

    /// `(fixture name, bytes)` for every figure VS fixture.
    const VS_FIXTURES: &[(&str, &[u8])] = &[
        (
            "daoc_figure_vs_15ecaea13070e5af.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_15ecaea13070e5af.bin"),
        ),
        (
            "daoc_figure_vs_31a064b8fe30b4ed.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_31a064b8fe30b4ed.bin"),
        ),
        (
            "daoc_figure_vs_4c76b4b8b4466a51.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_4c76b4b8b4466a51.bin"),
        ),
        (
            "daoc_figure_vs_8c436dc740ce3674.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_8c436dc740ce3674.bin"),
        ),
        (
            "daoc_figure_vs_cbb5aac252c08202.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_cbb5aac252c08202.bin"),
        ),
        (
            "daoc_figure_vs_d0e250f335b00f24.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_d0e250f335b00f24.bin"),
        ),
        (
            "daoc_figure_vs_d3c4595fec6bd3e2.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_d3c4595fec6bd3e2.bin"),
        ),
        (
            "daoc_figure_vs_dcf0475f8ef49250.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_dcf0475f8ef49250.bin"),
        ),
        (
            "daoc_figure_vs_e389281fb2b88a58.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_e389281fb2b88a58.bin"),
        ),
        (
            "daoc_figure_vs_e8b6913dccdf28ed.bin",
            include_bytes!("../dxso/emit/fixtures/daoc_figure_vs_e8b6913dccdf28ed.bin"),
        ),
    ];
    const CASTER_FIXTURE: &[u8] =
        include_bytes!("../dxso/emit/fixtures/daoc_caster_vs_f6e6b85a000b43b8.bin");

    /// Every fixture's `ProgramId` maps to its table entry.
    #[test]
    fn table_ids_match_fixtures() {
        for (name, bytes) in VS_FIXTURES {
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let id = ProgramId::from_tokens(&words).raw();
            assert!(figure_vs(id).is_some(), "{name}: no table entry");
            // The fixture name carries the same id.
            assert!(name.contains(&format!("{id:x}")), "{name} vs {id:x}");
            let program = parse(&words).expect(name);
            assert_eq!(
                program.shader_type,
                crate::dxso::ShaderType::Vertex,
                "{name}"
            );
        }
        assert_eq!(
            VS_FIXTURES.len(),
            FIGURE_VS.len(),
            "table and fixtures diverge"
        );
    }

    /// The 736 B shadow-caster VS (`oD0 = 0.75`) is not a figure shader.
    #[test]
    fn caster_is_not_a_figure() {
        let words: Vec<u32> = CASTER_FIXTURE
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let id = ProgramId::from_tokens(&words).raw();
        assert_eq!(id, 0xf6e6b85a000b43b8);
        assert!(figure_vs(id).is_none());
        // And it is not merely a missing entry: the caster has no CTAB light
        // constants at all, so no layout could describe it.
        let program = parse(&words).expect("caster parses");
        assert!(program.instructions.len() < 40);
    }

    /// `capture_before` is the first body instruction that reads the
    /// sun-direction register (`c12` for layout 1, `c13` for layout 2), and
    /// no earlier instruction does.
    #[test]
    fn capture_before_is_first_sun_read() {
        for (name, bytes) in VS_FIXTURES {
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let id = ProgramId::from_tokens(&words).raw();
            let entry = figure_vs(id).expect(name);
            let program = parse(&words).expect(name);
            let sun = if entry.layout == 1 { 12 } else { 13 };
            let first = program
                .instructions
                .iter()
                .position(|inst| {
                    inst.srcs
                        .iter()
                        .any(|s| s.reg.kind == RegKind::Const && s.reg.index == sun)
                })
                .unwrap_or_else(|| panic!("{name}: sun register c{sun} never read"));
            assert_eq!(first, entry.capture_before as usize, "{name}");
            // The capture point reads the skinned normal, which is what the
            // pixel stage needs.
            let inst = &program.instructions[first];
            assert!(
                inst.srcs
                    .iter()
                    .any(|s| s.reg.kind == RegKind::Temp
                        && s.reg.index == u16::from(entry.normal_reg)),
                "{name}: instruction at capture_before does not read the normal"
            );
        }
    }

    /// The `od0_tint` census: every fixture's `oD0` write must match its
    /// table entry — the pixel stage reproduces the oD0 write, so the table
    /// has to describe it exactly. The census found three shapes: a plain
    /// clamp (`min oD0`), the `$tintColor` constant (`mul oD0.xyz, r1, c11`)
    /// and the D3D9 vertex-colour modulation (`mul oD0, r0, v5`).
    #[test]
    fn od0_tint_matches_the_fixtures() {
        for (name, bytes) in VS_FIXTURES {
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let id = ProgramId::from_tokens(&words).raw();
            let entry = figure_vs(id).expect(name);
            let program = parse(&words).expect(name);
            let writes: Vec<&crate::dxso::Instruction> = program
                .instructions
                .iter()
                .filter(|inst| {
                    inst.dst
                        .as_ref()
                        .is_some_and(|dst| dst.reg.kind == RegKind::AttrOut && dst.reg.index == 0)
                })
                .collect();
            let rgb = writes
                .iter()
                .find(|inst| {
                    inst.dst
                        .as_ref()
                        .is_some_and(|dst| dst.write_mask.0 & 0b0111 == 0b0111)
                })
                .unwrap_or_else(|| panic!("{name}: no rgb oD0 write"));
            let src = rgb.srcs.last().expect("oD0 write has a tint source");
            match (entry.od0_tint, src.reg.kind, src.reg.index) {
                (Od0Tint::None, RegKind::Const, _) => {
                    assert_eq!(rgb.opcode, crate::dxso::Opcode::Min, "{name}");
                }
                (Od0Tint::Const(c), RegKind::Const, index) => {
                    assert_eq!(index, u16::from(c), "{name}");
                    assert_eq!(rgb.opcode, crate::dxso::Opcode::Mul, "{name}");
                }
                (Od0Tint::Input(v), RegKind::Input, index) => {
                    assert_eq!(index, u16::from(v), "{name}");
                    assert_eq!(rgb.opcode, crate::dxso::Opcode::Mul, "{name}");
                }
                (expected, kind, index) => {
                    panic!("{name}: oD0 tint {expected:?} vs fixture write from {kind:?} c{index}")
                }
            }
        }
    }

    type Mat = [[f64; 4]; 4];

    fn mul(a: &Mat, b: &Mat) -> Mat {
        let mut out = [[0.0; 4]; 4];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
            }
        }
        out
    }

    fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn unit3(a: [f64; 3]) -> [f64; 3] {
        let l = dot3(a, a).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    }

    /// `D3DXMatrixLookAtLH` (row vectors: `v_view = v_world · M`) and the
    /// camera's forward axis.
    fn look_at_lh(eye: [f64; 3], at: [f64; 3], up: [f64; 3]) -> (Mat, [f64; 3]) {
        let z = unit3(sub3(at, eye));
        let x = unit3(cross3(up, z));
        let y = cross3(z, x);
        (
            [
                [x[0], y[0], z[0], 0.0],
                [x[1], y[1], z[1], 0.0],
                [x[2], y[2], z[2], 0.0],
                [-dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1.0],
            ],
            z,
        )
    }

    /// `D3DXMatrixPerspectiveFovLH`.
    fn perspective_lh(fovy: f64, aspect: f64, zn: f64, zf: f64) -> Mat {
        let ys = 1.0 / (fovy / 2.0).tan();
        let q = zf / (zf - zn);
        [
            [ys / aspect, 0.0, 0.0, 0.0],
            [0.0, ys, 0.0, 0.0],
            [0.0, 0.0, q, 1.0],
            [0.0, 0.0, -zn * q, 0.0],
        ]
    }

    /// `D3DXMatrixOrthoLH`.
    fn ortho_lh(w: f64, h: f64, zn: f64, zf: f64) -> Mat {
        [
            [2.0 / w, 0.0, 0.0, 0.0],
            [0.0, 2.0 / h, 0.0, 0.0],
            [0.0, 0.0, 1.0 / (zf - zn), 0.0],
            [0.0, 0.0, zn / (zn - zf), 1.0],
        ]
    }

    /// The figure VS constants `c0..c6` as the game uploads them (transposed,
    /// so `dp4 r, c[k]` is column `k`): `c0..c3` the clip transform `clip`,
    /// `c4..c6` the light-space affine `diag(scale) + translate`.
    fn constants(clip: &Mat, scale: [f64; 3], translate: [f64; 3]) -> [[f32; 4]; 7] {
        let mut c = [[0.0f32; 4]; 7];
        for (k, row) in c.iter_mut().take(4).enumerate() {
            for (r, lane) in row.iter_mut().enumerate() {
                *lane = clip[r][k] as f32;
            }
        }
        for k in 0..3 {
            c[4 + k][k] = scale[k] as f32;
            c[4 + k][3] = translate[k] as f32;
        }
        c
    }

    /// A perspective camera: the row is the camera centre in light space,
    /// `(scale * eye + translate, 1)`, for a camera near the origin and one
    /// at DAoC zone-scale coordinates.
    #[test]
    fn figure_eye_recovers_a_perspective_camera() {
        let scale = [1.5, 1.5, 1.5];
        let translate = [-120.0, 40.0, 7.5];
        for (eye, at) in [
            ([12.0, -3.5, 40.0], [0.0, 0.0, 0.0]),
            (
                [35_210.0, 41_870.5, -1_480.0],
                [35_190.0, 41_900.0, -1_470.0],
            ),
        ] {
            let (view, _) = look_at_lh(eye, at, [0.0, 0.0, -1.0]);
            let clip = mul(&view, &perspective_lh(1.0, 4.0 / 3.0, 1.0, 20_000.0));
            let row = figure_eye_row(&constants(&clip, scale, translate));
            assert_eq!(row[3], 1.0, "a perspective camera is a point: {row:?}");
            for k in 0..3 {
                let want = scale[k] * eye[k] + translate[k];
                let tol = 2e-6 * (1.0 + want.abs());
                assert!(
                    (f64::from(row[k]) - want).abs() <= tol,
                    "lane {k}: {} vs {want} (eye {eye:?})",
                    row[k]
                );
            }
        }
    }

    /// An orthographic camera has no centre: the row is the unit direction
    /// toward the camera (`-forward`) through the light-space affine's
    /// linear part, with `w = 0`.
    #[test]
    fn figure_eye_orthographic_gives_the_direction_to_the_camera() {
        let eye = [200.0, -80.0, -50.0];
        let (view, forward) = look_at_lh(eye, [190.0, -60.0, -40.0], [0.0, 0.0, -1.0]);
        let clip = mul(&view, &ortho_lh(64.0, 48.0, 1.0, 500.0));
        let scale = [2.0, 1.0, 0.5];
        let row = figure_eye_row(&constants(&clip, scale, [3.0, 4.0, 5.0]));
        assert_eq!(row[3], 0.0, "{row:?}");
        let want = unit3([
            -forward[0] * scale[0],
            -forward[1] * scale[1],
            -forward[2] * scale[2],
        ]);
        for k in 0..3 {
            assert!(
                (f64::from(row[k]) - want[k]).abs() < 1e-5,
                "lane {k}: {row:?} vs {want:?}"
            );
        }
    }

    /// A clip transform with no single camera centre (zero, rank-deficient or
    /// non-finite rows) yields the fixed fallback, never a NaN.
    #[test]
    fn figure_eye_singular_falls_back() {
        let zero = [[0.0f32; 4]; 7];
        assert_eq!(figure_eye_row(&zero), FIGURE_EYE_FALLBACK);
        let mut rank2 = zero;
        rank2[0] = [1.0, 2.0, 3.0, 4.0];
        rank2[1] = [2.0, 4.0, 6.0, 8.0];
        rank2[3] = [0.0, 0.0, 1.0, 0.0];
        rank2[4] = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(figure_eye_row(&rank2), FIGURE_EYE_FALLBACK);
        let (view, _) = look_at_lh([1.0, 2.0, 3.0], [0.0; 3], [0.0, 0.0, -1.0]);
        let clip = mul(&view, &perspective_lh(1.0, 1.0, 1.0, 100.0));
        let mut nan = constants(&clip, [1.0; 3], [0.0; 3]);
        nan[1][2] = f32::NAN;
        assert_eq!(figure_eye_row(&nan), FIGURE_EYE_FALLBACK);
        let mut inf = constants(&clip, [1.0; 3], [0.0; 3]);
        inf[5][3] = f32::INFINITY;
        assert_eq!(figure_eye_row(&inf), FIGURE_EYE_FALLBACK);
        // An orthographic transform whose depth row ignores the view axis
        // cannot say which way the camera is.
        let ortho = mul(
            &look_at_lh([0.0; 3], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]).0,
            &ortho_lh(10.0, 10.0, 1.0, 100.0),
        );
        let mut flat = constants(&ortho, [1.0; 3], [0.0; 3]);
        flat[2] = [0.0, 0.0, 0.0, 0.5];
        assert_eq!(figure_eye_row(&flat), FIGURE_EYE_FALLBACK);
        assert!(FIGURE_EYE_FALLBACK.iter().all(|v| v.is_finite()));
    }
}
