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

#[cfg(test)]
mod tests {
    use super::{FIGURE_VS, Od0Tint, figure_vs};
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
}
