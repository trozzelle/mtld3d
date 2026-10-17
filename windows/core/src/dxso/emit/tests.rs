//! Emitter tests.
//!
//! We don't pin the exact output (formatting churn would break tests
//! without catching real bugs) — instead check that the generated MSL
//! contains the structural elements each instruction should produce.
//!
//! Each test emits VS and PS independently (matching the per-stage API) and
//! concatenates the two strings into one check target.

use mtld3d_shared::mtl::{
    PS_BOOL_CONST_SLOT, PS_DRAW_SLOT, PS_INT_CONST_SLOT, PS_LOD_BIAS_SLOT, VS_FLOAT_CONST_SLOT,
    VS_INT_CONST_SLOT, VS_LOD_SLOT, VS_POS_FIXUP_SLOT,
};
use mtld3d_types::{
    D3DFOG_LINEAR, D3DTA_DIFFUSE, D3DTA_SPECULAR, D3DTOP_DISABLE, D3DTOP_SELECTARG1,
};

use super::{
    POS_FIXUP_MSL, VariantFlags, VariantKey, VsSamplerKinds, declared_ps_samplers,
    emit_ps_programmable, emit_vs_programmable, emit_vs_programmable_named,
};
use crate::{
    dxso::{ir::TextureType, parser::parse},
    shader_key::ff_key_hash,
};

/// D3D enum constant at the key's narrow width.
fn narrow(v: u32) -> u8 {
    u8::try_from(v).expect("D3D9 fixed-function enum value ≤ u8::MAX")
}

const VS_HEADER: u32 = 0xFFFE_0200;
const PS_HEADER: u32 = 0xFFFF_0200;
const VS3_HEADER: u32 = 0xFFFE_0300;
const PS3_HEADER: u32 = 0xFFFF_0300;
const END_TOKEN: u32 = 0x0000_FFFF;
const SWIZ_IDENTITY: u8 = 0xE4;
const SWIZ_BGRA: u8 = 0xC6;
const SWIZ_BBBB: u8 = 0xAA;

#[test]
fn fetch4_emits_one_native_gather_and_keeps_ordinary_shaders_unchanged() {
    let ps = single_sampler_ps(0x9000_0000);
    let ordinary = emit_ps_programmable(&ps, VariantKey::default()).expect("ordinary sample");
    assert!(!ordinary.contains(".gather("));
    for (depth, alpha) in [(false, false), (false, true), (true, false)] {
        let variant = VariantKey {
            fetch4_mask: 1,
            fetch4_alpha_mask: u16::from(alpha),
            depth_sampler_mask: u16::from(depth),
            depth_fetch_mask: u16::from(depth),
            ..VariantKey::default()
        };
        let source = emit_ps_programmable(&ps, variant).expect("gather sample");
        assert_eq!(source.matches(".gather(").count(), 1);
        assert!(!source.contains(".sample("));
        assert!(!source.contains(".sample_compare("));
        assert!(!source.contains("lod_bias [[buffer"));
        assert!(source.contains(".zxyw"));
        assert_eq!(source.contains("component::w"), alpha);
        metal_compile_or_fail(&source);
    }
}

#[test]
fn fetch4_ignores_instruction_lod_and_preserves_projection_and_sm1_slots() {
    for opcode in [
        0x0300_0042,
        0x0301_0042,
        0x0302_0042,
        0x0300_005f,
        0x0500_005d,
    ] {
        let mut code = vec![
            PS3_HEADER,
            0x0200_001f,
            0x9000_0000,
            0xa00f_0803,
            0x0200_001f,
            0x8000_0005,
            0x900f_0000,
            opcode,
            0x800f_0000,
            0x90e4_0000,
            0xa0e4_0803,
        ];
        if opcode == 0x0500_005d {
            code.extend_from_slice(&[0x80e4_0001, 0x80e4_0002]);
        }
        code.extend_from_slice(&[0x0200_0001, 0x800f_0800, 0x80e4_0000, END_TOKEN]);
        let ps = parse(&code).expect("texture instruction");
        let source = emit_ps_programmable(
            &ps,
            VariantKey {
                fetch4_mask: 8,
                ..VariantKey::default()
            },
        )
        .expect("gather");
        assert!(source.contains("s3.gather(samp3,"), "{source}");
        assert!(!source.contains(".sample("));
        assert!(!source.contains("gradient2d("));
        assert!(!source.contains("level("));
        assert!(!source.contains("bias("));
        if opcode == 0x0301_0042 {
            assert!(
                source.contains("/ (in.texcoord0).w"),
                "projective divide survives"
            );
        }
        metal_compile_or_fail(&source);
    }
    for minor in 1..=3 {
        let ps = parse(&[
            0xffff_0100 | minor,
            0x0000_0042,
            0xb00f_0003,
            0x0000_0001,
            0x800f_0000,
            0xb0e4_0003,
            END_TOKEN,
        ])
        .expect("SM1 texture");
        let source = emit_ps_programmable(
            &ps,
            VariantKey {
                fetch4_mask: 8,
                tt_projected_mask: 8,
                ..VariantKey::default()
            },
        )
        .expect("SM1 gather");
        assert!(source.contains("s3.gather(samp3,"));
        assert!(!source.contains(".sample("));
        metal_compile_or_fail(&source);
    }
}

const TYPE_TEMP: u32 = 0;
const TYPE_INPUT: u32 = 1;
const TYPE_CONST: u32 = 2;
const TYPE_ADDR: u32 = 3;
const TYPE_RASTOUT: u32 = 4;
const TYPE_TEXCOORDOUT: u32 = 6;
const TYPE_COLOROUT: u32 = 8;
const TYPE_OUTPUT: u32 = 11;

/// `dcl_<usage>_<index>` token: bit 31 set + usage bits 0..4 + index bits 16..19.
fn dcl_usage_token(usage: u8, index: u8) -> u32 {
    0x8000_0000 | (u32::from(usage) & 0x1F) | ((u32::from(index) & 0xF) << 16)
}

const DCL_POSITION: u8 = 0;
const DCL_TEXCOORD: u8 = 5;
const DCL_COLOR: u8 = 10;

const OP_MOV: u16 = 1;
const OP_ADD: u16 = 2;
const OP_DP3: u16 = 8;
const OP_DP4: u16 = 9;
const OP_SLT: u16 = 12;
const OP_SGE: u16 = 13;
const OP_M3X4: u16 = 22;
const OP_M3X2: u16 = 24;
const OP_DCL: u16 = 31;
const OP_CRS: u16 = 33;
const OP_SGN: u16 = 34;
const OP_MOVA: u16 = 46;
const OP_EXPP: u16 = 78;
const OP_LOGP: u16 = 79;
const OP_DEF: u16 = 81;
const OP_CMP: u16 = 88;
const OP_LIT: u16 = 16;
const OP_NRM: u16 = 36;
const OP_DST: u16 = 17;
const OP_CND: u16 = 80;
const OP_LOOP: u16 = 27;
const OP_ENDLOOP: u16 = 29;
const OP_REP: u16 = 38;
const OP_ENDREP: u16 = 39;
const OP_IF: u16 = 40;
const OP_IFC: u16 = 41;
const OP_ELSE: u16 = 42;
const OP_ENDIF: u16 = 43;
const OP_BREAK: u16 = 44;
const OP_BREAKC: u16 = 45;
const OP_DEFI: u16 = 48;
const OP_SETP: u16 = 94;

const TYPE_PREDICATE: u32 = 19;
const TYPE_LABEL: u32 = 18;
const OP_CALL: u16 = 25;
const OP_CALLNZ: u16 = 26;
const OP_RET: u16 = 28;
const OP_LABEL: u16 = 30;

const TYPE_CONSTINT: u32 = 7;
const TYPE_LOOP: u32 = 15;
const OP_DP2ADD: u16 = 90;
const OP_DSX: u16 = 91;
const OP_DSY: u16 = 92;
const OP_TEXLDD: u16 = 93;
const OP_TEXLDL: u16 = 95;
const OP_TEXKILL: u16 = 65;

/// `.xxxx` swizzle (replicate component 0).
const SWIZ_XXXX: u8 = 0x00;

fn reg_bits(reg_type: u32, index: u16) -> u32 {
    let low = reg_type & 0x7;
    let high = (reg_type >> 3) & 0x3;
    // Bit 31 marks every dst/src parameter token in real D3D9 bytecode (all
    // shader models). SM1 operand counting keys off it, so the helpers must
    // set it to model the stream a shader compiler actually emits.
    0x8000_0000 | (low << 28) | (high << 11) | u32::from(index)
}

fn opcode_token(opcode: u16, token_count: u32) -> u32 {
    u32::from(opcode) | (token_count << 24)
}

fn dst_token(reg_type: u32, index: u16, write_mask: u8, saturate: bool) -> u32 {
    let mut t = reg_bits(reg_type, index);
    t |= (u32::from(write_mask) & 0xF) << 16;
    if saturate {
        t |= 1 << 20;
    }
    t
}

fn src_token(reg_type: u32, index: u16, swizzle: u8, modifier: u8) -> u32 {
    let mut t = reg_bits(reg_type, index);
    t |= u32::from(swizzle) << 16;
    t |= (u32::from(modifier) & 0xF) << 24;
    t
}

fn trivial_passthrough_vs() -> Vec<u32> {
    // dcl_position v0; mov oPos, v0;
    vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]
}

fn red_constant_ps() -> Vec<u32> {
    // def c0, 1, 0, 0, 1; mov oC0, c0;
    vec![
        PS_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]
}

fn emit_pair_for_tests(vs_bc: &[u32], ps_bc: &[u32], variant: VariantKey) -> String {
    let vs = parse(vs_bc).expect("VS parse");
    let ps = parse(ps_bc).expect("PS parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS");
    let ps_msl = emit_ps_programmable(&ps, variant).expect("emit PS");
    format!("{vs_msl}\n{ps_msl}")
}

#[test]
fn minimal_vs_plus_ps_emits_valid_msl_skeleton() {
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );

    assert!(
        msl.contains("#include <metal_stdlib>"),
        "missing stdlib:\n{msl}"
    );
    assert!(msl.contains("struct VertexIn"), "no VertexIn:\n{msl}");
    assert!(
        msl.contains("float4 v0 [[attribute(0)]]"),
        "no VertexIn::v0:\n{msl}"
    );
    assert!(msl.contains("struct Varyings"), "no Varyings:\n{msl}");
    assert!(
        msl.contains("float4 position [[position, invariant]]"),
        "no position field:\n{msl}"
    );
    assert!(
        msl.contains(&format!(
            "constant float4 *vs_c [[buffer({VS_FLOAT_CONST_SLOT})]]"
        )),
        "no VS constants slot:\n{msl}"
    );
    assert!(
        msl.contains("constant float4 *ps_c [[buffer(15)]]"),
        "no PS constants slot:\n{msl}"
    );
    assert!(
        msl.contains("vertex Varyings mtld3d_vs("),
        "no VS entry:\n{msl}"
    );
    assert!(
        msl.contains("fragment float4 mtld3d_ps("),
        "no PS entry:\n{msl}"
    );
    assert!(
        msl.contains("out.position = in.v0;"),
        "mov to oPos missing:\n{msl}"
    );
    assert!(
        msl.contains("float4 c0 = float4(1.0, 0.0, 0.0, 1.0);"),
        "def c0 missing:\n{msl}"
    );
    assert!(msl.contains("oC0 = c0;"), "mov oC0, c0 missing:\n{msl}");
}

#[test]
fn temporaries_start_at_zero() {
    // A temporary read before any write must not pick up what the GPU
    // register last held; both stages zero the register file.
    let vs = emit_vs_programmable(&parse(&trivial_passthrough_vs()).expect("vs parse"))
        .expect("emit vs");
    let ps = emit_ps_programmable(
        &parse(&red_constant_ps()).expect("ps parse"),
        VariantKey::default(),
    )
    .expect("emit ps");
    for msl in [&vs, &ps] {
        assert!(
            msl.contains("float4 r[32] = {};"),
            "temporaries must be zero-initialised:\n{msl}"
        );
        metal_compile_or_fail(msl);
    }
}

#[test]
fn varyings_put_texcoord_before_color() {
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );

    let tc0 = msl.find("texcoord0").expect("texcoord0 missing");
    let color0 = msl.find("color0").expect("color0 missing");
    assert!(
        tc0 < color0,
        "texcoord should precede color in Varyings to avoid Metal compiler crash"
    );
}

#[test]
fn emit_vs_1_1_real_bytecode_without_length_field() {
    // A literal vs_1_1 stream with no instruction-length fields. Real SM1
    // opcode tokens carry none, so the parser counts operands by the bit-31
    // run; each `mov` must therefore resolve its sources rather than parse
    // with zero (which would leave the emitter reading a missing `srcs[0]`).
    // This must translate cleanly: oPos → position, oD0 → color varying.
    let bc = vec![
        0xFFFE_0101, // vs_1_1
        0x0000_001F,
        0x8000_0000,
        0x900F_0000, // dcl_position v0
        0x0000_0001,
        0xC00F_0000,
        0x90E4_0000, // mov oPos, v0
        0x0000_0001,
        0xD00F_0000,
        0xA0E4_0000, // mov oD0, c0
        0x0000_FFFF, // end
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("vs_1_1 emit");
    assert!(
        msl.contains("out.position = in.v0;"),
        "mov oPos missing:\n{msl}"
    );
    assert!(
        msl.contains("out.color0 = vs_c[0];") || msl.contains("color0 = vs_c[0]"),
        "mov oD0, c0 should write color0 from constant 0:\n{msl}"
    );
}

#[test]
fn swizzle_and_write_mask_emit_correctly() {
    // vs_2_0 { dcl_position v0; mov r0.xy, v0.zwxy; mov oPos, r0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0b0011, false),     // .xy mask
        src_token(TYPE_INPUT, 0, 0b01_00_11_10, 0), // .zwxy
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());

    // r[0].xy = (in.v0).zwxy.xy;
    assert!(
        msl.contains("r[0].xy = (in.v0).zwxy.xy;") || msl.contains("r[0].xy = ((in.v0).zwxy).xy;"),
        "swizzle + write mask incorrect:\n{msl}"
    );
}

#[test]
fn saturate_wraps_in_saturate_call() {
    // vs_2_0 { dcl_position v0; mov_sat r0, v0; mov oPos, r0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, true), // saturate
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());

    assert!(
        msl.contains("r[0] = saturate(in.v0);"),
        "saturate missing:\n{msl}"
    );
}

#[test]
fn add_with_negate_modifier() {
    // vs_2_0 { dcl_position v0; add r0, v0, -c0; mov oPos, r0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_ADD, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 1), // modifier=neg
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());

    assert!(
        msl.contains("r[0] = (in.v0 + (-vs_c[0]));"),
        "add with neg modifier incorrect:\n{msl}"
    );
}

/// Build a src token that also carries a relative-addressing sub-token in the following u32.
///
/// The returned Vec contains two entries: the src token (with bit 13 set)
/// and the rel-addr token.
fn src_token_rel(
    reg_type: u32,
    index: u16,
    swizzle: u8,
    modifier: u8,
    rel_reg_type: u32,
    rel_index: u16,
    rel_swizzle: u8,
) -> [u32; 2] {
    let mut src = reg_bits(reg_type, index);
    src |= u32::from(swizzle) << 16;
    src |= (u32::from(modifier) & 0xF) << 24;
    src |= 1 << 13; // rel-addr flag
    let mut rel = reg_bits(rel_reg_type, rel_index);
    rel |= u32::from(rel_swizzle) << 16;
    [src, rel]
}

#[test]
fn mova_writes_address_register() {
    // vs_2_0 { dcl_position v0; mova a0, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("int4 a = int4(0);"),
        "address register declaration missing:\n{msl}"
    );
    assert!(
        msl.contains("a = int4(round(in.v0));"),
        "mova didn't emit round-to-int4 write:\n{msl}"
    );
}

#[test]
fn mova_respects_write_mask() {
    // vs_2_0 { dcl_position v0; mova a0.xy, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0b0011, false), // .xy mask
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("a.xy = (int4(round(in.v0))).xy;"),
        "mova write mask not honored:\n{msl}"
    );
}

#[test]
fn predicated_mova_preserves_false_address_components() {
    // vs_3_0 {
    //   dcl_position v0; dcl_position o0;
    //   def c0, 0, 1, 0, 1; def c1, 0.5, 0.5, 0.5, 0.5;
    //   def c2, 2, 2, 2, 2; def c3, 9, 9, 9, 9;
    //   setp_lt p0, c0, c1; mova a0, c2; (p0) mova a0, c3;
    //   mov o0, v0;
    // }
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24);
    let predicated_mova_token = u32::from(OP_MOVA) | (1u32 << 28) | (3u32 << 24);
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 1, 0xF, false),
        f32::to_bits(0.5),
        f32::to_bits(0.5),
        f32::to_bits(0.5),
        f32::to_bits(0.5),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 2, 0xF, false),
        f32::to_bits(2.0),
        f32::to_bits(2.0),
        f32::to_bits(2.0),
        f32::to_bits(2.0),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 3, 0xF, false),
        f32::to_bits(9.0),
        f32::to_bits(9.0),
        f32::to_bits(9.0),
        f32::to_bits(9.0),
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_CONST, 2, SWIZ_IDENTITY, 0),
        predicated_mova_token,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_PREDICATE, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 3, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    assert_eq!(bc.len(), 46, "keep the issue proof's exact bytecode words");
    assert!(
        vs.instructions[2].predicate.is_some(),
        "parser must preserve the mova predicate operand"
    );
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("a = select(a, int4(round(c3)), p0);"),
        "predicated mova must retain a0.yw when p0.yw is false:\n{vs_msl}"
    );
    metal_compile_or_fail(&vs_msl);
}

#[test]
fn predicated_mova_maps_replicated_negated_partial_masks() {
    // vs_3_0 {
    //   dcl_position v0; dcl_position o0; setp_lt p0, c0, c1;
    //   mova a0, c2; (p0.zzzz) mova a0.yw, c3;
    //   (!p0) mova a0.xz, c0; mov o0, v0;
    // }
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24);
    let predicated_mova_token = u32::from(OP_MOVA) | (1u32 << 28) | (3u32 << 24);
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_CONST, 2, SWIZ_IDENTITY, 0),
        predicated_mova_token,
        dst_token(TYPE_ADDR, 0, 0b1010, false),
        src_token(TYPE_PREDICATE, 0, 0xAA /* .zzzz */, 0),
        src_token(TYPE_CONST, 3, SWIZ_IDENTITY, 0),
        predicated_mova_token,
        dst_token(TYPE_ADDR, 0, 0b0101, false),
        src_token(TYPE_PREDICATE, 0, SWIZ_IDENTITY, 13 /* logical NOT */),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("a.yw = select(a.yw, (int4(round(vs_c[3]))).yw, p0.zz);"),
        "replicate predicate must cover only address components in the write mask:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("a.xz = select(a.xz, (int4(round(vs_c[0]))).xz, !(p0.xz));"),
        "predicate NOT must invert each address component in the write mask:\n{vs_msl}"
    );
    metal_compile_or_fail(&vs_msl);
}

#[test]
fn reading_addr_register_casts_to_float4() {
    // vs_2_0 { dcl_position v0; mova a0, v0; mov r0, a0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("r[0] = float4(a);"),
        "reading a0 didn't widen to float4:\n{msl}"
    );
}

#[test]
fn relative_addressed_const_read_uses_a() {
    // vs_2_0 { dcl_position v0; mova a0, v0; mov r0, c[a0.x + 5]; mov oPos, r0; }
    let mut bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        // A rel-addr src is two tokens, so the opcode token-count reflects
        // that: dst (1) + src (2) = 3 operand tokens.
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        5,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("r[0] = vs_c[a.x + 5];"),
        "relative-addressed const read not emitted correctly:\n{msl}"
    );
    let vs = parse(&bc).expect("VS parse");
    assert!(
        vs.uses_relative_const_addressing(),
        "rel-addr on const must be detected — the draw path gates \
         the full-constant-buffer upload on this flag"
    );
}

#[test]
fn relative_addressed_const_read_overlays_def_constants() {
    // def c2, 0.25, 0.5, 0.75, 1.0; mova a0, v0; mov r0, c[a0.x + 0]; mov oPos, r0
    // The relative read must see the `def`'d c2 (which the app never uploads),
    // not the empty uniform slot — so it routes through the overlay helper.
    let mut bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 2, 0xF, false),
        f32::to_bits(0.25),
        f32::to_bits(0.5),
        f32::to_bits(0.75),
        f32::to_bits(1.0),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        0,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("mtld3d_const_rel(a.x + 0, vs_c)"),
        "rel-addr read with def constants must route through the overlay:\n{msl}"
    );
    assert!(
        msl.contains("case 2: return float4(0.25"),
        "overlay helper must carry the def constant value:\n{msl}"
    );
    assert!(
        !msl.contains("r[0] = vs_c[a.x + 0];"),
        "rel-addr read must NOT bypass the overlay when defs exist:\n{msl}"
    );
}

#[test]
fn ps_relative_const_addressing_emits_a_indexed_buffer() {
    // `load_src` emits `ps_c[a.<comp> + N]` when a const source carries a
    // rel-addr operand, mirroring the VS path, so the PS prologue must
    // declare `a` or the MSL fails to compile. SM2.x PS does not permit
    // relative const addressing, so this construct is synthetic — but SM3
    // PS does, and the emission shape is pinned here so the SM3 PS path
    // rests on a working SM2 one.
    //
    // ps_2_0 { dcl t0; mov r0, c[t0.x + 5]; mov oC0, r0; }
    let mut bc = vec![
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x8000_0000, // POSITION usage (structural only on PS 2.0)
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        5,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let ps = parse(&bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        ps_msl.contains("int4 a = int4(0);"),
        "PS prologue must declare `a` so rel-addr `ps_c[a.<comp> + N]` compiles:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("ps_c[a.x + 5]"),
        "PS rel-addr emission shape:\n{ps_msl}"
    );
}

#[test]
fn ps_rel_addr_emission_has_preceding_a_declaration() {
    // Whenever `ps_c[a.` appears in emitted PS MSL, the `int4 a`
    // declaration must precede it in the same function. Defends against
    // future edits removing the declaration without spotting the rel-addr
    // emit path in `load_src`.
    let mut bc = vec![
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        7,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let ps = parse(&bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    let rel_pos = ps_msl
        .find("ps_c[a.")
        .expect("test bytecode must produce ps_c[a.<...>]");
    let a_decl_pos = ps_msl
        .find("int4 a")
        .expect("`int4 a` declaration missing in PS function");
    assert!(
        a_decl_pos < rel_pos,
        "`int4 a` must precede ps_c[a.<...>] usage:\n{ps_msl}"
    );
}

#[test]
fn uses_relative_const_addressing_is_false_for_static_reads() {
    // vs_2_0 { dcl_position v0; mov r0, c[5]; mov oPos, r0; }
    // Same program shape as above but with no rel-addr.
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 5, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS parse");
    assert!(!vs.uses_relative_const_addressing());
    // Plain static read still reports the right max so the fast path
    // keeps the short CB upload.
    assert_eq!(vs.max_const_reg(), Some(5));
}

#[test]
fn relative_addressed_const_read_inside_a_call_is_detected() {
    // vs_3_0 {
    //   dcl_position v0;
    //   dcl_position oT0;
    //   mova a0, v0;
    //   call l0;
    //   ret;
    //   label l0;
    //     mov r0, c[a0.x + 5];
    //     mov oT0, r0;
    //   ret;
    // }
    // The only relative read sits in the subroutine, which `call` inline-expands
    // into the emitted function, so the flag the draw path gates the
    // full-constant-buffer upload on has to see it.
    let mut bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_CALL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        5,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ]);
    let vs = parse(&bc).expect("VS3 parse");
    assert!(
        vs.uses_relative_const_addressing(),
        "a rel-addr const read reached only through `call` must set the flag"
    );
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("r[0] = vs_c[a.x + 5];"),
        "the inlined subroutine body must carry the relative read:\n{vs_msl}"
    );
}

#[test]
fn call_only_relative_read_emits_the_def_overlay_helper() {
    // vs_3_0 {
    //   def c2, 0.25, 0.5, 0.75, 1.0;
    //   dcl_position v0;
    //   dcl_position oT0;
    //   mova a0, v0;
    //   call l0;
    //   ret;
    //   label l0;
    //     mov r0, c[a0.x + 0];
    //     mov oT0, r0;
    //   ret;
    // }
    // `load_src` routes every rel-addr read through `mtld3d_const_rel` as soon as
    // the shader declares any `def` constant, so the helper's own emission gate
    // must walk the same instructions or the MSL calls a function it never
    // defines.
    let mut bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 2, 0xF, false),
        f32::to_bits(0.25),
        f32::to_bits(0.5),
        f32::to_bits(0.75),
        f32::to_bits(1.0),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_CALL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        0,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ]);
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("mtld3d_const_rel(a.x + 0, vs_c)"),
        "the inlined rel-addr read must route through the overlay:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("inline float4 mtld3d_const_rel(int idx, constant float4 *cb)"),
        "the overlay helper the inlined body calls must be defined:\n{vs_msl}"
    );
}

#[test]
fn cmp_emits_select_on_ge_zero() {
    // vs_2_0 { dcl_position v0; cmp r0, v0, c0, c1; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_CMP, 4),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // MSL select(s2, s1, s0 >= 0): cond=true returns s1 (=vs_c[0]).
    assert!(msl.contains("select("), "cmp must emit select(): {msl}");
    assert!(
        msl.contains(">= float4(0.0)"),
        "cmp condition missing: {msl}"
    );
}

#[test]
fn slt_emits_step_complement() {
    // vs_2_0 { dcl_position v0; slt r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_SLT, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // slt s0, s1 → (s0 < s1) ? 1 : 0 = 1 - step(s1, s0).
    assert!(
        msl.contains("(float4(1.0) - step(vs_c[0], in.v0))"),
        "slt emission: {msl}"
    );
}

#[test]
fn sge_emits_step() {
    // vs_2_0 { dcl_position v0; sge r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_SGE, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // sge s0, s1 → (s0 >= s1) ? 1 : 0 = step(s1, s0).
    assert!(msl.contains("step(vs_c[0], in.v0)"), "sge emission: {msl}");
}

#[test]
fn m3x2_emits_two_dp3s() {
    // vs_2_0 { dcl_position v0; m3x2 r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_M3X2, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // Two dp3 calls against vs_c[0] and vs_c[1], remaining lanes zeroed.
    // `.xyz` swizzle on both operands matches the DP3 lowering shape.
    assert!(msl.contains("dot((in.v0).xyz, (vs_c[0]).xyz)"), "{msl}");
    assert!(msl.contains("dot((in.v0).xyz, (vs_c[1]).xyz)"), "{msl}");
    // rows=2 pads with two 0.0 lanes.
    assert!(msl.contains("float4("), "{msl}");
}

#[test]
fn m3x4_emits_four_dp3s() {
    // vs_2_0 { dcl_position v0; m3x4 r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_M3X4, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // Four plain dp3 calls against rows vs_c[0..3], `.xyz` swizzle on both.
    for i in 0..4 {
        let expected = format!("dot((in.v0).xyz, (vs_c[{i}]).xyz)");
        assert!(msl.contains(&expected), "{msl}");
    }
}

#[test]
fn ps2_vreg_input_maps_to_color_not_position() {
    // PS 2.0 `dcl v0` encodes usage=0 (POSITION) in its DCL token — the
    // usage field is structural-only, the real semantic comes from the
    // register kind. A PS 2.0 v-reg read must resolve to `in.color0`
    // (interpolated vertex color), not Metal's `in.position` (fragment
    // screen coord): a composite shader that uses `v0.zzzz` / `v0.wwww`
    // as scene↔bloom LERP weight and bloom-squared gain would otherwise
    // read clip-space Z/W and blend a gradient across the whole screen.
    //
    // Mirrors the structure of a WoW composite PS:
    //   dcl t0; dcl t1; dcl v0;
    //   def c0, 1, 0, 0, 0;
    //   mov oC0, v0;  // write the input color straight through
    let bc = vec![
        PS_HEADER,
        // def c0, 1, 0, 0, 0
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        // dcl t0 (usage token 0x80000000 = POSITION, structural only on PS 2.0)
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        // dcl t1
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 1, 0xF, false),
        // dcl v0 (same POSITION-encoded usage; should still resolve to color0)
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        // mov oC0, v0
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &bc, VariantKey::default());
    assert!(
        msl.contains("in.color0"),
        "PS v0 read should resolve to in.color0:\n{msl}"
    );
    assert!(
        !msl.contains("oC0 = in.position"),
        "PS v0 must map to a color input, not in.position:\n{msl}"
    );
}

#[test]
fn programmable_ps_emits_fog_blend_when_variant_fog_mode_set() {
    let variant = VariantKey {
        linked_input_mask: 0,
        alpha_func: 0,
        fog_mode: narrow(D3DFOG_LINEAR),
        fog_table_mode: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        fetch4_mask: 0,
        fetch4_alpha_mask: 0,
        raw_depth_red_mask: 0,
        sample_mask: 0,
        volume_sampler_mask: 0,
        cube_sampler_mask: 0,
        tt_projected_mask: 0,
        color_out_mask: 0,
        flags: VariantFlags::empty(),
        ppl_mask: 0,
        ppl_directional_mask: 0,
        ppl_model: 0,
        ppl_knee: 0,
        ppl_figure: 0,
        reserved: 0,
    };
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &red_constant_ps(), variant);
    assert!(
        msl.contains("constant float4 *fog_data [[buffer(13)]]"),
        "programmable PS must bind fog_data on slot 13 when fog_mode != 0:\n{msl}"
    );
    assert!(
        msl.contains("mix(fog_data[0].rgb, oC0.rgb, saturate(in.fog.x))"),
        "programmable PS must blend fog color with oC0:\n{msl}"
    );
}

#[test]
fn programmable_ps_omits_fog_blend_when_variant_fog_mode_zero() {
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );
    assert!(!msl.contains("fog_data"), "{msl}");
    assert!(!msl.contains("in.fog.x"), "{msl}");
}

#[test]
fn vs_without_ofog_write_falls_back_to_output_specular_alpha() {
    // A VS that never writes oFog sources the fog factor from the OUTPUT
    // specular alpha — per the D3D9 spec the fallback is 1 - oD1.a.
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );
    assert!(
        msl.contains("out.fog = float4(out.color1.w);"),
        "VS without an oFog write must fall back to the output specular alpha:\n{msl}"
    );
}

#[test]
fn vs_writing_ofog_keeps_its_value() {
    // dcl_position v0; mov oPos, v0; mov oFog, v0 — RastOut index 1 is oFog.
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 1, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS parse");
    let msl = emit_vs_programmable(&vs).expect("emit");
    assert!(
        !msl.contains("out.fog = float4(out.color1.w);"),
        "a fog-writing VS must not emit the specular-alpha fallback:\n{msl}"
    );
    assert!(msl.contains("out.fog = "), "{msl}");
}

#[test]
fn crs_emits_cross_product() {
    // vs_2_0 { dcl_position v0; crs r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_CRS, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("cross((in.v0).xyz, (vs_c[0]).xyz)"),
        "crs must emit cross() on .xyz operands:\n{msl}"
    );
}

#[test]
fn sgn_emits_sign_builtin() {
    // vs_2_0 { dcl_position v0; sgn r0, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_SGN, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("r[0] = sign(in.v0);"),
        "sgn must emit sign() on full float4:\n{msl}"
    );
}

#[test]
fn dp2add_emits_dot2_plus_scalar() {
    // vs_2_0 { dcl_position v0; dp2add r0, v0, c0, c1; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DP2ADD, 4),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("dot((in.v0).xy, (vs_c[0]).xy) + (vs_c[1]).x"),
        "dp2add must combine 2-wide dot with scalar add:\n{msl}"
    );
}

#[test]
fn dsx_dsy_emit_metal_derivatives() {
    // ps_2_0 { dcl t0; dsx r0, t0; dsy r1, t0; mov oC0, r0; }
    let bc = vec![
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_DSX, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_DSY, 2),
        dst_token(TYPE_TEMP, 1, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &bc, VariantKey::default());
    assert!(msl.contains("dfdx(in.texcoord0)"), "dsx → dfdx:\n{msl}");
    assert!(msl.contains("dfdy(in.texcoord0)"), "dsy → dfdy:\n{msl}");
}

#[test]
fn lit_emits_lighting_coefficients() {
    // vs_2_0 { dcl_position v0; lit r0, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LIT, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("max((in.v0).x, 0.0)"),
        "lit must compute max(src.x, 0) for the diffuse term:\n{msl}"
    );
    assert!(
        msl.contains("pow((in.v0).y, clamp((in.v0).w, -127.9961, 127.9961))"),
        "lit must raise src.y to src.w clamped to the D3D9 exponent range:\n{msl}"
    );
    assert!(
        msl.contains("((in.v0).x > 0.0) && ((in.v0).y > 0.0)"),
        "lit must gate the specular term on src.x > 0 and src.y > 0:\n{msl}"
    );
    metal_compile_or_fail(&emit_vs_programmable(&parse(&bc).expect("vs parse")).expect("emit vs"));
}

#[test]
fn nrm_of_a_zero_vector_returns_the_source() {
    // ps_3_0 { dcl_texcoord0 v0; nrm r0, v0; mov oC0, r0; }
    // rsqrt(0) is inf and 0 * inf is NaN, so a zero-length source has to
    // bypass the scale.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_NRM, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        msl.contains(
            "r[0] = ((dot((in.texcoord0).xyz, (in.texcoord0).xyz) == 0.0) ? (in.texcoord0) \
             : ((in.texcoord0) * rsqrt(dot((in.texcoord0).xyz, (in.texcoord0).xyz))));"
        ),
        "nrm must return a zero-length source unchanged:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn dst_emits_distance_attenuation_vector() {
    // vs_2_0 { dcl_position v0; dst r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DST, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("(in.v0).y * (vs_c[0]).y"),
        "dst[1] = src0.y * src1.y:\n{msl}"
    );
    assert!(msl.contains("(in.v0).z"), "dst[2] = src0.z:\n{msl}");
    assert!(msl.contains("(vs_c[0]).w"), "dst[3] = src1.w:\n{msl}");
}

#[test]
fn cnd_emits_conditional_select_on_half() {
    // ps_1_x style conditional move. Use SM2 PS for the host test.
    // ps_2_0 { dcl t0; cnd r0, t0, c0, c1; mov oC0, r0; }
    let bc = vec![
        PS_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 1, 0xF, false),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_CND, 4),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &bc, VariantKey::default());
    assert!(
        msl.contains("select(c1, c0, (in.texcoord0).x > 0.5)"),
        "cnd must emit select gated on src0.x > 0.5:\n{msl}"
    );
}

#[test]
fn secondary_position_semantic_routes_through_position1_varying() {
    // vs_3_0 { dcl_position0 v0; dcl_position1 v1; dcl_position0 o0;
    //          dcl_position1 o1; mov o0,v0; mov o1,v1; }
    let vs_bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 1),
        dst_token(TYPE_INPUT, 1, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 1),
        dst_token(TYPE_OUTPUT, 1, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 1, 0xF, false),
        src_token(TYPE_INPUT, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&vs_bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position1 = in.v1;") || vs_msl.contains("out.position1 ="),
        "POSITION1 output must route to out.position1, not clobber out.position:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("float4 position1;"),
        "Varyings must carry the position1 field:\n{vs_msl}"
    );

    // ps_3_0 { dcl_position1 v0; mov oC0, v0; } — reads the position1 varying.
    let ps_bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 1),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&ps_bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("in.position1"),
        "PS dcl_position1 must read the position1 varying:\n{ps_msl}"
    );
}

#[test]
fn cnd_ps_1_4_compares_per_component_and_ps_1_1_coissue_selects_src1() {
    // Build `cnd r0, t0, c0, c1; mov r0, r0` (the trailing mov keeps r0 as the
    // ps_1_x colour output) under a given header + control bits.
    const OP_CND: u16 = 80;
    let emit = |header: u32, cnd_extra: u32| {
        let bc = vec![
            header,
            opcode_token(OP_DEF, 5),
            dst_token(TYPE_CONST, 0, 0xF, false),
            f32::to_bits(0.0),
            f32::to_bits(1.0),
            f32::to_bits(0.0),
            f32::to_bits(1.0),
            opcode_token(OP_DEF, 5),
            dst_token(TYPE_CONST, 1, 0xF, false),
            f32::to_bits(1.0),
            f32::to_bits(0.0),
            f32::to_bits(1.0),
            f32::to_bits(1.0),
            opcode_token(OP_CND, 4) | cnd_extra,
            dst_token(TYPE_TEMP, 0, 0xF, false),
            src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0), // ps_1_x t0
            src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
            src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
            END_TOKEN,
        ];
        let ps = parse(&bc).expect("PS parse");
        emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS")
    };
    // ps_1_4: per-component compare (whole-vector `> float4(0.5)`).
    let ps14 = emit(0xFFFF_0104, 0);
    assert!(
        ps14.contains("select(c1, c0, t[0] > float4(0.5))"),
        "ps_1_4 cnd must compare per component:\n{ps14}"
    );
    // ps_1_1 plain: scalar `.x > 0.5` broadcast.
    let ps11 = emit(0xFFFF_0101, 0);
    assert!(
        ps11.contains("select(c1, c0, (t[0]).x > 0.5)"),
        "ps_1_1 cnd must test the scalar .x lane:\n{ps11}"
    );
    // ps_1_1 co-issued (D3DSI_COISSUE, RGB write): selects src1 unconditionally.
    let ps11_coissue = emit(0xFFFF_0101, 0x4000_0000);
    assert!(
        !ps11_coissue.contains("> 0.5"),
        "co-issued non-alpha cnd must bypass the compare:\n{ps11_coissue}"
    );
    assert!(
        ps11_coissue.contains("r[0] = c0;"),
        "co-issued cnd must select src1 (c0) unconditionally:\n{ps11_coissue}"
    );
}

#[test]
fn expp_logp_lower_to_full_precision_builtins() {
    // ps 1.x partial-precision exp/log map to full-precision Metal
    // builtins on modern hardware.
    // vs_2_0 { dcl_position v0; expp r0, v0; logp r1, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_EXPP, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_LOGP, 2),
        dst_token(TYPE_TEMP, 1, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("exp2((in.v0).x)"),
        "expp must lower to exp2:\n{msl}"
    );
    assert!(
        msl.contains("log2(abs((in.v0).x))"),
        "logp must lower to log2:\n{msl}"
    );
}

#[test]
fn expp_vs_1_1_emits_legacy_four_component_result() {
    // Exact little-endian words from issue #486:
    // vs_1_1 { expp r0, c0.x; mov oPos, c1; mov oD0, r0; }
    let bc = [
        0xFFFE_0101,
        0x0000_004E,
        0x800F_0000,
        0xA000_0000,
        0x0000_0001,
        0xC00F_0000,
        0xA0E4_0001,
        0x0000_0001,
        0xD00F_0000,
        0x80E4_0000,
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("emit vs_1_1");
    metal_compile_or_fail(&msl);
    assert!(
        msl.contains(
            "float4(exp2(floor(((vs_c[0]).xxxx).x)), ((vs_c[0]).xxxx).x - \
             floor(((vs_c[0]).xxxx).x), exp2(((vs_c[0]).xxxx).x), 1.0)"
        ),
        "vs_1_1 expp must keep its four distinct result components:\n{msl}"
    );

    // A source modifier and non-x replicate swizzle are applied before the
    // operation, while the common destination store keeps a partial mask.
    let masked_bc = [
        0xFFFE_0101,
        0x0000_004E,
        dst_token(TYPE_TEMP, 0, 0b0110, false),
        src_token(TYPE_CONST, 3, 0xFF, 1), // -c3.wwww
        0x0000_0001,
        0xC00F_0000,
        0xA0E4_0001,
        END_TOKEN,
    ];
    let masked_vs = parse(&masked_bc).expect("masked vs_1_1 parse");
    let masked_msl = emit_vs_programmable(&masked_vs).expect("emit masked vs_1_1");
    assert!(
        masked_msl.contains("r[0].yz =")
            && masked_msl.contains("-(vs_c[3]).wwww")
            && masked_msl.contains("floor("),
        "vs_1_1 expp lost a source modifier, replicate swizzle, or destination mask:\n{masked_msl}"
    );
    metal_compile_or_fail(&masked_msl);
}

#[test]
fn texldl_emits_sample_with_explicit_lod() {
    // ps_3_0 { dcl_2d s0; dcl t0; texldl r0, t0, s0; mov oC0, r0; }
    // texldl carries the LOD in coord.w.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d sampler usage token (texture type 2D in bits 27..30)
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_TEXLDL, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("s0.sample(samp0, (in.texcoord0).xy, level((in.texcoord0).w))"),
        "texldl must pass coord.w as level():\n{ps_msl}"
    );
}

#[test]
fn texldp_divides_coord_by_w_before_sampling() {
    // ps_2_0 { dcl_2d s0; dcl t0; texldp r0, t0, s0; mov oC0, r0; }
    // The D3DSI_TEXLD_PROJECT control bit (0x00010000) makes texld sample at
    // coord.xy / coord.w; plain texld (no bit) samples raw.
    const OP_TEXLD: u16 = 66;
    let proj = |bc_extra: u32| {
        let bc = vec![
            PS_HEADER,
            opcode_token(OP_DCL, 2),
            0x9000_0000, // dcl_2d s0
            dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
            opcode_token(OP_DCL, 2),
            dcl_usage_token(DCL_TEXCOORD, 0),
            dst_token(TYPE_INPUT, 0, 0xF, false),
            opcode_token(OP_TEXLD, 3) | bc_extra,
            dst_token(TYPE_TEMP, 0, 0xF, false),
            src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
            src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
            opcode_token(OP_MOV, 2),
            dst_token(TYPE_COLOROUT, 0, 0xF, false),
            src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
            END_TOKEN,
        ];
        let ps = parse(&bc).expect("PS parse");
        emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS")
    };
    // texldp: coordinate divided by its .w before the .xy sampler swizzle.
    let projected = proj(0x0001_0000);
    assert!(
        projected.contains(".w)).xy"),
        "texldp must divide the coord by .w before sampling:\n{projected}"
    );
    // Plain texld: no projective divide.
    let plain = proj(0);
    assert!(
        !plain.contains(".w)).xy"),
        "plain texld must not project:\n{plain}"
    );
}

#[test]
fn texldb_adds_coord_w_to_the_sample_bias() {
    // ps_3_0 { dcl_2d s0; dcl_texcoord0 v0; texldb r0, v0, s0; mov oC0, r0; }
    const OP_TEXLD: u16 = 66;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000,
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_TEXLD, 3) | 0x0002_0000,
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS parse");

    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit texldb");
    assert!(
        plain.contains("bias((in.texcoord0).w)"),
        "texldb must bias by coord.w:\n{plain}"
    );
    assert!(
        !plain.contains("lod_bias"),
        "zero sampler bias must not add the uniform:\n{plain}"
    );

    let sampler_biased = emit_ps_programmable(&ps, lod_bias_variant()).expect("emit texldb");
    assert!(
        sampler_biased.contains("bias((in.texcoord0).w + lod_bias[0].x)"),
        "texldb must add coord.w and sampler bias:\n{sampler_biased}"
    );
    metal_compile_or_fail(&sampler_biased);
}

#[test]
fn texldd_emits_sample_with_gradient2d_for_2d_sampler() {
    // ps_3_0 { dcl_2d s0; dcl t0; texldd r0, t0, s0, r1, r2; mov oC0, r0; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000,
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_TEXLDD, 5),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 2, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("gradient2d((r[1]).xy, (r[2]).xy)"),
        "texldd on a 2D sampler must use gradient2d() with .xy gradients:\n{ps_msl}"
    );
}

#[test]
fn depth_sampler_mask_emits_depth2d_binding_and_widens_sample_result() {
    // ps_3_0 { dcl_2d s0; dcl t0; texld r0, t0, s0; mov oC0, r0; }
    // With depth_sampler_mask bit 0 set, the slot binding must be
    // `depth2d<float>` instead of `texture2d<float>`, and the sample
    // call must be wrapped in `float4(...)` so downstream code reading
    // `.xyzw` keeps compiling. Mirrors how WoW's shadow PS samples
    // a D24X8 texture bound via CreateTexture(DEPTHSTENCIL).
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d s0
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");

    // Without the mask: standard color path.
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit plain");
    assert!(
        plain.contains("texture2d<float> s0 [[texture(0)]]"),
        "plain bind must be texture2d<float>:\n{plain}"
    );
    assert!(
        !plain.contains("depth2d<float>"),
        "plain bind must NOT mention depth2d:\n{plain}"
    );
    assert!(
        plain.contains("s0.sample(samp0, (in.texcoord0).xy)"),
        "plain sample must be the bare s0.sample expression:\n{plain}"
    );
    assert!(
        !plain.contains("float4(s0.sample"),
        "plain sample must NOT be wrapped in float4():\n{plain}"
    );
    assert!(
        !plain.contains("saturate"),
        "color path must NOT saturate — the clamp is depth-branch only:\n{plain}"
    );

    // With the mask: depth2d binding + sample_compare with the reference
    // depth saturate()d to [0,1], wrapped in float4 for downstream .xyzw
    // reads. This is the D3D9 hardware-shadow PCF idiom — `tex2D(s_shadow,
    // float3(uv, z_ref))` against a depth-format texture returns the
    // comparison result, not raw depth. The saturate() replicates the
    // clamp a D24 UNORM target gives for free; Depth32Float (Apple
    // Silicon's only depth format) does not. See `sample_or_compare`.
    let depth_variant = VariantKey {
        depth_sampler_mask: 0b0001,
        depth_fetch_mask: 0,
        fetch4_mask: 0,
        fetch4_alpha_mask: 0,
        raw_depth_red_mask: 0,
        ..VariantKey::default()
    };
    let depth = emit_ps_programmable(&ps, depth_variant).expect("emit depth");
    assert!(
        depth.contains("depth2d<float> s0 [[texture(0)]]"),
        "depth bind must be depth2d<float>:\n{depth}"
    );
    assert!(
        !depth.contains("texture2d<float>"),
        "depth bind must NOT mention texture2d:\n{depth}"
    );
    assert!(
        depth.contains(
            "float4(s0.sample_compare(samp0, (in.texcoord0).xy, saturate((in.texcoord0).z), level(0)))"
        ),
        "depth sample must be sample_compare with saturate()d ref and level(0):\n{depth}"
    );
    assert!(
        !depth.contains("s0.sample(samp0,"),
        "depth slot must NOT use plain sample():\n{depth}"
    );

    // Smoke-compile under Metal. The structural assertions above don't
    // catch wrong sampler return types — a buggy emitter that drops
    // the `float4(...)` wrap would still match the `s0.sample_compare(...)`
    // substring, but Metal would refuse to compile because
    // sample_compare returns `float`, not `float4`. Run the same compile
    // the production unix .so uses so the bug surfaces here.
    metal_compile_or_fail(&depth);
}

#[test]
fn ps_sampler_index_8_emits_slot_8_binding_and_compiles() {
    // ps_3_0 { dcl_2d s8; dcl t0; texld r0, t0, s8; mov oC0, r0; }
    //
    // PS3.0 allows sampler slots s0–s15. WoW's HD shadow receivers declare
    // `dcl_2d s8` for a 4th cascade-shadow-map tile (the `else`-branch in
    // the cascade-select ladder), so the emitter must carry the
    // SetTexture(8, …) binding through. A cap of 8 PS sampler stages would
    // silently drop it, producing Apple "Missing Fragment Texture s8"
    // warnings and a visibly missing far-cascade shadow. This test locks
    // the emitter path so the s8 binding can never be quietly regressed.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d sampler usage token (texture type 2D in bits 27..30)
        dst_token(10 /* TYPE_SAMPLER */, 8, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 8, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");

    // Plain color path — slot 8 must appear in the MSL signature and
    // be sampled by `s8.sample(samp8, …)`.
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit plain");
    assert!(
        plain.contains("texture2d<float> s8 [[texture(8)]]"),
        "PS sampler 8 must bind at Metal slot 8:\n{plain}"
    );
    assert!(
        plain.contains("sampler samp8 [[sampler(8)]]"),
        "PS sampler-state 8 must bind at Metal sampler slot 8:\n{plain}"
    );
    assert!(
        plain.contains("s8.sample(samp8, (in.texcoord0).xy)"),
        "PS body must sample s8 via samp8:\n{plain}"
    );

    // Depth-mask path with bit 8 set — slot 8 becomes a depth2d binding
    // and the sample turns into sample_compare. Mirrors WoW's shadow
    // receiver path with the cascade-3 depth texture bound on stage 8.
    let depth_variant = VariantKey {
        depth_sampler_mask: 1 << 8,
        depth_fetch_mask: 0,
        fetch4_mask: 0,
        fetch4_alpha_mask: 0,
        raw_depth_red_mask: 0,
        ..VariantKey::default()
    };
    let depth = emit_ps_programmable(&ps, depth_variant).expect("emit depth");
    assert!(
        depth.contains("depth2d<float> s8 [[texture(8)]]"),
        "depth_sampler_mask bit 8 must rewrite slot 8 binding to depth2d:\n{depth}"
    );
    assert!(
        depth.contains("s8.sample_compare(samp8,"),
        "depth slot 8 must use sample_compare:\n{depth}"
    );

    // Smoke-compile both variants under Metal so a future emitter
    // refactor that breaks slot-8 codegen surfaces here instead of
    // shipping to the unix .so.
    metal_compile_or_fail(&plain);
    metal_compile_or_fail(&depth);
}

#[test]
fn fog_msl_compiles_under_metal() {
    use crate::dxso::ff::{FfPsKey, FfStage, emit_ps_ff};
    // Every fog blend shape through a real Metal compile: `precise::exp`,
    // the `in.position` fragcoord read, and the two-row `fog_data` binding
    // must all be valid MSL on both the FF and programmable PS emitters.
    let ps_key = FfPsKey {
        stages: [FfStage {
            color_op: narrow(D3DTOP_DISABLE),
            ..FfStage::default()
        }; 8],
        specular_add: false,
        tt_projected_mask: 0,
    };
    let variants = [
        (0u8, 1u8, false), // EXP, Z source
        (0, 2, true),      // EXP2, W source
        (0, 3, false),     // LINEAR, Z source
        (0, 3, true),      // LINEAR, W source
        (3, 0, false),     // vertex fog
    ];
    for (fog_mode, fog_table_mode, fog_source_w) in variants {
        let mut flags = VariantFlags::empty();
        flags.set(VariantFlags::FOG_SOURCE_W, fog_source_w);
        let variant = VariantKey {
            fog_mode,
            fog_table_mode,
            flags,
            ..VariantKey::default()
        };
        metal_compile_or_fail(&emit_ps_ff(&ps_key, variant));
        let ps = parse(&red_constant_ps()).expect("PS parse");
        let msl = emit_ps_programmable(&ps, variant).expect("emit PS");
        metal_compile_or_fail(&msl);
    }
}

/// Compile MSL through `MTLDevice::newLibraryWithSource_options_error`.
///
/// Uses the same options the production unix .so uses (`metal/shader.rs`).
/// Skips if no Metal device is available (headless / no-GPU test runner).
fn metal_compile_or_fail(msl: &str) {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLMathMode,
    };
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping Metal-compile check");
        return;
    };
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    options.setPreserveInvariance(true);
    options.setMathMode(MTLMathMode::Fast);
    let source = NSString::from_str(msl);
    if let Err(err) = device.newLibraryWithSource_options_error(&source, Some(&options)) {
        panic!("MSL failed Metal compilation: {err}\n--- MSL ---\n{msl}");
    }
}

#[test]
fn texldd_uses_gradientcube_for_cube_sampler() {
    // ps_3_0 { dcl_cube s0; dcl t0; texldd r0, t0, s0, r1, r2; mov oC0, r0; }
    // Cube samplers consume float3 coord + float3 gradients. The cube form is
    // selected by the cube texture bound to the slot, so the variant carries
    // the binding the declaration expects.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9800_0000, // dcl_cube — texture type CUBE = 3 in bits 27..30
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_TEXLDD, 5),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 2, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let cube_bound = VariantKey {
        cube_sampler_mask: 0b0001,
        ..VariantKey::default()
    };
    let ps_msl = emit_ps_programmable(&ps, cube_bound).expect("emit PS3");
    assert!(
        ps_msl.contains("(in.texcoord0).xyz"),
        "cube samplers need .xyz coord:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("gradientcube((r[1]).xyz, (r[2]).xyz)"),
        "texldd on cube sampler must use gradientcube() with .xyz gradients:\n{ps_msl}"
    );
}

// ── SM3 ──

#[test]
fn sm3_vs_position_output_resolves_via_dcl() {
    // vs_3_0 { dcl_position o2; mov o2, v0; }
    // SM3 unifies outputs under reg type 11 (RegKind::Output); the
    // `dcl_position` carries the semantic — register index alone is not
    // sufficient (could be o0, o2, o7…).
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_OUTPUT, 2, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 2, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "VS3 oN with dcl_position should resolve to out.position:\n{vs_msl}"
    );
}

#[test]
fn sm3_vs_outputs_via_texcoordout_kind_resolve_through_dcl() {
    // WoW's HLSL compiler ships SM3 outputs as `RegKind::TexcoordOut`
    // (D3DSPR_TEXCRDOUT, type 6) — the type aliases D3DSPR_OUTPUT in SM3,
    // with the dcl carrying the actual semantic. The output map keys on
    // (kind, index) so these resolve through the dcl; matching only
    // `RegKind::Output` (type 11) would leave them to fall through the SM2
    // default that maps TexcoordOut[0] to `out.texcoord0`, so a
    // dcl_position output would never write clip-space position and the
    // geometry would collapse.
    //
    // vs_3_0 { dcl_position oT0; mov oT0, v0; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "VS3 dcl_position oT0 must resolve to out.position via dcl, not out.texcoord0:\n{vs_msl}"
    );
    assert!(
        !vs_msl.contains("out.texcoord0 = in.v0;"),
        "VS3 dcl_position oT0 must not fall back to texcoord0:\n{vs_msl}"
    );
}

#[test]
fn sm3_vs_color_and_texcoord_outputs_resolve_via_dcl() {
    // vs_3_0 {
    //     dcl_position o0;
    //     dcl_color0 o3;
    //     dcl_texcoord2 o5;
    //     mov o0, v0;
    //     mov o3, v0;
    //     mov o5, v0;
    // }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_COLOR, 0),
        dst_token(TYPE_OUTPUT, 3, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 2),
        dst_token(TYPE_OUTPUT, 5, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 3, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 5, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "VS3 dcl_position o0 → out.position:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.color0 = in.v0;"),
        "VS3 dcl_color0 o3 → out.color0 (varying = usage_index, not reg index):\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.texcoord2 = in.v0;"),
        "VS3 dcl_texcoord2 o5 → out.texcoord2:\n{vs_msl}"
    );
}

#[test]
fn sm3_ps_input_texcoord_resolves_via_dcl() {
    // ps_3_0 { dcl_texcoord3 v5; mov oC0, v5; }
    // The PS input map must distinguish color from texcoord by the dcl
    // usage. SM2 always assumed color — that breaks SM3.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 3),
        dst_token(TYPE_INPUT, 5, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 5, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("oC0 = in.texcoord3;"),
        "PS3 v5 with dcl_texcoord3 must read from in.texcoord3:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("oC0 = in.color5"),
        "PS3 v5 must not fall back to in.color5 (SM2 assumption):\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_input_color_uses_usage_index_not_reg_index() {
    // ps_3_0 { dcl_color2 v7; mov oC0, v7; }
    // Varying slot is `usage_index` (=2), not `reg.index` (=7).
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_COLOR, 2),
        dst_token(TYPE_INPUT, 7, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 7, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let linked = VariantKey {
        linked_input_mask: 1,
        ..VariantKey::default()
    };
    let ps_msl = emit_ps_programmable(&ps, linked).expect("emit PS3");
    assert!(
        ps_msl.contains("oC0 = in.color2;"),
        "PS3 dcl_color2 v7 must read from in.color2:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
    // COLOR2 is no fixed-function varying: behind a vertex shader that does
    // not output it, the input reads zero.
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(ps_msl.contains("oC0 = float4(0.0);"), "{ps_msl}");
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn sincos_emits_cos_sin_pair() {
    // vs_3_0 { dcl_position v0; sincos r1.xy, r0.x; mov oPos, v0; }
    // SM3 single-source form: dst.x = cos(src.x), dst.y = sin(src.x).
    // Write mask 0b0011 picks only .xy from the float4 we synthesize.
    const OP_SINCOS: u16 = 37;
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_SINCOS, 2),
        dst_token(TYPE_TEMP, 1, 0b0011, false),
        src_token(TYPE_TEMP, 0, 0x00 /* .xxxx */, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("cos(") && vs_msl.contains("sin("),
        "sincos must emit both cos() and sin() builtins:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("r[1].xy"),
        "sincos write_mask 0b0011 must land in r[1].xy:\n{vs_msl}"
    );
}

#[test]
fn call_inline_expands_subroutine_body() {
    // vs_3_0 {
    //   dcl_position v0;
    //   dcl_position oT0;
    //   call l0;
    //   ret;
    //   label l0;
    //     mov oT0, v0;
    //   ret;
    // }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_CALL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "subroutine body must inline-expand at the call site:\n{vs_msl}"
    );
}

#[test]
fn callnz_wraps_inlined_body_in_conditional() {
    // vs_3_0 {
    //   dcl_position v0;
    //   dcl_position oT0;
    //   callnz l0, c0;
    //   label l0;
    //     mov oT0, v0;
    //   ret;
    // }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_CALLNZ, 2),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("if ((vs_c[0]).x != 0.0) {"),
        "callnz must gate the inlined body on the condition src:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "callnz body must inline-expand:\n{vs_msl}"
    );
}

#[test]
fn setp_lt_emits_componentwise_predicate_assignment() {
    // ps_3_0 { dcl t0; setp_lt p0, t0, t0; mov oC0, t0; }
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24); // cmp=Lt(4), token_count=3
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("bool4 p0 = bool4(false);"),
        "PS prologue must declare p0:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("p0 = (in.texcoord0 < in.texcoord0);"),
        "setp_lt must emit a componentwise bool4 assignment:\n{ps_msl}"
    );
}

#[test]
fn setp_honours_its_write_mask_and_predicate() {
    // ps_3_0 { setp_gt p0.x, c0, c1; setp_lt p0.y, c0, c1;
    // (p0.x) setp_eq p0.zw, c0, c1; mov oC0, c0; }
    // Each setp writes only its masked lanes, and a predicated one keeps the
    // lanes whose predicate component is false.
    let setp = |cmp: u32| u32::from(OP_SETP) | (cmp << 16) | (3u32 << 24);
    let bc = vec![
        PS3_HEADER,
        setp(1),
        dst_token(TYPE_PREDICATE, 0, 0x1, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        setp(4),
        dst_token(TYPE_PREDICATE, 0, 0x2, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        u32::from(OP_SETP) | (2u32 << 16) | (4u32 << 24) | (1 << 28),
        dst_token(TYPE_PREDICATE, 0, 0xC, false),
        src_token(TYPE_PREDICATE, 0, SWIZ_XXXX, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        msl.contains("p0.x = (ps_c[0] > ps_c[1]).x;"),
        "setp p0.x must write only .x:\n{msl}"
    );
    assert!(
        msl.contains("p0.y = (ps_c[0] < ps_c[1]).y;"),
        "setp p0.y must write only .y:\n{msl}"
    );
    assert!(
        msl.contains("p0.zw = select(p0.zw, (ps_c[0] == ps_c[1]).zw, p0.xx);"),
        "a predicated setp must keep the lanes its predicate rejects:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn if_reads_predicate_source_with_replicate_swizzle() {
    // ps_3_0 { setp_lt p0, c0, c1; if p0.x; mov oC0, c0; endif; }
    let bc = vec![
        PS3_HEADER,
        u32::from(OP_SETP) | (4u32 << 16) | (3u32 << 24),
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        0x0100_0028,
        0xb000_1000,
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("if (((float4(p0)).xxxx).x != 0.0) {"),
        "if must read the replicated p0.x source:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn callnz_reads_negated_predicate_source_with_replicate_swizzle() {
    // vs_3_0 { callnz l0, !p0.z; ret; label l0; mov r0, c0; ret; }
    let bc = vec![
        VS3_HEADER,
        0x0200_001a,
        0xa0e4_1000,
        0xbdaa_1000,
        opcode_token(OP_RET, 0),
        0x0100_001e,
        0xa0e4_1000,
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("if ((float4(!bool4((float4(p0)).zzzz))).x != 0.0) {"),
        "callnz must read and negate the replicated p0.z source:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("r[0] = vs_c[0];"),
        "callnz body must inline-expand:\n{vs_msl}"
    );
    metal_compile_or_fail(&vs_msl);
}

#[test]
fn predicated_instruction_selects_destination_components_from_p0() {
    // ps_3_0 {
    //   def c0, 0, 1, 0, 1; def c1, 0.5, 0.5, 0.5, 0.5;
    //   def c2, 1, 1, 1, 1; def c3, 0, 0, 0, 0;
    //   mov r0, c3; mov r1, c0; setp_lt p0, r1, c1;
    //   (p0) mov r0, c2; mov oC0, r0;
    // }
    // Token format for the predicated mov: opcode bits, predicate flag
    // (bit 28), token_count covering predicate operand + dst + src.
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24);
    let predicated_mov_token = u32::from(OP_MOV) | (1u32 << 28) | (3u32 << 24); // predicated, count=3
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 1, 0xF, false),
        f32::to_bits(0.5),
        f32::to_bits(0.5),
        f32::to_bits(0.5),
        f32::to_bits(0.5),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 2, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(1.0),
        f32::to_bits(1.0),
        f32::to_bits(1.0),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 3, 0xF, false),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 3, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 1, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        predicated_mov_token,
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_PREDICATE, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 2, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("r[0] = select(r[0], c2, p0);"),
        "predicated mov must select each destination component from p0:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn negated_predicate_inverts_each_destination_component() {
    // ps_3_0 { setp_lt p0, c0, c1; (!p0) mov r0, c2; mov oC0, r0; }
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24);
    let predicated_mov_token = u32::from(OP_MOV) | (1u32 << 28) | (3u32 << 24);
    let bc = vec![
        PS3_HEADER,
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        predicated_mov_token,
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_PREDICATE, 0, SWIZ_IDENTITY, 13 /* logical NOT */),
        src_token(TYPE_CONST, 2, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("r[0] = select(r[0], ps_c[2], !(p0));"),
        "predicate NOT must invert every selected component:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn replicate_predicate_intersects_partial_destination_mask() {
    // ps_3_0 { setp_lt p0, c0, c1; (p0.z) mov r0.yw, c2; mov oC0, r0; }
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24);
    let predicated_mov_token = u32::from(OP_MOV) | (1u32 << 28) | (3u32 << 24);
    let bc = vec![
        PS3_HEADER,
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        predicated_mov_token,
        dst_token(TYPE_TEMP, 0, 0b1010, false),
        src_token(TYPE_PREDICATE, 0, 0xAA /* .zzzz */, 0),
        src_token(TYPE_CONST, 2, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("r[0].yw = select(r[0].yw, (ps_c[2]).yw, p0.zz);"),
        "replicate predicate must cover only destination-written components:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn predicate_intersects_single_component_destination_mask() {
    // ps_3_0 { setp_lt p0, c0, c1; (p0) mov r0.w, c2; mov oC0, r0; }
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24);
    let predicated_mov_token = u32::from(OP_MOV) | (1u32 << 28) | (3u32 << 24);
    let bc = vec![
        PS3_HEADER,
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        predicated_mov_token,
        dst_token(TYPE_TEMP, 0, 0b1000, false),
        src_token(TYPE_PREDICATE, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 2, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("r[0].w = select(r[0].w, (ps_c[2]).w, p0.w);"),
        "predicate must narrow to the single destination component:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn breakp_emits_predicate_gated_break() {
    // vs_3_0 { defi i0, 4,0,1,0; loop aL, i0; breakp p0.w; endloop; ... }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        0x0100_0060,
        0xb0ff_1000,
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let breakp = vs
        .instructions
        .iter()
        .find(|inst| inst.opcode == super::Opcode::BreakP)
        .expect("breakp instruction");
    assert!(
        breakp.predicate.is_none(),
        "breakp must not use the instruction predication operand"
    );
    assert_eq!(breakp.srcs.len(), 1, "breakp must have one regular source");
    assert_eq!(breakp.srcs[0].reg.kind, super::RegKind::Predicate);
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("if (((float4(p0)).wwww).x != 0.0) break;"),
        "breakp must gate break on the predicate operand:\n{vs_msl}"
    );
    metal_compile_or_fail(&vs_msl);
}

#[test]
fn defi_emits_int4_local() {
    // vs_3_0 { defi i0, 4, 0, 1, 0; dcl_position v0; mov oPos, v0; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("int4 i0 = int4(4, 0, 1, 0);"),
        "defi must emit an `int4 iN = int4(...)` local:\n{vs_msl}"
    );
}

#[test]
fn loop_emits_for_with_named_al_counter() {
    // vs_3_0 { defi i0, 4, 0, 1, 0; loop aL, i0; mov r0, c[aL]; endloop; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("for (int aL_0 ="),
        "loop must allocate aL_0:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("aL_0 += _aL_step_0"),
        "loop step must reference its own counters:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("float4(aL_0)"),
        "RegKind::Loop reads inside the body must use aL_0:\n{vs_msl}"
    );
}

#[test]
fn loop_relative_const_addressing_indexes_by_al() {
    // vs_3_0 { defi i0, 4, 0, 1, 0; dcl_position v0;
    //          loop aL, i0; mov r0, c[aL + 8]; endloop;
    //          mov oTexcoord0, v0; }
    // `c[aL + N]` carries a relative-addressing token whose index register is
    // the loop counter (TYPE_LOOP), not the address register — it must resolve
    // to the enclosing loop's `aL_<n>` local, indexing the constant buffer.
    let mut bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        // mov r0, c[aL + 8] — dst (1) + rel-addr src (2) = 3 operand tokens.
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        8,
        SWIZ_IDENTITY,
        0,
        TYPE_LOOP,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("r[0] = vs_c[aL_0 + 8];"),
        "c[aL + N] must index the constant buffer by the loop counter:\n{vs_msl}"
    );
    assert!(
        vs.uses_relative_const_addressing(),
        "rel-addr on const must be detected for the full-constant-buffer upload gate"
    );
}

/// The `aL` operand of `loop aL, iN` is not loaded; the counter and a `c[aL + N]` read are.
///
/// The `Loop` arm declares the counter, so loading that operand first would
/// resolve an `aL` read with no loop frame open and log a false warning.
#[test]
fn loop_counter_operand_is_not_loaded_before_its_loop() {
    // ps_3_0 { defi i0, 1, 18, 1, 0; loop aL, i0; mov r0, c[aL + 2]; endloop;
    //          mov oC0, r0; }
    let bc = [
        0xFFFF_0300,
        0x0500_0030,
        0xF00F_0000,
        1,
        18,
        1,
        0,
        0x0200_001B,
        0xF0E4_0800,
        0xF0E4_0000,
        0x0300_0001,
        0x800F_0000,
        0xA0E4_2002,
        0xF000_0800,
        0x0000_001D,
        0x0200_0001,
        0x800F_0800,
        0x80E4_0000,
        0x0000_FFFF,
    ];
    let ps = parse(&bc).expect("ps_3_0 parse");
    let loop_inst = ps
        .instructions
        .iter()
        .find(|inst| inst.opcode == super::Opcode::Loop)
        .expect("loop instruction");
    assert!(
        !super::loads_source(loop_inst, 0),
        "the aL operand names the counter"
    );
    assert!(super::loads_source(loop_inst, 1), "the iN operand is read");
    let mov = ps
        .instructions
        .iter()
        .find(|inst| inst.srcs.first().is_some_and(|src| src.rel_addr.is_some()))
        .expect("relative read");
    assert!(
        super::loads_source(mov, 0),
        "c[aL + N] is read inside the loop"
    );
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        msl.contains("r[0] = ps_c[aL_0 + 2];"),
        "the loop body still indexes by its counter:\n{msl}"
    );
}

#[test]
fn dynamic_int_constant_reads_the_runtime_vs_i_buffer() {
    // vs_3_0 { dcl_position v0; loop aL, i0; mov r0, c[aL + 8]; endloop;
    //          mov oTexcoord0, v0; }
    // i0 has NO `defi` — it is a dynamic integer constant fed by
    // SetVertexShaderConstantI, so the loop counter must read the runtime
    // `vs_i` buffer (slot 14), not a baked `int4 i0` local.
    let mut bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        8,
        SWIZ_IDENTITY,
        0,
        TYPE_LOOP,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let vs = parse(&bc).expect("VS3 parse");
    assert!(
        vs.uses_dynamic_int_constants(),
        "a non-defi iN read must be flagged as a dynamic integer constant"
    );
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains(&format!(
            "constant int4 *vs_i [[buffer({VS_INT_CONST_SLOT})]]"
        )),
        "a dynamic-int-const shader must declare the vs_i buffer at slot 14:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("vs_i[0]"),
        "the dynamic i0 loop counter must read vs_i[0]:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("vs_c[aL_0 + 8]"),
        "the aL-relative const read must still index vs_c by the loop counter:\n{vs_msl}"
    );
}

#[test]
fn defi_int_constant_stays_a_baked_local_without_vs_i() {
    // vs_3_0 { defi i0, 4, 0, 1, 0; dcl_position v0; loop aL, i0; mov r0, c0;
    //          endloop; mov oTexcoord0, v0; }
    // A defi'd i0 is a compile-time local; the shader must NOT declare vs_i.
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    assert!(
        !vs.uses_dynamic_int_constants(),
        "a defi'd iN is static, not a dynamic integer constant"
    );
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        !vs_msl.contains("vs_i"),
        "a defi-only shader must not declare or read vs_i:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("int4 i0 = int4(4, 0, 1, 0);"),
        "the defi'd i0 must stay a baked local:\n{vs_msl}"
    );
}

#[test]
fn dynamic_int_constant_inside_a_subroutine_declares_vs_i() {
    // vs_3_0 {
    //   dcl_position v0;
    //   dcl_position oT0;
    //   call l0;
    //   ret;
    //   label l0;
    //     rep i0;
    //       mov oT0, v0;
    //     endrep;
    //   ret;
    // }
    // The only iN read sits in the subroutine body, which `call`
    // inline-expands into the entry point, so the emitted body references
    // vs_i and the signature has to declare it.
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_CALL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_REP, 1),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDREP, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    assert!(
        vs.uses_dynamic_int_constants(),
        "an iN read reachable only through a call is still a dynamic integer constant"
    );
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains(&format!(
            "constant int4 *vs_i [[buffer({VS_INT_CONST_SLOT})]]"
        )),
        "the inlined `rep i0` must declare the vs_i buffer:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("vs_i[0]"),
        "the inlined `rep i0` counter must read vs_i[0]:\n{vs_msl}"
    );
    // The substring assertions above cannot see an undeclared argument;
    // Metal refuses the source outright when the body names vs_i and the
    // signature does not.
    metal_compile_or_fail(&vs_msl);
}

#[test]
fn dynamic_int_constant_inside_a_subroutine_declares_ps_i() {
    // ps_3_0 { call l0; ret; label l0; rep i0; mov oC0, c0; endrep; ret; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_CALL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_REP, 1),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDREP, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    assert!(
        ps.uses_dynamic_int_constants(),
        "an iN read reachable only through a call is still a dynamic integer constant"
    );
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains(&format!(
            "constant int4 *ps_i [[buffer({PS_INT_CONST_SLOT})]]"
        )),
        "the inlined `rep i0` must declare the ps_i file:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("ps_i[0]"),
        "the inlined `rep i0` counter must read ps_i[0]:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn rep_emits_for_loop_without_al() {
    // vs_3_0 { defi i0, 8, 0, 0, 0; rep i0; mov r0, r1; endrep; ... }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        8u32,
        0u32,
        0u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_REP, 1),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDREP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("for (int _rep_0 = 0; _rep_0 < (float4(i0)).x; ++_rep_0) {"),
        "rep must emit a counted for loop:\n{vs_msl}"
    );
}

#[test]
fn break_emits_msl_break_inside_loop() {
    // vs_3_0 { defi i0, 4,0,1,0; loop aL, i0; break; endloop; ... }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_BREAK, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("    break;\n"),
        "break must emit `break;`:\n{vs_msl}"
    );
}

#[test]
fn breakc_lt_emits_conditional_break() {
    // breakc_lt s0, s1 — opcode 45 with cmp = 4 (Lt) in bits 16-23.
    let breakc_lt_token = u32::from(OP_BREAKC) | ((4u32) << 16) | (2u32 << 24);
    // vs_3_0 { defi i0, 4,0,1,0; loop aL, i0; breakc_lt s0, s1; endloop; ... }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        breakc_lt_token,
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("if ((r[0]).x < (r[1]).x) break;"),
        "breakc_lt must emit a conditional break:\n{vs_msl}"
    );
}

#[test]
fn nested_loops_get_distinct_al_indices() {
    // Two nested `loop` blocks must allocate aL_0 and aL_1 so reads
    // bind to the correct enclosing scope.
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(vs_msl.contains("aL_0"), "outer loop missing:\n{vs_msl}");
    assert!(
        vs_msl.contains("aL_1"),
        "inner loop must get distinct aL_1:\n{vs_msl}"
    );
}

#[test]
fn if_emits_msl_branch_on_x_lane() {
    // ps_3_0 { dcl t0; if t0; mov oC0, t0; endif; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_IF, 1),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("if ((in.texcoord0).x != 0.0) {"),
        "`if src` must check src.x != 0:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("    }\n"),
        "endif must emit a closing brace:\n{ps_msl}"
    );
}

#[test]
fn ifc_lt_emits_msl_strict_less_than() {
    // Build an `ifc_lt` instruction: opcode 41 with cmp = 4 (Lt) in
    // bits 16-23 of the instruction token.
    // ps_3_0 { dcl t0; ifc_lt t0, t0; mov oC0, t0; endif; }
    let ifc_lt_token = u32::from(OP_IFC) | ((4u32) << 16) | (2u32 << 24); // cmp=4 (Lt), token_count=2
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        ifc_lt_token,
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("if ((in.texcoord0).x < (in.texcoord0).x) {"),
        "ifc_lt must compare src0.x < src1.x:\n{ps_msl}"
    );
}

#[test]
fn if_else_endif_balanced_braces() {
    // ps_3_0 { dcl t0; if t0; mov oC0, t0; else; mov oC0, t0; endif; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_IF, 1),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ELSE, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("} else {"),
        "else must emit a `}} else {{`:\n{ps_msl}"
    );
    let opens = ps_msl.matches("if (").count();
    let closes = ps_msl.matches("    }\n").count();
    assert!(
        closes >= opens,
        "every `if (` must have a matching `}}`:\n{ps_msl}"
    );
}

#[test]
fn vs_writing_psize_via_opts_routes_through_storage_local() {
    // vs_2_0 { dcl_position v0; mov oPts, c0; mov oPos, v0; }
    // SM2 RastOut[2] is `oPts`. The Varyings field is scalar but the
    // emit goes through a `_psize_storage` float4 so `store_dst`'s
    // write-mask path stays uniform.
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 2, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS");
    assert!(
        vs_msl.contains("float point_size [[point_size]]"),
        "Varyings must declare a [[point_size]] field:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("float4 _psize_storage = float4(vs_draw.point.x);"),
        "VS prologue must seed _psize_storage with D3DRS_POINTSIZE:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("_psize_storage = vs_c[0];"),
        "oPts write must land in _psize_storage:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains(
            "out.point_size = clamp(_psize_storage.x, vs_draw.point.y, vs_draw.point.z) * pos_fixup.w;"
        ),
        "VS epilogue must extract scalar point size from storage:\n{vs_msl}"
    );
}

#[test]
fn sm3_vs_writing_psize_via_dcl_routes_through_storage_local() {
    // vs_3_0 { dcl_position oT0; dcl_psize oT4; mov oT0, v0; mov oT4, c0; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(4 /* DeclUsage::PSize */, 0),
        dst_token(TYPE_TEXCOORDOUT, 4, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 4, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("_psize_storage = vs_c[0];"),
        "SM3 dcl_psize write must land in _psize_storage:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains(
            "out.point_size = clamp(_psize_storage.x, vs_draw.point.y, vs_draw.point.z) * pos_fixup.w;"
        ),
        "VS epilogue must extract scalar point size:\n{vs_msl}"
    );
}

#[test]
fn ps_writing_odepth_returns_psout_struct_with_depth_field() {
    // ps_3_0 { dcl t0; mov oDepth, t0.x; mov oC0, t0; }
    // DepthOut writes flip the PS return type to a struct that
    // exposes both `oC0 [[color(0)]]` and `oDepth [[depth(any)]]`.
    const TYPE_DEPTHOUT: u32 = 9;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_DEPTHOUT, 0, 0x1 /* .x only */, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("struct PsOut"),
        "PS writing oDepth must emit a PsOut struct:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("float4 oC0 [[color(0)]];"),
        "PsOut must bind oC0 to color(0):\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("float oDepth [[depth(any)]];"),
        "PsOut must bind oDepth to depth(any):\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("fragment PsOut mtld3d_ps("),
        "fragment must return PsOut:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("_depth_storage"),
        "DepthOut writes must route through _depth_storage:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("_ps_out.oDepth = _depth_storage.x;"),
        "scalar oDepth value extracted from _depth_storage.x at return:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("return _ps_out;"),
        "fragment must return the struct:\n{ps_msl}"
    );
}

#[test]
fn ps_writing_odepth_only_in_a_subroutine_exports_depth() {
    // ps_3_0 { call l0; mov oC0, c0; ret; label l0; mov oDepth, c1.x; ret; }
    const TYPE_DEPTHOUT: u32 = 9;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_CALL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_DEPTHOUT, 0, 0xF, false),
        src_token(TYPE_CONST, 1, SWIZ_XXXX, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    metal_compile_or_fail(&msl);
    assert!(
        msl.contains("_depth_storage = (ps_c[1]).xxxx;") && msl.contains("oDepth [[depth(any)]]"),
        "an oDepth write in a subroutine must reach the depth output:\n{msl}"
    );
}

#[test]
fn ps_without_odepth_keeps_float4_return_for_simplicity() {
    // PS without oDepth writes stays on the bare-`float4` return path —
    // no struct, no extra storage local.
    let ps = parse(&red_constant_ps()).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        !ps_msl.contains("struct PsOut"),
        "no PsOut struct unless shader writes oDepth:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("fragment float4 mtld3d_ps("),
        "fragment stays float4 → return oC0:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("_depth_storage"),
        "no _depth_storage local without oDepth:\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_vpos_via_misctype_reads_in_position_no_duplicate_position() {
    // ps_3_0 { dcl_position vPos; mov oC0, vPos; }
    // SM3 dedicated `vPos` register lives on RegKind::MiscType index 0 and
    // reads the screen-space pixel coord — which IS the `[[position]]` the
    // `Varyings` struct already declares, so vPos reads `in.position`. A
    // SECOND `float4 v_pos [[position]]` fragment arg is a duplicate-
    // `[[position]]` MSL error that fails the shader to compile (it then
    // never renders). D3D9 vPos is the integer pixel coord vs Metal's
    // pixel-centre `[[position]]`, hence the `- 0.5`.
    const TYPE_MISCTYPE: u32 = 17;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_MISCTYPE, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_MISCTYPE, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        !ps_msl.contains("v_pos [[position]]"),
        "vPos must NOT add a second [[position]] arg (duplicate is an MSL error):\n{ps_msl}"
    );
    assert_eq!(
        ps_msl.matches("[[position").count(),
        1,
        "exactly one [[position...]] (the Varyings field):\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("oC0 = (in.position - 0.5);"),
        "vPos read must resolve to (in.position - 0.5):\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_vpos_under_the_render_scale_variant_reads_the_scaled_floored_position() {
    // ps_3_0 { dcl_position vPos; mov oC0, vPos; } under `VPOS_SCALE`: the
    // function takes the `PsDraw` uniform and the register is the pixel
    // centre scaled into the reported space and floored, so `frc(vPos)`
    // stays zero. The struct is emitted ahead of the function.
    const TYPE_MISCTYPE: u32 = 17;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_MISCTYPE, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_MISCTYPE, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let variant = VariantKey {
        flags: VariantFlags::VPOS_SCALE,
        ..VariantKey::default()
    };
    let ps_msl = emit_ps_programmable(&ps, variant).expect("emit PS3");
    assert!(
        ps_msl.contains("struct PsDraw {"),
        "PsDraw declared:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains(&format!(
            "constant PsDraw &ps_draw [[buffer({PS_DRAW_SLOT})]]"
        )),
        "the uniform is a fragment argument:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains(
            "oC0 = float4(floor(in.position.xy * ps_draw.vpos_scale.xy), in.position.zw);"
        ),
        "vPos read resolves to the scaled, floored position:\n{ps_msl}"
    );
    assert_eq!(
        ps_msl.matches("[[position").count(),
        1,
        "still one [[position]]:\n{ps_msl}"
    );
}

#[test]
fn the_render_scale_variant_is_a_shader_cache_key_of_its_own() {
    // The compensation is not baked into a shared library: it reaches the
    // function through the `PsDraw` uniform, and the flag that makes the
    // function take that uniform is part of the key the library is cached
    // under. A key that dropped the bit would serve the identity emission to
    // a draw into a scaled target, so pin the separation on the hash the
    // on-disk cache is addressed by, over the same `(program, variant)` pair
    // the draw path hashes.
    let plain = VariantKey::default();
    let scaled = VariantKey {
        flags: VariantFlags::VPOS_SCALE,
        ..VariantKey::default()
    };
    assert_ne!(plain, scaled, "the flag is part of the key's identity");
    let program = 0x1234_5678_9abc_def0u64;
    assert_ne!(
        ff_key_hash(&(program, plain)),
        ff_key_hash(&(program, scaled)),
        "one shader's two variants address two cache entries"
    );
}

#[test]
fn the_render_scale_variant_leaves_a_shader_without_vpos_unchanged() {
    // The bit rides only on a shader that declares `vPos`; for any other the
    // draw path never sets it, and the emitter ignores it either way, so the
    // MSL is byte-identical to the default variant's.
    let ps = parse(&red_constant_ps()).expect("PS parse");
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit");
    let variant = VariantKey {
        flags: VariantFlags::VPOS_SCALE,
        ..VariantKey::default()
    };
    let flagged = emit_ps_programmable(&ps, variant).expect("emit");
    assert_eq!(plain, flagged);
    assert!(!flagged.contains("PsDraw"));
}

#[test]
fn sm3_ps_vface_via_misctype_converts_bool_to_signed_float() {
    // ps_3_0 { dcl_face vFace; mov oC0, vFace; }
    // SM3 vFace lives on RegKind::MiscType index 1; D3D9 convention is
    // +1.0 for front-facing, -1.0 for back. MSL gives a bool through
    // [[front_facing]], so the prologue computes the signed float.
    const TYPE_MISCTYPE: u32 = 17;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_MISCTYPE, 1, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_MISCTYPE, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("bool v_face_in [[front_facing]]"),
        "vFace dcl must add a [[front_facing]] fragment-function arg:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("float v_face = v_face_in ? 1.0 : -1.0;"),
        "vFace prologue must convert bool → ±1.0 float:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("oC0 = float4(v_face);"),
        "vFace read must broadcast the float to float4:\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_without_misctype_dcl_omits_position_and_face_args() {
    // Shaders that don't use vPos/vFace must not pay the cost of always-on
    // [[position]] / [[front_facing]] args. Metal accepts them either way
    // but the emitted MSL stays minimal.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_COLOR, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        !ps_msl.contains("[[position]]") || ps_msl.matches("[[position]]").count() == 1,
        "Varyings already has one [[position]] field; no extra:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("[[front_facing]]"),
        "no [[front_facing]] arg unless shader declares vFace:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("v_face"),
        "no v_face local unless shader declares vFace:\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_input_fog_resolves_to_fog_varying() {
    // ps_3_0 { dcl_fog v3; mov oC0, v3; }
    // The PS3 fog varying must read `in.fog`, mirroring VS3 `dcl_fog oN`
    // writes. Without a dedicated fog arm the input hits the wildcard and
    // returns `in.color0`, mis-sampling distant fog.
    const DCL_FOG: u8 = 11;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_FOG, 0),
        dst_token(TYPE_INPUT, 3, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 3, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("oC0 = in.fog;"),
        "PS3 dcl_fog v3 must read from in.fog:\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_input_position_resolves_to_position_varying() {
    // ps_3_0 { dcl_position v0; mov oC0, v0; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("oC0 = in.position;"),
        "PS3 dcl_position v0 must read from in.position (screen-space coord post-rasterizer):\n{ps_msl}"
    );
}

#[test]
fn sm2_ps_input_mapping_unaffected_by_sm3_changes() {
    // Companion to `ps2_vreg_input_maps_to_color_not_position`: SM2 PS with
    // `dcl v0` (Position usage encoded structurally) must still resolve to
    // in.color0, not in.texcoord0 or in.position.
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );
    assert!(
        msl.contains("oC0 = c0;"),
        "SM2 PS must resolve oC0 to c0:\n{msl}"
    );
}

#[test]
fn dp4_emits_plain_dot() {
    // dp4 lowers to plain MSL `dot(a, b)`. Apple Silicon has hardware
    // dot-product; let the compiler use it. Cross-shader bit-invariance
    // is not the goal here — per-pipeline matrix bytes genuinely differ
    // between FF and programmable paths, so no emit-shape trick can
    // bridge it, and the only depth bias is the application's own.
    //
    // vs_2_0 { dcl_position v0; dp4 r0.x, v0, c0; mov oPos, r0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DP4, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("dot(in.v0, vs_c[0])"),
        "dp4 must lower to plain MSL dot():\n{msl}"
    );
    assert!(
        !msl.contains("fma_dot4_invariant"),
        "fma_dot4_invariant helper should be gone:\n{msl}"
    );
}

#[test]
fn dp3_emits_plain_dot() {
    // Same rationale as `dp4_emits_plain_dot`. Plain `dot()` on the
    // `.xyz` swizzle of both operands.
    //
    // vs_2_0 { dcl_position v0; dp3 r0.x, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DP3, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("dot((in.v0).xyz, (vs_c[0]).xyz)"),
        "dp3 must lower to plain MSL dot() on .xyz swizzle:\n{msl}"
    );
    assert!(
        !msl.contains("fma_dot3_invariant"),
        "fma_dot3_invariant helper should be gone:\n{msl}"
    );
}

#[test]
fn vertex_blend_msl_compiles_under_metal() {
    use crate::dxso::ff::{FfVsFlags, FfVsKey, emit_vs_ff};
    // Exercise the three blend shapes through a real Metal compile so the
    // emitted code is syntactically and semantically valid MSL — same
    // discipline as `every_emitted_msl_compiles_under_metal` for the SM3
    // corpus.
    let mut sequential = FfVsKey {
        ppl_knee: 0,
        ppl_mask: 0,
        ppl_model: 0,
        flags: FfVsFlags::HAS_NORMAL | FfVsFlags::COLOR_VERTEX,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0,
        light_directional_mask: 0,
        light_spot_mask: 0,
        diffuse_source: 1,
        ambient_source: 0,
        specular_source: 2,
        emissive_source: 0,
        fog_mode: 0,
        tci: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 3,
        declared_weights_count: 2,
        clip_plane_count: 0,
        passthrough: [0; 8],
    };
    metal_compile_or_fail(&emit_vs_ff(&sequential));

    sequential.vertex_blend_count = 4;
    sequential
        .flags
        .insert(FfVsFlags::VERTEX_BLEND_INDEXED | FfVsFlags::DECLARED_INDICES);
    sequential.declared_weights_count = 3;
    metal_compile_or_fail(&emit_vs_ff(&sequential));

    let indexed_only = FfVsKey {
        vertex_blend_count: 1,
        declared_weights_count: 0,
        clip_plane_count: 0,
        passthrough: [0; 8],
        ..sequential
    };
    metal_compile_or_fail(&emit_vs_ff(&indexed_only));
}

#[test]
fn ff_vs_lit_specular_msl_compiles_under_metal() {
    use crate::dxso::ff::{FfVsFlags, FfVsKey, emit_vs_ff};
    // Lit + specular + one directional, one point, and one spot light,
    // through a real Metal compile — covers the Blinn-Phong block, the
    // per-light specular-row reads, and the spot cone factor.
    let key = FfVsKey {
        ppl_knee: 0,
        ppl_mask: 0,
        ppl_model: 0,
        flags: FfVsFlags::HAS_NORMAL
            | FfVsFlags::COLOR_VERTEX
            | FfVsFlags::LIGHTING_ENABLED
            | FfVsFlags::SPECULAR_ENABLE,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0b111,
        light_directional_mask: 0b001,
        light_spot_mask: 0b100,
        diffuse_source: 1,
        ambient_source: 0,
        specular_source: 2,
        emissive_source: 0,
        fog_mode: 0,
        tci: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
        clip_plane_count: 0,
        passthrough: [0; 8],
    };
    metal_compile_or_fail(&emit_vs_ff(&key));
}

#[test]
fn ppl_vs_msl_compiles_under_metal() {
    use crate::dxso::ff::{FfVsFlags, FfVsKey, emit_vs_ff};
    let points = FfVsKey {
        ppl_knee: 0,
        flags: FfVsFlags::HAS_NORMAL
            | FfVsFlags::HAS_COLOR0
            | FfVsFlags::COLOR_VERTEX
            | FfVsFlags::LIGHTING_ENABLED,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0b111,
        light_directional_mask: 0b001,
        light_spot_mask: 0,
        diffuse_source: 1,
        ambient_source: 0,
        specular_source: 2,
        emissive_source: 0,
        fog_mode: 0,
        tci: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
        clip_plane_count: 0,
        passthrough: [0; 8],
        ppl_mask: 0b110,
        ppl_model: 0,
    };
    metal_compile_or_fail(&emit_vs_ff(&points));
    metal_compile_or_fail(&emit_vs_ff(&FfVsKey {
        ppl_mask: 0b111,
        ..points.clone()
    }));
    metal_compile_or_fail(&emit_vs_ff(&FfVsKey {
        vertex_blend_count: 2,
        declared_weights_count: 1,
        ..points
    }));
}

#[test]
fn hemi_vs_msl_compiles_under_metal() {
    use crate::dxso::ff::{FfVsFlags, FfVsKey, emit_vs_ff};
    // No active light: the normal is declared for the hemi weight alone.
    let no_lights = FfVsKey {
        ppl_knee: 0,
        ppl_mask: 0,
        ppl_model: 0,
        flags: FfVsFlags::HAS_NORMAL
            | FfVsFlags::HAS_COLOR0
            | FfVsFlags::COLOR_VERTEX
            | FfVsFlags::LIGHTING_ENABLED
            | FfVsFlags::HEMI_AMBIENT,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0,
        light_directional_mask: 0,
        light_spot_mask: 0,
        diffuse_source: 1,
        ambient_source: 0,
        specular_source: 2,
        emissive_source: 0,
        fog_mode: 0,
        tci: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
        clip_plane_count: 0,
        passthrough: [0; 8],
    };
    metal_compile_or_fail(&emit_vs_ff(&no_lights));
    metal_compile_or_fail(&emit_vs_ff(&FfVsKey {
        light_active_mask: 0b111,
        light_directional_mask: 0b001,
        ..no_lights
    }));
}

#[test]
fn ff_vs_with_clip_planes_emits_clip_distances_and_compiles() {
    use crate::dxso::ff::{FfVsFlags, FfVsKey, emit_vs_ff};
    // Two enabled planes: the VS-only Varyings member carries two lanes (MSL
    // wants the attribute between the name and the dimension), the world
    // position comes back through the inverse view, and one distance is
    // written per plane. The PS struct must stay free of the member.
    let key = FfVsKey {
        ppl_knee: 0,
        ppl_mask: 0,
        ppl_model: 0,
        flags: FfVsFlags::HAS_COLOR0 | FfVsFlags::COLOR_VERTEX,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0,
        light_directional_mask: 0,
        light_spot_mask: 0,
        diffuse_source: 1,
        ambient_source: 0,
        specular_source: 2,
        emissive_source: 0,
        fog_mode: 0,
        tci: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
        clip_plane_count: 2,
        passthrough: [0; 8],
    };
    let msl = emit_vs_ff(&key);
    assert!(
        msl.contains("float clip_distance [[clip_distance]] [2];"),
        "VS Varyings must declare two clip lanes:\n{msl}"
    );
    assert!(
        msl.contains("dot(pos_view, vs_draw.inv_view[3])"),
        "FF clip planes go through the inverse view:\n{msl}"
    );
    assert!(
        msl.contains("out.clip_distance[1] = dot(world_pos, vs_draw.clip[1]);"),
        "one distance per enabled plane:\n{msl}"
    );
    metal_compile_or_fail(&msl);
    let no_clip = FfVsKey {
        clip_plane_count: 0,
        passthrough: [0; 8],
        ..key
    };
    assert!(
        !emit_vs_ff(&no_clip).contains("clip_distance"),
        "a draw without planes pays nothing"
    );
}

#[test]
fn programmable_vs_with_clip_planes_emits_clip_distances_and_compiles() {
    // vs_1_1 { dcl_position v0; mov oPos, v0 } with three planes: the
    // distances are taken against the shader's own clip-space position,
    // before the half-pixel fixup.
    let bc = [
        0xFFFE_0101,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable_named(
        &vs,
        "clip_vs",
        u16::MAX,
        3,
        VsSamplerKinds::default(),
        None,
        None,
    )
    .expect("emit");
    assert!(
        msl.contains("float clip_distance [[clip_distance]] [3];"),
        "VS Varyings must declare three clip lanes:\n{msl}"
    );
    let clip = msl
        .find("out.clip_distance[2] = dot(out.position, vs_draw.clip[2]);")
        .expect("third clip distance");
    let fixup = msl
        .find("out.position.x += pos_fixup.x * out.position.w;")
        .expect("pos fixup");
    assert!(
        clip < fixup,
        "clip distances use the unfixed position:\n{msl}"
    );
    metal_compile_or_fail(&msl);
    assert!(
        !emit_vs_programmable(&vs)
            .expect("emit")
            .contains("clip_distance"),
        "the zero-plane variant has no clip lanes"
    );
}

#[test]
fn ff_ps_specular_add_msl_compiles_under_metal() {
    use crate::dxso::ff::{FfPsKey, FfStage, emit_ps_ff};
    // End-of-cascade specular add plus a D3DTA_SPECULAR stage argument,
    // through a real Metal compile.
    let mut stages = [FfStage {
        color_op: narrow(D3DTOP_DISABLE),
        ..FfStage::default()
    }; 8];
    stages[0] = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_SPECULAR),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        ..FfStage::default()
    };
    let key = FfPsKey {
        stages,
        specular_add: true,
        tt_projected_mask: 0,
    };
    metal_compile_or_fail(&emit_ps_ff(&key, VariantKey::default()));
}

#[test]
fn texkill_ps20_full_mask_kills_on_all_components() {
    // ps_2_0 { def c0, 1, 0, 0, 1; texkill r0; mov oC0, c0; }
    // r0 is uninitialised — runtime behaviour is not the point here, only
    // that the emitted MSL reads r0 with the full write-mask (`.xyzw`).
    // Decoding the operand in SRC form instead would turn the mask bits
    // into a `.wwxx` swizzle.
    let ps_bc = vec![
        PS_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_TEXKILL, 1),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&ps_bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        ps_msl.contains("if (any((r[0]).xyzw < 0.0)) discard_fragment();"),
        "texkill r0 must read r0 with full .xyzw mask:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains(".wwxx"),
        "texkill must not leak the SRC-form .wwxx swizzle:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn texkill_ps20_partial_mask_honored() {
    // ps_2_0 { def c0, 1, 0, 0, 1; texkill r0.xyz; mov oC0, c0; }
    // SM2+ honors the dst write_mask — without it, some
    // post-processing shaders (e.g. ENB-style effect chains) kill
    // every pixel.
    let ps_bc = vec![
        PS_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_TEXKILL, 1),
        dst_token(TYPE_TEMP, 0, 0b0111, false), // .xyz
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&ps_bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        ps_msl.contains("if (any((r[0]).xyz < 0.0)) discard_fragment();"),
        "texkill r0.xyz must emit .xyz mask:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn texld_dw_modifier_emits_perspective_divide() {
    // ps_2_0 { texld r0, t0_dw, s0; mov oC0, r0; }
    // FXC encodes HLSL `tex2Dproj(s0, t0)` as `texld` with the Dw
    // modifier on the coord source. The emitter must divide the whole
    // texcoord vector by `.w` before sampling — dropping the divide
    // mis-samples shadow cascades (foliage self-shadow flicker).
    // Modifier value 10 = Dw per `parser.rs`.
    const SRC_MOD_DW: u8 = 10;
    let ps_bc = vec![
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d s0
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, SRC_MOD_DW), // t0 with Dw
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&ps_bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        ps_msl.contains("in.texcoord0") && ps_msl.contains("/ ("),
        "Dw modifier must emit a perspective divide:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains(").w)"),
        "Dw modifier must divide by .w (not .z):\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("not implemented"),
        "Dw must no longer trigger the warn-and-passthrough stub:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn depth_sample_compare_gets_level_zero_by_default() {
    // ps_3_0 { dcl_2d s0; dcl t0; texld r0, t0, s0; mov oC0, r0; }
    // Cascade shadow maps have no mips; `sample_compare` with implicit
    // gradients is undefined when neighbour fragments in the 2×2 quad
    // ran `discard_fragment` (alpha-cut foliage receiver). Force
    // `level(0)` so the mip pick doesn't depend on derivatives.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d s0
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS parse");

    let depth_variant = VariantKey {
        depth_sampler_mask: 0b0001,
        depth_fetch_mask: 0,
        fetch4_mask: 0,
        fetch4_alpha_mask: 0,
        raw_depth_red_mask: 0,
        ..VariantKey::default()
    };
    let depth = emit_ps_programmable(&ps, depth_variant).expect("emit depth");
    assert!(
        depth.contains(", level(0)))"),
        "depth sample_compare must pin LOD with level(0):\n{depth}"
    );

    // Non-depth path must NOT acquire a level(0) — the implicit-gradient
    // fix only applies to sample_compare against a depth sampler.
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit plain");
    assert!(
        !plain.contains("level(0)"),
        "non-depth s0.sample must not be pinned to level(0):\n{plain}"
    );
    metal_compile_or_fail(&depth);
}

// ── Shader Model 1 ──

const OP_TEXCOORD: u16 = 64;
const OP_TEX: u16 = 66;
const OP_TEXBEM: u16 = 67;
const OP_TEXBEML: u16 = 68;
const OP_TEXDEPTH: u16 = 87;
const OP_TEXM3X2PAD: u16 = 71;
const OP_TEXM3X3PAD: u16 = 73;
const OP_TEXM3X3VSPEC: u16 = 77;
const OP_TEXM3X2DEPTH: u16 = 84;

#[test]
fn vs_1_1_passthrough_uses_implicit_position_output() {
    // vs_1_1 { dcl_position v0; mov oPos, v0; } — vs_1_1 has no dcl for the
    // implicit oPos output; the RastOut[0] register kind alone routes it.
    let bc = [
        0xFFFE_0101,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("emit vs_1_1");
    assert!(
        msl.contains("out.position = in.v0;"),
        "vs_1_1 oPos write missing:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn programmable_vs_emits_half_pixel_pos_fixup() {
    // Every DXSO VS declares the buffer-13 `pos_fixup` uniform and applies a
    // half-pixel window→NDC fixup in the position epilogue (after every
    // instruction, so per-op `oPos` writes stay verbatim) so on-boundary
    // geometry matches the D3D9 reference filling convention.
    let bc = [
        0xFFFE_0101,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("emit vs_1_1");
    assert!(
        msl.contains(&format!(
            "constant PosFixup &pos_fixup [[buffer({VS_POS_FIXUP_SLOT})]]"
        )),
        "VS must declare the pos_fixup uniform at its slot:\n{msl}"
    );
    assert!(
        msl.contains(POS_FIXUP_MSL),
        "VS must declare the PosFixup struct:\n{msl}"
    );
    assert!(
        msl.contains("out.position.x += pos_fixup.x * out.position.w;")
            && msl.contains("out.position.y += pos_fixup.y * out.position.w;"),
        "VS must apply the half-pixel pos_fixup epilogue:\n{msl}"
    );
    // The `mov oPos, v0` still lands verbatim before the epilogue.
    assert!(
        msl.contains("out.position = in.v0;"),
        "oPos write must survive the epilogue:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn programmable_vs_zeroes_the_texcoords_it_never_writes() {
    // `mov oPos, v0` writes no texture coordinate, so a PS reading `t0`
    // must see zero, as it does on D3D9 hardware, rather than whatever the
    // register held.
    let bc = [
        0xFFFE_0101,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("emit vs_1_1");
    for i in 0..16 {
        assert!(
            msl.contains(&format!("out.texcoord{i} = float4(0.0);")),
            "texcoord{i} must start at zero:\n{msl}"
        );
    }
}

#[test]
fn programmable_vs_adds_the_depth_bias_after_fog_z() {
    // `D3DRS_DEPTHBIAS` is an absolute depth offset, which Metal's
    // `setDepthBias` cannot express on a float depth buffer, so the vertex
    // shader adds it, scaled by `w` to survive the perspective divide. It
    // lands after `fog_z`: the table-fog source adds the raw bias itself and
    // must read the unbiased depth.
    let bc = [
        0xFFFE_0101,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("emit vs_1_1");
    let bias = "float _depth_biased = _pos.z + pos_fixup.depth_bias * _pos.w;";
    let bias_at = msl.find(bias).expect("VS applies the depth bias");
    let fog_z = msl.find("out.fog_z =").expect("VS writes fog_z");
    assert!(
        fog_z < bias_at,
        "fog_z must read the unbiased depth:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_tex_and_texcoord_use_t_register_array() {
    // ps_1_1 { texcoord t0; tex t1; mov oC0, t1; }
    // t0 receives the (clamped) iterated texcoord; t1 samples stage 1 using
    // its own iterated coord and writes the result back to t1.
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEXCOORD, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 1, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_ADDR, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_1");
    assert!(msl.contains("float4 t[8];"), "no t[] register file:\n{msl}");
    assert!(
        msl.contains("t[0] = in.texcoord0;"),
        "t[] not seeded from texcoord varyings:\n{msl}"
    );
    assert!(
        msl.contains("t[0] = float4(saturate(in.texcoord0).xyz, 1.0);"),
        "texcoord must clamp the iterated coord and set w to 1:\n{msl}"
    );
    // `tex t1` samples stage 1 (implicit sampler) at the coord in t[1].
    assert!(
        msl.contains("s1.sample(samp1, (t[1]).xy)"),
        "tex must sample stage 1 from t[1]:\n{msl}"
    );
    assert!(
        msl.contains("texture2d<float> s1 [[texture(1)]]"),
        "implicit SM1 sampler not synthesized:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_implicit_r0_is_the_colour_output() {
    // ps_1_1 { tex t0; mov r0, t0; }
    // SM1 PS has no D3DSPR_COLOROUT register — the final pixel colour is
    // whatever the shader left in r0. The emitter must bridge `oC0 = r[0]`
    // after the body, or every such shader returns the float4(0.0) `oC0`
    // default (black).
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_1");
    assert!(
        msl.contains("oC0 = r[0];"),
        "SM1 PS must route r0 to the colour output:\n{msl}"
    );
    // The bridge must precede `return oC0;` so the returned colour is r0.
    let bridge = msl.find("oC0 = r[0];").expect("bridge present");
    let ret = msl.find("return oC0;").expect("return present");
    assert!(bridge < ret, "bridge must come before return:\n{msl}");
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_4_texld_samples_destination_stage() {
    // ps_1_4 { texcrd r0, t0; phase; texld r1, r0; mov oC0, r1; }
    // ps_1_4 texld has no sampler operand — the sampler index is the dst
    // register number (r1 → sampler 1).
    const OP_PHASE: u16 = 0xFFFD; // D3DSIO_PHASE
    const OP_TEXLD: u16 = 66;
    let bc = [
        0xFFFF_0104,
        opcode_token(OP_TEXCOORD, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_PHASE, 0),
        opcode_token(OP_TEXLD, 2),
        dst_token(TYPE_TEMP, 1, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_4 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_4");
    assert!(
        msl.contains("s1.sample(samp1, (r[0]).xy)"),
        "ps_1_4 texld must sample dst-numbered stage from the coord src:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_add_x2_result_modifier_scales() {
    // ps_1_1 { texcoord t0; add_x2 r0, t0, t0; mov oC0, r0; }
    // The `_x2` result modifier (shift_scale = +1) doubles the result.
    let mut add_dst = dst_token(TYPE_TEMP, 0, 0xF, false);
    add_dst |= 1 << 24; // shift_scale = +1 → ×2
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEXCOORD, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_ADD, 3),
        add_dst,
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_1");
    assert!(
        msl.contains("* 2"),
        "add_x2 must scale the result by 2:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

/// Emits `ps_1_1 { tex t0; <op> t1, t0; mov oC0, t1; }` under `key`.
///
/// `op` is `texbem` or `texbeml`, which perturb stage 1's coord by the bump
/// matrix applied to t0 and then sample stage 1.
fn emit_ps_1_1_texbem(op: u16, key: VariantKey) -> String {
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(op, 2),
        dst_token(TYPE_ADDR, 1, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_ADDR, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    emit_ps_programmable(&ps, key).expect("emit ps_1_1")
}

#[test]
fn ps_1_1_texbem_emits_bump_uniform_and_perturb() {
    // The per-stage bump matrix comes from buffer(12).
    let msl = emit_ps_1_1_texbem(OP_TEXBEM, VariantKey::default());
    assert!(
        msl.contains("constant float4 *bump_env [[buffer(12)]]"),
        "texbem must bind the bump-env uniform on slot 12:\n{msl}"
    );
    assert!(
        msl.contains("bump_env[2]"),
        "texbem on stage 1 must read bump_env[2] (= stage*2):\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_texbem_into_a_cube_keeps_the_stage_z() {
    // The displacement moves the direction's x and y; its z is stage 1's own,
    // and a cube stage takes no projective divide even when the stage asks.
    for op in [OP_TEXBEM, OP_TEXBEML] {
        for tt_projected_mask in [0, 0b0010] {
            let msl = emit_ps_1_1_texbem(
                op,
                VariantKey {
                    cube_sampler_mask: 0b0010,
                    tt_projected_mask,
                    ..VariantKey::default()
                },
            );
            assert!(
                msl.contains("texturecube<float> s1 [[texture(1)]]"),
                "a cube-bound op {op} stage binds texturecube:\n{msl}"
            );
            assert!(
                msl.contains(", (t[1]).z, (t[1]).w)).xyz)"),
                "op {op}: the cube direction carries stage 1's z into the xyz sample:\n{msl}"
            );
            assert!(
                !msl.contains("/ (t[1]).w"),
                "op {op}: a cube stage is never projected:\n{msl}"
            );
            metal_compile_or_fail(&msl);
        }
    }
}

#[test]
fn ps_1_1_texbem_into_a_volume_keeps_the_stage_z() {
    for op in [OP_TEXBEM, OP_TEXBEML] {
        let msl = emit_ps_1_1_texbem(
            op,
            VariantKey {
                volume_sampler_mask: 0b0010,
                ..VariantKey::default()
            },
        );
        assert!(
            msl.contains("texture3d<float> s1 [[texture(1)]]"),
            "a volume-bound op {op} stage binds texture3d:\n{msl}"
        );
        assert!(
            msl.contains(", (t[1]).z, (t[1]).w)).xyz)"),
            "op {op}: the volume coordinate carries stage 1's z into the xyz sample:\n{msl}"
        );
        metal_compile_or_fail(&msl);
    }
}

#[test]
fn ps_1_1_texbem_into_a_projected_volume_divides_the_stage_z() {
    // D3DTTFF_PROJECTED divides the stage's x, y and z by its w before the
    // displacement, with the same zero-w guard on each component.
    for op in [OP_TEXBEM, OP_TEXBEML] {
        let msl = emit_ps_1_1_texbem(
            op,
            VariantKey {
                volume_sampler_mask: 0b0010,
                tt_projected_mask: 0b0010,
                ..VariantKey::default()
            },
        );
        for c in ['x', 'y'] {
            assert!(
                msl.contains(&format!(
                    "(((t[1]).w != 0.0) ? (t[1]).{c} / (t[1]).w : 0.0)"
                )),
                "op {op}: the projected {c} is divided by w:\n{msl}"
            );
        }
        assert!(
            msl.contains(", (((t[1]).w != 0.0) ? (t[1]).z / (t[1]).w : 0.0), 1.0)).xyz)"),
            "op {op}: the projected z is divided by w and reaches the xyz sample:\n{msl}"
        );
        metal_compile_or_fail(&msl);
    }
}

#[test]
fn ps_1_1_texbem_into_a_depth_stage_compares_against_the_stage_z() {
    // A 2D depth stage samples with a comparison whose reference is the
    // coordinate's z, so that reference is stage 1's own z, divided by its w
    // when the stage is projected.
    for (tt_projected_mask, reference) in [
        (0, ", (t[1]).z, (t[1]).w)).z), level(0)))"),
        (
            0b0010,
            ", (((t[1]).w != 0.0) ? (t[1]).z / (t[1]).w : 0.0), 1.0)).z), level(0)))",
        ),
    ] {
        let msl = emit_ps_1_1_texbem(
            OP_TEXBEM,
            VariantKey {
                depth_sampler_mask: 0b0010,
                tt_projected_mask,
                ..VariantKey::default()
            },
        );
        assert!(
            msl.contains("depth2d<float> s1 [[texture(1)]]"),
            "a depth-bound texbem stage binds depth2d:\n{msl}"
        );
        assert!(
            msl.contains("s1.sample_compare(samp1, (float4(") && msl.contains(reference),
            "the compare reference is stage 1's z (projected: {tt_projected_mask}):\n{msl}"
        );
        metal_compile_or_fail(&msl);
    }
}

#[test]
fn ps_1_3_texdepth_writes_depth_output() {
    // ps_1_3 { tex t0; texdepth r5; } — texdepth writes fragment depth from
    // r5.x / r5.y, so the function must return the PsOut depth struct.
    let bc = [
        0xFFFF_0103,
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_TEXDEPTH, 1),
        dst_token(TYPE_TEMP, 5, 0xF, false),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_3 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_3");
    assert!(
        msl.contains("oDepth [[depth(any)]]") && msl.contains("_depth_storage"),
        "texdepth must route through the PsOut depth path:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_3_texm3x2depth_with_a_zero_w_writes_the_far_plane() {
    // ps_1_3 { tex t0; texm3x2pad t1, t0; texm3x2depth t2, t0; mov r0, t0; }
    let bc = [
        0xFFFF_0103,
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_TEXM3X2PAD, 2),
        dst_token(TYPE_ADDR, 1, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_TEXM3X2DEPTH, 2),
        dst_token(TYPE_ADDR, 2, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_3 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_3");
    assert!(
        msl.contains("saturate((t[1].x) / (dot((t[2]).xyz, (t[0]).xyz))) : 1.0);"),
        "texm3x2depth must write depth 1.0 when w is zero:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_texm3x3vspec_reads_the_eye_vector_from_the_iterated_coordinates() {
    // ps_1_1 { tex t0; texm3x3pad t1, t0; texm3x3pad t2, t0;
    // texm3x3vspec t3, t0; mov r0, t3; }
    // The pads overwrite t1 and t2 with their dot products, so the eye vector
    // has to come from the interpolated texture coordinates' .w.
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_TEXM3X3PAD, 2),
        dst_token(TYPE_ADDR, 1, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_TEXM3X3PAD, 2),
        dst_token(TYPE_ADDR, 2, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_TEXM3X3VSPEC, 2),
        dst_token(TYPE_ADDR, 3, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 3, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_1");
    assert!(
        msl.contains("float3(in.texcoord1.w, in.texcoord2.w, in.texcoord3.w)"),
        "vspec must build the eye vector from the iterated coordinates:\n{msl}"
    );
    assert!(
        !msl.contains("t[1].w") && !msl.contains("t[2].w"),
        "vspec must not read .w from the pad registers:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_2_x_static_flow_control_compiles() {
    // ps_2_x { if b0; mov oC0, c0; endif } reads the runtime boolean file;
    // ps_2_x { defi i0, 2, 0, 0, 0; mov r0, c0; rep i0; add r0, r0, c1;
    // endrep; mov oC0, r0 } runs a defined loop count.
    let bool_branch = [
        0xFFFF_0201,
        opcode_token(OP_IF, 1),
        src_token(TYPE_CONSTBOOL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let int_loop = [
        0xFFFF_0201,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        2,
        0,
        0,
        0,
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_REP, 1),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ADD, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDREP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    for bc in [&bool_branch[..], &int_loop[..]] {
        let ps = parse(bc).expect("ps_2_x parse");
        assert!(!ps.violates_constant_register_limits());
        let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_2_x");
        metal_compile_or_fail(&msl);
    }
}

#[test]
fn constant_register_limits_reject_out_of_range_files() {
    // Addressing a constant register past the model's file must be caught
    // so CreateShader returns INVALIDCALL. The bytecode is hand-assembled
    // to use the out-of-range indices the native assembler refuses.

    // vs_1_1 { def c255; add r0, v0, c255; mov oPos, r0 } — c255 is the last
    // in-range vertex float constant.
    let vs_float_in_range = [
        0xFFFE_0101_u32,
        0x0000_001F,
        0x8000_0000,
        0x900F_0000,
        0x0000_0051,
        0xA00F_00FF,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x0000_0002,
        0x800F_0000,
        0x90E4_0000,
        0xA0E4_00FF,
        0x0000_0001,
        0xC00F_0000,
        0x80E4_0000,
        0x0000_FFFF,
    ];
    assert!(
        !parse(&vs_float_in_range)
            .expect("vs_float_in_range parse")
            .violates_constant_register_limits(),
        "c255 is the last in-range vertex float constant"
    );

    // Same shader at c256 — one past the 256-entry vertex float file.
    let vs_float_over = [
        0xFFFE_0101_u32,
        0x0000_001F,
        0x8000_0000,
        0x900F_0000,
        0x0000_0051,
        0xA00F_0100,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x0000_0002,
        0x800F_0000,
        0x90E4_0000,
        0xA0E4_0100,
        0x0000_0001,
        0xC00F_0000,
        0x80E4_0000,
        0x0000_FFFF,
    ];
    assert!(
        parse(&vs_float_over)
            .expect("vs_float_over parse")
            .violates_constant_register_limits(),
        "c256 overflows the vertex float file"
    );

    // vs_3_0 { defi i16; rep i16; add r0,r0,v0; endrep; mov o0,r0 } — i16 is one
    // past the 16-entry integer file.
    let vs_int_over = [
        0xFFFE_0300_u32,
        0x0200_001F,
        0x8000_0000,
        0x900F_0000,
        0x0200_001F,
        0x8000_0000,
        0xE00F_0000,
        0x0500_0030,
        0xF00F_0010,
        0x0000_0001,
        0x0000_0001,
        0x0000_0001,
        0x0000_0001,
        0x0100_0026,
        0xF0E4_0010,
        0x0300_0002,
        0x800F_0000,
        0x80E4_0000,
        0x90E4_0000,
        0x0000_0027,
        0x0200_0001,
        0xE00F_0000,
        0x80E4_0000,
        0x0000_FFFF,
    ];
    assert!(
        parse(&vs_int_over)
            .expect("vs_int_over parse")
            .violates_constant_register_limits(),
        "i16 overflows the integer constant file"
    );
}

/// `ps_3_0 { def c0, 0,1,0,0; def c1, 0,0,1,0; mov oC0, c0; mov oC1, c1; }`
///
/// The two-target shape a deferred renderer's G-buffer pass uses.
fn two_target_ps() -> Vec<u32> {
    vec![
        PS3_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 1, 0xF, false),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 1, 0xF, false),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]
}

#[test]
fn color_out_mask_reports_written_targets() {
    let ps = parse(&two_target_ps()).expect("ps parse");
    assert_eq!(ps.color_out_mask(), 0b11);
    let single = parse(&red_constant_ps()).expect("ps parse");
    assert_eq!(single.color_out_mask(), 0b1);
    // A write beyond oC3 is outside the D3D9 limit and is not reported.
    let mut beyond = two_target_ps();
    let oc1_dst = dst_token(TYPE_COLOROUT, 1, 0xF, false);
    let slot = beyond
        .iter()
        .position(|&t| t == oc1_dst)
        .expect("oC1 dst present");
    beyond[slot] = dst_token(TYPE_COLOROUT, 4, 0xF, false);
    assert_eq!(parse(&beyond).expect("ps parse").color_out_mask(), 0b1);
}

#[test]
fn ps_oc1_exports_color1_when_the_attachment_is_present() {
    let ps = parse(&two_target_ps()).expect("ps parse");
    let variant = VariantKey {
        color_out_mask: 0b11,
        ..VariantKey::default()
    };
    let msl = emit_ps_programmable(&ps, variant).expect("emit");
    assert!(
        msl.contains("fragment PsOut "),
        "two targets return a struct:\n{msl}"
    );
    assert!(msl.contains("float4 oC0 [[color(0)]];"), "{msl}");
    assert!(msl.contains("float4 oC1 [[color(1)]];"), "{msl}");
    assert!(msl.contains("_ps_out.oC1 = oC1;"), "{msl}");
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_oc1_sinks_when_the_attachment_is_absent() {
    // The default key means render target 0 only: the oC1 store lands in a
    // plain local and the function keeps its bare float4 return, so a pass
    // with one colour attachment never sees an output it cannot bind.
    let ps = parse(&two_target_ps()).expect("ps parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit");
    assert!(msl.contains("fragment float4 "), "{msl}");
    assert!(!msl.contains("color(1)"), "{msl}");
    assert!(msl.contains("float4 oC1 = float4(0.0);"), "{msl}");
    assert!(
        msl.contains("    oC1 = c1;"),
        "the write still has a target:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_oc1_with_depth_out_returns_every_member() {
    const TYPE_DEPTHOUT: u32 = 9;
    let mut bc = two_target_ps();
    bc.pop();
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_DEPTHOUT, 0, 0x1, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let ps = parse(&bc).expect("ps parse");
    let variant = VariantKey {
        color_out_mask: 0b11,
        flags: VariantFlags::SRGB_WRITE,
        ..VariantKey::default()
    };
    let msl = emit_ps_programmable(&ps, variant).expect("emit");
    assert!(msl.contains("float4 oC1 [[color(1)]];"), "{msl}");
    assert!(msl.contains("float oDepth [[depth(any)]];"), "{msl}");
    assert!(msl.contains("_ps_out.oDepth = _depth_storage.x;"), "{msl}");
    assert!(
        msl.contains("oC0.rgb = mtld3d_linear_to_srgb(oC0.rgb);"),
        "{msl}"
    );
    assert!(
        msl.contains("oC1.rgb = mtld3d_linear_to_srgb(oC1.rgb);"),
        "{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn color_out_mask_bit0_is_implied_for_single_target_shaders() {
    // A present mask that names extra attachments the shader never writes
    // must not change a single-output shader's MSL at all.
    let ps = parse(&red_constant_ps()).expect("ps parse");
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit");
    let variant = VariantKey {
        color_out_mask: 0b1111,
        ..VariantKey::default()
    };
    let masked = emit_ps_programmable(&ps, variant).expect("emit");
    assert_eq!(plain, masked);
}

// An unbound-sampler pixel shader: `dcl_<dim> s0; dcl t0; texld r0, t0, s0;
// mov oC0, r0`. The dcl token's high nibble selects the sampler dimension.
const UNBOUND_PS_2D: [u32; 12] = [
    0xffff_0200,
    0x0200_001f,
    0x9000_0000,
    0xa00f_0800,
    0x0200_001f,
    0x8000_0000,
    0xb00f_0000,
    0x0300_0042,
    0x800f_0000,
    0xb0e4_0000,
    0xa0e4_0800,
    0x0000_ffff,
];

#[test]
fn declared_ps_samplers_reads_the_sampler_dimension() {
    // 2D (0x90000000), cube (0x98000000), volume (0xa0000000): the dcl token's
    // high nibble carries the sampler dimensionality.
    for (dcl, expected) in [
        (0x9000_0000u32, TextureType::Texture2D),
        (0x9800_0000, TextureType::TextureCube),
        (0xa000_0000, TextureType::Texture3D),
    ] {
        let mut bc = UNBOUND_PS_2D;
        bc[2] = dcl;
        let prog = parse(&bc).expect("unbound-sampler ps_2_0 should parse");
        let samplers = declared_ps_samplers(&prog);
        assert_eq!(samplers.len(), 1, "one declared sampler");
        assert_eq!(
            samplers.get(&0),
            Some(&expected),
            "sampler s0 dimensionality"
        );
    }
}

const TYPE_CONSTBOOL: u32 = 14;
const OP_DEFB: u16 = 47;

#[test]
fn defb_emits_bool_local() {
    // vs_3_0 { defb b0, true; dcl_position v0; if b0 mov oT0, v0 endif }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFB, 2),
        dst_token(TYPE_CONSTBOOL, 0, 0xF, false),
        1u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_IF, 1),
        src_token(TYPE_CONSTBOOL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    assert!(
        !vs.uses_dynamic_bool_constants(),
        "a defb-defined bool is not dynamic"
    );
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("bool b0 = true;"),
        "defb must emit a `bool bN` local:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("if ((float4(float(b0))).x != 0.0)"),
        "`if b0` must test the local:\n{vs_msl}"
    );
    assert!(
        !vs_msl.contains("vs_b"),
        "no runtime bitmask uniform for a defb-only shader:\n{vs_msl}"
    );
}

#[test]
fn dynamic_bool_constant_reads_the_runtime_vs_b_bitmask() {
    // vs_3_0 { dcl_position v0; if b3 mov oT0, v0 endif }
    // b3 has NO `defb`: it is fed by SetVertexShaderConstantB, so the branch
    // must read bit 3 of the `vs_b` uniform.
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_IF, 1),
        src_token(TYPE_CONSTBOOL, 3, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    assert!(vs.uses_dynamic_bool_constants());
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("constant uint &vs_b [[buffer(26)]]"),
        "a dynamic bool declares the bitmask uniform:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("(vs_b >> 3u) & 1u"),
        "`if b3` reads bit 3 of the bitmask:\n{vs_msl}"
    );
}

#[test]
fn sm3_texcoord_index_above_seven_links_through_the_varyings() {
    // vs_3_0 { dcl_position v0; dcl_texcoord8 o0; mov o0, v0; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 8),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("float4 texcoord8;") && vs_msl.contains("float4 texcoord15;"),
        "the varyings carry every SM3 texcoord index:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.texcoord8"),
        "`dcl_texcoord8 o0` writes the matching varying:\n{vs_msl}"
    );
}

#[test]
fn ps_writing_odepth_drops_the_export_without_a_depth_attachment() {
    // ps_3_0 { dcl t0; mov oDepth, t0.x; mov oC0, t0; } against a pass with
    // no depth attachment: the value is still computed, the return type stays
    // a bare float4 and nothing binds `[[depth(any)]]`.
    const TYPE_DEPTHOUT: u32 = 9;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_DEPTHOUT, 0, 0x1, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let variant = VariantKey {
        flags: VariantFlags::NO_DEPTH_ATTACHMENT,
        ..VariantKey::default()
    };
    let ps_msl = emit_ps_programmable(&ps, variant).expect("emit PS3");
    assert!(
        !ps_msl.contains("[[depth(any)]]") && !ps_msl.contains("_ps_out.oDepth"),
        "no depth export against a depth-less pass:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("fragment float4 mtld3d_ps("),
        "a single colour output keeps the bare float4 return:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("_depth_storage"),
        "the oDepth write still has a target:\n{ps_msl}"
    );
}

#[test]
fn dynamic_int_constant_reads_the_runtime_ps_i_buffer() {
    // ps_3_0 { rep i0; mov r0, c0; endrep; mov oC0, r0 }
    // i0 has NO `defi`: it is fed by SetPixelShaderConstantI, so the loop
    // count must read the runtime `ps_i` file, the fragment twin of `vs_i`.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_REP, 1),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDREP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    assert!(ps.uses_dynamic_int_constants());
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains(&format!(
            "constant int4 *ps_i [[buffer({PS_INT_CONST_SLOT})]]"
        )),
        "a dynamic-int-const PS declares the ps_i file:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("ps_i[0]"),
        "`rep i0` reads the runtime file, not a baked local:\n{ps_msl}"
    );
}

#[test]
fn dynamic_bool_constant_reads_the_runtime_ps_b_bitmask() {
    // ps_3_0 { if b3 mov oC0, c0 endif }
    // b3 has NO `defb`: it is fed by SetPixelShaderConstantB, so the branch
    // must read bit 3 of the `ps_b` uniform.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_IF, 1),
        src_token(TYPE_CONSTBOOL, 3, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    assert!(ps.uses_dynamic_bool_constants());
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains(&format!(
            "constant uint &ps_b [[buffer({PS_BOOL_CONST_SLOT})]]"
        )),
        "a dynamic bool declares the bitmask uniform:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("(ps_b >> 3u) & 1u"),
        "`if b3` reads bit 3 of the bitmask:\n{ps_msl}"
    );
}

/// A `ps_3_0` wrapping one sampling instruction against `s0` and `t0`.
///
/// `ps_3_0 { dcl_2d s0; dcl_texcoord0 t0; <op>; mov oC0, r0; }`, so the
/// LOD-bias variant tests below differ only in the instruction they pass.
fn ps3_sampling_program(op: u16, extra_srcs: &[u32]) -> Vec<u32> {
    let mut bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000,
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(
            op,
            u32::try_from(3 + extra_srcs.len()).expect("operand count fits u32"),
        ),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
    ];
    bc.extend_from_slice(extra_srcs);
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    bc
}

/// A `ps_3_0` sample whose sampler result swaps red and blue.
fn ps3_sampler_swizzle_program(op: u16, control: u32, extra_srcs: &[u32]) -> Vec<u32> {
    let mut bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000,
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(0.25),
        f32::to_bits(0.5),
        f32::to_bits(0.75),
        f32::to_bits(1.0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 1, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 2, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(
            op,
            u32::try_from(3 + extra_srcs.len()).expect("operand count fits u32"),
        ) | control,
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_BGRA, 0),
    ];
    bc.extend_from_slice(extra_srcs);
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    bc
}

#[test]
fn ps3_sampling_forms_apply_sampler_result_swizzles() {
    const OP_TEXLD: u16 = 66;
    const TEXLD_PROJECT: u32 = 1 << 16;
    const TEXLD_BIAS: u32 = 2 << 16;
    let gradients = [
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 2, SWIZ_IDENTITY, 0),
    ];
    for (name, op, control, extra_srcs) in [
        ("texld", OP_TEXLD, 0, &[][..]),
        ("texldb", OP_TEXLD, TEXLD_BIAS, &[][..]),
        ("texldp", OP_TEXLD, TEXLD_PROJECT, &[][..]),
        ("texldl", OP_TEXLDL, 0, &[][..]),
        ("texldd", OP_TEXLDD, 0, &gradients[..]),
    ] {
        let ps = parse(&ps3_sampler_swizzle_program(op, control, extra_srcs))
            .unwrap_or_else(|err| panic!("parse {name}: {err:?}"));
        let msl = emit_ps_programmable(&ps, VariantKey::default())
            .unwrap_or_else(|err| panic!("emit {name}: {err:?}"));
        assert!(
            msl.contains(")).zyxw"),
            "{name} must apply the sampler's .bgra result swizzle after sampling:\n{msl}"
        );
        metal_compile_or_fail(&msl);
    }
}

#[test]
fn sampler_result_swizzle_precedes_saturate_mask_and_predication() {
    const OP_TEXLD: u16 = 66;
    let setp_lt_token = u32::from(OP_SETP) | (4 << 16) | (3 << 24);
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000,
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 1, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(1.0),
        f32::to_bits(1.0),
        f32::to_bits(1.0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_TEXLD, 4) | (1 << 28),
        dst_token(TYPE_TEMP, 0, 0xA, true),
        src_token(TYPE_PREDICATE, 0, SWIZ_XXXX, 0),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_BBBB, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        msl.contains(
            "r[0].yw = select(r[0].yw, \
             (saturate((s0.sample(samp0, (in.texcoord0).xy)).zzzz)).yw, p0.xx);"
        ),
        "the component predicate must select the sampled, swizzled, saturated, and masked value:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

fn lod_bias_variant() -> VariantKey {
    VariantKey {
        flags: VariantFlags::LOD_BIAS,
        ..VariantKey::default()
    }
}

#[test]
fn lod_bias_variant_biases_an_implicit_lod_sample() {
    const OP_TEXLD: u16 = 66;
    let ps = parse(&ps3_sampling_program(OP_TEXLD, &[])).expect("PS3 parse");

    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        !plain.contains("lod_bias"),
        "an unbiased draw must keep the shader it had:\n{plain}"
    );

    let biased = emit_ps_programmable(&ps, lod_bias_variant()).expect("emit PS3");
    assert!(
        biased.contains(&format!(
            "constant float4 *lod_bias [[buffer({PS_LOD_BIAS_SLOT})]]"
        )),
        "the biased variant must take the per-slot bias table:\n{biased}"
    );
    assert!(
        biased.contains("bias(lod_bias[0].x)"),
        "texld must apply the bound slot's bias:\n{biased}"
    );
    metal_compile_or_fail(&biased);
}

#[test]
fn lod_table_variant_offsets_and_clamps_an_explicit_lod_sample() {
    // Metal ignores sampler LOD clamps at an explicit level, so `texldl`
    // reads the stage's row: the texture LOD and the game bias shift the
    // level, the finest level clamps it. MSL takes one LOD option per sample,
    // so it carries no `bias()` as well.
    let ps = parse(&ps3_sampling_program(OP_TEXLDL, &[])).expect("PS3 parse");
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        plain.contains("level((in.texcoord0).w)"),
        "without the table texldl keeps its own LOD:\n{plain}"
    );
    let biased = emit_ps_programmable(&ps, lod_bias_variant()).expect("emit PS3");
    assert!(
        biased.contains("level(max((in.texcoord0).w + lod_bias[0].z, lod_bias[0].w))"),
        "texldl offsets and clamps through its slot's row:\n{biased}"
    );
    assert!(
        !biased.contains("bias("),
        "texldl must not also carry a bias:\n{biased}"
    );
    metal_compile_or_fail(&biased);
}

#[test]
fn lod_table_variant_clamps_a_depth_sample_level() {
    let ps = parse(&ps3_sampling_program(OP_TEX, &[])).expect("PS3 parse");
    for fetch in [0, 1] {
        let variant = VariantKey {
            depth_sampler_mask: 1,
            depth_fetch_mask: fetch,
            flags: VariantFlags::LOD_BIAS,
            ..VariantKey::default()
        };
        let msl = emit_ps_programmable(&ps, variant).expect("emit depth");
        assert!(
            msl.contains(", level(max(lod_bias[0].w, 0.0))"),
            "a depth sample with no level of its own samples the finest level:\n{msl}"
        );
        metal_compile_or_fail(&msl);
    }
}

#[test]
fn explicit_lod_samplers_name_the_texldl_slots() {
    let texldl = parse(&ps3_sampling_program(OP_TEXLDL, &[])).expect("PS3 parse");
    assert_eq!(super::explicit_lod_samplers(&texldl), 1);
    let texld = parse(&ps3_sampling_program(OP_TEX, &[])).expect("PS3 parse");
    assert_eq!(super::explicit_lod_samplers(&texld), 0);
}

#[test]
fn ff_depth_samples_under_the_lod_table_compile_under_metal() {
    // The fixed-function depth and raw-depth stages pin the table's finest
    // level, and the cascade must still compile with the table declared.
    use mtld3d_types::D3DTA_TEXTURE;

    use crate::dxso::{FfPsKey, FfStage, FfStageFlags, emit_ps_ff};
    let mut stages = [FfStage {
        color_op: narrow(D3DTOP_DISABLE),
        ..FfStage::default()
    }; 8];
    stages[0] = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_TEXTURE),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_TEXTURE),
        flags: FfStageFlags::HAS_TEXTURE,
        ..FfStage::default()
    };
    let ps_key = FfPsKey {
        stages,
        specular_add: false,
        tt_projected_mask: 0,
    };
    for (fetch, red) in [(0, 0), (1, 0), (1, 1)] {
        let msl = emit_ps_ff(
            &ps_key,
            VariantKey {
                depth_sampler_mask: 1,
                depth_fetch_mask: fetch,
                raw_depth_red_mask: red,
                flags: VariantFlags::LOD_BIAS,
                ..VariantKey::default()
            },
        );
        assert!(
            msl.contains("level(max(lod_bias[0].w, 0.0))"),
            "the depth stage pins the table's finest level:\n{msl}"
        );
        metal_compile_or_fail(&msl);
    }
}

#[test]
fn lod_bias_variant_scales_texldd_gradients() {
    // A gradient sample cannot take `bias()` as well, so the bias rides on the
    // derivatives: scaling both by `exp2(bias)` shifts the computed LOD by it.
    let extra = [
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 2, SWIZ_IDENTITY, 0),
    ];
    let ps = parse(&ps3_sampling_program(OP_TEXLDD, &extra)).expect("PS3 parse");
    let biased = emit_ps_programmable(&ps, lod_bias_variant()).expect("emit PS3");
    assert!(
        biased.contains("gradient2d((r[1]).xy * lod_bias[0].y, (r[2]).xy * lod_bias[0].y)"),
        "texldd must scale both gradients by the slot's exp2 lane:\n{biased}"
    );
    assert!(
        !biased.contains("bias("),
        "the gradient sample carries no second LOD option:\n{biased}"
    );
    metal_compile_or_fail(&biased);
}

#[test]
fn programmable_ps_writes_the_sample_mask_when_the_variant_carries_one() {
    // Metal takes a coverage mask only from a `[[sample_mask]]` fragment
    // output, so `D3DRS_MULTISAMPLEMASK` turns the bare `float4` return into a
    // struct carrying the mask beside the colour.
    let variant = VariantKey {
        sample_mask: 0b0011,
        flags: VariantFlags::SAMPLE_MASK,
        ..VariantKey::default()
    };
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &red_constant_ps(), variant);
    assert!(
        msl.contains("uint oMask [[sample_mask]];"),
        "the PS output struct must declare the mask:\n{msl}"
    );
    assert!(
        msl.contains("_ps_out.oMask = 3u;"),
        "and write the variant's value:\n{msl}"
    );
}

#[test]
fn programmable_ps_keeps_the_bare_return_without_a_sample_mask() {
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );
    assert!(!msl.contains("sample_mask"), "{msl}");
    assert!(!msl.contains("struct PsOut"), "{msl}");
}

/// `ps_3_0 { dcl_<dim> s0; dcl t0; texld r0, t0, s0; mov oC0, r0; }`.
///
/// `sampler_usage` is the `dcl` usage token whose bits 27..30 carry the
/// declared texture type: `0x9000_0000` for `dcl_2d`, `0xA000_0000` for
/// `dcl_volume`, `0x9800_0000` for `dcl_cube`.
fn single_sampler_ps(sampler_usage: u32) -> crate::dxso::DxsoProgram {
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        sampler_usage,
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    parse(&bc).expect("PS3 parse")
}

#[test]
fn a_2d_declared_slot_bound_to_a_volume_binds_texture3d() {
    // D3D9 samples the texture the application bound, not the kind the
    // shader's `dcl` names, so a `dcl_2d` slot carrying a volume texture
    // reads the volume with a three-component coordinate.
    let ps = single_sampler_ps(0x9000_0000);
    assert_eq!(
        declared_ps_samplers(&ps).get(&0),
        Some(&TextureType::Texture2D),
        "the declaration itself stays 2D"
    );

    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit plain");
    assert!(
        plain.contains("texture2d<float> s0 [[texture(0)]]"),
        "an unbound or 2D-bound slot keeps the 2D binding:\n{plain}"
    );
    assert!(
        plain.contains("s0.sample(samp0, (in.texcoord0).xy)"),
        "and samples with .xy:\n{plain}"
    );

    let volume = emit_ps_programmable(
        &ps,
        VariantKey {
            volume_sampler_mask: 0b0001,
            ..VariantKey::default()
        },
    )
    .expect("emit volume");
    assert!(
        volume.contains("texture3d<float> s0 [[texture(0)]]"),
        "a volume-bound slot binds texture3d whatever it declared:\n{volume}"
    );
    assert!(
        !volume.contains("texture2d<float>"),
        "and nothing keeps the 2D binding:\n{volume}"
    );
    assert!(
        volume.contains("s0.sample(samp0, (in.texcoord0).xyz)"),
        "a texture3d sample takes a float3 coordinate:\n{volume}"
    );
    metal_compile_or_fail(&volume);

    let cube = emit_ps_programmable(
        &ps,
        VariantKey {
            cube_sampler_mask: 0b0001,
            ..VariantKey::default()
        },
    )
    .expect("emit cube");
    assert!(
        cube.contains("texturecube<float> s0 [[texture(0)]]"),
        "a cube-bound slot binds texturecube whatever it declared:\n{cube}"
    );
    assert!(
        cube.contains("s0.sample(samp0, (in.texcoord0).xyz)"),
        "a texturecube sample takes a float3 direction:\n{cube}"
    );
    metal_compile_or_fail(&cube);
}

#[test]
fn a_volume_declared_slot_bound_to_a_2d_texture_binds_texture2d() {
    // The reverse mismatch, which Star Wars: The Old Republic renders water
    // with: the shader declares `dcl_volume` and the application binds a 2D
    // texture, which D3D9 samples with the coordinate's first two components.
    let ps = single_sampler_ps(0xA000_0000);
    assert_eq!(
        declared_ps_samplers(&ps).get(&0),
        Some(&TextureType::Texture3D),
        "the declaration itself stays a volume"
    );

    let flat = emit_ps_programmable(&ps, VariantKey::default()).expect("emit flat");
    assert!(
        flat.contains("texture2d<float> s0 [[texture(0)]]"),
        "a 2D-bound slot binds texture2d whatever it declared:\n{flat}"
    );
    assert!(
        !flat.contains("texture3d<float>"),
        "and nothing keeps the 3D binding:\n{flat}"
    );
    assert!(
        flat.contains("s0.sample(samp0, (in.texcoord0).xy)"),
        "a texture2d sample takes a float2 coordinate:\n{flat}"
    );
    metal_compile_or_fail(&flat);

    let volume = emit_ps_programmable(
        &ps,
        VariantKey {
            volume_sampler_mask: 0b0001,
            ..VariantKey::default()
        },
    )
    .expect("emit volume");
    assert!(
        volume.contains("texture3d<float> s0 [[texture(0)]]"),
        "the matching binding still emits texture3d:\n{volume}"
    );
}

#[test]
fn a_depth_bound_slot_stays_depth2d_over_the_volume_and_cube_masks() {
    // Every depth-format texture is 2D, so the depth binding outranks the
    // other kinds rather than leaving a slot typed from a mask that cannot
    // apply to it.
    let ps = single_sampler_ps(0xA000_0000);
    let depth = emit_ps_programmable(
        &ps,
        VariantKey {
            depth_sampler_mask: 0b0001,
            ..VariantKey::default()
        },
    )
    .expect("emit depth");
    assert!(
        depth.contains("depth2d<float> s0 [[texture(0)]]"),
        "a depth-bound volume-declared slot binds depth2d:\n{depth}"
    );
    metal_compile_or_fail(&depth);
}

/// `vs_3_0 { dcl_<dim> s0; dcl_position v0; texldl r0, c0, s0; mov oPos, r0; }`
///
/// `dcl` selects the declared sampler dimension in bits 27..30 of the usage
/// token: `0x9000_0000` is 2D, `0x9800_0000` cube, `0xa000_0000` volume.
fn vertex_fetch_shader(dcl: u32) -> Vec<u32> {
    vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl,
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_TEXLDL, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]
}

#[test]
fn vertex_fetch_slot_bound_to_a_volume_texture_types_the_argument_3d() {
    // The shader declares `dcl_2d s0`, the game binds a volume texture to
    // D3DVERTEXTEXTURESAMPLER0. Metal type-checks the binding against the
    // signature, so the argument and the coordinate follow the binding.
    let vs = parse(&vertex_fetch_shader(0x9000_0000)).expect("vs_3_0 parse");
    let kinds = VsSamplerKinds {
        volume_mask: 0b0001,
        ..VsSamplerKinds::default()
    };
    let msl =
        emit_vs_programmable_named(&vs, "vtf_vs", u16::MAX, 0, kinds, None, None).expect("emit");
    assert!(
        msl.contains("texture3d<float> s0 [[texture(0)]]"),
        "a volume binding types the argument texture3d:\n{msl}"
    );
    assert!(
        msl.contains("s0.sample(samp0, (vs_c[0]).xyz, level((vs_c[0]).w))"),
        "a volume binding samples with a three-component coord:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn vertex_fetch_slot_bound_to_a_2d_texture_types_the_argument_2d() {
    // The mirror direction: the shader declares `dcl_volume s0` and the game
    // binds an ordinary 2D texture. Typing from the declaration would emit
    // `texture3d<float>` sampled with a `float2`, which does not even
    // compile, so the whole vertex library would be lost.
    let vs = parse(&vertex_fetch_shader(0xa000_0000)).expect("vs_3_0 parse");
    let msl = emit_vs_programmable_named(
        &vs,
        "vtf_vs",
        u16::MAX,
        0,
        VsSamplerKinds::default(),
        None,
        None,
    )
    .expect("emit");
    assert!(
        msl.contains("texture2d<float> s0 [[texture(0)]]"),
        "a 2D binding types the argument texture2d:\n{msl}"
    );
    assert!(
        msl.contains("s0.sample(samp0, (vs_c[0]).xy, level((vs_c[0]).w))"),
        "a 2D binding samples with a two-component coord:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn vertex_fetch_slot_bound_to_a_cube_texture_types_the_argument_cube() {
    let vs = parse(&vertex_fetch_shader(0x9000_0000)).expect("vs_3_0 parse");
    let kinds = VsSamplerKinds {
        cube_mask: 0b0001,
        ..VsSamplerKinds::default()
    };
    let msl =
        emit_vs_programmable_named(&vs, "vtf_vs", u16::MAX, 0, kinds, None, None).expect("emit");
    assert!(
        msl.contains("texturecube<float> s0 [[texture(0)]]"),
        "a cube binding types the argument texturecube:\n{msl}"
    );
    assert!(
        msl.contains("s0.sample(samp0, (vs_c[0]).xyz, level((vs_c[0]).w))"),
        "a cube binding samples with a three-component coord:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn vertex_texldl_reads_its_row_of_the_lod_table_only_under_the_table() {
    // Metal ignores sampler LOD clamps at an explicit level and has no
    // sampler bias, so under the table the level a vertex `texldl` names
    // counts from the texture LOD plus the bias, and is clamped by the slot's
    // finest level. Without it the shader is unchanged and declares no table.
    let vs = parse(&vertex_fetch_shader(0x9000_0000)).expect("vs_3_0 parse");
    let plain = emit_vs_programmable_named(
        &vs,
        "vtf_vs",
        u16::MAX,
        0,
        VsSamplerKinds::default(),
        None,
        None,
    )
    .expect("emit");
    assert!(
        !plain.contains("vs_lod"),
        "no table without the key bit:\n{plain}"
    );
    let kinds = VsSamplerKinds {
        lod_table: true,
        ..VsSamplerKinds::default()
    };
    let msl =
        emit_vs_programmable_named(&vs, "vtf_vs", u16::MAX, 0, kinds, None, None).expect("emit");
    assert!(
        msl.contains(&format!(
            "constant float2 *vs_lod [[buffer({VS_LOD_SLOT})]]"
        )),
        "the table is a vertex argument:\n{msl}"
    );
    assert!(
        msl.contains(
            "s0.sample(samp0, (vs_c[0]).xy, level(max((vs_c[0]).w + vs_lod[0].x, vs_lod[0].y)))"
        ),
        "texldl offsets and clamps its level by the slot's row:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn a_vertex_shader_without_samplers_declares_no_lod_table() {
    // The key bit only follows a `texldl` slot, but a shader with no sampler
    // has no row to read, so the emitter leaves the argument out regardless.
    let vs = parse(&[
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ])
    .expect("vs_3_0 parse");
    let kinds = VsSamplerKinds {
        lod_table: true,
        ..VsSamplerKinds::default()
    };
    let msl =
        emit_vs_programmable_named(&vs, "plain_vs", u16::MAX, 0, kinds, None, None).expect("emit");
    assert!(!msl.contains("vs_lod"), "no sampler, no table:\n{msl}");
}

#[test]
fn vertex_sampler_kinds_reads_one_slot_at_a_time() {
    let mut kinds = VsSamplerKinds::default();
    kinds.set_slot(1, true, false);
    kinds.set_slot(3, false, true);
    assert_eq!(
        kinds.kind(0),
        TextureType::Texture2D,
        "untouched slot is 2D"
    );
    assert_eq!(kinds.kind(1), TextureType::Texture3D, "slot 1 volume");
    assert_eq!(
        kinds.kind(2),
        TextureType::Texture2D,
        "untouched slot is 2D"
    );
    assert_eq!(kinds.kind(3), TextureType::TextureCube, "slot 3 cube");
    assert_eq!(
        kinds.kind(4),
        TextureType::Texture2D,
        "D3D9 defines four vertex fetch slots; past them reads 2D"
    );
    kinds.set_slot(1, false, false);
    assert_eq!(
        kinds.kind(1),
        TextureType::Texture2D,
        "unbinding a slot returns it to 2D"
    );
    assert_eq!(
        kinds,
        VsSamplerKinds {
            cube_mask: 0b1000,
            ..VsSamplerKinds::default()
        }
    );
}

#[test]
fn ff_temp_register_compiles_with_simultaneous_color_alpha_updates() {
    use mtld3d_types::{
        D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_TEMP, D3DTOP_DISABLE, D3DTOP_SELECTARG1,
    };

    use crate::dxso::{FfPsKey, FfStage, FfStageResult, emit_ps_ff};
    let narrow = |value| u8::try_from(value).expect("D3D stage value fits one byte");
    let mut stages = [FfStage {
        color_op: narrow(D3DTOP_DISABLE),
        ..FfStage::default()
    }; 8];
    stages[0] = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_TEMP | D3DTA_ALPHAREPLICATE),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_TEMP | D3DTA_COMPLEMENT),
        ..FfStage::default()
    };
    stages[0].set_result(FfStageResult::Temp);
    stages[1] = stages[0];
    stages[1].set_result(FfStageResult::Current);
    let mut key = FfPsKey {
        stages,
        specular_add: false,
        tt_projected_mask: 0,
    };
    metal_compile_or_fail(&emit_ps_ff(&key, VariantKey::default()));
    key.stages[0].color_op = narrow(mtld3d_types::D3DTOP_DOTPRODUCT3);
    metal_compile_or_fail(&emit_ps_ff(&key, VariantKey::default()));
    key.stages[0].color_arg1 = narrow(mtld3d_types::D3DTA_TEXTURE);
    metal_compile_or_fail(&emit_ps_ff(&key, VariantKey::default()));
}

#[test]
fn ff_disabled_alpha_keeping_the_register_alpha_compiles() {
    use mtld3d_types::{
        D3DTA_CURRENT, D3DTA_TEMP, D3DTA_TEXTURE, D3DTOP_DISABLE, D3DTOP_MODULATE,
        D3DTOP_SELECTARG1,
    };

    use crate::dxso::{FfPsKey, FfStage, FfStageFlags, FfStageResult, emit_ps_ff};
    let narrow = |value| u8::try_from(value).expect("D3D stage value fits one byte");
    let mut stages = [FfStage {
        color_op: narrow(D3DTOP_DISABLE),
        ..FfStage::default()
    }; 8];
    stages[0] = FfStage {
        color_op: narrow(D3DTOP_MODULATE),
        color_arg1: narrow(D3DTA_TEXTURE),
        color_arg2: narrow(D3DTA_CURRENT),
        alpha_op: narrow(D3DTOP_DISABLE),
        alpha_arg1: narrow(D3DTA_TEXTURE),
        alpha_arg2: narrow(D3DTA_CURRENT),
        flags: FfStageFlags::HAS_TEXTURE,
    };
    stages[1] = stages[0];
    stages[2] = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_TEMP),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_TEMP),
        ..FfStage::default()
    };
    let mut key = FfPsKey {
        stages,
        specular_add: false,
        tt_projected_mask: 0,
    };
    metal_compile_or_fail(&emit_ps_ff(&key, VariantKey::default()));
    key.stages[0].set_result(FfStageResult::Temp);
    key.stages[1].set_result(FfStageResult::Temp);
    metal_compile_or_fail(&emit_ps_ff(&key, VariantKey::default()));
}

/// A pixel shader reading `oDepth` as a source parses, and the emitter rejects it.
///
/// The end-to-end suite draws this shader as the one whose library fails to
/// build, so the parser must keep accepting it.
#[test]
fn a_ps_reading_its_depth_output_parses_but_does_not_emit() {
    const TYPE_DEPTHOUT: u32 = 9;
    let bytecode = [
        PS_HEADER,
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_DEPTHOUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    assert_eq!(
        bytecode[3], 0x90E4_0800,
        "the token the end-to-end suite spells out"
    );
    let ps = parse(&bytecode).expect("the parser accepts a depth-output source");
    assert!(emit_ps_programmable(&ps, VariantKey::default()).is_err());
}

// ── SM3 linkage by semantic ──

const DCL_NORMAL: u8 = 3;

/// `dcl_<usage><index> <reg_type>N.<mask>` as its three tokens.
fn dcl(usage: u8, index: u8, reg_type: u32, reg: u16, mask: u8) -> [u32; 3] {
    [
        opcode_token(OP_DCL, 2),
        dcl_usage_token(usage, index),
        dst_token(reg_type, reg, mask, false),
    ]
}

/// The member names of the emitted `Varyings` struct, in order.
fn varyings_members(msl: &str) -> Vec<String> {
    let body = msl
        .split("struct Varyings {\n")
        .nth(1)
        .and_then(|rest| rest.split("};").next())
        .expect("Varyings struct");
    body.lines()
        .filter_map(|line| {
            let decl = line.trim().strip_suffix(';')?;
            let decl = decl.split(" [[").next()?;
            decl.split_whitespace().nth(1).map(str::to_owned)
        })
        .collect()
}

/// `vs_3_0`: position in `o0`, NORMAL0 in `o1` and COLOR2 in `o2`, both from `v1`.
fn vs3_normal_color2() -> Vec<u32> {
    let mut bc = vec![VS3_HEADER];
    for d in [
        dcl(DCL_POSITION, 0, TYPE_INPUT, 0, 0xF),
        dcl(DCL_NORMAL, 0, TYPE_INPUT, 1, 0xF),
        dcl(DCL_POSITION, 0, TYPE_TEXCOORDOUT, 0, 0xF),
        dcl(DCL_NORMAL, 0, TYPE_TEXCOORDOUT, 1, 0xF),
        dcl(DCL_COLOR, 2, TYPE_TEXCOORDOUT, 2, 0xF),
    ] {
        bc.extend_from_slice(&d);
    }
    for (out, input) in [(0, 0), (1, 1), (2, 1)] {
        bc.extend_from_slice(&[
            opcode_token(OP_MOV, 2),
            dst_token(TYPE_TEXCOORDOUT, out, 0xF, false),
            src_token(TYPE_INPUT, input, SWIZ_IDENTITY, 0),
        ]);
    }
    bc.push(END_TOKEN);
    bc
}

/// `ps_3_0 { dcl_normal v0; dcl_color2 v1; add oC0, v0, v1; }`.
fn ps3_normal_color2() -> Vec<u32> {
    let mut bc = vec![PS3_HEADER];
    bc.extend_from_slice(&dcl(DCL_NORMAL, 0, TYPE_INPUT, 0, 0xF));
    bc.extend_from_slice(&dcl(DCL_COLOR, 2, TYPE_INPUT, 1, 0xF));
    bc.extend_from_slice(&[
        opcode_token(OP_ADD, 3),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_INPUT, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    bc
}

#[test]
fn sm3_vertex_outputs_outside_the_fixed_set_get_their_own_members() {
    let vs = parse(&vs3_normal_color2()).expect("VS3 parse");
    let msl = emit_vs_programmable(&vs).expect("emit VS3");
    let members = varyings_members(&msl);
    for member in ["normal0", "color2", "texcoord15", "color1", "fog"] {
        assert!(members.iter().any(|m| m == member), "{member}:\n{msl}");
    }
    assert!(msl.contains("out.normal0 = in.v1;"), "{msl}");
    assert!(msl.contains("out.color2 = in.v1;"), "{msl}");
    assert!(!msl.contains("_rastout_discard = in.v1"), "{msl}");
    metal_compile_or_fail(&msl);
}

#[test]
fn sm3_pixel_inputs_outside_the_fixed_set_read_the_linked_member_or_zero() {
    let ps = parse(&ps3_normal_color2()).expect("PS3 parse");
    let linked = emit_ps_programmable(
        &ps,
        VariantKey {
            linked_input_mask: 0b11,
            ..VariantKey::default()
        },
    )
    .expect("emit PS3 linked");
    assert!(
        linked.contains("oC0 = (in.normal0 + in.color2);"),
        "{linked}"
    );
    assert!(!linked.contains("in.color0"), "{linked}");
    metal_compile_or_fail(&linked);

    // The vertex side declares every member the linked pixel side reads, so
    // Metal, which links the two stages by member name, accepts the pair.
    let vs = parse(&vs3_normal_color2()).expect("VS3 parse");
    let vs_members = varyings_members(&emit_vs_programmable(&vs).expect("emit VS3"));
    for member in varyings_members(&linked) {
        assert!(vs_members.contains(&member), "VS lacks {member}");
    }

    let unlinked = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3 unlinked");
    assert!(
        unlinked.contains("oC0 = (float4(0.0) + float4(0.0));"),
        "{unlinked}"
    );
    let members = varyings_members(&unlinked);
    assert!(!members.iter().any(|m| m == "normal0" || m == "color2"));
    metal_compile_or_fail(&unlinked);

    let normal_only = emit_ps_programmable(
        &ps,
        VariantKey {
            linked_input_mask: 0b01,
            ..VariantKey::default()
        },
    )
    .expect("emit PS3 normal");
    assert!(normal_only.contains("oC0 = (in.normal0 + float4(0.0));"));
    metal_compile_or_fail(&normal_only);
}

#[test]
fn sm3_packed_registers_link_each_semantic_by_its_lanes() {
    // vs_3_0 { dcl_position o0; dcl_texcoord0 o1.xy; dcl_texcoord1 o1.zw;
    //          mov o0, v0; mov o1.xy, v1; mov o1.zw, v1.xyxy; }
    let mut bc = vec![VS3_HEADER];
    for d in [
        dcl(DCL_POSITION, 0, TYPE_INPUT, 0, 0xF),
        dcl(DCL_TEXCOORD, 0, TYPE_INPUT, 1, 0xF),
        dcl(DCL_POSITION, 0, TYPE_TEXCOORDOUT, 0, 0xF),
        dcl(DCL_TEXCOORD, 0, TYPE_TEXCOORDOUT, 1, 0x3),
        dcl(DCL_TEXCOORD, 1, TYPE_TEXCOORDOUT, 1, 0xC),
    ] {
        bc.extend_from_slice(&d);
    }
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 1, 0x3, false),
        src_token(TYPE_INPUT, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 1, 0xC, false),
        src_token(TYPE_INPUT, 1, 0x44, 0),
        END_TOKEN,
    ]);
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    for line in [
        "_o0.xy = (in.v1).xy;",
        "_o0.zw = ((in.v1).xyxy).zw;",
        "out.texcoord0.xy = _o0.xy;",
        "out.texcoord1.zw = _o0.zw;",
    ] {
        assert!(vs_msl.contains(line), "{line}:\n{vs_msl}");
    }
    metal_compile_or_fail(&vs_msl);

    // ps_3_0 packs the same two semantics into another register in reverse
    // declaration order: { dcl_texcoord1 v3.zw; dcl_texcoord0 v3.xy; mov oC0, v3; }
    let mut bc = vec![PS3_HEADER];
    bc.extend_from_slice(&dcl(DCL_TEXCOORD, 1, TYPE_INPUT, 3, 0xC));
    bc.extend_from_slice(&dcl(DCL_TEXCOORD, 0, TYPE_INPUT, 3, 0x3));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 3, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains(
            "float4 _v3 = float4(in.texcoord0.x, in.texcoord0.y, in.texcoord1.z, in.texcoord1.w);"
        ),
        "{ps_msl}"
    );
    assert!(ps_msl.contains("oC0 = _v3;"), "{ps_msl}");
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn sm3_packed_input_lanes_follow_the_point_sprite_substitution() {
    let mut bc = vec![PS3_HEADER];
    bc.extend_from_slice(&dcl(DCL_TEXCOORD, 0, TYPE_INPUT, 0, 0x3));
    bc.extend_from_slice(&dcl(DCL_TEXCOORD, 1, TYPE_INPUT, 0, 0xC));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let ps = parse(&bc).expect("PS3 parse");
    let mut variant = VariantKey::default();
    variant.flags.insert(VariantFlags::POINT_SPRITE);
    let msl = emit_ps_programmable(&ps, variant).expect("emit PS3 sprite");
    let substituted = msl
        .find("in.texcoord0 = float4(point_coord")
        .expect("sprite prologue");
    let assembled = msl.find("float4 _v0 = ").expect("assembled input");
    assert!(substituted < assembled, "{msl}");
    metal_compile_or_fail(&msl);
}

// ── Per-pixel lighting (ppl) ──

/// The game's terrain `ps_2_0`: eight splat samples, then `mul_pp r3, r3, v0`.
fn daoc_terrain_ps20() -> crate::dxso::DxsoProgram {
    let bytes = include_bytes!("fixtures/daoc_terrain_ps20.bin");
    let tokens: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    parse(&tokens).expect("terrain ps_2_0 parses")
}

fn ppl_variant(ppl_mask: u8, ppl_directional_mask: u8, ppl_model: u8) -> VariantKey {
    VariantKey {
        ppl_mask,
        ppl_directional_mask,
        ppl_model,
        ..VariantKey::default()
    }
}

#[test]
fn ppl_programmable_ps_maps_v0_to_lit() {
    let ps = daoc_terrain_ps20();
    let off = emit_ps_programmable(&ps, VariantKey::default()).expect("emit off");
    assert!(
        off.contains("in.color0"),
        "v0 reads the vertex colour when off:\n{off}"
    );
    assert!(!off.contains("lit0"), "{off}");
    let on = emit_ps_programmable(&ps, ppl_variant(0b110, 0, 0)).expect("emit on");
    let lit_line = on
        .lines()
        .find(|l| l.contains("float4 lit0 = ppl_light("))
        .unwrap_or_else(|| panic!("no lit0 prologue:\n{on}"));
    assert!(lit_line.contains("in.color0"), "{on}");
    assert_eq!(on.matches("in.color0").count(), 1, "v0 reads lit0:\n{on}");
    assert!(on.contains("[[buffer(7)]]"), "{on}");
    let v0_reads = off.matches("in.color0").count();
    assert_eq!(
        on.matches("lit0").count(),
        v0_reads + 1,
        "each v0 read became lit0:\n{on}"
    );
}

#[test]
fn ppl_ps_msl_compiles_under_metal() {
    use crate::{
        dxso::ff::{emit_ps_ff, tests::ppl_ff_ps_key},
        ppl::{PPL_MODEL_DEBUG_TERM, PPL_MODEL_HALF_LAMBERT, PPL_MODEL_SMOOTH_FALLOFF},
    };
    let ps = daoc_terrain_ps20();
    let variants = [
        ppl_variant(0b110, 0, 0),
        ppl_variant(0b110, 0, PPL_MODEL_HALF_LAMBERT | PPL_MODEL_SMOOTH_FALLOFF),
        ppl_variant(0b110, 0, PPL_MODEL_DEBUG_TERM),
        ppl_variant(0b111, 0b001, 0),
        VariantKey {
            ppl_knee: 80,
            ..ppl_variant(0b110, 0, 0)
        },
    ];
    for variant in variants {
        metal_compile_or_fail(&emit_ps_ff(&ppl_ff_ps_key(), variant));
        metal_compile_or_fail(&emit_ps_programmable(&ps, variant).expect("emit terrain"));
    }
}

#[test]
fn ppl_shoulder_replaces_saturate() {
    let variant = VariantKey {
        ppl_knee: 80,
        ..ppl_variant(0b110, 0, 0)
    };
    let msl = emit_ps_programmable(&daoc_terrain_ps20(), variant).expect("emit terrain");
    assert!(
        msl.contains("ppl_shoulder(base.rgb + sum.rgb, 0.8)"),
        "{msl}"
    );
    assert!(!msl.contains("saturate(float4(base.rgb + sum.rgb"), "{msl}");
    metal_compile_or_fail(&msl);
}

#[test]
fn ppl_shoulder_off_unchanged() {
    let msl =
        emit_ps_programmable(&daoc_terrain_ps20(), ppl_variant(0b110, 0, 0)).expect("emit terrain");
    assert!(
        msl.contains("saturate(float4(base.rgb + sum.rgb, base.a))"),
        "{msl}"
    );
    assert!(!msl.contains("ppl_shoulder"), "{msl}");
}

// ── Phase 4: figure VS per-vertex capture ──

/// `(fixture name, bytes)` for every figure VS fixture, mirroring the table
/// in [`crate::ppl::figures`].
const FIGURE_VS_FIXTURES: &[(&str, &[u8])] = &[
    (
        "daoc_figure_vs_15ecaea13070e5af.bin",
        include_bytes!("fixtures/daoc_figure_vs_15ecaea13070e5af.bin"),
    ),
    (
        "daoc_figure_vs_31a064b8fe30b4ed.bin",
        include_bytes!("fixtures/daoc_figure_vs_31a064b8fe30b4ed.bin"),
    ),
    (
        "daoc_figure_vs_4c76b4b8b4466a51.bin",
        include_bytes!("fixtures/daoc_figure_vs_4c76b4b8b4466a51.bin"),
    ),
    (
        "daoc_figure_vs_8c436dc740ce3674.bin",
        include_bytes!("fixtures/daoc_figure_vs_8c436dc740ce3674.bin"),
    ),
    (
        "daoc_figure_vs_cbb5aac252c08202.bin",
        include_bytes!("fixtures/daoc_figure_vs_cbb5aac252c08202.bin"),
    ),
    (
        "daoc_figure_vs_d0e250f335b00f24.bin",
        include_bytes!("fixtures/daoc_figure_vs_d0e250f335b00f24.bin"),
    ),
    (
        "daoc_figure_vs_d3c4595fec6bd3e2.bin",
        include_bytes!("fixtures/daoc_figure_vs_d3c4595fec6bd3e2.bin"),
    ),
    (
        "daoc_figure_vs_dcf0475f8ef49250.bin",
        include_bytes!("fixtures/daoc_figure_vs_dcf0475f8ef49250.bin"),
    ),
    (
        "daoc_figure_vs_e389281fb2b88a58.bin",
        include_bytes!("fixtures/daoc_figure_vs_e389281fb2b88a58.bin"),
    ),
    (
        "daoc_figure_vs_e8b6913dccdf28ed.bin",
        include_bytes!("fixtures/daoc_figure_vs_e8b6913dccdf28ed.bin"),
    ),
];

fn figure_fixture_program(name: &str, bytes: &[u8]) -> crate::dxso::DxsoProgram {
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    crate::dxso::parse(&words).unwrap_or_else(|e| panic!("{name}: parse: {e:?}"))
}

/// Which of the xyz lanes body instruction `index` writes on temp `reg`
/// (the census shaders may build a register one lane per `dp3`).
fn written_lanes(instructions: &[crate::dxso::ir::Instruction], index: usize, reg: u8) -> u8 {
    instructions[index]
        .dst
        .as_ref()
        .filter(|dst| dst.reg.kind == crate::dxso::RegKind::Temp && dst.reg.index == u16::from(reg))
        .map_or(0, |dst| dst.write_mask.0 & 0b0111)
}

/// The capture must sit where the census put it: the two capture lines appear
/// exactly once, immediately before the emitted sun `dp3`, and — checked on the
/// parsed IR, not the MSL text — both captured registers hold their final light
/// input values there (written before, not rewritten between the capture and
/// their last light read).
#[test]
fn figure_capture_reads_live_regs() {
    use crate::dxso::{RegKind, ir::Instruction, opcode::Opcode};

    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let entry = crate::ppl::figures::figure_vs(id).unwrap_or_else(|| panic!("{name}"));
        let msl = emit_vs_programmable_named(
            &program,
            "clip_vs",
            u16::MAX,
            0,
            VsSamplerKinds::default(),
            Some(entry),
            None,
        )
        .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));

        let normal_line = format!(
            "out.texcoord14 = float4(r[{}].xyz, length(r[{}].xyz));",
            entry.normal_reg, entry.normal_reg
        );
        let pos_line = format!("out.texcoord15 = float4(r[{}].xyz, 1.0);", entry.pos_reg);
        assert_eq!(msl.matches(&normal_line).count(), 1, "{name}:\n{msl}");
        assert_eq!(msl.matches(&pos_line).count(), 1, "{name}:\n{msl}");
        // The capture sits immediately before the emitted sun dp3 (the tint
        // line ships the oD0 modulation factor and sits between them).
        let after = msl
            .split_once(&pos_line)
            .map(|(_, rest)| rest)
            .expect("pos line present");
        let next = after
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.contains("out.texcoord13 ="))
            .next()
            .unwrap_or("");
        assert!(
            next.contains("dot(") && next.contains(&format!("r[{}]", entry.normal_reg)),
            "{name}: capture not immediately before the sun dp3: {next:?}"
        );

        // The capture ships the oD0 modulation factor through texcoord13
        // (the emitter's zero-init block writes it first; the capture's write
        // is the one the prologue reads).
        let expected_tint = match entry.od0_tint {
            crate::ppl::figures::Od0Tint::None => "float4(float3(1.0, 1.0, 1.0), 0.0)".to_owned(),
            crate::ppl::figures::Od0Tint::Const(reg) => format!("float4(vs_c[{reg}].xyz, 0.0)"),
            crate::ppl::figures::Od0Tint::Input(reg) => format!("float4(in.v{reg}.xyz, 0.0)"),
        };
        assert!(
            msl.contains(&format!("out.texcoord13 = {expected_tint};")),
            "{name}: capture must ship the oD0 tint {expected_tint:?}:\n{msl}"
        );
        // Liveness on the parsed IR: both registers are written xyz before the
        // capture, and no instruction between the capture and the register's
        // last light read rewrites their xyz.
        let capture = entry.capture_before as usize;
        let sun = if entry.layout == 1 { 12 } else { 13 };
        let plight_positions: &[u16] = if entry.layout == 1 {
            &[15, 18, 21]
        } else {
            &[16, 19, 22]
        };
        let is_light_read = |index: usize, reg: u8, light_consts: &[u16]| -> bool {
            let inst = &program.instructions[index];
            let (reads_reg, reads_light) = inst.srcs.iter().fold((false, false), |(r, l), s| {
                (
                    r || (s.reg.kind == RegKind::Temp && s.reg.index == u16::from(reg)),
                    l || (s.reg.kind == RegKind::Const && light_consts.contains(&s.reg.index)),
                )
            });
            reads_reg && reads_light
        };
        for (reg, light_consts, opcode) in [
            (entry.normal_reg, &[sun][..], Opcode::Dp3),
            (entry.pos_reg, plight_positions, Opcode::Add),
        ] {
            let covered = (0..capture).fold(0u8, |acc, i| {
                acc | written_lanes(&program.instructions, i, reg)
            });
            assert_eq!(
                covered & 0b0111,
                0b0111,
                "{name}: r{reg} not fully written (xyz) before the capture"
            );
            let last_light_read = (capture..program.instructions.len())
                .filter(|&i| {
                    program.instructions[i].opcode == opcode && is_light_read(i, reg, light_consts)
                })
                .next_back()
                .unwrap_or_else(|| panic!("{name}: r{reg} has no light read after the capture"));
            let rewritten = (capture..=last_light_read)
                .any(|i| written_lanes(&program.instructions, i, reg) != 0);
            assert!(
                !rewritten,
                "{name}: r{reg} rewritten between the capture and its last light read"
            );
        }
    }
}

/// The figure specular term's view vector rests on one relation of the figure
/// VS, checked here on every fixture: `oPos` is `dp4` of one temp `R` with
/// `c0..c3` (lane `k` from `c[k]`, written straight to `oPos` or to a temp
/// moved whole to it), the light-space position `pos_reg.xyz` is `dp4` of the
/// same `R` with `c4..c6`, `R` is not rewritten between the first and the last
/// of those reads, and `R.w` is a `def` constant 1.0. So the camera centre
/// recovered from `c0`, `c1`, `c3` maps to light space through `c4..c6`
/// (`crate::ppl::figures::figure_eye_row`).
#[test]
fn figure_clip_and_light_rows_share_the_skinned_position() {
    use crate::dxso::{
        RegKind,
        ir::{Instruction, SrcModifier, Swizzle},
        opcode::Opcode,
    };

    // `R` when `inst` is `dp4 <dst>, R, c[row]` (plain temp, no swizzle).
    let dp4_row = |inst: &Instruction, row: u16| -> Option<u16> {
        let [r, c] = inst.srcs.as_slice() else {
            return None;
        };
        (inst.opcode == Opcode::Dp4
            && r.reg.kind == RegKind::Temp
            && r.swizzle == Swizzle::IDENTITY
            && r.modifier == SrcModifier::None
            && r.rel_addr.is_none()
            && c.reg.kind == RegKind::Const
            && c.reg.index == row
            && c.rel_addr.is_none())
        .then_some(r.reg.index)
    };
    let writes = |inst: &Instruction, kind: RegKind, index: u16| -> u8 {
        inst.dst
            .as_ref()
            .filter(|dst| dst.reg.kind == kind && dst.reg.index == index)
            .map_or(0, |dst| dst.write_mask.0)
    };
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let entry = crate::ppl::figures::figure_vs(id).unwrap_or_else(|| panic!("{name}"));
        let insts = &program.instructions;
        let capture = usize::from(entry.capture_before);
        // (instruction index, R) per read.
        let mut reads: Vec<(usize, u16)> = Vec::new();
        // Light space: the last `dp4 pos_reg.<lane>, R, c[4 + lane]` before
        // the capture (the value the capture ships).
        for lane in 0..3u16 {
            let (i, r) = (0..capture)
                .rev()
                .find_map(|i| {
                    (writes(&insts[i], RegKind::Temp, u16::from(entry.pos_reg)) == 1 << lane)
                        .then(|| dp4_row(&insts[i], 4 + lane).map(|r| (i, r)))
                        .flatten()
                })
                .unwrap_or_else(|| {
                    panic!("{name}: no dp4 r{}.{lane}, R, c{}", entry.pos_reg, 4 + lane)
                });
            // Nothing rewrites that lane between it and the capture.
            assert!(
                (i + 1..capture).all(|j| writes(
                    &insts[j],
                    RegKind::Temp,
                    u16::from(entry.pos_reg)
                ) & (1 << lane)
                    == 0),
                "{name}: r{}.{lane} rewritten before the capture",
                entry.pos_reg
            );
            reads.push((i, r));
        }
        // Clip space: exactly one `dp4 <oPos or T>.<lane>, R, c[lane]` per lane.
        let mut clip_temp: Option<u16> = None;
        let mut last_clip = 0;
        for lane in 0..4u16 {
            let hits: Vec<(usize, u16)> = insts
                .iter()
                .enumerate()
                .filter_map(|(i, inst)| dp4_row(inst, lane).map(|r| (i, r)))
                .collect();
            assert_eq!(hits.len(), 1, "{name}: dp4 with c{lane}: {hits:?}");
            let (i, r) = hits[0];
            let dst = insts[i].dst.as_ref().expect("dp4 writes");
            assert_eq!(dst.write_mask.0, 1 << lane, "{name}: c{lane} lane");
            match dst.reg.kind {
                RegKind::RastOut => assert_eq!(dst.reg.index, 0, "{name}: oPos"),
                RegKind::Temp => {
                    assert!(
                        clip_temp.is_none_or(|t| t == dst.reg.index),
                        "{name}: clip lanes in two temps"
                    );
                    clip_temp = Some(dst.reg.index);
                }
                kind => panic!("{name}: clip row c{lane} writes {kind:?}"),
            }
            last_clip = last_clip.max(i);
            reads.push((i, r));
        }
        if let Some(t) = clip_temp {
            // The temp reaches `oPos` whole and unchanged.
            let mov = (last_clip + 1..insts.len())
                .find(|&i| writes(&insts[i], RegKind::RastOut, 0) != 0)
                .unwrap_or_else(|| panic!("{name}: no oPos write"));
            let inst = &insts[mov];
            assert_eq!(inst.opcode, Opcode::Mov, "{name}");
            assert_eq!(writes(inst, RegKind::RastOut, 0), 0b1111, "{name}");
            assert_eq!(inst.srcs[0].reg.kind, RegKind::Temp, "{name}");
            assert_eq!(inst.srcs[0].reg.index, t, "{name}");
            assert_eq!(inst.srcs[0].swizzle, Swizzle::IDENTITY, "{name}");
            assert_eq!(inst.srcs[0].modifier, SrcModifier::None, "{name}");
            assert!(
                (last_clip + 1..mov).all(|j| writes(&insts[j], RegKind::Temp, t) == 0),
                "{name}: r{t} rewritten before the oPos move"
            );
        }
        // One skinned register for all seven rows, unchanged between its
        // first and last read.
        let r = reads[0].1;
        assert!(reads.iter().all(|&(_, x)| x == r), "{name}: {reads:?}");
        let first = reads.iter().map(|&(i, _)| i).min().expect("reads");
        let last = reads.iter().map(|&(i, _)| i).max().expect("reads");
        assert!(
            (first + 1..last).all(|j| writes(&insts[j], RegKind::Temp, r) == 0),
            "{name}: r{r} rewritten between its clip/light reads"
        );
        // `R.w` is the constant 1.0: its last write before the first read is
        // `mov R.w, cN.<lane>` with a `def` lane of 1.0.
        let w_write = (0..first)
            .rev()
            .find(|&i| writes(&insts[i], RegKind::Temp, r) & 0b1000 != 0)
            .unwrap_or_else(|| panic!("{name}: r{r}.w never written"));
        let inst = &insts[w_write];
        assert_eq!(inst.opcode, Opcode::Mov, "{name}");
        let src = &inst.srcs[0];
        assert_eq!(src.reg.kind, RegKind::Const, "{name}");
        let lane = usize::from(src.swizzle.0[3]);
        let def = program
            .def_constants
            .iter()
            .find(|d| d.reg.kind == RegKind::Const && d.reg.index == src.reg.index)
            .unwrap_or_else(|| panic!("{name}: c{} is not a def", src.reg.index));
        assert_eq!(def.value[lane], 1.0, "{name}: r{r}.w");
        assert_eq!(src.modifier, SrcModifier::None, "{name}");
    }
}

/// Without the figure plan the emission is exactly today's output: the capture
/// is a pure two-line insertion, so removing it from the figure emission
/// reproduces the stock MSL byte-for-byte.
#[test]
fn figure_off_vs_unchanged() {
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let entry = crate::ppl::figures::figure_vs(id).unwrap_or_else(|| panic!("{name}"));
        let plain = emit_vs_programmable_named(
            &program,
            "clip_vs",
            u16::MAX,
            0,
            VsSamplerKinds::default(),
            None,
            None,
        )
        .expect("emit");
        assert!(
            !plain.contains("out.texcoord14 = float4(r["),
            "{name}:\n{plain}"
        );
        let figure = emit_vs_programmable_named(
            &program,
            "clip_vs",
            u16::MAX,
            0,
            VsSamplerKinds::default(),
            Some(entry),
            None,
        )
        .expect("emit");
        let tint_expr = match entry.od0_tint {
            crate::ppl::figures::Od0Tint::None => "float3(1.0, 1.0, 1.0)".to_owned(),
            crate::ppl::figures::Od0Tint::Const(reg) => format!("vs_c[{reg}].xyz"),
            crate::ppl::figures::Od0Tint::Input(reg) => format!("in.v{reg}.xyz"),
        };
        let normal_line = format!(
            "\n    out.texcoord13 = float4({}, 0.0);\n    out.texcoord14 = float4(r[{}].xyz, length(r[{}].xyz));\n    out.texcoord15 = float4(r[{}].xyz, 1.0);",
            tint_expr, entry.normal_reg, entry.normal_reg, entry.pos_reg
        );
        let without_capture = figure.replacen(&normal_line, "", 1);
        if without_capture != plain {
            for (i, (a, b)) in without_capture.lines().zip(plain.lines()).enumerate() {
                if a != b {
                    eprintln!("DIFF {name} line {i}:\n  got:  {a}\n  want: {b}");
                    break;
                }
            }
            panic!("{name}: capture is not a pure insertion");
        }
    }
}

/// Every figure fixture with the capture plan compiles under Metal, and the
/// extra outputs link to a pixel shader that does not read them (the stock
/// figure PS declares the same `Varyings` shape, so stage-in linkage by field
/// index is unchanged).
#[test]
fn figure_vs_msl_compiles_under_metal() {
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let entry = crate::ppl::figures::figure_vs(id).unwrap_or_else(|| panic!("{name}"));
        let msl = emit_vs_programmable_named(
            &program,
            "clip_vs",
            u16::MAX,
            0,
            VsSamplerKinds::default(),
            Some(entry),
            None,
        )
        .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));
        metal_compile_or_fail(&msl);
    }
}

// ── World effects Task 1: figure fog ──

/// Emit a figure fixture with only the figure-fog plan set (or not).
fn emit_figure_fog(program: &crate::dxso::DxsoProgram, clip_planes: u8, fog: bool) -> String {
    let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
    let entry = crate::ppl::figures::figure_vs(id).expect("figure VS");
    emit_vs_programmable_named(
        program,
        "clip_vs",
        u16::MAX,
        clip_planes,
        VsSamplerKinds::default(),
        None,
        fog.then_some(entry),
    )
    .expect("emit")
}

/// The linear figure fog line for a fog-params register.
fn figure_fog_line(reg: u8) -> String {
    format!(
        "    out.fog = float4(saturate((vs_c[{reg}].y - out.position.w) / max(vs_c[{reg}].z, 1e-6)));\n"
    )
}

/// With `fog.figures` at its default no figure draw is flagged, and the stock
/// emission carries no fog override; turning it on is a pure one-line
/// insertion into that stock MSL.
#[test]
fn figure_fog_off_unchanged() {
    let off = crate::fog::FogConfig::default();
    let on = crate::fog::FogConfig { figures: true };
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        assert!(!crate::fog::figure_fog_flag(&off, id), "{name}");
        assert!(crate::fog::figure_fog_flag(&on, id), "{name}");
        let stock = emit_vs_programmable_named(
            &program,
            "clip_vs",
            u16::MAX,
            0,
            VsSamplerKinds::default(),
            None,
            None,
        )
        .expect("emit");
        assert_eq!(emit_figure_fog(&program, 0, false), stock, "{name}");
        assert!(!stock.contains("max(vs_c["), "{name}");
        let entry = crate::ppl::figures::figure_vs(id).expect("figure VS");
        let line = figure_fog_line(crate::ppl::figures::fog_params_register(entry.layout));
        let fogged = emit_figure_fog(&program, 0, true);
        assert_eq!(
            fogged.replacen(&line, "", 1),
            stock,
            "{name}: not a pure insertion"
        );
    }
    // A non-figure shader is never flagged.
    assert!(!crate::fog::figure_fog_flag(&on, 0x1234));
}

/// Layout-1 figure shaders read their fog params from `c11`, layout-2 from `c12`.
#[test]
fn figure_fog_uses_layout_register() {
    assert_eq!(crate::ppl::figures::fog_params_register(1), 11);
    assert_eq!(crate::ppl::figures::fog_params_register(2), 12);
    let mut layouts = [0usize; 3];
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let layout = crate::ppl::figures::figure_vs(id)
            .expect("figure VS")
            .layout;
        layouts[usize::from(layout)] += 1;
        let msl = emit_figure_fog(&program, 0, true);
        let (want, other) = if layout == 1 { (11, 12) } else { (12, 11) };
        assert!(msl.contains(&figure_fog_line(want)), "{name}:\n{msl}");
        assert!(!msl.contains(&figure_fog_line(other)), "{name}");
    }
    assert!(layouts[1] > 0 && layouts[2] > 0, "both layouts covered");
}

/// The fog factor is `saturate((end - w) / max(end - start, 1e-6))` on the
/// clip-space `w` the VS wrote to `oPos`: it replaces the shader's own fog
/// result (after the oFog fallback) and reads `w` before the clip distances
/// and the half-pixel position epilogue.
#[test]
fn figure_fog_is_one_inside_start() {
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let msl = emit_figure_fog(&program, 1, true);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let layout = crate::ppl::figures::figure_vs(id)
            .expect("figure VS")
            .layout;
        let line = figure_fog_line(crate::ppl::figures::fog_params_register(layout));
        let at = msl.find(&line).unwrap_or_else(|| panic!("{name}:\n{msl}"));
        let last_other_fog = msl.rfind("out.fog = float4(out.color1.w)");
        assert!(
            last_other_fog.is_none_or(|i| i < at),
            "{name}: fallback after override"
        );
        let clip = msl.find("out.clip_distance[0] =").expect("clip distance");
        let fixup = msl.find("out.position.x += pos_fixup.x").expect("fixup");
        assert!(at < clip && at < fixup, "{name}: override after epilogue");
        // Nothing after the override writes the fog again.
        assert!(!msl[at + line.len()..].contains("out.fog ="), "{name}");
    }
    // The factor itself: 1 inside start, 0 past end, linear between.
    let factor =
        |w: f32, start: f32, end: f32| ((end - w) / (end - start).max(1e-6)).clamp(0.0, 1.0);
    assert_eq!(factor(5.0, 10.0, 20.0), 1.0);
    assert_eq!(factor(25.0, 10.0, 20.0), 0.0);
    assert_eq!(factor(15.0, 10.0, 20.0), 0.5);
}

/// Every figure fixture with the fog override (alone and with the per-pixel
/// capture) compiles under Metal.
#[test]
fn figure_fog_msl_compiles_under_metal() {
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        metal_compile_or_fail(&emit_figure_fog(&program, 0, true));
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let entry = crate::ppl::figures::figure_vs(id).expect("figure VS");
        let both = emit_vs_programmable_named(
            &program,
            "clip_vs",
            u16::MAX,
            0,
            VsSamplerKinds::default(),
            Some(entry),
            Some(entry),
        )
        .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));
        metal_compile_or_fail(&both);
    }
}

// ── Phase 4 Task 4: figure pixel term ──

/// `(fixture name, bytes)` for every figure PS fixture, mirroring the pairing
/// in `dev/lighting/ppl/notes4.md` (every PS the traces pair with a figure VS
/// is `ps_1_x`).
const FIGURE_PS_FIXTURES: &[(&str, &[u8])] = &[
    (
        "daoc_figure_ps_1615e1aad1107b9d.bin",
        include_bytes!("fixtures/daoc_figure_ps_1615e1aad1107b9d.bin"),
    ),
    (
        "daoc_figure_ps_2c8276e3c548b27d.bin",
        include_bytes!("fixtures/daoc_figure_ps_2c8276e3c548b27d.bin"),
    ),
    (
        "daoc_figure_ps_41c41873567c9808.bin",
        include_bytes!("fixtures/daoc_figure_ps_41c41873567c9808.bin"),
    ),
    (
        "daoc_figure_ps_52c6a08ed862f461.bin",
        include_bytes!("fixtures/daoc_figure_ps_52c6a08ed862f461.bin"),
    ),
    (
        "daoc_figure_ps_5328f11d00beaa6e.bin",
        include_bytes!("fixtures/daoc_figure_ps_5328f11d00beaa6e.bin"),
    ),
    (
        "daoc_figure_ps_7c2629241d2358f2.bin",
        include_bytes!("fixtures/daoc_figure_ps_7c2629241d2358f2.bin"),
    ),
    (
        "daoc_figure_ps_87bf99b5173433dd.bin",
        include_bytes!("fixtures/daoc_figure_ps_87bf99b5173433dd.bin"),
    ),
    (
        "daoc_figure_ps_9e8e6f54d49c139b.bin",
        include_bytes!("fixtures/daoc_figure_ps_9e8e6f54d49c139b.bin"),
    ),
    (
        "daoc_figure_ps_a552075efb97215c.bin",
        include_bytes!("fixtures/daoc_figure_ps_a552075efb97215c.bin"),
    ),
    (
        "daoc_figure_ps_d41abd53f69b44bf.bin",
        include_bytes!("fixtures/daoc_figure_ps_d41abd53f69b44bf.bin"),
    ),
];

/// A per-pixel figure variant: the sun plus three point lights of `layout`.
fn figure_variant(ppl_figure: u8, ppl_model: u8, ppl_knee: u8) -> VariantKey {
    VariantKey {
        ppl_mask: 0b1111,
        ppl_directional_mask: 0,
        ppl_model,
        ppl_knee,
        ppl_figure,
        ..VariantKey::default()
    }
}

/// The figure PS fixtures are `ps_1_x` shaders; parse them like the VS ones.
fn figure_ps_fixture_program(name: &str, bytes: &[u8]) -> crate::dxso::DxsoProgram {
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    crate::dxso::parse(&words).unwrap_or_else(|e| panic!("{name}: parse: {e:?}"))
}

/// Each paired figure PS reads the vertex colour exactly once — inside the
/// `fig_light` prologue — and every other diffuse read goes through `lit0`.
#[test]
fn figure_ps_maps_v0_to_lit() {
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        let off = emit_ps_programmable(&ps, VariantKey::default()).expect("emit off");
        assert!(
            !off.contains("lit0"),
            "{name} off keeps stock reads:\n{off}"
        );
        assert!(off.contains("in.color0"), "{name}:\n{off}");
        let v0_reads = off.matches("in.color0").count();
        let on = emit_ps_programmable(&ps, figure_variant(1, 0, 0)).expect("emit on");
        let prologue = on
            .lines()
            .find(|l| l.contains("float4 lit0 = fig_light("))
            .unwrap_or_else(|| panic!("{name}: no fig_light prologue:\n{on}"));
        assert!(prologue.contains("in.color0"), "{name}:\n{on}");
        assert_eq!(
            on.matches("in.color0").count(),
            1,
            "{name}: the prologue is the only v0 read:\n{on}"
        );
        assert_eq!(
            on.matches("lit0").count(),
            v0_reads + 1,
            "{name}: each v0 read became lit0:\n{on}"
        );
        assert!(
            on.contains("[[buffer(7)]]"),
            "{name} binds the VS constants:\n{on}"
        );
    }
}

/// The figure light terms carry their own colours, so the vertex colour
/// enters the lighting only as `base.a`: on every figure PS fixture the
/// prologue's whole-`in.color0` read is the only one, no rgb/xyz swizzle of
/// it reaches the lighting, and `fig_light` touches no other lane of `base`.
#[test]
fn figure_prologue_reads_only_vertex_alpha() {
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        let msl = emit_ps_programmable(&ps, figure_variant(1, 0, 0))
            .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));
        // The prologue is the only `in.color0` read, and it passes the
        // colour whole (its `.a` is what `fig_light` consumes).
        let prologue_lines: Vec<&str> = msl.lines().filter(|l| l.contains("in.color0")).collect();
        assert_eq!(
            prologue_lines.len(),
            1,
            "{name}: in.color0 must be read only in the prologue:\n{msl}"
        );
        assert!(
            prologue_lines[0].contains("float4 lit0 = fig_light("),
            "{name}: the only in.color0 read is not the prologue:\n{}",
            prologue_lines[0]
        );
        // No swizzled read of the vertex colour reaches the lighting.
        for swizzle in [
            "in.color0.rgb",
            "in.color0.xyz",
            "in.color0.r",
            "in.color0.g",
            "in.color0.b",
        ] {
            assert!(
                !msl.contains(swizzle),
                "{name}: {swizzle} reaches the lighting:\n{msl}"
            );
        }
        // The function body (signature excluded): every `base` use reads `.a` only.
        let start = msl
            .find("static inline float4 fig_light(")
            .expect("fig_light present");
        let open = start + msl[start..].find('{').expect("fig_light opens");
        let close = open + msl[open..].find("}\n\n").expect("fig_light closes");
        let body = &msl[open..close];
        let uses: Vec<&str> = body
            .match_indices("base")
            .map(|(i, _)| &body[i..])
            .collect();
        assert!(
            !uses.is_empty(),
            "{name}: fig_light must consume the vertex colour's alpha:\n{body}"
        );
        for use_ in uses {
            let lane = use_
                .strip_prefix("base")
                .and_then(|rest| rest.chars().next())
                .unwrap_or('?');
            assert_eq!(lane, '.', "{name}: {use_} must be a swizzle");
            let comp = use_
                .strip_prefix("base.")
                .and_then(|rest| rest.chars().next())
                .unwrap_or('?');
            assert_eq!(
                comp, 'a',
                "{name}: fig_light reads {comp} of base, not alpha:\n{body}"
            );
        }
        metal_compile_or_fail(&msl);
    }
}

/// A fragment sitting exactly at a point light must light nothing, not go
/// NaN: the emitted `fig_light` clamps the `rsqrt` denominator and selects
/// the whole term away when `d2` is zero.
#[test]
fn figure_point_term_guards_zero_distance() {
    let msl = emit_ps_programmable(&daoc_terrain_ps20(), figure_variant(1, 0, 0)).expect("emit");
    assert!(
        msl.contains("rsqrt(max(d2, 1e-8))"),
        "rsqrt denominator unguarded:\n{msl}"
    );
    assert_eq!(
        msl.matches("step(1e-8, d2)").count(),
        3,
        "each point term needs the d2 = 0 select:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

/// The figure layouts read their own constant rows: layout 1 the sun at
/// `c[12]`… third point light `c[23]`, layout 2 everything one register
/// higher (`c[13]`…`c[24]`).
#[test]
fn figure_layouts_read_their_registers() {
    for (layout, first, last) in [(1u8, 12u32, 23u32), (2, 13, 24)] {
        let msl =
            emit_ps_programmable(&daoc_terrain_ps20(), figure_variant(layout, 0, 0)).expect("emit");
        for reg in first..=last {
            let row = format!("c[{reg}]");
            assert!(msl.contains(&row), "layout {layout} misses {row}:\n{msl}");
        }
        let before = format!("c[{}]", first - 1);
        assert!(
            !msl.contains(&before),
            "layout {layout} must not read {before}:\n{msl}"
        );
        let after = format!("c[{}]", last + 1);
        assert!(
            !msl.contains(&after),
            "layout {layout} must not read {after}:\n{msl}"
        );
    }
}

/// Every figure PS fixture compiles under Metal for the stock model, the
/// half-Lambert sun, the debug term and the soft shoulder.
#[test]
fn figure_ps_msl_compiles_under_metal() {
    use crate::ppl::{PPL_MODEL_DEBUG_TERM, PPL_MODEL_HALF_LAMBERT};
    let variants = [
        figure_variant(1, 0, 0),
        figure_variant(1, PPL_MODEL_HALF_LAMBERT, 0),
        figure_variant(1, PPL_MODEL_DEBUG_TERM, 0),
        figure_variant(1, 0, 80),
        // Layout 2 shifts every constant row; compile it too.
        figure_variant(2, PPL_MODEL_HALF_LAMBERT, 80),
    ];
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        for variant in &variants {
            let msl = emit_ps_programmable(&ps, *variant)
                .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));
            metal_compile_or_fail(&msl);
        }
    }
}

/// The figure pixel stage compiles against the figure vertex stage: both
/// libraries build (the harness has no pipeline-link helper, so both are
/// compiled separately — the shared `Varyings` struct is emitted by the same
/// `emit_varyings` code, so stage-in linkage is by construction).
#[test]
fn figure_vs_and_ps_msl_compile_as_a_pair() {
    use crate::dxso::VsSamplerKinds;
    let (vs_name, vs_bytes) = FIGURE_VS_FIXTURES[5];
    let (ps_name, ps_bytes) = FIGURE_PS_FIXTURES[6];
    let vs = figure_fixture_program(vs_name, vs_bytes);
    let ps = figure_ps_fixture_program(ps_name, ps_bytes);
    let id = crate::ids::ProgramId::from_tokens(&vs.bytecode()).raw();
    let entry = crate::ppl::figures::figure_vs(id).expect("figure VS");
    let vs_msl = emit_vs_programmable_named(
        &vs,
        "clip_vs",
        u16::MAX,
        0,
        VsSamplerKinds::default(),
        Some(entry),
        None,
    )
    .expect("emit VS");
    let ps_msl = emit_ps_programmable(&ps, figure_variant(1, 0, 0)).expect("emit PS");
    metal_compile_or_fail(&vs_msl);
    metal_compile_or_fail(&ps_msl);
}

/// The skinned normal is not unit length and the figure VS lights with it
/// unnormalized: the prologue rebuilds the interpolated skinned normal
/// (direction re-scaled by the interpolated per-vertex length in `.w`) and
/// `fig_light` dots with THAT normal — no renormalization after the scaling.
#[test]
fn figure_light_uses_the_scaled_skinned_normal() {
    let msl = emit_ps_programmable(&daoc_terrain_ps20(), figure_variant(1, 0, 0)).expect("emit");
    let prologue = msl
        .lines()
        .find(|l| l.contains("float4 lit0 = fig_light("))
        .expect("prologue present");
    assert!(
        prologue.contains("ppl_safe_normalize(in.texcoord14.xyz) * in.texcoord14.w"),
        "the prologue must re-scale the direction by the captured length:\n{prologue}"
    );
    let start = msl
        .find("static inline float4 fig_light(")
        .expect("fig_light present");
    let open = start + msl[start..].find('{').expect("fig_light opens");
    let close = open + msl[open..].find("}\n\n").expect("fig_light closes");
    let body = &msl[open..close];
    assert!(
        body.contains("dot(n,"),
        "fig_light must dot with the scaled normal:\n{body}"
    );
    assert!(
        !body.contains("normalize(n)") && !body.contains("ppl_safe_normalize(n)"),
        "fig_light must not renormalize the scaled skinned normal:\n{body}"
    );
    metal_compile_or_fail(&msl);
}

// ── Per-pixel lighting (ppl): Blinn-Phong specular ──

/// A points-only per-pixel variant with the specular bit and `model`.
fn specular_variant(model: u8) -> VariantKey {
    ppl_variant(0b110, 0, crate::ppl::PPL_MODEL_SPECULAR | model)
}

/// With the specular bit off, every emitter's output is byte-identical to
/// the output before the specular term existed (xxh3 of the MSL, recorded
/// from the emitters of commit 5f07b60), figure variants included
/// (`figure_specular_off_unchanged` covers every figure fixture).
#[test]
fn specular_off_unchanged() {
    use xxhash_rust::xxh3::xxh3_64;

    use crate::{
        dxso::ff::{
            FfVsKey, emit_ps_ff, emit_vs_ff,
            tests::{ppl_ff_ps_key, ppl_lit_vs_key},
        },
        ppl::{PPL_MODEL_DEBUG_TERM, PPL_MODEL_HALF_LAMBERT, PPL_MODEL_SMOOTH_FALLOFF},
    };
    let ps = daoc_terrain_ps20();
    let mut hashes = Vec::new();
    for vs in [
        ppl_lit_vs_key(0),
        ppl_lit_vs_key(0b110),
        FfVsKey {
            ppl_model: PPL_MODEL_HALF_LAMBERT | PPL_MODEL_SMOOTH_FALLOFF,
            ..ppl_lit_vs_key(0b111)
        },
    ] {
        hashes.push(xxh3_64(emit_vs_ff(&vs).as_bytes()));
    }
    let variants = [
        VariantKey::default(),
        ppl_variant(0b110, 0, 0),
        ppl_variant(
            0b111,
            0b001,
            PPL_MODEL_HALF_LAMBERT | PPL_MODEL_SMOOTH_FALLOFF,
        ),
        VariantKey {
            ppl_knee: 80,
            fog_mode: narrow(D3DFOG_LINEAR),
            ..ppl_variant(0b110, 0, PPL_MODEL_DEBUG_TERM)
        },
    ];
    for variant in variants {
        hashes.push(xxh3_64(emit_ps_ff(&ppl_ff_ps_key(), variant).as_bytes()));
        hashes.push(xxh3_64(
            emit_ps_programmable(&ps, variant)
                .expect("emit terrain")
                .as_bytes(),
        ));
    }
    for layout in [1u8, 2] {
        let off = emit_ps_programmable(&ps, figure_variant(layout, PPL_MODEL_HALF_LAMBERT, 80))
            .expect("emit figure");
        hashes.push(xxh3_64(off.as_bytes()));
    }
    let golden: [u64; 13] = [
        0x9b71_6aeb_d6e7_1896,
        0xe24d_b9e8_18ea_bb4a,
        0xa232_50da_bb94_ae8f,
        0x8dd4_3d44_45e6_3bc2,
        0x814a_8dc3_1295_22f0,
        0x1df4_9c31_47c9_dd5e,
        0xfe2c_8617_f851_2466,
        0x29e5_d0d8_9c12_eb85,
        0x2c7d_47b7_d7c2_faf0,
        0x8579_a8d1_9876_db6f,
        0x74ef_51f2_fd61_ed7e,
        0xe449_e70f_25cc_e6de,
        0xd754_61f4_d767_cd3e,
    ];
    assert_eq!(hashes, golden, "{hashes:#x?}");
}

#[test]
fn specular_added_before_fog() {
    use crate::dxso::ff::{emit_ps_ff, tests::ppl_ff_ps_key};
    let fogged = VariantKey {
        fog_mode: narrow(D3DFOG_LINEAR),
        ..specular_variant(0)
    };
    let ff = emit_ps_ff(&ppl_ff_ps_key(), fogged);
    let add = ff
        .find("    current.rgb = saturate(current.rgb + ppl_spec);\n")
        .unwrap_or_else(|| panic!("no FF highlight add:\n{ff}"));
    let fog = ff.find("mix(fog_data[0].rgb").expect("fog blend");
    assert!(add < fog, "{ff}");
    // After the cascade: no stage writes `current` past the add.
    assert!(
        ff[..add].contains("    current = "),
        "the cascade precedes the add:\n{ff}"
    );
    assert!(!ff[add + 1..].contains("    current = "), "{ff}");

    let terrain = emit_ps_programmable(&daoc_terrain_ps20(), fogged).expect("emit terrain");
    let add = terrain
        .find("    oC0.rgb = saturate(oC0.rgb + ppl_spec);\n")
        .unwrap_or_else(|| panic!("no programmable highlight add:\n{terrain}"));
    let fog = terrain.find("mix(fog_data[0].rgb").expect("fog blend");
    let ret = terrain.find("    return oC0;").expect("return");
    assert!(add < fog && fog < ret, "{terrain}");
    // Every shader write of `oC0` precedes the add; after it only fog reads it.
    for line in terrain[add..].lines().skip(1) {
        assert!(
            !line.contains("oC0") || line.contains("fog") || line.contains("return"),
            "{line} after the add:\n{terrain}"
        );
    }
}

#[test]
fn specular_scale_is_not_in_the_key() {
    use crate::{
        dxso::ff::{
            FfVsKey, emit_ps_ff, emit_vs_ff,
            tests::{ppl_ff_ps_key, ppl_lit_vs_key},
        },
        ppl::{PplConfig, PplMode, PsPairing, ppl_key_fields, ppl_masks},
    };
    let base = PplConfig {
        mode: PplMode::All,
        specular: true,
        ..PplConfig::default()
    };
    let configs = [
        base,
        PplConfig {
            specular_scale_pct: 250,
            ..base
        },
        PplConfig {
            specular_power: Some(40),
            ..base
        },
        PplConfig {
            specular_default_pct: Some([10, 20, 30]),
            ..base
        },
    ];
    let ps = daoc_terrain_ps20();
    let outputs = configs.map(|cfg| {
        let vs = ppl_lit_vs_key(0);
        let masks = ppl_masks(&cfg, &vs, PsPairing::FixedFunction);
        let (ppl_mask, ppl_model, ppl_knee) = ppl_key_fields(&cfg, masks);
        let vs = FfVsKey {
            ppl_mask,
            ppl_model,
            ppl_knee,
            ..vs
        };
        let variant = VariantKey {
            ppl_knee,
            ..ppl_variant(ppl_mask, masks.directional, ppl_model)
        };
        (
            vs.clone(),
            variant,
            emit_vs_ff(&vs),
            emit_ps_ff(&ppl_ff_ps_key(), variant),
            emit_ps_programmable(&ps, variant).expect("emit terrain"),
        )
    });
    for (vs, variant, ..) in &outputs {
        assert_ne!(vs.ppl_mask, 0);
        assert_ne!(variant.ppl_mask, 0);
        assert_ne!(vs.ppl_model & crate::ppl::PPL_MODEL_SPECULAR, 0);
        assert_ne!(variant.ppl_model & crate::ppl::PPL_MODEL_SPECULAR, 0);
    }
    for other in &outputs[1..] {
        assert!(
            outputs[0] == *other,
            "a scale/power/default change keyed a shader"
        );
    }
}

#[test]
fn specular_msl_compiles_under_metal() {
    use crate::{
        dxso::ff::{
            FfVsKey, emit_ps_ff, emit_vs_ff,
            tests::{ppl_ff_ps_key, ppl_lit_vs_key},
        },
        ppl::{
            PPL_MODEL_DEBUG_TERM, PPL_MODEL_HALF_LAMBERT, PPL_MODEL_SMOOTH_FALLOFF,
            PPL_MODEL_SPECULAR,
        },
    };
    for (mask, model) in [
        (0b110, PPL_MODEL_SPECULAR),
        (
            0b111,
            PPL_MODEL_SPECULAR | PPL_MODEL_HALF_LAMBERT | PPL_MODEL_SMOOTH_FALLOFF,
        ),
    ] {
        metal_compile_or_fail(&emit_vs_ff(&FfVsKey {
            ppl_model: model,
            ..ppl_lit_vs_key(mask)
        }));
    }
    let ps = daoc_terrain_ps20();
    let variants = [
        // `points`.
        specular_variant(0),
        // `all`, half-Lambert, smooth falloff and the shoulder.
        VariantKey {
            ppl_knee: 80,
            ..ppl_variant(
                0b111,
                0b001,
                PPL_MODEL_SPECULAR | PPL_MODEL_HALF_LAMBERT | PPL_MODEL_SMOOTH_FALLOFF,
            )
        },
        specular_variant(PPL_MODEL_DEBUG_TERM),
        VariantKey {
            fog_mode: narrow(D3DFOG_LINEAR),
            ..specular_variant(PPL_MODEL_HALF_LAMBERT)
        },
    ];
    for variant in variants {
        metal_compile_or_fail(&emit_ps_ff(&ppl_ff_ps_key(), variant));
        metal_compile_or_fail(&emit_ps_programmable(&ps, variant).expect("emit terrain"));
    }
}

// ── Per-pixel lighting (ppl): figure specular ──

/// The `fig_light` function of `msl`, signature through its last body line.
fn fig_light_fn(msl: &str) -> &str {
    let start = msl
        .find("static inline float4 fig_light(")
        .expect("fig_light present");
    let end = start + msl[start..].find("}\n\n").expect("fig_light closes");
    &msl[start..end]
}

/// With the specular bit off, every figure VS and figure PS fixture emits the
/// MSL it emitted before the figure specular term existed (xxh3, recorded
/// from the emitters of commit e864a91); with the bit on but no figure layout,
/// the FF and terrain pixel stages still emit that commit's output.
#[test]
fn figure_specular_off_unchanged() {
    use xxhash_rust::xxh3::xxh3_64;

    use crate::{
        dxso::ff::{emit_ps_ff, tests::ppl_ff_ps_key},
        ppl::{PPL_MODEL_HALF_LAMBERT, PPL_MODEL_SMOOTH_FALLOFF, PPL_MODEL_SPECULAR},
    };
    let mut hashes = Vec::new();
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let entry = crate::ppl::figures::figure_vs(id).unwrap_or_else(|| panic!("{name}"));
        let msl = emit_vs_programmable_named(
            &program,
            "clip_vs",
            u16::MAX,
            0,
            VsSamplerKinds::default(),
            Some(entry),
            None,
        )
        .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));
        hashes.push(xxh3_64(msl.as_bytes()));
    }
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        for variant in [
            figure_variant(1, 0, 0),
            figure_variant(2, PPL_MODEL_HALF_LAMBERT, 80),
        ] {
            let msl = emit_ps_programmable(&ps, variant)
                .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));
            hashes.push(xxh3_64(msl.as_bytes()));
        }
    }
    let terrain = daoc_terrain_ps20();
    for variant in [
        specular_variant(0),
        VariantKey {
            ppl_knee: 80,
            ..ppl_variant(
                0b111,
                0b001,
                PPL_MODEL_SPECULAR | PPL_MODEL_HALF_LAMBERT | PPL_MODEL_SMOOTH_FALLOFF,
            )
        },
    ] {
        // Bit-on FF/terrain: no figure term and no figure-only `sp[2]` read.
        for msl in [
            emit_ps_ff(&ppl_ff_ps_key(), variant),
            emit_ps_programmable(&terrain, variant).expect("emit terrain"),
        ] {
            assert!(!msl.contains("fig_light"), "{msl}");
            assert!(!msl.contains("sp[2]"), "{msl}");
        }
    }
    let golden: [u64; 30] = [
        0x3377_04fa_8981_f5ae,
        0x6ff8_a548_b888_26b5,
        0x7af4_877a_a072_44a3,
        0x4b9a_4e21_a973_9eb4,
        0x5a30_c7a2_8607_7977,
        0xe8c4_8240_37b3_9c6e,
        0x81b3_70ca_2e7d_b24a,
        0xff9f_7f75_ee9e_970b,
        0xa2b3_0086_95bc_1bc7,
        0xe629_e336_74dc_203e,
        0x6b10_2fda_9879_0c84,
        0x1a39_df8f_0066_9f6d,
        0xbb70_6d59_05f9_03fc,
        0xa68c_9efa_3e17_4391,
        0xdb23_028f_aa80_6bf6,
        0x02ad_b4f6_5f50_d173,
        0xb538_8bbc_144c_cfcd,
        0x6361_176b_2ce2_f15c,
        0x60d9_0984_8db1_3720,
        0x8da4_25cb_40a1_5e32,
        0xab92_cb6b_551a_c572,
        0x13e8_4a50_d99e_9aff,
        0xff90_100f_7449_54d1,
        0xd562_2889_506e_37db,
        0x7858_2b72_baa5_e8d5,
        0xde22_8f34_55f9_19d2,
        0x8c74_14b8_ddb0_4da0,
        0xece5_7ec1_61f8_099c,
        0xf9fc_6de4_c186_e1cb,
        0x4466_50ab_78ae_b3ce,
    ];
    assert_eq!(hashes, golden, "{hashes:#x?}");
}

#[test]
fn figure_specular_gated_on_ndotl_and_ndoth() {
    use crate::ppl::{PPL_MODEL_HALF_LAMBERT, PPL_MODEL_SPECULAR};
    for (layout, sun) in [(1u8, 12u32), (2, 13)] {
        for model in [0, PPL_MODEL_HALF_LAMBERT] {
            let msl = emit_ps_programmable(
                &daoc_terrain_ps20(),
                figure_variant(layout, PPL_MODEL_SPECULAR | model, 0),
            )
            .expect("emit");
            let f = fig_light_fn(&msl);
            assert!(
                f.contains("constant float4 *c, constant float4 *sp, thread float3 &spec) {"),
                "fig_light writes the highlight out:\n{f}"
            );
            // The view vector: toward the light-space eye point (`w = 1`), or
            // the bound direction to the camera (`w = 0`).
            assert!(
                f.contains(
                    "float3 V = sp[2].w > 0.5 ? ppl_safe_normalize(sp[2].xyz - p) : sp[2].xyz;"
                ),
                "{f}"
            );
            // N·H on the direction only; the skinned normal stays scaled
            // for the diffuse terms.
            assert!(f.contains("float3 N = ppl_safe_normalize(n);"), "{f}");
            assert!(
                f.contains("float power = sp[0].y > 0.0 ? sp[0].y : 10.0;"),
                "{f}"
            );
            // The sun: its direction row as the stock term dots it, its
            // diffuse row as the colour, no attenuation.
            assert!(
                f.contains(&format!("float3 L = ppl_safe_normalize(c[{sun}].xyz);")),
                "{f}"
            );
            assert!(
                f.contains(&format!(
                    "spec_sum += (dot(n, L) > 0.0 && ndoth > 0.0) ? spec_f * c[{}].xyz : float3(0.0);",
                    sun + 1
                )),
                "{f}"
            );
            // The point lights: their colour rows, the diffuse term's
            // attenuation and liveness.
            for k in 0..3 {
                let colour = sun + 4 + 3 * k;
                assert!(
                    f.contains(&format!(
                        "spec_sum += (live > 0.0 && dot(n, L) > 0.0 && ndoth > 0.0) ? atten * spec_f * c[{colour}].xyz : float3(0.0);"
                    )),
                    "point light {k}:\n{f}"
                );
            }
            assert_eq!(
                f.matches("float3 H = ppl_safe_normalize(L + V);").count(),
                4,
                "{f}"
            );
            assert_eq!(f.matches("float ndoth = dot(N, H);").count(), 4, "{f}");
            // Clamped `pow` base and exponent: no fast-math NaN.
            assert_eq!(
                f.matches("float spec_f = pow(max(ndoth, 1e-6), max(power, 1e-4));")
                    .count(),
                4,
                "{f}"
            );
            assert!(!f.contains("pow(ndoth"), "{f}");
            // No material: the `specularDefault` colour, scaled.
            assert!(
                f.contains("spec = spec_sum * sp[1].rgb * sp[1].w * sp[0].x;"),
                "{f}"
            );
        }
    }
}

/// The highlight is added to the pixel stage's final colour, rgb only and
/// not through the oD0 tint, and `fig_light`'s returns are those of the
/// bit-off variant.
#[test]
fn figure_specular_untinted_leaves_alpha() {
    use crate::ppl::{PPL_MODEL_DEBUG_TERM, PPL_MODEL_SPECULAR};
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        let msl = emit_ps_programmable(&ps, figure_variant(1, PPL_MODEL_SPECULAR, 0))
            .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));
        assert!(
            msl.contains(
                "    float3 ppl_spec = float3(0.0);\n    float4 lit0 = fig_light(in.color0, ppl_safe_normalize(in.texcoord14.xyz) * in.texcoord14.w, in.texcoord15.xyz, in.texcoord13.xyz, ppl_c, ppl_sp, ppl_spec);\n"
            ),
            "{name}:\n{msl}"
        );
        assert!(
            msl.contains("constant float4 *ppl_sp [[buffer(6)]]"),
            "{name}:\n{msl}"
        );
        let add = msl
            .find("    oC0.rgb = saturate(oC0.rgb + ppl_spec);\n")
            .unwrap_or_else(|| panic!("{name}: no highlight add:\n{msl}"));
        let ret = msl.rfind("    return oC0;").expect("return");
        assert!(add < ret, "{name}:\n{msl}");
        assert_eq!(msl.matches("ppl_spec").count(), 3, "{name}:\n{msl}");
        // The highlight never meets the tint (the signature line excluded).
        let f = fig_light_fn(&msl);
        for line in f.lines().skip(1).filter(|l| l.contains("spec")) {
            assert!(!line.contains("tint"), "{name}: {line}");
        }
        // The returns are the bit-off variant's, in every mode.
        for (extra, knee) in [(0, 0), (PPL_MODEL_DEBUG_TERM, 0), (0, 80)] {
            let ret = |m: &str| {
                fig_light_fn(m)
                    .lines()
                    .filter(|l| l.trim_start().starts_with("return"))
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            };
            let on = emit_ps_programmable(&ps, figure_variant(1, PPL_MODEL_SPECULAR | extra, knee))
                .expect("emit on");
            let off = emit_ps_programmable(&ps, figure_variant(1, extra, knee)).expect("emit off");
            assert_eq!(ret(&on), ret(&off), "{name}");
            assert!(!ret(&on).is_empty(), "{name}");
        }
    }
}

/// Every figure fixture compiles under Metal with the specular term: the ten
/// VS with their capture, and the ten PS for both layouts (each PS's paired
/// VS is one of them), with half-Lambert, the shoulder and the debug term.
#[test]
fn figure_specular_msl_compiles_under_metal() {
    use crate::ppl::{PPL_MODEL_DEBUG_TERM, PPL_MODEL_HALF_LAMBERT, PPL_MODEL_SPECULAR};
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let program = figure_fixture_program(name, bytes);
        let id = crate::ids::ProgramId::from_tokens(&program.bytecode()).raw();
        let entry = crate::ppl::figures::figure_vs(id).unwrap_or_else(|| panic!("{name}"));
        metal_compile_or_fail(
            &emit_vs_programmable_named(
                &program,
                "clip_vs",
                u16::MAX,
                0,
                VsSamplerKinds::default(),
                Some(entry),
                None,
            )
            .unwrap_or_else(|e| panic!("{name}: emit: {e:?}")),
        );
    }
    let variants = [
        figure_variant(1, PPL_MODEL_SPECULAR, 0),
        figure_variant(1, PPL_MODEL_SPECULAR | PPL_MODEL_HALF_LAMBERT, 80),
        figure_variant(2, PPL_MODEL_SPECULAR | PPL_MODEL_HALF_LAMBERT, 80),
        figure_variant(2, PPL_MODEL_SPECULAR | PPL_MODEL_DEBUG_TERM, 0),
    ];
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        for variant in variants {
            let msl = emit_ps_programmable(&ps, variant)
                .unwrap_or_else(|e| panic!("{name}: emit: {e:?}"));
            assert!(msl.contains("ppl_sp"), "{name}: no specular term:\n{msl}");
            metal_compile_or_fail(&msl);
        }
    }
}
