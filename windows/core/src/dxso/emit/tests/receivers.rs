//! Sun-shadow receivers: the `VariantKey::scene_vis` pixel stages.

use core::hash::{Hash, Hasher};

use xxhash_rust::xxh3::xxh3_64;

use super::{
    DCL_COLOR, END_TOKEN, FIGURE_PS_FIXTURES, FIGURE_VS_FIXTURES, OP_MOV, PS3_HEADER,
    SWIZ_IDENTITY, TYPE_COLOROUT, TYPE_INPUT, daoc_terrain_ps20, dcl, dst_token,
    emit_ps_programmable, emit_vs_programmable, figure_fixture_program, figure_ps_fixture_program,
    figure_variant, metal_compile_or_fail, opcode_token, parse, ppl_variant, src_token,
};
use crate::{
    dxso::{
        SCENE_VIS_SUN, VariantFlags, VariantKey,
        ff::{
            FfVsKey, emit_ps_ff, emit_vs_ff,
            tests::{ppl_ff_ps_key, ppl_lit_vs_key},
        },
        scene_vis_hash_tag,
    },
    ppl::{PPL_MODEL_HALF_LAMBERT, PPL_MODEL_SPECULAR},
};

/// `variant` reading the sun's visibility.
fn sun(variant: VariantKey) -> VariantKey {
    VariantKey {
        scene_vis: SCENE_VIS_SUN,
        ..variant
    }
}

/// The text of the MSL function whose signature starts with `head`, signature included.
fn function<'a>(msl: &'a str, head: &str) -> &'a str {
    let start = msl
        .find(head)
        .unwrap_or_else(|| panic!("no {head}:\n{msl}"));
    let end = start + msl[start..].find("\n}\n").expect("the function closes");
    &msl[start..end]
}

/// The `{ … }` block of light slot `i` in a `ppl_light` body (`lc[15 + 6i + 1]`
/// for a directional light).
fn directional_block(body: &str, i: usize) -> &str {
    let marker = format!("float3 L = -lc[{}].xyz;", 15 + 6 * i + 1);
    let start = body
        .find(&marker)
        .unwrap_or_else(|| panic!("no directional slot {i}:\n{body}"));
    let end = start + body[start..].find("    }\n").expect("the block closes");
    &body[start..end]
}

/// The read of the sun's visibility at the pixel's own position.
const VIS_READ: &str = "float scene_vis_v = scene_vis.read(uint2(in.position.xy)).r;";
/// The visibility texture argument.
const VIS_ARG: &str = "texture2d<float, access::read> scene_vis [[texture(16)]]";
/// The receiver uniform argument.
const RECEIVER_ARG: &str = "constant SceneReceiver &scene_rx [[buffer(4)]]";

/// `ps_3_0 { dcl_color v0; mov oC0, v0; }`: a shader-model-3 stage that
/// passes the vertex colour through.
fn ps3_color_passthrough() -> crate::dxso::DxsoProgram {
    let mut bc = vec![PS3_HEADER];
    bc.extend_from_slice(&dcl(DCL_COLOR, 0, TYPE_INPUT, 0, 0xF));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    parse(&bc).expect("ps_3_0 parses")
}

/// The stock shaders a receiver variant derives from, with `scene_vis` zero,
/// grouped as `(group, [MSL])`: the fixed-function pixel and vertex stages
/// (stock and per pixel), the terrain's mixed `ps_2_0`, every figure pixel
/// stage (stock and with the figure terms) and vertex stage, and the
/// `SpeedTree` leaf vertex stage.
fn stock_receiver_shaders() -> Vec<(&'static str, Vec<String>)> {
    let ff_ps = [
        VariantKey::default(),
        ppl_variant(0b111, 0b001, 0),
        ppl_variant(0b111, 0b001, PPL_MODEL_SPECULAR),
    ]
    .map(|variant| emit_ps_ff(&ppl_ff_ps_key(), variant))
    .to_vec();
    let ff_vs = [0, 0b111, 0b001]
        .map(|mask| emit_vs_ff(&ppl_lit_vs_key(mask)))
        .to_vec();
    let terrain = daoc_terrain_ps20();
    let mixed = [
        VariantKey::default(),
        ppl_variant(0b001, 0b001, 0),
        ppl_variant(0b111, 0b001, PPL_MODEL_SPECULAR),
    ]
    .map(|variant| emit_ps_programmable(&terrain, variant).expect("emit terrain"))
    .to_vec();
    let mut figure = Vec::new();
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        for variant in [
            VariantKey::default(),
            figure_variant(1, 0, 0),
            figure_variant(2, PPL_MODEL_SPECULAR, 0),
        ] {
            figure.push(
                emit_ps_programmable(&ps, variant).unwrap_or_else(|e| panic!("{name}: {e:?}")),
            );
        }
    }
    for (name, bytes) in FIGURE_VS_FIXTURES {
        let vs = figure_fixture_program(name, bytes);
        figure.push(emit_vs_programmable(&vs).unwrap_or_else(|e| panic!("{name}: {e:?}")));
    }
    let leaf: &[u8] = include_bytes!("../../../light_controls/fixtures/vs_speedtree_388.bin");
    let speedtree = vec![
        emit_vs_programmable(&figure_fixture_program("vs_speedtree_388.bin", leaf))
            .expect("emit SpeedTree leaf"),
    ];
    vec![
        ("ff pixel", ff_ps),
        ("ff vertex", ff_vs),
        ("mixed terrain", mixed),
        ("figure", figure),
        ("speedtree", speedtree),
    ]
}

/// One digest over a group's shaders, in order.
fn group_digest(shaders: &[String]) -> u64 {
    xxh3_64(shaders.join("\0").as_bytes())
}

/// With every `shadows.*`/`rt.*` key absent no draw carries a `scene_vis`
/// bit, and the shaders it draws with are the ones the emitters wrote
/// before the receivers existed: the digests were taken at the commit
/// before them (be655ab).
#[test]
fn scene_vis_off_unchanged() {
    const PINNED: [(&str, u64); 5] = [
        ("ff pixel", 0x0ae4_0b3a_4801_2968),
        ("ff vertex", 0xf070_2685_3411_02a4),
        ("mixed terrain", 0xd685_779c_d219_ef6f),
        ("figure", 0x9da4_2457_8c94_81c9),
        ("speedtree", 0x3e44_009f_e802_31c4),
    ];
    let groups = stock_receiver_shaders();
    let actual: Vec<(&str, u64)> = groups
        .iter()
        .map(|(group, shaders)| (*group, group_digest(shaders)))
        .collect();
    for (group, shaders) in &groups {
        for msl in shaders {
            for receiver in ["scene_vis", "[[texture(16)]]", "[[buffer(4)]]"] {
                assert!(!msl.contains(receiver), "{group}: {receiver}:\n{msl}");
            }
        }
    }
    assert_eq!(actual, PINNED, "stock shaders changed: {actual:#x?}");
}

/// The sun-only receivers (`shadows.*` alone, or `rt.shadows`): every
/// pixel stage reading [`SCENE_VIS_SUN`] and nothing else, per pixel and
/// vertex lit, on the fixed-function stage, the mixed terrain, a `ps_3_0`
/// and every figure pixel stage.
fn sun_receiver_shaders() -> Vec<String> {
    let mut out: Vec<String> = [
        VariantKey::default(),
        ppl_variant(0b111, 0b001, 0),
        ppl_variant(0b011, 0b011, PPL_MODEL_SPECULAR),
        ppl_variant(0b001, 0b001, PPL_MODEL_HALF_LAMBERT),
    ]
    .map(|variant| emit_ps_ff(&ppl_ff_ps_key(), sun(variant)))
    .to_vec();
    let terrain = daoc_terrain_ps20();
    for variant in [
        VariantKey::default(),
        ppl_variant(0b001, 0b001, 0),
        ppl_variant(0b111, 0b001, PPL_MODEL_SPECULAR),
    ] {
        out.push(emit_ps_programmable(&terrain, sun(variant)).expect("emit terrain"));
    }
    out.push(
        emit_ps_programmable(&ps3_color_passthrough(), sun(VariantKey::default()))
            .expect("emit ps_3_0"),
    );
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        for variant in [
            VariantKey::default(),
            figure_variant(1, 0, 0),
            figure_variant(2, PPL_MODEL_SPECULAR, 0),
        ] {
            out.push(
                emit_ps_programmable(&ps, sun(variant)).unwrap_or_else(|e| panic!("{name}: {e:?}")),
            );
        }
    }
    out
}

/// With only the sun's keys on (`rt.ao`, `rt.gi` and `rt.pointShadows`
/// absent) the receivers read what they read before the ray-traced terms
/// existed: the digest was taken at the commit before them (3ffb50f).
#[test]
fn ao_gi_points_off_unchanged() {
    const PINNED: u64 = 0xabd8_61fc_9bbf_5191;
    let digest = group_digest(&sun_receiver_shaders());
    assert_eq!(digest, PINNED, "sun receivers changed: {digest:#x}");
}

/// A hasher that keeps the bytes it is fed.
struct HashBytes(Vec<u8>);

impl Hasher for HashBytes {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
}

fn hash_bytes(value: &impl Hash) -> Vec<u8> {
    let mut hasher = HashBytes(Vec::new());
    value.hash(&mut hasher);
    hasher.0
}

/// The receiver bits hash under `0x20 | bits`, only when set, after every
/// other optional field: no two keys that differ in `water`, `fog_style`,
/// `color_style` or `scene_vis` feed the hasher the same bytes, a stock key
/// feeds what it fed before the field existed, and the padding byte is never
/// hashed. The tags also stay apart from the fixed-function VS key's scene
/// tag (`0x10 | scene`).
#[test]
fn scene_vis_hash_tag_disjoint() {
    let base = VariantKey {
        fog_mode: 3,
        ..VariantKey::default()
    };
    let mut streams = std::collections::HashMap::new();
    for water in 0..=1u8 {
        for fog_style in 0..8u8 {
            for color_style in 0..4u8 {
                for scene_vis in 0..16u8 {
                    let key = VariantKey {
                        water,
                        fog_style,
                        color_style,
                        scene_vis,
                        ..base
                    };
                    if let Some(other) = streams.insert(hash_bytes(&key), key) {
                        panic!("{key:?} hashes like {other:?}");
                    }
                }
            }
        }
    }
    // The stock stream gains nothing; a receiver appends exactly its tag.
    let stock = hash_bytes(&base);
    for bits in 1..16u8 {
        let tagged = hash_bytes(&VariantKey {
            scene_vis: bits,
            ..base
        });
        assert_eq!(tagged[..stock.len()], stock[..]);
        assert_eq!(tagged[stock.len()..], [scene_vis_hash_tag(bits)]);
        assert_eq!(scene_vis_hash_tag(bits), 0x20 | bits);
    }
    assert_eq!(
        hash_bytes(&VariantKey {
            reserved: 0xff,
            ..base
        }),
        stock,
        "the padding byte is never hashed"
    );
    // The fixed-function VS key's scene tags, read off its hash stream.
    let ff = ppl_lit_vs_key(0);
    let ff_stock = hash_bytes(&ff);
    for scene in 1..16u8 {
        let stream = hash_bytes(&FfVsKey { scene, ..ff });
        let ff_tag = stream[ff_stock.len()..].to_vec();
        for bits in 1..16u8 {
            assert_ne!(ff_tag, [scene_vis_hash_tag(bits)], "scene {scene}");
        }
    }
}

/// `ppl_light` scales the sun's `atten` by the visibility, which the
/// prologue reads at the pixel and passes last; diffuse and highlight both
/// go through `atten`. Point lights keep their attenuation.
#[test]
fn sun_atten_multiplied_by_vis() {
    for model in [0, PPL_MODEL_SPECULAR, PPL_MODEL_HALF_LAMBERT] {
        let msl = emit_ps_ff(&ppl_ff_ps_key(), sun(ppl_variant(0b111, 0b001, model)));
        let body = function(&msl, "static inline float4 ppl_light(");
        let signature = body.lines().next().expect("signature");
        assert!(signature.ends_with(", float vis) {"), "{signature}");
        let block = directional_block(body, 0);
        assert!(block.contains("float atten = vis;"), "{block}");
        assert!(
            block.contains("sum += atten * ndotl * (lc[17] * matDiffuse);"),
            "{block}"
        );
        if model == PPL_MODEL_SPECULAR {
            assert!(block.contains("? atten * spec_f * lc[17].rgb"), "{block}");
        }
        assert_eq!(
            body.matches("vis").count(),
            2,
            "signature and atten:\n{body}"
        );
        assert!(msl.contains(VIS_ARG), "{msl}");
        assert!(
            !msl.contains("[[buffer(4)]]"),
            "a per-pixel sun reads no k:\n{msl}"
        );
        let prologue: Vec<&str> = msl
            .lines()
            .skip_while(|l| !l.contains(VIS_READ))
            .take(3)
            .collect();
        assert!(
            prologue.iter().any(
                |l| l.contains("lit0 = ppl_light(in.color0,") && l.ends_with(", scene_vis_v);")
            ),
            "{msl}"
        );
        assert_eq!(msl.matches("in.color0").count(), 1, "{msl}");
    }
}

/// Only the sun — the lowest directional slot — takes the visibility: a
/// second directional light keeps `atten = 1`, and a sun behind a point
/// light in the compacted slots is still the one shadowed.
#[test]
fn second_directional_light_is_not_shadowed() {
    let msl = emit_ps_ff(&ppl_ff_ps_key(), sun(ppl_variant(0b011, 0b011, 0)));
    let body = function(&msl, "static inline float4 ppl_light(");
    assert!(
        directional_block(body, 0).contains("float atten = vis;"),
        "{body}"
    );
    assert!(
        directional_block(body, 1).contains("float atten = 1.0;"),
        "{body}"
    );
    assert_eq!(body.matches("float atten = vis;").count(), 1, "{body}");
    let late = emit_ps_ff(&ppl_ff_ps_key(), sun(ppl_variant(0b111, 0b110, 0)));
    let body = function(&late, "static inline float4 ppl_light(");
    assert!(
        directional_block(body, 1).contains("float atten = vis;"),
        "{body}"
    );
    assert!(
        directional_block(body, 2).contains("float atten = 1.0;"),
        "{body}"
    );
}

/// The terrain's mixed draw (fixed-function lit VS, `ps_2_0` splat shader
/// multiplied by `v0`) takes the forced per-pixel sun and its visibility:
/// every `v0` read is the shadowed `lit0`.
#[test]
fn mixed_terrain_receives_the_sun() {
    let ps = daoc_terrain_ps20();
    let stock = emit_ps_programmable(&ps, VariantKey::default()).expect("emit stock");
    let msl = emit_ps_programmable(&ps, sun(ppl_variant(0b001, 0b001, 0))).expect("emit");
    let body = function(&msl, "static inline float4 ppl_light(");
    assert!(
        directional_block(body, 0).contains("float atten = vis;"),
        "{body}"
    );
    assert!(msl.contains(VIS_ARG), "{msl}");
    assert!(msl.contains(VIS_READ), "{msl}");
    assert_eq!(msl.matches("in.color0").count(), 1, "{msl}");
    assert_eq!(
        msl.matches("lit0").count(),
        stock.matches("in.color0").count() + 1,
        "each v0 read became lit0:\n{msl}"
    );
}

/// `fig_light` splits the sun from its ambient: the sun's diffuse and its
/// highlight are scaled by the visibility, the ambient `c[S+2]` and the
/// point lights are not.
#[test]
fn fig_sun_multiplied_by_vis() {
    let (name, bytes) = FIGURE_PS_FIXTURES[0];
    let ps = figure_ps_fixture_program(name, bytes);
    for (layout, s) in [(1u8, 12u32), (2, 13)] {
        let msl = emit_ps_programmable(&ps, sun(figure_variant(layout, PPL_MODEL_SPECULAR, 0)))
            .expect("emit figure");
        let body = function(&msl, "static inline float4 fig_light(");
        let signature = body.lines().next().expect("signature");
        assert!(
            signature.ends_with("thread float3 &spec, float vis) {"),
            "{signature}"
        );
        assert!(
            body.contains(&format!(
                "    float3 fig_sun = saturate(dot(n, c[{s}].xyz)) * c[{}].xyz;\n    sum = fig_sun * vis + c[{}].xyz;",
                s + 1,
                s + 2
            )),
            "{body}"
        );
        assert!(
            body.contains(&format!("? spec_f * c[{}].xyz * vis :", s + 1)),
            "{body}"
        );
        assert_eq!(
            body.matches("vis").count(),
            3,
            "signature, sun, highlight:\n{body}"
        );
        assert!(msl.contains(VIS_ARG), "{msl}");
        assert!(!msl.contains("[[buffer(4)]]"), "{msl}");
        let lit = msl
            .lines()
            .find(|l| l.contains("float4 lit0 = fig_light("))
            .expect("prologue");
        assert!(lit.ends_with(", ppl_spec, scene_vis_v);"), "{lit}");
    }
    let half = emit_ps_programmable(&ps, sun(figure_variant(1, PPL_MODEL_HALF_LAMBERT, 0)))
        .expect("emit half-Lambert figure");
    assert!(
        half.contains(
            "    float3 fig_sun = saturate(hw) * c[13].xyz;\n    sum = fig_sun * vis + c[14].xyz;"
        ),
        "{half}"
    );
}

/// A lit draw outside per-pixel lighting scales its vertex-lit colour by
/// `mix(k, 1, vis)`, `k` from the receiver uniform; every diffuse read sees
/// the scaled colour, on the fixed-function pixel stage and on a
/// programmable one of any model.
#[test]
fn vs_lit_fallback_uses_ratio() {
    let fallback =
        "    float4 lit0 = in.color0;\n    lit0.rgb *= mix(scene_rx.k, 1.0, scene_vis_v);\n";
    let ff = emit_ps_ff(&ppl_ff_ps_key(), sun(VariantKey::default()));
    let terrain = emit_ps_programmable(&daoc_terrain_ps20(), sun(VariantKey::default()))
        .expect("emit terrain");
    let ps3 = emit_ps_programmable(&ps3_color_passthrough(), sun(VariantKey::default()))
        .expect("emit ps_3_0");
    for msl in [&ff, &terrain, &ps3] {
        assert!(
            msl.contains(&format!("    {VIS_READ}\n{fallback}")),
            "{msl}"
        );
        assert!(
            msl.contains("struct SceneReceiver {\n    float k;\n};"),
            "{msl}"
        );
        assert!(msl.contains(VIS_ARG), "{msl}");
        assert!(msl.contains(RECEIVER_ARG), "{msl}");
        assert_eq!(msl.matches("in.color0").count(), 1, "{msl}");
        assert!(!msl.contains("ppl_light"), "{msl}");
    }
    assert!(ff.contains("float4 current = lit0;"), "{ff}");
    assert!(ps3.contains("lit0"), "{ps3}");
}

/// Every receiver variant compiles: per pixel (fixed-function, mixed and
/// figure, with the specular, half-Lambert and shoulder models) and vertex
/// lit (fixed-function, point sprites, `ps_2_0`, `ps_3_0`).
#[test]
fn scene_vis_msl_compiles_under_metal() {
    let ff = ppl_ff_ps_key();
    for variant in [
        ppl_variant(0b111, 0b001, 0),
        ppl_variant(0b011, 0b011, PPL_MODEL_SPECULAR),
        ppl_variant(0b001, 0b001, PPL_MODEL_HALF_LAMBERT),
        VariantKey {
            ppl_knee: 80,
            ..ppl_variant(0b111, 0b001, 0)
        },
        VariantKey::default(),
        VariantKey {
            flags: VariantFlags::POINT_SPRITE,
            ..VariantKey::default()
        },
    ] {
        metal_compile_or_fail(&emit_ps_ff(&ff, sun(variant)));
    }
    let terrain = daoc_terrain_ps20();
    for variant in [
        ppl_variant(0b001, 0b001, 0),
        ppl_variant(0b111, 0b001, PPL_MODEL_SPECULAR),
        VariantKey::default(),
    ] {
        metal_compile_or_fail(&emit_ps_programmable(&terrain, sun(variant)).expect("terrain"));
    }
    metal_compile_or_fail(
        &emit_ps_programmable(&ps3_color_passthrough(), sun(VariantKey::default()))
            .expect("ps_3_0"),
    );
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        for variant in [
            figure_variant(1, 0, 0),
            figure_variant(2, PPL_MODEL_SPECULAR, 80),
            figure_variant(1, PPL_MODEL_HALF_LAMBERT, 0),
        ] {
            metal_compile_or_fail(
                &emit_ps_programmable(&ps, sun(variant))
                    .unwrap_or_else(|e| panic!("{name}: {e:?}")),
            );
        }
    }
}

/// `variant` reading `bits` of the scene textures.
fn with_bits(variant: VariantKey, bits: u8) -> VariantKey {
    VariantKey {
        scene_vis: bits,
        ..variant
    }
}

const AO_ARG: &str = "texture2d<float, access::read> scene_ao [[texture(17)]]";
const GI_ARG: &str = "texture2d<float, access::read> scene_gi [[texture(18)]]";
const POINTS_ARG: &str = "texture2d<float, access::read> scene_points [[texture(19)]]";

/// R12: under `SCENE_AMBIENT` the lit FF VS writes its ambient part —
/// global and per-light ambient times the material ambient, without the
/// emissive or any N·L term — to `texcoord8`; the stock key writes nothing
/// there and is unchanged.
#[test]
fn ff_vs_writes_the_ambient_part() {
    let stock = emit_vs_ff(&ppl_lit_vs_key(0b111));
    assert!(!stock.contains("out.texcoord8"), "{stock}");
    let key = FfVsKey {
        scene: FfVsKey::SCENE_AMBIENT,
        ..ppl_lit_vs_key(0b111)
    };
    let msl = emit_vs_ff(&key);
    assert!(
        msl.contains("    float4 sceneAmbient = vs_c[9] * vs_c[11];\n"),
        "{msl}"
    );
    for light in 0..3 {
        let line = format!(
            "        sceneAmbient += atten * (vs_c[{}] * vs_c[11]);",
            15 + 6 * light + 3
        );
        assert!(msl.contains(&line), "{line}:\n{msl}");
    }
    assert!(
        msl.contains("    out.texcoord8 = float4(sceneAmbient.rgb, 0.0);\n"),
        "{msl}"
    );
    assert_eq!(msl.matches("sceneAmbient +=").count(), 3, "{msl}");
    // Everything else is the stock function's.
    let stripped: String = msl
        .lines()
        .filter(|l| !l.contains("sceneAmbient"))
        .map(|l| format!("{l}\n"))
        .collect();
    assert_eq!(stripped, stock);
    // The bit is not a scene replay variant, and keys apart.
    assert_eq!(key.scene_variant(), crate::dxso::VsScene::Stock);
    assert_ne!(hash_bytes(&key), hash_bytes(&ppl_lit_vs_key(0b111)));
    metal_compile_or_fail(&msl);
}

/// A per-pixel fixed-function receiver of occlusion and bounce light
/// lights on `scene_base`: the colour with its ambient part (`texcoord8`)
/// replaced by the occluded ambient blended toward the bounce light times
/// the material diffuse (`texcoord13`) by `rt.gi` (R44); `ppl_light`
/// itself is unchanged.
#[test]
fn ppl_receivers_occlude_the_ambient_and_add_bounce_light() {
    use crate::dxso::{SCENE_VIS_AO, SCENE_VIS_GI};
    let stock = emit_ps_ff(&ppl_ff_ps_key(), sun(ppl_variant(0b111, 0b001, 0)));
    let msl = emit_ps_ff(
        &ppl_ff_ps_key(),
        with_bits(
            ppl_variant(0b111, 0b001, 0),
            SCENE_VIS_SUN | SCENE_VIS_AO | SCENE_VIS_GI,
        ),
    );
    for line in [
        "    float scene_ao_v = scene_ao.read(uint2(in.position.xy)).r;\n",
        "    float3 scene_gi_v = scene_gi.read(uint2(in.position.xy)).rgb;\n",
        "    float4 scene_base = in.color0;\n",
        "    float3 scene_ambient = in.texcoord8.rgb;\n",
        "    scene_base.rgb = max(scene_base.rgb - scene_ambient, float3(0.0)) + max(mix(scene_ambient * max(mix(1.0, scene_ao_v, scene_rx.ao), 0.0), scene_gi_v * in.texcoord13.rgb, scene_rx.gi), float3(0.0));\n",
    ] {
        assert!(msl.contains(line), "{line}:\n{msl}");
    }
    assert!(
        msl.lines()
            .any(|l| l.contains("lit0 = ppl_light(scene_base,") && l.ends_with(", scene_vis_v);")),
        "{msl}"
    );
    for arg in [
        VIS_ARG,
        AO_ARG,
        GI_ARG,
        "constant SceneReceiver &scene_rx [[buffer(4)]]",
    ] {
        assert!(msl.contains(arg), "{arg}:\n{msl}");
    }
    assert!(!msl.contains(POINTS_ARG));
    assert!(msl.contains(crate::shadows::SCENE_RECEIVER_RT_MSL));
    assert_eq!(
        function(&msl, "static inline float4 ppl_light("),
        function(&stock, "static inline float4 ppl_light("),
        "the lighting function is the sun receiver's"
    );
    assert_eq!(msl.matches("in.color0").count(), 1, "{msl}");
    // Without per-pixel lighting the bounce light is dropped and the
    // occlusion scales the whole colour by its ambient share.
    let vertex_lit = emit_ps_ff(
        &ppl_ff_ps_key(),
        with_bits(VariantKey::default(), SCENE_VIS_AO | SCENE_VIS_GI),
    );
    assert!(
        vertex_lit.contains(
            "    float4 lit0 = in.color0;\n    lit0.rgb *= max(mix(1.0, scene_ao_v, scene_rx.ao * scene_rx.k), 0.0);\n"
        ),
        "{vertex_lit}"
    );
    assert!(!vertex_lit.contains("scene_gi"), "{vertex_lit}");
    assert!(!vertex_lit.contains("in.texcoord8"), "{vertex_lit}");
}

/// R44: bounce light alone still splits the ambient off (`texcoord8`, no
/// occlusion factor) and blends it toward the bounce light; occlusion alone
/// keeps the ambient, occluded.
#[test]
fn gi_replaces_the_ambient_with_or_without_ao() {
    use crate::dxso::{SCENE_VIS_AO, SCENE_VIS_GI};
    let gi = emit_ps_ff(
        &ppl_ff_ps_key(),
        with_bits(ppl_variant(0b111, 0b001, 0), SCENE_VIS_GI),
    );
    assert!(
        gi.contains(
            "    scene_base.rgb = max(scene_base.rgb - scene_ambient, float3(0.0)) + max(mix(scene_ambient, scene_gi_v * in.texcoord13.rgb, scene_rx.gi), float3(0.0));\n"
        ),
        "{gi}"
    );
    assert!(!gi.contains("scene_ao"), "{gi}");
    let ao = emit_ps_ff(
        &ppl_ff_ps_key(),
        with_bits(ppl_variant(0b111, 0b001, 0), SCENE_VIS_AO),
    );
    assert!(
        ao.contains(
            "    scene_base.rgb = max(scene_base.rgb - scene_ambient, float3(0.0)) + scene_ambient * max(mix(1.0, scene_ao_v, scene_rx.ao), 0.0);\n"
        ),
        "{ao}"
    );
    assert!(!ao.contains("scene_gi"), "{ao}");
}

/// `point_match_by_position` (text): with `rt.pointShadows` the per-pixel
/// stage reads the point visibility, and every point light's block — not
/// the sun's — scales its attenuation by the channel of the scene point
/// within one unit of its own packed position (`lc[b]`), after the range.
#[test]
fn point_match_by_position() {
    use crate::dxso::SCENE_VIS_POINTS;
    let msl = emit_ps_ff(
        &ppl_ff_ps_key(),
        with_bits(
            ppl_variant(0b111, 0b001, 0),
            SCENE_VIS_SUN | SCENE_VIS_POINTS,
        ),
    );
    let body = function(&msl, "static inline float4 ppl_light(");
    assert!(
        body.lines()
            .next()
            .expect("signature")
            .ends_with(", float vis, float4 pvis, constant SceneReceiver &rx) {"),
        "{body}"
    );
    for slot in [1usize, 2] {
        let b = 15 + 6 * slot;
        let block_start = body
            .find(&format!("float3 toL = lc[{b}].xyz - posEye;"))
            .expect("point block");
        let block = &body[block_start..];
        let block = &block[..block.find("    }\n").expect("closes")];
        let range = block.find("atten *= step(dist, k.w);").expect("range cut");
        let vis = block
            .find(&format!("atten *= scene_point_vis(lc[{b}].xyz, pvis, rx);"))
            .unwrap_or_else(|| panic!("slot {slot}:\n{block}"));
        assert!(vis > range);
    }
    assert_eq!(body.matches("scene_point_vis").count(), 2, "{body}");
    assert!(directional_block(body, 0).contains("float atten = vis;"));
    assert!(msl.contains(POINTS_ARG), "{msl}");
    assert!(
        msl.contains("    float4 scene_points_v = scene_points.read(uint2(in.position.xy));\n")
    );
    assert!(
        msl.lines()
            .any(|l| l.contains("lit0 = ppl_light(in.color0,")
                && l.ends_with(", scene_vis_v, scene_points_v, scene_rx);")),
        "{msl}"
    );
    // The match: a point within one unit, channel `k`; none matched is 1.
    let rt = crate::shadows::SCENE_RECEIVER_RT_MSL;
    assert!(rt.contains("if (rx.points[k].w > 0.5 && distance(p, rx.points[k].xyz) <= 1.0) {\n            v = vis[k];"));
    assert!(rt.contains("    float v = 1.0;\n"));
    // A per-pixel stage with no point light reads no point visibility.
    let sun_only = emit_ps_ff(
        &ppl_ff_ps_key(),
        with_bits(
            ppl_variant(0b001, 0b001, 0),
            SCENE_VIS_SUN | SCENE_VIS_POINTS,
        ),
    );
    assert!(!sun_only.contains("scene_points"), "{sun_only}");
}

/// Figures take occlusion on their ambient row `c[S+2]`, which bounce
/// light replaces by `rt.gi` (R44), never point visibility; the sun stays
/// as it was.
#[test]
fn figures_take_ao_and_gi() {
    use crate::dxso::{SCENE_VIS_AO, SCENE_VIS_GI, SCENE_VIS_POINTS};
    let (name, bytes) = FIGURE_PS_FIXTURES[0];
    let ps = figure_ps_fixture_program(name, bytes);
    let msl = emit_ps_programmable(
        &ps,
        with_bits(
            figure_variant(1, 0, 0),
            SCENE_VIS_SUN | SCENE_VIS_AO | SCENE_VIS_GI | SCENE_VIS_POINTS,
        ),
    )
    .expect("emit figure");
    let body = function(&msl, "static inline float4 fig_light(");
    assert!(
        body.lines()
            .next()
            .expect("signature")
            .ends_with(", float vis, float ao, float3 gi, float gi_w) {"),
        "{body}"
    );
    assert!(
        body.contains(
            "    sum = fig_sun * vis + max(mix(c[14].xyz * ao, gi, gi_w), float3(0.0));\n"
        ),
        "{body}"
    );
    assert!(!body.contains("sum += gi"), "{body}");
    assert!(!msl.contains("scene_points"), "{msl}");
    assert!(!msl.contains("in.texcoord8"), "{msl}");
    assert!(
        msl.lines().any(|l| l.contains("lit0 = fig_light(in.color0,")
            && l.ends_with(
                ", scene_vis_v, max(mix(1.0, scene_ao_v, scene_rx.ao), 0.0), scene_gi_v, scene_rx.gi);"
            )),
        "{msl}"
    );
    // Occlusion alone, no sun: the stock sun line with the ambient occluded.
    let ao = emit_ps_programmable(&ps, with_bits(figure_variant(1, 0, 0), SCENE_VIS_AO))
        .expect("emit figure");
    let body = function(&ao, "static inline float4 fig_light(");
    assert!(
        body.contains("    sum = saturate(dot(n, c[12].xyz)) * c[13].xyz + c[14].xyz * ao;\n"),
        "{body}"
    );
}

/// Every ray-traced receiver compiles: the fixed-function stage per pixel
/// (all terms, each model) and vertex lit, the mixed terrain, a `ps_3_0`,
/// every figure pixel stage, and the ambient-writing vertex stages.
#[test]
fn rt_msl_compiles_under_metal() {
    use crate::dxso::{SCENE_VIS_AO, SCENE_VIS_GI, SCENE_VIS_POINTS};
    let all = SCENE_VIS_SUN | SCENE_VIS_AO | SCENE_VIS_GI | SCENE_VIS_POINTS;
    let rt_only = SCENE_VIS_AO | SCENE_VIS_GI | SCENE_VIS_POINTS;
    let ff = ppl_ff_ps_key();
    for bits in [all, rt_only, SCENE_VIS_POINTS, SCENE_VIS_AO, SCENE_VIS_GI] {
        for variant in [
            ppl_variant(0b111, 0b001, 0),
            ppl_variant(0b011, 0b011, PPL_MODEL_SPECULAR),
            ppl_variant(0b110, 0, PPL_MODEL_HALF_LAMBERT),
            VariantKey {
                ppl_knee: 80,
                ..ppl_variant(0b111, 0b001, 0)
            },
            VariantKey::default(),
        ] {
            metal_compile_or_fail(&emit_ps_ff(&ff, with_bits(variant, bits)));
        }
    }
    let terrain = daoc_terrain_ps20();
    for variant in [
        ppl_variant(0b111, 0b001, PPL_MODEL_SPECULAR),
        VariantKey::default(),
    ] {
        metal_compile_or_fail(
            &emit_ps_programmable(&terrain, with_bits(variant, all)).expect("terrain"),
        );
    }
    metal_compile_or_fail(
        &emit_ps_programmable(
            &ps3_color_passthrough(),
            with_bits(VariantKey::default(), all),
        )
        .expect("ps_3_0"),
    );
    for (name, bytes) in FIGURE_PS_FIXTURES {
        let ps = figure_ps_fixture_program(name, bytes);
        for variant in [
            figure_variant(1, 0, 0),
            figure_variant(2, PPL_MODEL_SPECULAR, 80),
        ] {
            for bits in [all, SCENE_VIS_AO | SCENE_VIS_GI, SCENE_VIS_GI] {
                metal_compile_or_fail(
                    &emit_ps_programmable(&ps, with_bits(variant, bits))
                        .unwrap_or_else(|e| panic!("{name}: {e:?}")),
                );
            }
        }
    }
    for flags in [
        crate::dxso::FfVsFlags::empty(),
        crate::dxso::FfVsFlags::HEMI_AMBIENT,
    ] {
        let base = ppl_lit_vs_key(0b111);
        metal_compile_or_fail(&emit_vs_ff(&FfVsKey {
            scene: FfVsKey::SCENE_AMBIENT,
            flags: base.flags | flags,
            ..base
        }));
    }
}
