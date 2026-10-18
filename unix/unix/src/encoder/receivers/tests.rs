//! Scene receivers: which draws read the sun's visibility, their bindings and uniform, and the GPU read.

use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    config::Mtld3dConfig,
    draw_data::{FixedVsSource, ShaderRef, VsSource},
    dxso::{FfVsFlags, FfVsKey, SCENE_VIS_SUN, VariantFlags, VariantKey, tci_entry},
    encoder_data::PostBoundary,
    ids::ProgramId,
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
    render_scale::RenderScale,
    rt::RtConfig,
    shadows::ShadowConfig,
    vs_draw::VS_DRAW_BYTES,
};
use mtld3d_shared::{
    Command, CommandType, MetalHandle, VertexAttrDesc,
    mtl::{
        CullMode, PS_FF_LIGHT_SLOT, PS_SCENE_RECEIVER_SLOT, PS_SCENE_VIS_TEXTURE, PixelFormat,
        PrimitiveType, VS_DRAW_SLOT, VS_FLOAT_CONST_SLOT, VS_POS_FIXUP_SLOT, VertexFormat,
        VertexStepFunction,
    },
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLDevice, MTLPixelFormat};

use crate::{
    encoder::{
        FrameEncoder,
        replay::tests::{Gpu, built, ff_ps},
        scene_capture::tests::{ff_vs, programmable_vs, speedtree_id},
        tests::{encoder_with_config, post_boundary_op, post_reset_frame},
    },
    metal::handle::IntoRetained,
};

const BACK_BUFFER: u64 = 0xB000;

fn sun() -> Mtld3dConfig {
    Mtld3dConfig {
        shadows: ShadowConfig {
            sun: true,
            ..ShadowConfig::default()
        },
        ..Mtld3dConfig::default()
    }
}

/// What `begin_frame` does for the receivers: a fresh pass state, a submission, the scene textures.
fn begin(enc: &mut FrameEncoder, size: (u32, u32), samples: u8) {
    post_reset_frame(enc, size, samples, true);
    enc.scene.begin_submission(false);
    enc.prepare_scene_textures(size, samples);
}

fn encoder(
    coherent: &AtomicU64,
    retained: &AtomicU64,
    config: Mtld3dConfig,
    size: (u32, u32),
) -> Option<(Retained<ProtocolObject<dyn MTLDevice>>, FrameEncoder)> {
    let (device, mut enc) = encoder_with_config(coherent, retained, config)?;
    begin(&mut enc, size, 1);
    Some((device, enc))
}

/// The id of a figure VS fixture (in `mtld3d_core::ppl::figures`' table).
fn figure_id() -> ProgramId {
    let bytes: &[u8] = include_bytes!(
        "../../../../../windows/core/src/dxso/emit/fixtures/daoc_figure_vs_15ecaea13070e5af.bin"
    );
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    ProgramId::from_tokens(&words)
}

/// The pixel variant of a draw that lights its sun (slot 0) per pixel.
const PER_PIXEL_SUN: VariantKey = VariantKey {
    ppl_mask: 0b1,
    ppl_directional_mask: 0b1,
    ..zero_variant()
};

const fn zero_variant() -> VariantKey {
    VariantKey {
        alpha_func: 0,
        fog_mode: 0,
        fog_table_mode: 0,
        linked_input_mask: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        fetch4_mask: 0,
        fetch4_alpha_mask: 0,
        raw_depth_red_mask: 0,
        volume_sampler_mask: 0,
        cube_sampler_mask: 0,
        tt_projected_mask: 0,
        color_out_mask: 0,
        sample_mask: 0,
        flags: VariantFlags::empty(),
        ppl_mask: 0,
        ppl_directional_mask: 0,
        ppl_model: 0,
        ppl_knee: 0,
        ppl_figure: 0,
        water: 0,
        fog_style: 0,
        color_style: 0,
        scene_vis: 0,
        reserved: 0,
    }
}

/// `vs` with its fixed-function key changed by `edit`.
fn with_key(vs: VsSource, edit: impl FnOnce(&mut FfVsKey)) -> VsSource {
    let VsSource::FixedFunction(mut source) = vs else {
        panic!("a fixed-function source")
    };
    edit(&mut source.key);
    VsSource::FixedFunction(source)
}

/// Only lit draws with a sun term into the back buffer before the boundary
/// read the visibility: a reflection-target draw reads nothing, the world
/// draw reads bit 0, and so does nothing after the boundary until the next
/// frame. Unlit, pre-transformed, normal-less (no N·L, so no sun term in
/// the vertex colour, whatever pixel stage it pairs with) and sun-less draws
/// and other programmable shaders never read it; the `SpeedTree` leaf and
/// the figures do.
#[test]
fn receiver_bit_only_on_backbuffer_world() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(&coherent, &retained, sun(), (64, 64)) else {
        return;
    };
    let lit = ff_vs(FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_NORMAL, 0);
    let normal_less = ff_vs(FfVsFlags::LIGHTING_ENABLED, 0);
    let unlit = ff_vs(FfVsFlags::HAS_NORMAL, 0);
    let rhw = ff_vs(FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_RHW, 0);
    let no_sun = with_key(
        ff_vs(FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_NORMAL, 0),
        |key| key.light_directional_mask = 0,
    );
    let (leaf, figure, other) = (
        programmable_vs(speedtree_id()),
        programmable_vs(figure_id()),
        programmable_vs(ProgramId::from_tokens(&[0xFFFE_0101, 0x0000_FFFF])),
    );
    let stock = zero_variant();
    let bits = |enc: &FrameEncoder, vs: &VsSource, variant: &VariantKey| {
        enc.scene_receiver_bits(vs.as_view(), variant)
    };

    // SAFETY: an opaque handle the pass state only names.
    let reflection = unsafe { MetalHandle::new(0xC000) };
    enc.pass_state.set_color_render_target(
        reflection,
        32,
        32,
        PixelFormat::Bgra8Unorm,
        RenderScale::IDENTITY,
    );
    for vs in [&lit, &leaf, &figure] {
        assert_eq!(bits(&enc, vs, &PER_PIXEL_SUN), 0, "the reflection pass");
        assert_eq!(bits(&enc, vs, &stock), 0, "the reflection pass");
    }

    // SAFETY: as above, the back buffer `post_reset_frame` named.
    let back_buffer = unsafe { MetalHandle::new(BACK_BUFFER) };
    enc.pass_state.set_color_render_target(
        back_buffer,
        64,
        64,
        PixelFormat::Bgra8Unorm,
        RenderScale::IDENTITY,
    );
    assert_eq!(bits(&enc, &lit, &PER_PIXEL_SUN), SCENE_VIS_SUN, "per pixel");
    assert_eq!(bits(&enc, &lit, &stock), SCENE_VIS_SUN, "vertex-lit");
    assert_eq!(
        bits(&enc, &leaf, &stock),
        SCENE_VIS_SUN,
        "the SpeedTree leaf"
    );
    assert_eq!(bits(&enc, &figure, &stock), SCENE_VIS_SUN, "a figure");
    assert_eq!(
        bits(
            &enc,
            &figure,
            &VariantKey {
                ppl_figure: 1,
                ppl_mask: 0b1111,
                ..stock
            }
        ),
        SCENE_VIS_SUN,
        "a figure lit per pixel"
    );
    for (name, vs, variant) in [
        ("unlit", &unlit, &stock),
        ("pre-transformed", &rhw, &stock),
        ("no sun", &no_sun, &stock),
        ("no normals", &normal_less, &stock),
        (
            "points per pixel",
            &no_sun,
            &VariantKey {
                ppl_mask: 0b1,
                ..stock
            },
        ),
        ("another programmable VS", &other, &stock),
    ] {
        assert_eq!(bits(&enc, vs, variant), 0, "{name}");
    }

    enc.post_boundary(&post_boundary_op());
    for vs in [&lit, &leaf, &figure] {
        assert_eq!(bits(&enc, vs, &PER_PIXEL_SUN), 0, "after the boundary");
        assert_eq!(bits(&enc, vs, &stock), 0, "after the boundary");
    }
    begin(&mut enc, (64, 64), 1);
    assert_eq!(
        bits(&enc, &lit, &PER_PIXEL_SUN),
        SCENE_VIS_SUN,
        "the next frame"
    );
    // A multisampled back buffer: the scene passes step aside.
    begin(&mut enc, (64, 64), 4);
    assert_eq!(bits(&enc, &lit, &PER_PIXEL_SUN), 0, "multisampled");
    enc.shutdown_cleanup();
}

/// Each key produces its own bits and nothing else: with every key absent
/// no draw reads anything; `rt.ao` alone gives every lit draw occlusion and
/// no sun; `rt.gi` only per-pixel draws; `rt.pointShadows` only draws that
/// light a point light per pixel (figures keep theirs unshadowed); all on,
/// each draw reads what it can apply.
#[test]
fn receiver_bits_follow_the_keys() {
    use mtld3d_core::dxso::{SCENE_VIS_AO, SCENE_VIS_GI, SCENE_VIS_POINTS};
    let lit = ff_vs(FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_NORMAL, 0);
    let with_point = with_key(
        ff_vs(FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_NORMAL, 0),
        |key| {
            key.light_active_mask = 0b11;
            key.light_directional_mask = 0b01;
        },
    );
    let sun_and_point = VariantKey {
        ppl_mask: 0b11,
        ppl_directional_mask: 0b01,
        ..zero_variant()
    };
    let figure_pp = VariantKey {
        ppl_figure: 1,
        ppl_mask: 0b1111,
        ..zero_variant()
    };
    let rt = |edit: fn(&mut RtConfig)| {
        let mut rt = RtConfig::default();
        edit(&mut rt);
        Mtld3dConfig {
            rt,
            ..Mtld3dConfig::default()
        }
    };
    let (leaf, figure) = (
        programmable_vs(speedtree_id()),
        programmable_vs(figure_id()),
    );
    let cases: [(&str, Mtld3dConfig, Vec<(&VsSource, VariantKey, u8)>); 5] = [
        (
            "every key absent",
            Mtld3dConfig::default(),
            vec![
                (&lit, zero_variant(), 0),
                (&lit, PER_PIXEL_SUN, 0),
                (&leaf, zero_variant(), 0),
            ],
        ),
        (
            "rt.ao",
            rt(|rt| rt.ao_pct = 100),
            vec![
                (&lit, zero_variant(), SCENE_VIS_AO),
                (&lit, PER_PIXEL_SUN, SCENE_VIS_AO),
                (&leaf, zero_variant(), SCENE_VIS_AO),
                (&figure, figure_pp, SCENE_VIS_AO),
            ],
        ),
        (
            "rt.gi",
            rt(|rt| rt.gi_pct = 100),
            vec![
                (&lit, zero_variant(), 0),
                (&lit, PER_PIXEL_SUN, SCENE_VIS_GI),
                (&leaf, zero_variant(), 0),
                (&figure, zero_variant(), 0),
                (&figure, figure_pp, SCENE_VIS_GI),
            ],
        ),
        (
            "rt.pointShadows",
            rt(|rt| rt.point_shadows = true),
            vec![
                (&lit, PER_PIXEL_SUN, 0),
                (&with_point, sun_and_point, SCENE_VIS_POINTS),
                (&with_point, zero_variant(), 0),
                (&figure, figure_pp, 0),
            ],
        ),
        (
            "everything",
            rt(|rt| {
                rt.shadows = true;
                rt.ao_pct = 100;
                rt.gi_pct = 100;
                rt.point_shadows = true;
            }),
            vec![
                (
                    &with_point,
                    sun_and_point,
                    SCENE_VIS_SUN | SCENE_VIS_AO | SCENE_VIS_GI | SCENE_VIS_POINTS,
                ),
                (&with_point, zero_variant(), SCENE_VIS_SUN | SCENE_VIS_AO),
                (
                    &figure,
                    figure_pp,
                    SCENE_VIS_SUN | SCENE_VIS_AO | SCENE_VIS_GI,
                ),
                (&leaf, zero_variant(), SCENE_VIS_SUN | SCENE_VIS_AO),
            ],
        ),
    ];
    for (name, config, draws) in cases {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let Some((_device, mut enc)) = encoder(&coherent, &retained, config, (64, 64)) else {
            return;
        };
        // SAFETY: the back buffer `post_reset_frame` named.
        let back_buffer = unsafe { MetalHandle::new(BACK_BUFFER) };
        enc.pass_state.set_color_render_target(
            back_buffer,
            64,
            64,
            PixelFormat::Bgra8Unorm,
            RenderScale::IDENTITY,
        );
        for (i, (vs, variant, expected)) in draws.into_iter().enumerate() {
            assert_eq!(
                enc.scene_receiver_bits(vs.as_view(), &variant),
                expected,
                "{name}: draw {i}"
            );
        }
        enc.shutdown_cleanup();
    }
}

/// The receiver commands of `pass`: `(texture at slot 16, (uniform address, length) at slot 4)`.
fn receiver_binds(commands: &[Command]) -> (Vec<u64>, Vec<(u64, u64)>) {
    let textures = commands
        .iter()
        .filter(|c| {
            c.cmd == CommandType::SetFragmentTexture as u32 && c.param_a == PS_SCENE_VIS_TEXTURE
        })
        .map(|c| c.param_b)
        .collect();
    let uniforms = commands
        .iter()
        .filter(|c| {
            c.cmd == CommandType::SetFragmentBytesAt as u32 && c.param_a == PS_SCENE_RECEIVER_SLOT
        })
        .map(|c| (c.param_b, c.param_c))
        .collect();
    (textures, uniforms)
}

/// The `k` a uniform at `address` holds.
fn k_at(address: u64) -> f32 {
    // SAFETY: the live encoder's scratch copy of a `SceneReceiver`.
    let bytes = unsafe { core::slice::from_raw_parts(address as *const u8, 4) };
    f32::from_le_bytes(bytes.try_into().expect("four bytes"))
}

/// The first receiver of each pass binds `vis` and the submission's one
/// uniform, the rest of the pass reuses them. The uniform holds full sun
/// until finalize fills it: from the boundary's ambient and sun when the
/// boundary fired in the submission, `k = 1` (no darkening) when it did not.
#[test]
fn receiver_uniform_filled_at_finalize() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(&coherent, &retained, sun(), (64, 64)) else {
        return;
    };
    // Nothing bound, nothing to fill.
    enc.fill_scene_receiver();
    let vis = enc.scene_textures().expect("live").vis.raw();
    enc.begin_render_pass_if_needed();
    enc.bind_scene_receivers();
    enc.bind_scene_receivers();
    enc.pass_state.end_current_pass("test");
    enc.begin_render_pass_if_needed();
    enc.bind_scene_receivers();
    enc.pass_state.end_current_pass("test");
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 2);
    let (first, second) = (
        receiver_binds(passes[0].commands()),
        receiver_binds(passes[1].commands()),
    );
    assert_eq!(first.0, [vis], "vis once in the first pass");
    assert_eq!(second.0, [vis], "and again in the next");
    assert_eq!(first.1.len(), 1);
    assert_eq!(first.1, second.1, "one uniform per submission");
    let (uniform, len) = first.1[0];
    assert_eq!(len, 80, "the whole SceneReceiver");
    assert_eq!(k_at(uniform), 1.0, "full sun until finalize");

    enc.post_boundary(&PostBoundary {
        ambient: [0.2; 3],
        sun_color: [0.6; 3],
        sun_dir_eye: Some([0.0, 0.0, 1.0]),
        sun_above_horizon: true,
        ..post_boundary_op()
    });
    enc.fill_scene_receiver();
    assert!((k_at(uniform) - 0.25).abs() < 1e-6, "{}", k_at(uniform));

    // The next submission's boundary has not fired: no darkening.
    begin(&mut enc, (64, 64), 1);
    enc.begin_render_pass_if_needed();
    enc.bind_scene_receivers();
    enc.pass_state.end_current_pass("test");
    let (_, uniforms) = receiver_binds(enc.pass_state.passes()[0].commands());
    assert_eq!(uniforms.len(), 1);
    enc.fill_scene_receiver();
    assert_eq!(k_at(uniforms[0].0), 1.0);
    enc.shutdown_cleanup();
}

// ── The receiver on the GPU ──

/// The side of the GPU test's target.
const SIDE: u32 = 16;
/// Bytes per vertex: a float3 position, then a float3 normal.
const STRIDE: u32 = 24;
/// D3D9's global ambient (row 9) and the sun's diffuse (row 17), grey.
const AMBIENT: f32 = 0.2;
const SUN: f32 = 0.6;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A lit fixed-function VS with a normal and one directional light, the sun, per pixel when `per_pixel`.
fn lit_vs(per_pixel: bool) -> VsSource {
    VsSource::FixedFunction(FixedVsSource {
        key: FfVsKey {
            ppl_knee: 0,
            color_style: 0,
            scene: 0,
            ppl_mask: u8::from(per_pixel),
            ppl_model: 0,
            flags: FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_NORMAL,
            input_tex_coord_count: 0,
            tex_coord_count: 0,
            light_active_mask: 1,
            light_directional_mask: 1,
            light_spot_mask: 0,
            diffuse_source: 0,
            ambient_source: 0,
            specular_source: 0,
            emissive_source: 0,
            fog_mode: 0,
            tci: [tci_entry(0, 0); 8],
            tex_coord_dims: [0; 8],
            tt_flags: [0; 8],
            vertex_blend_count: 0,
            declared_weights_count: 0,
            clip_plane_count: 0,
            passthrough: [0; 8],
        },
        max_row_count: 96,
        reserved: [0; 2],
    })
}

/// Identity world-view, a perspective from 1 to 10, the grey ambient, a
/// white material and the sun shining down the view axis.
fn lit_constants() -> Vec<u8> {
    let (near, far) = (1.0f32, 10.0f32);
    let a = far / (far - near);
    let mut rows = vec![[0.0f32; 4]; 96];
    rows[..8].copy_from_slice(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, a, -near * a],
        [0.0, 0.0, 1.0, 0.0],
    ]);
    rows[9] = [AMBIENT, AMBIENT, AMBIENT, 1.0];
    rows[10] = [1.0; 4];
    rows[11] = [1.0; 4];
    // Light 0: a directional light travelling +z (away from the camera).
    rows[16] = [0.0, 0.0, 1.0, 0.0];
    rows[17] = [SUN, SUN, SUN, 1.0];
    bytes(rows.as_flattened())
}

/// A full-target quad at eye z 5 facing the camera, as `(x, y, z, nx, ny, nz)`.
fn quad() -> Vec<u8> {
    let corner = |x: f32, y: f32| [x, y, 5.0, 0.0, 0.0, -1.0];
    let (a, b, c, d) = (
        corner(-10.0, -10.0),
        corner(10.0, -10.0),
        corner(-10.0, 10.0),
        corner(10.0, 10.0),
    );
    bytes([a, b, c, b, d, c].as_flattened())
}

fn snapshot(vs_fn: u64, ps_fn: u64) -> PipelineSnapshot {
    let mut stream_layouts = [StreamLayout::UNUSED; 16];
    stream_layouts[0] = StreamLayout {
        stride: STRIDE,
        step: VertexStepFunction::PerVertex,
        step_rate: 1,
    };
    PipelineSnapshot {
        // SAFETY: the encoder's library cache owns both functions.
        vs_fn: unsafe { MetalHandle::new(vs_fn) },
        // SAFETY: as above.
        ps_fn: unsafe { MetalHandle::new(ps_fn) },
        vdecl_hash: 0x5157,
        stream_layouts,
        color_format: PixelFormat::Bgra8Unorm,
        attach: PipelineAttachFlags::HAS_DEPTH | PipelineAttachFlags::HAS_COLOR_OUTPUT,
        rs: PipelineRsBits {
            color_write_mask: 0xF,
            ..PipelineRsBits::default()
        },
        extra: ExtraColorAttachments::NONE,
        ps_color_out_mask: 1,
        sample_count: 1,
    }
}

const ATTRS: [VertexAttrDesc; 2] = [
    VertexAttrDesc {
        attr_index: 0,
        buffer_index: 0,
        offset: 0,
        format: VertexFormat::Float3,
    },
    VertexAttrDesc {
        attr_index: 1,
        buffer_index: 0,
        offset: 12,
        format: VertexFormat::Float3,
    },
];

/// A lit fixed-function quad over a `vis` whose left half is black: on that
/// half the sun's term vanishes and the ambient remains; on the right the
/// sun adds in full. Per pixel (`ppl_light` scales the sun's `atten`) and
/// vertex-lit (`mix(k, 1, vis)` with the boundary's `k`) alike, each drawn
/// with the variant [`FrameEncoder::scene_receiver_bits`] gives it and the
/// bindings [`FrameEncoder::bind_scene_receivers`] records.
#[test]
fn receiver_reads_vis_at_pixel() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = encoder(&coherent, &retained, sun(), (SIDE, SIDE)) else {
        return;
    };
    let gpu = Gpu::new(Retained::clone(&device));
    let textures = *enc.scene_textures().expect("live scene textures");
    let vis = textures.vis.into_retained().expect("vis");
    let half = SIDE / 2;
    let halves: Vec<u16> = (0..SIDE * SIDE)
        .map(|i| if i % SIDE < half { 0x0000 } else { 0x3C00 })
        .collect();
    gpu.upload_halves(&vis, SIDE, &halves);

    // Each draw's variant, as `emit_draw_view` gates it during the world.
    let draws = [true, false].map(|per_pixel| {
        let vs = lit_vs(per_pixel);
        let stock = if per_pixel {
            PER_PIXEL_SUN
        } else {
            zero_variant()
        };
        let bits = enc.scene_receiver_bits(vs.as_view(), &stock);
        assert_eq!(bits, SCENE_VIS_SUN, "per pixel {per_pixel}");
        (
            per_pixel,
            vs,
            VariantKey {
                scene_vis: bits,
                ..stock
            },
        )
    });
    // The receiver bindings, as the first receiver of a pass records them.
    enc.begin_render_pass_if_needed();
    enc.bind_scene_receivers();
    enc.pass_state.end_current_pass("test");
    let receiver: Vec<Command> = enc.pass_state.passes()[0]
        .commands()
        .iter()
        .filter(|c| {
            (c.cmd == CommandType::SetFragmentTexture as u32 && c.param_a == PS_SCENE_VIS_TEXTURE)
                || (c.cmd == CommandType::SetFragmentBytesAt as u32
                    && c.param_a == PS_SCENE_RECEIVER_SLOT)
        })
        .copied()
        .collect();
    assert_eq!(receiver.len(), 2, "vis and the uniform");
    enc.post_boundary(&PostBoundary {
        ambient: [AMBIENT; 3],
        sun_color: [SUN; 3],
        sun_dir_eye: Some([0.0, 0.0, 1.0]),
        sun_above_horizon: true,
        ..post_boundary_op()
    });
    enc.fill_scene_receiver();

    let constants = lit_constants();
    let mut vs_draw = vec![0u8; VS_DRAW_BYTES];
    vs_draw[..16].copy_from_slice(&bytes(&[1.0, 1.0, 64.0, 0.0]));
    let pixel = 1.0 / f32::from(u16::try_from(SIDE).expect("small"));
    let pos_fixup = bytes(&[pixel, -pixel, 0.0, 1.0, 0.0]);
    let vertices = gpu.buffer(&quad());
    let ps = ff_ps();
    let raw = |data: &[u8]| {
        (
            data.as_ptr() as u64,
            u32::try_from(data.len()).expect("small"),
        )
    };
    for (per_pixel, vs, variant) in draws {
        let vs_lib = built(&mut enc, |enc| enc.resolve_vs_library(vs.as_view()));
        let ps_lib = built(&mut enc, |enc| {
            enc.resolve_ps_library(ps.as_view(), variant)
        });
        let shaders = ShaderRef {
            vs: vs.as_view(),
            ps: ps.as_view(),
            variant,
        };
        let pipeline = built(&mut enc, |enc| {
            enc.get_or_create_pipeline(
                &snapshot(vs_lib.func.raw(), ps_lib.func.raw()),
                &ATTRS[..],
                &shaders,
            )
        });
        let (c_ptr, c_len) = raw(&constants);
        let (d_ptr, d_len) = raw(&vs_draw);
        let (f_ptr, f_len) = raw(&pos_fixup);
        let mut commands = vec![
            Command::set_viewport(0, 0, SIDE, SIDE, 0.0, 1.0),
            Command::set_render_pipeline_state(pipeline),
            Command::set_cull_mode(CullMode::None),
            Command::set_vertex_buffer(Retained::as_ptr(&vertices) as u64, 0, 0),
            Command::set_vertex_bytes_at(c_ptr, c_len, VS_FLOAT_CONST_SLOT),
            Command::set_vertex_bytes_at(d_ptr, d_len, VS_DRAW_SLOT),
            Command::set_vertex_bytes_at(f_ptr, f_len, VS_POS_FIXUP_SLOT),
        ];
        if per_pixel {
            commands.push(Command::set_fragment_bytes_at(
                c_ptr,
                c_len,
                PS_FF_LIGHT_SLOT,
            ));
        }
        commands.extend_from_slice(&receiver);
        commands.push(Command::draw_primitives(PrimitiveType::Triangle, 0, 6));
        let color = gpu.target(MTLPixelFormat::BGRA8Unorm);
        let depth = gpu.target(MTLPixelFormat::Depth32Float);
        gpu.run(Some(&color), &depth, &commands);
        let texels = gpu.texels_at(&color, SIDE, 0, 4);
        let red = |x: u32, y: u32| f32::from(texels[((y * SIDE + x) * 4 + 2) as usize]) / 255.0;
        for y in [1, half, SIDE - 2] {
            for x in [1, half - 2] {
                assert!(
                    (red(x, y) - AMBIENT).abs() < 0.02,
                    "per pixel {per_pixel}: shadowed ({x}, {y}) {} is the ambient",
                    red(x, y)
                );
            }
            for x in [half + 1, SIDE - 2] {
                assert!(
                    (red(x, y) - (AMBIENT + SUN)).abs() < 0.02,
                    "per pixel {per_pixel}: lit ({x}, {y}) {} is ambient and sun",
                    red(x, y)
                );
            }
        }
    }
    enc.shutdown_cleanup();
}

/// The Classic blob's blend state: `ZERO` / `INVSRCCOLOR`.
fn blob_blend() -> PipelineRsBits {
    PipelineRsBits {
        flags: mtld3d_core::pipeline_state::PipelineRsFlags::BLEND_ENABLE,
        src_blend: mtld3d_types::D3DBLEND_ZERO as u8,
        dst_blend: mtld3d_types::D3DBLEND_INVSRCCOLOR as u8,
        ..PipelineRsBits::default()
    }
}

/// A Classic blob is left out only while the sun's visibility replaces it
/// (`shadows.sun`, `rt.shadows`), into the back buffer and before the
/// boundary, with the scene textures live: with every key absent, or only
/// `rt.ao`, it draws, as it does into another target, after the boundary
/// or on a multisampled back buffer.
#[test]
fn classic_blob_skipped_only_with_the_keys() {
    let unlit = ff_vs(FfVsFlags::empty(), 0);
    let blend = blob_blend();
    let skipped = |enc: &FrameEncoder| enc.classic_blob_skipped(unlit.as_view(), &blend, false, 4);
    // SAFETY: an opaque handle the pass state only names.
    let back_buffer = unsafe { MetalHandle::new(BACK_BUFFER) };
    let on_back_buffer = |enc: &mut FrameEncoder| {
        enc.pass_state.set_color_render_target(
            back_buffer,
            64,
            64,
            PixelFormat::Bgra8Unorm,
            RenderScale::IDENTITY,
        );
    };

    for (name, config) in [
        ("every key absent", Mtld3dConfig::default()),
        (
            "rt.ao alone",
            Mtld3dConfig {
                rt: RtConfig {
                    ao_pct: 100,
                    ..RtConfig::default()
                },
                ..Mtld3dConfig::default()
            },
        ),
    ] {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let Some((_device, mut enc)) = encoder(&coherent, &retained, config, (64, 64)) else {
            return;
        };
        on_back_buffer(&mut enc);
        assert!(!skipped(&enc), "{name}");
        enc.shutdown_cleanup();
    }

    for config in [
        sun(),
        Mtld3dConfig {
            rt: RtConfig {
                shadows: true,
                ..RtConfig::default()
            },
            ..Mtld3dConfig::default()
        },
    ] {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let Some((_device, mut enc)) = encoder(&coherent, &retained, config, (64, 64)) else {
            return;
        };
        // SAFETY: as above.
        let reflection = unsafe { MetalHandle::new(0xC000) };
        enc.pass_state.set_color_render_target(
            reflection,
            32,
            32,
            PixelFormat::Bgra8Unorm,
            RenderScale::IDENTITY,
        );
        assert!(!skipped(&enc), "another target");
        on_back_buffer(&mut enc);
        assert!(skipped(&enc), "the back buffer");
        assert!(
            !enc.classic_blob_skipped(unlit.as_view(), &blend, false, 6),
            "not a quad"
        );
        enc.post_boundary(&post_boundary_op());
        assert!(!skipped(&enc), "after the boundary");
        // A multisampled back buffer: the scene passes step aside.
        begin(&mut enc, (64, 64), 4);
        on_back_buffer(&mut enc);
        assert!(!skipped(&enc), "multisampled");
        enc.shutdown_cleanup();
    }
}

// ── The ray-traced receivers on the GPU ──

/// The receiver commands the first receiver of a pass records: every scene
/// texture bind and the uniform.
fn receiver_commands(enc: &mut FrameEncoder) -> Vec<Command> {
    enc.begin_render_pass_if_needed();
    enc.bind_scene_receivers();
    enc.pass_state.end_current_pass("test");
    let commands = enc.pass_state.passes()[0]
        .commands()
        .iter()
        .filter(|c| {
            (c.cmd == CommandType::SetFragmentTexture as u32
                && (PS_SCENE_VIS_TEXTURE..PS_SCENE_VIS_TEXTURE + 4).contains(&c.param_a))
                || (c.cmd == CommandType::SetFragmentBytesAt as u32
                    && c.param_a == PS_SCENE_RECEIVER_SLOT)
        })
        .copied()
        .collect();
    commands
}

/// The red channel of [`quad`] drawn with `vs` under `constants`, its pixel
/// variant `variant`, the receiver commands `receiver` and, per pixel, the
/// constants at the light slot too.
fn draw_quad(
    enc: &mut FrameEncoder,
    gpu: &Gpu,
    (vs, variant): (&VsSource, VariantKey),
    constants: &[u8],
    receiver: &[Command],
) -> Vec<f32> {
    let mut vs_draw = vec![0u8; VS_DRAW_BYTES];
    vs_draw[..16].copy_from_slice(&bytes(&[1.0, 1.0, 64.0, 0.0]));
    let pixel = 1.0 / f32::from(u16::try_from(SIDE).expect("small"));
    let pos_fixup = bytes(&[pixel, -pixel, 0.0, 1.0, 0.0]);
    let vertices = gpu.buffer(&quad());
    let ps = ff_ps();
    let raw = |data: &[u8]| {
        (
            data.as_ptr() as u64,
            u32::try_from(data.len()).expect("small"),
        )
    };
    let vs_lib = built(enc, |enc| enc.resolve_vs_library(vs.as_view()));
    let ps_lib = built(enc, |enc| enc.resolve_ps_library(ps.as_view(), variant));
    let shaders = ShaderRef {
        vs: vs.as_view(),
        ps: ps.as_view(),
        variant,
    };
    let pipeline = built(enc, |enc| {
        enc.get_or_create_pipeline(
            &snapshot(vs_lib.func.raw(), ps_lib.func.raw()),
            &ATTRS[..],
            &shaders,
        )
    });
    let (c_ptr, c_len) = raw(constants);
    let (d_ptr, d_len) = raw(&vs_draw);
    let (f_ptr, f_len) = raw(&pos_fixup);
    let mut commands = vec![
        Command::set_viewport(0, 0, SIDE, SIDE, 0.0, 1.0),
        Command::set_render_pipeline_state(pipeline),
        Command::set_cull_mode(CullMode::None),
        Command::set_vertex_buffer(Retained::as_ptr(&vertices) as u64, 0, 0),
        Command::set_vertex_bytes_at(c_ptr, c_len, VS_FLOAT_CONST_SLOT),
        Command::set_vertex_bytes_at(d_ptr, d_len, VS_DRAW_SLOT),
        Command::set_vertex_bytes_at(f_ptr, f_len, VS_POS_FIXUP_SLOT),
    ];
    if variant.ppl_mask != 0 {
        commands.push(Command::set_fragment_bytes_at(
            c_ptr,
            c_len,
            PS_FF_LIGHT_SLOT,
        ));
    }
    commands.extend_from_slice(receiver);
    commands.push(Command::draw_primitives(PrimitiveType::Triangle, 0, 6));
    let color = gpu.target(MTLPixelFormat::BGRA8Unorm);
    let depth = gpu.target(MTLPixelFormat::Depth32Float);
    gpu.run(Some(&color), &depth, &commands);
    let texels = gpu.texels_at(&color, SIDE, 0, 4);
    (0..SIDE * SIDE)
        .map(|i| f32::from(texels[(i * 4 + 2) as usize]) / 255.0)
        .collect()
}

/// R12 end to end: ambient occlusion darkens the ambient alone. A lit quad
/// (ambient 0.2, sun 0.6 straight on) over an occlusion texture whose left
/// half is 0: per pixel, the VS keyed `SCENE_AMBIENT` writes the ambient
/// part to `texcoord8` and the pixel stage takes it out (0.8 → 0.6); vertex
/// lit, the colour scales by `mix(1, ao, rt.ao · k)` with `k` 0.25, the
/// same 0.6. The open half keeps 0.8.
#[test]
fn ao_scales_the_ambient_part() {
    use mtld3d_core::dxso::SCENE_VIS_AO;
    use mtld3d_shared::mtl::PS_SCENE_AO_TEXTURE;
    let config = Mtld3dConfig {
        rt: RtConfig {
            ao_pct: 100,
            ..RtConfig::default()
        },
        ..Mtld3dConfig::default()
    };
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = encoder(&coherent, &retained, config, (SIDE, SIDE)) else {
        return;
    };
    let gpu = Gpu::new(Retained::clone(&device));
    let textures = *enc.scene_textures().expect("live scene textures");
    let ao = textures.ao.into_retained().expect("ao");
    let half = SIDE / 2;
    let halves: Vec<u16> = (0..SIDE * SIDE)
        .map(|i| if i % SIDE < half { 0x0000 } else { 0x3C00 })
        .collect();
    gpu.upload_halves(&ao, SIDE, &halves);
    let receiver = receiver_commands(&mut enc);
    assert!(
        receiver
            .iter()
            .any(|c| c.cmd == CommandType::SetFragmentTexture as u32
                && c.param_a == PS_SCENE_AO_TEXTURE
                && c.param_b == textures.ao.raw()),
        "the occlusion is bound"
    );
    // Each draw's bits, as `emit_draw_view` gates them during the world.
    let bits = [true, false].map(|per_pixel| {
        let stock = if per_pixel {
            PER_PIXEL_SUN
        } else {
            zero_variant()
        };
        enc.scene_receiver_bits(lit_vs(per_pixel).as_view(), &stock)
    });
    assert_eq!(bits, [SCENE_VIS_AO; 2]);
    enc.post_boundary(&PostBoundary {
        ambient: [AMBIENT; 3],
        sun_color: [SUN; 3],
        sun_dir_eye: Some([0.0, 0.0, 1.0]),
        sun_above_horizon: true,
        ..post_boundary_op()
    });
    enc.fill_scene_receiver();
    let constants = lit_constants();
    for (per_pixel, bits) in [true, false].into_iter().zip(bits) {
        let stock = if per_pixel {
            PER_PIXEL_SUN
        } else {
            zero_variant()
        };
        let vs = lit_vs(per_pixel);
        // The ambient-writing VS `emit_draw_view` swaps in for a per-pixel draw.
        let vs = with_key(vs, |key| {
            if per_pixel {
                key.scene = FfVsKey::SCENE_AMBIENT;
            }
        });
        let red = draw_quad(
            &mut enc,
            &gpu,
            (
                &vs,
                VariantKey {
                    scene_vis: bits,
                    ..stock
                },
            ),
            &constants,
            &receiver,
        );
        let at = |x: u32, y: u32| red[(y * SIDE + x) as usize];
        for y in [1, half, SIDE - 2] {
            for x in [1, half - 2] {
                assert!(
                    (at(x, y) - SUN).abs() < 0.02,
                    "per pixel {per_pixel}: occluded ({x}, {y}) {} keeps the sun alone",
                    at(x, y)
                );
            }
            for x in [half + 1, SIDE - 2] {
                assert!(
                    (at(x, y) - (AMBIENT + SUN)).abs() < 0.02,
                    "per pixel {per_pixel}: open ({x}, {y}) {} keeps the ambient",
                    at(x, y)
                );
            }
        }
    }
    enc.shutdown_cleanup();
}

/// R44 end to end: bounce light replaces the ambient. A per-pixel lit quad
/// (white material, ambient 0.2, sun 0.6 straight on) at `rt.gi = 1` over a
/// bounce light whose right half is the ambient tinted by a white sky —
/// what an open, sky-facing surface gathers — matches the stock draw there
/// (0.8); the left half, where the bounce light is 0, keeps the sun alone.
#[test]
fn gi_at_full_strength_on_an_open_plane_is_the_stock_ambient() {
    use mtld3d_core::dxso::SCENE_VIS_GI;
    let config = Mtld3dConfig {
        rt: RtConfig {
            gi_pct: 100,
            ..RtConfig::default()
        },
        ..Mtld3dConfig::default()
    };
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = encoder(&coherent, &retained, config, (SIDE, SIDE)) else {
        return;
    };
    let gpu = Gpu::new(Retained::clone(&device));
    let textures = *enc.scene_textures().expect("live scene textures");
    let gi = textures.gi.into_retained().expect("gi");
    let half = SIDE / 2;
    // 0.2 as a half float.
    let ambient_half = 0x3266;
    let texels: Vec<u16> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let v = if i % SIDE < half { 0 } else { ambient_half };
            [v, v, v, 0x3C00]
        })
        .collect();
    gpu.upload_texels(&gi, SIDE, MTLPixelFormat::RGBA16Float, &texels);
    let receiver = receiver_commands(&mut enc);
    let bits = enc.scene_receiver_bits(lit_vs(true).as_view(), &PER_PIXEL_SUN);
    assert_eq!(bits, SCENE_VIS_GI);
    enc.post_boundary(&PostBoundary {
        ambient: [AMBIENT; 3],
        sun_color: [SUN; 3],
        sun_dir_eye: Some([0.0, 0.0, 1.0]),
        sun_above_horizon: true,
        ..post_boundary_op()
    });
    enc.fill_scene_receiver();
    let constants = lit_constants();
    let stock = draw_quad(
        &mut enc,
        &gpu,
        (&lit_vs(true), PER_PIXEL_SUN),
        &constants,
        &receiver,
    );
    // The ambient-writing VS `emit_draw_view` swaps in for the receiver.
    let vs = with_key(lit_vs(true), |key| key.scene = FfVsKey::SCENE_AMBIENT);
    let red = draw_quad(
        &mut enc,
        &gpu,
        (
            &vs,
            VariantKey {
                scene_vis: bits,
                ..PER_PIXEL_SUN
            },
        ),
        &constants,
        &receiver,
    );
    let at = |v: &[f32], x: u32, y: u32| v[(y * SIDE + x) as usize];
    for y in [1, half, SIDE - 2] {
        for x in [half + 1, SIDE - 2] {
            assert!(
                (at(&stock, x, y) - (AMBIENT + SUN)).abs() < 0.02,
                "stock ({x}, {y}) {}",
                at(&stock, x, y)
            );
            assert!(
                (at(&red, x, y) - at(&stock, x, y)).abs() < 0.01,
                "open ({x}, {y}) {} against stock {}",
                at(&red, x, y),
                at(&stock, x, y)
            );
        }
        for x in [1, half - 2] {
            assert!(
                (at(&red, x, y) - SUN).abs() < 0.02,
                "no bounce ({x}, {y}) {} keeps the sun alone",
                at(&red, x, y)
            );
        }
    }
    enc.shutdown_cleanup();
}

/// `point_match_by_position` on the GPU: a quad lit by one point light per
/// pixel over a point visibility whose left half is 0 in channel 0. When
/// scene point 0 lies where the draw's light is, the left half loses the
/// light (the ambient alone) and the right keeps it; when it lies 1.5
/// units away, nothing matches and both halves are lit alike.
#[test]
fn point_match_by_position_on_the_gpu() {
    use mtld3d_core::{dxso::SCENE_VIS_POINTS, rt::ScenePoint};
    let config = Mtld3dConfig {
        rt: RtConfig {
            point_shadows: true,
            ..RtConfig::default()
        },
        ..Mtld3dConfig::default()
    };
    let light = [0.0f32, 0.0, 2.0];
    for (offset, matched) in [(0.0f32, true), (0.5, true), (1.5, false)] {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let Some((device, mut enc)) = encoder(&coherent, &retained, config.clone(), (SIDE, SIDE))
        else {
            return;
        };
        let gpu = Gpu::new(Retained::clone(&device));
        let textures = *enc.scene_textures().expect("live scene textures");
        let points = textures.points.into_retained().expect("points");
        let half = SIDE / 2;
        // RGBA16Float: channel 0 is 0 on the left, every other channel 1.
        let texels: Vec<u16> = (0..SIDE * SIDE)
            .flat_map(|i| {
                let first = if i % SIDE < half { 0x0000 } else { 0x3C00 };
                [first, 0x3C00, 0x3C00, 0x3C00]
            })
            .collect();
        gpu.upload_texels(&points, SIDE, MTLPixelFormat::RGBA16Float, &texels);
        let receiver = receiver_commands(&mut enc);
        // One point light per pixel, at eye (0, 0, 2), range 100; its bits
        // as `emit_draw_view` gates them during the world.
        let vs = with_key(lit_vs(true), |key| key.light_directional_mask = 0);
        let stock = VariantKey {
            ppl_mask: 1,
            ..zero_variant()
        };
        let bits = enc.scene_receiver_bits(vs.as_view(), &stock);
        assert_eq!(bits, SCENE_VIS_POINTS);
        let mut scene_points = [ScenePoint::default(); 4];
        scene_points[0] = ScenePoint {
            position: [light[0], light[1] + offset, light[2]],
            range: 100.0,
            diffuse: [SUN; 3],
            attenuation: [1.0, 0.0, 0.0],
        };
        enc.post_boundary(&PostBoundary {
            ambient: [AMBIENT; 3],
            points: scene_points,
            ..post_boundary_op()
        });
        enc.fill_scene_receiver();
        let mut rows = vec![[0.0f32; 4]; 96];
        rows.copy_from_slice(
            &lit_constants()
                .as_chunks::<16>()
                .0
                .iter()
                .map(|row| {
                    let f = |i: usize| f32::from_le_bytes(row[i..i + 4].try_into().expect("4"));
                    [f(0), f(4), f(8), f(12)]
                })
                .collect::<Vec<_>>(),
        );
        rows[15] = [light[0], light[1], light[2], 1.0];
        rows[16] = [0.0; 4];
        rows[19] = [1.0, 0.0, 0.0, 100.0];
        let constants = bytes(rows.as_flattened());
        let red = draw_quad(
            &mut enc,
            &gpu,
            (
                &vs,
                VariantKey {
                    scene_vis: bits,
                    ..stock
                },
            ),
            &constants,
            &receiver,
        );
        let at = |x: u32, y: u32| red[(y * SIDE + x) as usize];
        for y in [2, half, SIDE - 3] {
            for x in [2, half - 2] {
                let mirror = SIDE - 1 - x;
                assert!(
                    at(mirror, y) > AMBIENT + 0.2,
                    "offset {offset}: ({mirror}, {y}) {} is lit",
                    at(mirror, y)
                );
                if matched {
                    assert!(
                        (at(x, y) - AMBIENT).abs() < 0.02,
                        "offset {offset}: shadowed ({x}, {y}) {} is the ambient",
                        at(x, y)
                    );
                } else {
                    // The mirror pixel's centre sits half a pixel off
                    // symmetric: lit alike within that.
                    assert!(
                        at(x, y) > AMBIENT + 0.2 && (at(x, y) - at(mirror, y)).abs() < 0.06,
                        "offset {offset}: ({x}, {y}) {} unmatched, lit as ({mirror}) {}",
                        at(x, y),
                        at(mirror, y)
                    );
                }
            }
        }
        enc.shutdown_cleanup();
    }
}
