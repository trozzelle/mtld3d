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

/// With no sun key — every key absent, or only a key that captures the
/// world for something else (`rt.ao`) — no draw reads the visibility.
#[test]
fn receivers_off_without_the_sun_keys() {
    let lit = ff_vs(FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_NORMAL, 0);
    for config in [
        Mtld3dConfig::default(),
        Mtld3dConfig {
            rt: RtConfig {
                ao_pct: 100,
                ..RtConfig::default()
            },
            ..Mtld3dConfig::default()
        },
    ] {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let Some((_device, mut enc)) = encoder(&coherent, &retained, config, (64, 64)) else {
            return;
        };
        for variant in [zero_variant(), PER_PIXEL_SUN] {
            assert_eq!(enc.scene_receiver_bits(lit.as_view(), &variant), 0);
        }
        enc.shutdown_cleanup();
    }
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(
        &coherent,
        &retained,
        Mtld3dConfig {
            rt: RtConfig {
                shadows: true,
                ..RtConfig::default()
            },
            ..Mtld3dConfig::default()
        },
        (64, 64),
    ) else {
        return;
    };
    assert_eq!(
        enc.scene_receiver_bits(lit.as_view(), &PER_PIXEL_SUN),
        SCENE_VIS_SUN,
        "rt.shadows reads it too"
    );
    enc.shutdown_cleanup();
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
    assert_eq!(len, 4);
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
