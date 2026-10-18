//! The capture of vertex-shader geometry on the GPU: a skinned figure's
//! eye positions against the raster of the same draw, the BLAS built from
//! them, the base vertex, the store's bound, and where the pass goes.

use core::ptr::NonNull;
use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    config::Mtld3dConfig,
    depth_stencil_state::DepthStencilSnapshot,
    draw_data::{
        AttrSnapshot, DeclarationHeader, ProgrammableVsSource, PsSource, PsSourcePtr, ShaderRef,
        ShaderSourceFlags, VsSource, VsSourcePtr,
    },
    dxso::{VariantKey, VsSamplerKinds, VsScene},
    encoder_data::PostBoundary,
    ids::ProgramId,
    passes::Pass,
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
    shadows::{
        ShadowConfig,
        cascades::{Mat4, projection},
    },
    vs_draw::VS_DRAW_BYTES,
};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, Command, CommandType, MetalHandle, VertexAttrDesc,
    mtl::{
        CullMode, IndexType, PixelFormat, PrimitiveType, VS_CAPTURE_POS_SLOT, VS_CAPTURE_UV_SLOT,
        VertexFormat, VertexStepFunction,
    },
};
use mtld3d_types::{D3DCMP_GREATEREQUAL, D3DCMP_LESSEQUAL};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLLibrary, MTLPixelFormat, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState,
};

use super::{
    CaptureTarget, MASK_ALPHA, MASK_FIGURE, Planned,
    alpha_tests::{checker, opaque_texel, sampler, trace},
    capture_draw, inverse_projection_bytes,
    tests::{blit_pass, build_with_capture, count, length, next_frame, probe, rt_encoder},
};
use crate::{
    encoder::{
        FrameEncoder, pass_to_descriptor,
        replay::tests::{Gpu, SIZE, built, ff_ps},
        scene::assemble,
        scene_capture::{
            BufferContent, CapturedDraw, CapturedIndex, DrawFacts, ReplayBinding, Stage0,
            VertexBind, encode_captured_draw,
        },
        tests::{encoder_with_config, post_reset_frame},
    },
    metal::{handle::IntoRetained, rt::BlasBuild, rt_kernels::UV_CAPTURED},
};

/// Bytes per vertex: position float3 @0, coordinate float2 @12, bone
/// indices float4 @20, weights float3 @36.
const STRIDE: u32 = 48;
/// The camera's projection terms `[_11, _22, _33, _43]`: near 1, far 10.
const PROJ: [f32; 4] = [1.0, 1.0, 10.0 / 9.0, -10.0 / 9.0];
/// A sentinel no capture writes.
const SENTINEL: f32 = -12345.0;
/// The first constant row of the figure's bone palette, three rows a bone.
const BONE_BASE: usize = 26;

/// A bone: the rows of its object-to-eye 3x4 matrix.
type Bone = [[f32; 4]; 3];

/// One figure vertex.
#[derive(Clone, Copy)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
    bones: [u8; 3],
    weights: [f32; 3],
}

impl Vertex {
    const fn new(pos: [f32; 3], uv: [f32; 2], bones: [u8; 3], weights: [f32; 3]) -> Self {
        Self {
            pos,
            uv,
            bones,
            weights,
        }
    }

    /// Its eye position under `bones`: the weighted sum of its bones' transforms.
    fn skinned(&self, bones: &[Bone]) -> [f32; 3] {
        let p = [self.pos[0], self.pos[1], self.pos[2], 1.0];
        let mut eye = [0.0f32; 3];
        for (&bone, &weight) in self.bones.iter().zip(&self.weights) {
            let m = &bones[usize::from(bone)];
            for (c, out) in eye.iter_mut().enumerate() {
                *out += weight * (0..4).map(|k| m[c][k] * p[k]).sum::<f32>();
            }
        }
        eye
    }
}

/// Little-endian bytes of `values`.
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The vertex stream of `vertices`; the bone index `b` as the figure reads it, `(3b + 0.5) / 765`.
fn vertex_bytes(vertices: &[Vertex]) -> Vec<u8> {
    let floats: Vec<f32> = vertices
        .iter()
        .flat_map(|v| {
            let index = |b: u8| (3.0 * f32::from(b) + 0.5) / 765.0;
            [
                v.pos[0],
                v.pos[1],
                v.pos[2],
                v.uv[0],
                v.uv[1],
                index(v.bones[0]),
                index(v.bones[1]),
                index(v.bones[2]),
                0.0,
                v.weights[0],
                v.weights[1],
                v.weights[2],
            ]
        })
        .collect();
    bytes(&floats)
}

/// The figure's constants: rows 0..3 the clip transform (`P` for a
/// column vector, as the shader's `dp4`s read it), the bones from row 26.
fn figure_constants(bones: &[Bone]) -> Vec<u8> {
    let p = projection(PROJ);
    let mut rows = vec![[0.0f32; 4]; 96];
    for (r, row) in rows.iter_mut().take(4).enumerate() {
        *row = [p[0][r], p[1][r], p[2][r], p[3][r]];
    }
    for (b, bone) in bones.iter().enumerate() {
        rows[BONE_BASE + 3 * b..BONE_BASE + 3 * b + 3].copy_from_slice(bone);
    }
    bytes(rows.as_flattened())
}

/// Three bones: a move, a quarter turn about z and a move, a doubling and a move.
const BONES: [Bone; 3] = [
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 4.0],
    ],
    [
        [0.0, -1.0, 0.0, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 5.0],
    ],
    [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 3.0],
    ],
];

/// `bones` moved by `t` in eye space.
fn moved(bones: [Bone; 3], t: [f32; 3]) -> [Bone; 3] {
    bones.map(|mut bone| {
        for (row, d) in bone.iter_mut().zip(t) {
            row[3] += d;
        }
        bone
    })
}

/// A quad of four vertices, each blending the three bones differently.
fn quad() -> [Vertex; 4] {
    [
        Vertex::new([-1.0, -1.0, 0.0], [0.0, 1.0], [0, 1, 2], [0.7, 0.2, 0.1]),
        Vertex::new([1.0, -1.0, 0.2], [1.0, 1.0], [1, 0, 2], [0.6, 0.3, 0.1]),
        Vertex::new([-1.0, 1.0, 0.5], [0.0, 0.0], [2, 1, 0], [0.2, 0.3, 0.5]),
        Vertex::new([1.0, 1.0, 0.3], [1.0, 0.0], [0, 2, 1], [0.5, 0.25, 0.25]),
    ]
}

/// Vertices a draw skips by its base or first vertex, far from every probe.
fn skipped(n: usize) -> Vec<Vertex> {
    vec![Vertex::new([40.0, 40.0, 40.0], [0.5, 0.5], [0, 0, 0], [1.0, 0.0, 0.0]); n]
}

/// The figure's sources, declaration and draw-local bytes.
struct Figure {
    vs: VsSource,
    ps: PsSource,
    header: Box<DeclarationHeader>,
    attrs: Box<[VertexAttrDesc; 4]>,
    vs_draw: Vec<u8>,
    pos_fixup: Vec<u8>,
}

impl Figure {
    /// The figure fixture `daoc_figure_vs_d0e250f335b00f24`, registered with `enc`.
    fn new(enc: &mut FrameEncoder) -> Self {
        let raw: &[u8] = include_bytes!(
            "../../../../../windows/core/src/dxso/emit/fixtures/daoc_figure_vs_d0e250f335b00f24.bin"
        );
        let words: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let id = ProgramId::from_tokens(&words);
        assert!(
            mtld3d_core::ppl::figures::figure_vs(id.raw()).is_some(),
            "a figure of the table"
        );
        enc.register_program(
            id,
            mtld3d_core::dxso::parse(&words).expect("the fixture parses"),
        );
        let attr = |attr_index, offset, format| VertexAttrDesc {
            attr_index,
            buffer_index: 0,
            offset,
            format,
        };
        let mut vs_draw = vec![0u8; VS_DRAW_BYTES];
        vs_draw[..16].copy_from_slice(&bytes(&[1.0, 1.0, 64.0, 0.0]));
        Self {
            vs: VsSource::Programmable(ProgrammableVsSource {
                vs_id: id,
                max_const_used: 96,
                provided_input_mask: 0b1111,
                flags: ShaderSourceFlags::empty(),
                clip_plane_count: 0,
                sampler_kinds: VsSamplerKinds::default(),
                scene: 0,
                reserved: [0; 6],
            }),
            ps: ff_ps(),
            header: Box::new(DeclarationHeader {
                vdecl_hash: 0xF16,
                extents: [STRIDE; 16],
                count: 4,
                used_streams: 1,
                reserved: 0,
            }),
            attrs: Box::new([
                attr(0, 0, VertexFormat::Float3),
                attr(1, 12, VertexFormat::Float2),
                attr(2, 20, VertexFormat::Float4),
                attr(3, 36, VertexFormat::Float3),
            ]),
            vs_draw,
            // The camera's half-pixel offset; a capture binds a neutral one.
            pos_fixup: bytes(&[1.0 / 64.0, -1.0 / 64.0, 0.0, 1.0, 0.0]),
        }
    }

    fn token(&self) -> AttrSnapshot {
        // SAFETY: both boxes outlive every capture the test makes.
        unsafe {
            AttrSnapshot::new(
                NonNull::from(&self.attrs[0]),
                NonNull::from(self.header.as_ref()),
            )
        }
    }

    /// The world pass's snapshot: `Bgra8Unorm` colour over depth, the figure's stream.
    fn snapshot(vs_fn: u64, ps_fn: u64) -> PipelineSnapshot {
        let mut stream_layouts = [StreamLayout::UNUSED; 16];
        stream_layouts[0] = StreamLayout {
            stride: STRIDE,
            step: VertexStepFunction::PerVertex,
            step_rate: 1,
        };
        PipelineSnapshot {
            // SAFETY: the encoder's library cache owns both functions (or they are null).
            vs_fn: unsafe { MetalHandle::new(vs_fn) },
            // SAFETY: as above.
            ps_fn: unsafe { MetalHandle::new(ps_fn) },
            vdecl_hash: 0xF16,
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

    /// Capture its draw of `vertex_buffer` through `index` under `constants`, as `emit_draw_view` offers it.
    fn capture(
        &self,
        enc: &mut FrameEncoder,
        snapshot: &PipelineSnapshot,
        (pipeline, vertex_buffer): (u64, u64),
        index: CapturedIndex,
        constants: &[u8],
    ) -> CapturedDraw {
        self.capture_tested(
            enc,
            snapshot,
            (pipeline, vertex_buffer),
            (index, constants),
            None,
        )
    }

    /// [`Self::capture`], alpha-tested `GREATEREQUAL` 0.5 against `stage0` when given.
    fn capture_tested(
        &self,
        enc: &mut FrameEncoder,
        snapshot: &PipelineSnapshot,
        (pipeline, vertex_buffer): (u64, u64),
        (index, constants): (CapturedIndex, &[u8]),
        stage0: Option<Stage0>,
    ) -> CapturedDraw {
        let alpha_ref = 0.5f32.to_le_bytes();
        enc.begin_scene_binds();
        enc.stage_scene_bind(VertexBind::Buffer {
            handle: vertex_buffer,
            offset: 0,
            slot: 0,
            content: BufferContent::default(),
        });
        // SAFETY: the sources outlive the test's captures and do not move.
        let vs = unsafe { VsSourcePtr::new(NonNull::from(&self.vs)) };
        // SAFETY: as above.
        let ps = unsafe { PsSourcePtr::new(NonNull::from(&self.ps)) };
        let before = enc.scene_draws().len();
        enc.capture_scene_draw(&DrawFacts {
            snapshot,
            attrs: self.token(),
            fetch_attrs: None,
            vs,
            ps,
            variant: VariantKey {
                alpha_func: if stage0.is_some() {
                    u8::try_from(D3DCMP_GREATEREQUAL).expect("a D3DCMP value")
                } else {
                    0
                },
                ..VariantKey::default()
            },
            pipeline,
            depth_stencil: 0xD5,
            depth_write: true,
            blend: false,
            index,
            primitive: PrimitiveType::Triangle,
            instances: 1,
            vs_constants: constants,
            vs_draw: &self.vs_draw,
            vs_int: &[],
            vs_bool: &[],
            pos_fixup: &self.pos_fixup,
            alpha_ref: if stage0.is_some() { &alpha_ref } else { &[] },
            cull: CullMode::None,
            slope_scale: 0.0,
            stage0,
        });
        assert_eq!(
            enc.scene_draws().len(),
            before + 1,
            "the figure is captured"
        );
        enc.scene_draws()[before].clone()
    }
}

/// Build the capture variant of `d`'s vertex shader and its vertex-only pipeline now.
fn prebuild(enc: &mut FrameEncoder, d: &CapturedDraw) {
    let vs = built(enc, |enc| enc.scene_vs(d.vs.as_ref(), VsScene::Capture));
    built(enc, |enc| {
        enc.capture_pipeline(&d.snapshot, d.vertex_attrs(), vs)
    });
}

/// The boundary of a frame whose camera projects by [`PROJ`].
fn boundary() -> PostBoundary {
    PostBoundary {
        proj: PROJ,
        ..PostBoundary::default()
    }
}

/// A Shared buffer of `n` copies of `value`.
fn filled(gpu: &Gpu, n: usize, value: &[f32]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    gpu.buffer(&bytes(&value.repeat(n)))
}

/// `n` items of `N` floats from the start of `buffer`.
fn read<const N: usize>(buffer: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<[f32; N]> {
    assert!(buffer.length() >= n * N * 4);
    // SAFETY: the Shared buffer holds `n` items and the GPU is done with it.
    unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr().cast::<[f32; N]>(), n) }
        .to_vec()
}

/// The raw handle of a buffer the test keeps alive.
fn raw(buffer: &ProtocolObject<dyn MTLBuffer>) -> u64 {
    core::ptr::from_ref(buffer) as u64
}

/// Little-endian 16-bit indices.
fn indices(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// Run `commands` as a vertex-only scene pass ([`Pass::scene_vertex`]) through the real replay.
fn run_capture(gpu: &Gpu, commands: Vec<Command>) {
    let pass = Pass::scene_vertex(commands);
    gpu.run_descriptors(&[pass_to_descriptor(&pass, MetalHandle::NULL)]);
}

/// [`capture_draw`] once its builds have landed: the commands it encoded.
fn captured(enc: &mut FrameEncoder, d: &CapturedDraw, target: &CaptureTarget) -> Vec<Command> {
    let mut out = Vec::new();
    built(enc, |enc| {
        out.clear();
        capture_draw(enc, &mut out, d, target)
    });
    out
}

/// A depth-only pipeline whose vertex function returns the clip position `clip[vertex_id]`.
fn clip_pipeline(gpu: &Gpu) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
    const MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
vertex float4 clip_vs(device const float4 *clip [[buffer(0)]], uint vid [[vertex_id]]) {
    return clip[vid];
}
";
    let library = gpu
        .device
        .newLibraryWithSource_options_error(&NSString::from_str(MSL), None)
        .expect("the clip shader compiles");
    let function = library
        .newFunctionWithName(&NSString::from_str("clip_vs"))
        .expect("clip_vs");
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&function));
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    gpu.device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .expect("the clip pipeline")
}

/// The capture of a skinned figure, its eye positions projected by the
/// camera, rasterises as the figure's own draw does: the same pixels at the
/// same depths. The draw is indexed from base vertex 2, so the ids it
/// stores count from 2 and ids 0 and 1 stay as they were; each coordinate
/// is the vertex's own, and each position the blend of its bones.
#[test]
fn capture_matches_raster_positions() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let variant = VariantKey::default();
    let vs_lib = built(&mut enc, |enc| enc.resolve_vs_library(figure.vs.as_view()));
    let ps_lib = built(&mut enc, |enc| {
        enc.resolve_ps_library(figure.ps.as_view(), variant)
    });
    let snapshot = Figure::snapshot(vs_lib.func.raw(), ps_lib.func.raw());
    let shaders = ShaderRef {
        vs: figure.vs.as_view(),
        ps: figure.ps.as_view(),
        variant,
    };
    let pipeline = built(&mut enc, |enc| {
        enc.get_or_create_pipeline(&snapshot, &figure.attrs[..], &shaders)
    });

    const BASE: usize = 2;
    let quad = quad();
    let mut vertices = skipped(BASE);
    vertices.extend(quad);
    let vb = gpu.buffer(&vertex_bytes(&vertices));
    let list = [0u16, 1, 2, 1, 3, 2];
    let ib = gpu.buffer(&indices(&list));
    let constants = figure_constants(&BONES);
    let d = figure.capture(
        &mut enc,
        &snapshot,
        (pipeline, raw(&vb)),
        CapturedIndex::Buffer {
            handle: raw(&ib),
            offset: 0,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 2,
            content: BufferContent::default(),
        },
        &constants,
    );
    assert_eq!(
        super::capture_extent(&d),
        Ok(6),
        "the vertex buffer's length in vertices"
    );

    // The figure's own raster, under a neutral `pos_fixup` (no half-pixel offset).
    let depth_state = enc.get_or_create_depth_stencil(
        &DepthStencilSnapshot {
            depth_enable: 1,
            depth_write: 1,
            depth_func: u8::try_from(D3DCMP_LESSEQUAL).expect("a D3DCMP value"),
            ..DepthStencilSnapshot::inert()
        },
        false,
    );
    let mut world = vec![Command::set_viewport(0, 0, SIZE, SIZE, 0.0, 1.0)];
    encode_captured_draw(
        &mut world,
        &d,
        &ReplayBinding {
            pipeline: d.pipeline,
            depth_stencil: depth_state,
            cull: CullMode::None,
            depth_bias: (0.0, 0.0),
            pos_fixup: crate::encoder::replay::neutral_pos_fixup(),
            extra_vs: &[],
        },
    );
    let color = gpu.target(MTLPixelFormat::BGRA8Unorm);
    let raster = gpu.target(MTLPixelFormat::Depth32Float);
    gpu.run(Some(&color), &raster, &world);

    // The capture, into buffers one id longer than the draw reaches.
    let pos = filled(&gpu, 7, &[SENTINEL; 4]);
    let uv = filled(&gpu, 7, &[SENTINEL; 2]);
    let target = CaptureTarget {
        pos_buf: raw(&pos),
        pos_offset: 0,
        uv_buf: raw(&uv),
        uv_offset: 0,
        vertex_count: 6,
        p_inv: inverse_projection_bytes(&mut enc, PROJ).expect("a perspective"),
    };
    let commands = captured(&mut enc, &d, &target);
    assert!(
        commands
            .iter()
            .any(|c| c.cmd == CommandType::SetVertexBuffer as u32
                && c.param_a == VS_CAPTURE_POS_SLOT
                && c.param_b == raw(&pos)),
        "the positions are bound at their slot"
    );
    assert!(
        commands
            .iter()
            .any(|c| c.cmd == CommandType::SetVertexBuffer as u32
                && c.param_a == VS_CAPTURE_UV_SLOT
                && c.param_b == raw(&uv))
    );
    run_capture(&gpu, commands);
    let (eye, coords) = (read::<4>(&pos, 7), read::<2>(&uv, 7));
    for id in [0, 1, 6] {
        assert_eq!(eye[id], [SENTINEL; 4], "id {id} is no vertex of the draw");
        assert_eq!(coords[id], [SENTINEL; 2], "id {id}");
    }
    for (i, v) in quad.iter().enumerate() {
        let (got, want) = (eye[BASE + i], v.skinned(&BONES));
        assert!(
            (0..3).all(|c| (got[c] - want[c]).abs() < 1e-4) && got[3] == 1.0,
            "vertex {i}: {got:?} vs {want:?}"
        );
        assert_eq!(coords[BASE + i], v.uv, "vertex {i}");
    }

    // The captured positions under the camera, rasterised by a pass-through.
    let p: Mat4 = projection(PROJ);
    let clip: Vec<f32> = list
        .iter()
        .flat_map(|&index| {
            let e = eye[BASE + usize::from(index)];
            core::array::from_fn::<f32, 4, _>(|c| (0..4).map(|k| e[k] * p[k][c]).sum())
        })
        .collect();
    let clip_buffer = gpu.buffer(&bytes(&clip));
    let clip_pipeline = clip_pipeline(&gpu);
    let replayed = vec![
        Command::set_viewport(0, 0, SIZE, SIZE, 0.0, 1.0),
        Command::set_render_pipeline_state(Retained::as_ptr(&clip_pipeline) as u64),
        Command::set_depth_stencil_state(depth_state),
        Command::set_cull_mode(CullMode::None),
        Command::set_vertex_buffer(raw(&clip_buffer), 0, 0),
        Command::draw_primitives(PrimitiveType::Triangle, 0, 6),
    ];
    let projected = gpu.target(MTLPixelFormat::Depth32Float);
    gpu.run(None, &projected, &replayed);

    let (raster, projected) = (
        gpu.depths_at(&raster, SIZE, 0),
        gpu.depths_at(&projected, SIZE, 0),
    );
    let clear = 1.0f32.to_bits();
    let covered = raster.iter().filter(|&&depth| depth != clear).count();
    assert!(
        covered > 8 && covered < raster.len(),
        "{covered} pixels covered"
    );
    for (index, (&a, &b)) in raster.iter().zip(&projected).enumerate() {
        assert_eq!(a == clear, b == clear, "pixel {index}: coverage");
        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
        assert!(
            (a - b).abs() < 1e-3,
            "pixel {index}: raster {a} vs captured {b}"
        );
    }
}

/// A vertex id at or past the capture's count stores nothing: the draw
/// reaches ids 2..6, the target counts 5, and id 5 keeps its sentinel while
/// ids 2..4 are written.
#[test]
fn capture_store_bounded_by_count() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let mut vertices = skipped(2);
    vertices.extend(quad());
    let vb = gpu.buffer(&vertex_bytes(&vertices));
    let ib = gpu.buffer(&indices(&[0, 1, 2, 1, 3, 2]));
    let d = figure.capture(
        &mut enc,
        &Figure::snapshot(0, 0),
        (0x91, raw(&vb)),
        CapturedIndex::Buffer {
            handle: raw(&ib),
            offset: 0,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 2,
            content: BufferContent::default(),
        },
        &figure_constants(&BONES),
    );
    let pos = filled(&gpu, 8, &[SENTINEL; 4]);
    let uv = filled(&gpu, 8, &[SENTINEL; 2]);
    let target = CaptureTarget {
        pos_buf: raw(&pos),
        pos_offset: 0,
        uv_buf: raw(&uv),
        uv_offset: 0,
        vertex_count: 5,
        p_inv: inverse_projection_bytes(&mut enc, PROJ).expect("a perspective"),
    };
    let commands = captured(&mut enc, &d, &target);
    run_capture(&gpu, commands);
    let (eye, coords) = (read::<4>(&pos, 8), read::<2>(&uv, 8));
    for id in 2..5 {
        assert_ne!(eye[id], [SENTINEL; 4], "id {id} is stored");
    }
    for id in [0, 1, 5, 6, 7] {
        assert_eq!(eye[id], [SENTINEL; 4], "id {id} is not stored");
        assert_eq!(coords[id], [SENTINEL; 2], "id {id}");
    }

    // A non-indexed draw sizes its capture by its own range, and one
    // running past its vertex buffer (6 vertices) is left out, counted.
    let plain = |enc: &mut FrameEncoder, start_vertex| {
        figure.capture(
            enc,
            &Figure::snapshot(0, 0),
            (0x91, raw(&vb)),
            CapturedIndex::None {
                start_vertex,
                vertex_count: 3,
            },
            &figure_constants(&BONES),
        )
    };
    assert_eq!(super::capture_extent(&plain(&mut enc, 3)), Ok(6));
    let past = plain(&mut enc, 4);
    assert_eq!(super::capture_extent(&past), Err(super::Skip::Extent));
    enc.scene.note_boundary(&boundary());
    let (tlas, _, capture) = build_with_capture(&mut enc, &[past]);
    assert!(tlas.is_none() && capture.is_empty(), "nothing captured");
    assert_eq!(enc.rt_stats().skipped, 1);
}

/// The BLASes this frame's blits build.
fn blas_builds(blits: &[BlitCommand]) -> Vec<BlasBuild> {
    blits
        .iter()
        .filter(|b| b.cmd == BlitCommandType::BuildBlas as u32)
        // SAFETY: a BuildBlas names its `BlasBuild` in the live encoder's scratch arena.
        .map(|b| unsafe { core::ptr::read_unaligned(b.byte_size as *const BlasBuild) })
        .collect()
}

/// The centroid of three eye positions.
fn centroid(points: [[f32; 3]; 3]) -> [f32; 3] {
    core::array::from_fn(|c| (points[0][c] + points[1][c] + points[2][c]) / 3.0)
}

/// Capture `draws`, build the frame's TLAS and run the capture pass and
/// the builds in one command buffer: the TLAS and the blits.
fn trace_frame(
    enc: &mut FrameEncoder,
    gpu: &Gpu,
    draws: &[CapturedDraw],
) -> (super::TlasHandle, Vec<BlitCommand>) {
    enc.scene.note_boundary(&boundary());
    let (tlas, blits, capture) = build_with_capture(enc, draws);
    let tlas = tlas.expect("a TLAS");
    assert!(!capture.is_empty(), "the figure is captured");
    let pass = Pass::scene_vertex(capture);
    gpu.run_descriptors(&[
        pass_to_descriptor(&pass, MetalHandle::NULL),
        blit_pass(&blits),
    ]);
    (tlas, blits)
}

/// A ray from the eye at each skinned triangle's centroid hits the BLAS
/// built from the captured positions, under the identity, at the centroid's
/// distance. The next frame's bones move the figure: its BLAS is built
/// anew (never cached nor compacted) and the rays follow it.
#[test]
fn captured_blas_hit() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let snapshot = Figure::snapshot(0, 0);
    let q = quad();
    let list = [q[0], q[1], q[2], q[1], q[3], q[2]];
    let vb = gpu.buffer(&vertex_bytes(&list));
    let draw_index = CapturedIndex::None {
        start_vertex: 0,
        vertex_count: 6,
    };
    let triangles = |bones: &[Bone]| {
        [
            centroid([0, 1, 2].map(|i| list[i].skinned(bones))),
            centroid([3, 4, 5].map(|i| list[i].skinned(bones))),
        ]
    };

    let d = figure.capture(
        &mut enc,
        &snapshot,
        (0x91, raw(&vb)),
        draw_index,
        &figure_constants(&BONES),
    );
    prebuild(&mut enc, &d);
    let (tlas, blits) = trace_frame(&mut enc, &gpu, &[d]);
    let stats = enc.rt_stats();
    assert_eq!((stats.built, stats.captured), (0, 1), "{stats:?}");
    assert_eq!(tlas.instances.len(), 1);
    assert_eq!(tlas.instances[0].transform, super::IDENTITY_ROWS);
    assert_eq!(
        tlas.instances[0].mask,
        super::MASK_OPAQUE | super::MASK_FIGURE
    );
    assert_eq!(count(&blits, BlitCommandType::BuildBlas), 1);
    let first = triangles(&BONES);
    for (i, hit) in probe(&gpu, &tlas, &first).into_iter().enumerate() {
        let (distance, id) = hit.unwrap_or_else(|| panic!("triangle {i} is hit"));
        assert_eq!(id, 0);
        assert!(
            (distance - length(first[i])).abs() < 1e-3,
            "triangle {i}: {distance}"
        );
    }

    // The next frame moves every bone a unit and a half to the right.
    next_frame(&mut enc, &coherent);
    let bones = moved(BONES, [1.5, 0.0, 0.0]);
    let d = figure.capture(
        &mut enc,
        &snapshot,
        (0x91, raw(&vb)),
        draw_index,
        &figure_constants(&bones),
    );
    let (tlas, blits) = trace_frame(&mut enc, &gpu, &[d]);
    let stats = enc.rt_stats();
    assert_eq!(
        (stats.built, stats.reused, stats.captured),
        (0, 0, 1),
        "{stats:?}"
    );
    assert_eq!(count(&blits, BlitCommandType::BuildBlas), 1, "built anew");
    assert_eq!(
        count(&blits, BlitCommandType::CompactBlas),
        0,
        "never compacted"
    );
    let second = triangles(&bones);
    let hits = probe(&gpu, &tlas, &[second[0], second[1], first[0]]);
    for (i, hit) in hits[..2].iter().enumerate() {
        let (distance, _) = hit.unwrap_or_else(|| panic!("moved triangle {i} is hit"));
        assert!(
            (distance - length(second[i])).abs() < 1e-3,
            "triangle {i}: {distance}"
        );
    }
    assert_eq!(hits[2], None, "where the figure was");
}

/// An indexed figure drawn from base vertex 3: its ids count from 3, so its
/// BLAS starts three positions into its captured range, and a ray at each
/// triangle's skinned centroid hits it there; the vertices it skips are
/// never part of it.
#[test]
fn captured_base_vertex_hits() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let q = quad();
    let mut vertices = skipped(3);
    vertices.extend(q);
    let vb = gpu.buffer(&vertex_bytes(&vertices));
    // A leading index the draw skips by its offset.
    let ib = gpu.buffer(&indices(&[7, 0, 1, 2, 1, 3, 2]));
    let d = figure.capture(
        &mut enc,
        &Figure::snapshot(0, 0),
        (0x91, raw(&vb)),
        CapturedIndex::Buffer {
            handle: raw(&ib),
            offset: 2,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 3,
            content: BufferContent::default(),
        },
        &figure_constants(&BONES),
    );
    assert_eq!(super::capture_extent(&d), Ok(7));
    prebuild(&mut enc, &d);
    let (tlas, blits) = trace_frame(&mut enc, &gpu, &[d]);
    let builds = blas_builds(&blits);
    assert_eq!(builds.len(), 1);
    assert_eq!(
        builds[0].geometry.vertex_offset,
        3 * 16,
        "the base vertex folded in"
    );
    assert_eq!(
        builds[0].geometry.index_buffer,
        raw(&ib),
        "the draw's own indices"
    );
    assert_eq!(builds[0].geometry.index_offset, 2);
    let targets = [
        centroid([0, 1, 2].map(|i| q[i].skinned(&BONES))),
        centroid([1, 3, 2].map(|i| q[i].skinned(&BONES))),
    ];
    for (i, hit) in probe(&gpu, &tlas, &targets).into_iter().enumerate() {
        let (distance, id) = hit.unwrap_or_else(|| panic!("triangle {i} is hit"));
        assert_eq!(id, 0);
        assert!(
            (distance - length(targets[i])).abs() < 1e-3,
            "triangle {i}: {distance}"
        );
    }
}

/// Under `rt.shadows` the frame's scene work captures the figure in one
/// vertex-only pass spliced in ahead of the world, whose leading blits
/// then build the figure's BLAS from the very buffer the pass stores into.
/// With the `rt.*` keys absent (every key, or `shadows.sun` alone, which
/// captures the world for its replays) no pass is vertex-only and nothing
/// binds the capture's slots.
#[test]
fn capture_pass_precedes_its_build() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let vb = gpu.buffer(&vertex_bytes(&quad()));
    enc.prepare_scene_textures((64, 64), 1);
    enc.pass_state
        .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
    let d = figure.capture(
        &mut enc,
        &Figure::snapshot(0, 0),
        (0x91, raw(&vb)),
        CapturedIndex::None {
            start_vertex: 1,
            vertex_count: 3,
        },
        &figure_constants(&BONES),
    );
    prebuild(&mut enc, &d);
    enc.scene.note_boundary(&boundary());
    enc.pass_state.end_current_pass("test");
    assemble(&mut enc, false);
    let passes = enc.pass_state.passes();
    let world = passes
        .iter()
        .position(|p| p.color_texture().raw() == 0xB000)
        .expect("the world pass");
    let vertex_only: Vec<usize> = (0..passes.len())
        .filter(|&i| passes[i].is_vertex_only())
        .collect();
    assert_eq!(vertex_only.len(), 1, "one capture pass");
    let capture = vertex_only[0];
    assert!(capture < world, "the capture runs before the world");
    assert!(
        passes[capture..world].iter().all(Pass::is_scene),
        "inside the scene work"
    );
    let stored = passes[capture]
        .commands()
        .iter()
        .find(|c| c.cmd == CommandType::SetVertexBuffer as u32 && c.param_a == VS_CAPTURE_POS_SLOT)
        .expect("the positions' bind")
        .param_b;
    let builds = blas_builds(passes[world].leading_blits());
    assert_eq!(builds.len(), 1);
    assert_eq!(
        builds[0].geometry.vertex_buffer, stored,
        "built from what the pass stores"
    );
    assert_eq!(
        builds[0].geometry.vertex_offset, 16,
        "from the first vertex, id 1"
    );
    let kinds: Vec<u32> = passes[world]
        .leading_blits()
        .iter()
        .map(|b| b.cmd)
        .collect();
    assert_eq!(
        kinds[..2],
        [
            BlitCommandType::BuildBlas as u32,
            BlitCommandType::BuildTlas as u32
        ]
    );
    drop(enc);

    let sun = Mtld3dConfig {
        shadows: ShadowConfig {
            sun: true,
            ..ShadowConfig::default()
        },
        ..Mtld3dConfig::default()
    };
    for (name, config) in [("no key", Mtld3dConfig::default()), ("shadows.sun", sun)] {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let (_device, mut enc) =
            encoder_with_config(&coherent, &retained, config).expect("the device");
        post_reset_frame(&mut enc, (64, 64), 1, true);
        enc.scene.begin_submission(false);
        let figure = Figure::new(&mut enc);
        let vb = gpu.buffer(&vertex_bytes(&quad()));
        enc.prepare_scene_textures((64, 64), 1);
        enc.pass_state
            .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
        let before = enc.scene_draws().len();
        if name == "shadows.sun" {
            figure.capture(
                &mut enc,
                &Figure::snapshot(0, 0),
                (0x91, raw(&vb)),
                CapturedIndex::None {
                    start_vertex: 0,
                    vertex_count: 3,
                },
                &figure_constants(&BONES),
            );
            assert_eq!(enc.scene_draws().len(), before + 1, "kept for the replays");
        }
        enc.scene.note_boundary(&boundary());
        enc.pass_state.end_current_pass("test");
        assemble(&mut enc, false);
        let passes = enc.pass_state.passes();
        assert!(
            passes.iter().all(|p| !p.is_vertex_only()),
            "{name}: no capture pass"
        );
        let capture_slots = [
            VS_CAPTURE_POS_SLOT,
            VS_CAPTURE_UV_SLOT,
            mtld3d_shared::mtl::VS_CAPTURE_COUNT_SLOT,
        ];
        assert!(
            passes
                .iter()
                .flat_map(Pass::commands)
                .all(|c| !capture_slots.contains(&c.param_a)
                    || !matches!(
                        CommandType::from_repr(c.cmd),
                        Some(CommandType::SetVertexBuffer | CommandType::SetVertexBytesAt)
                    )),
            "{name}: nothing binds the capture's slots"
        );
        assert!(
            passes
                .iter()
                .flat_map(|p| p.leading_blits())
                .all(|b| !BlitCommandType::from_repr(b.cmd)
                    .is_some_and(BlitCommandType::is_acceleration_structure)),
            "{name}: no acceleration structure"
        );
        assert!(enc.rt_frame_tlas().is_none(), "{name}");
    }
}

/// Where the ray from the eye along the unit `dir` crosses `tri`: its
/// distance and the barycentrics `(u, v)` of the second and third corners.
fn crossing(dir: [f32; 3], tri: [[f32; 3]; 3]) -> Option<(f32, f32, f32)> {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (e1, e2) = (sub(tri[1], tri[0]), sub(tri[2], tri[0]));
    let p = cross(dir, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-9 {
        return None;
    }
    let s = sub([0.0; 3], tri[0]);
    let u = dot(s, p) / det;
    let q = cross(s, e1);
    let v = dot(dir, q) / det;
    let t = dot(e2, q) / det;
    (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0).then_some((t, u, v))
}

/// What a ray from the eye at `target` should meet among `triangles` (eye
/// positions and coordinates) under the 4x4 alpha checker: the distance of
/// its nearest opaque crossing, `None` through holes alone; `Err` when a
/// crossing falls too near a texel's edge to tell.
fn expected_hit(
    target: [f32; 3],
    triangles: &[([[f32; 3]; 3], [[f32; 2]; 3])],
) -> Result<Option<f32>, ()> {
    let len = length(target);
    let dir = target.map(|c| c / len);
    let mut nearest: Option<f32> = None;
    for (positions, uvs) in triangles {
        let Some((t, u, v)) = crossing(dir, *positions) else {
            continue;
        };
        let w = 1.0 - u - v;
        let uv: [f32; 2] = core::array::from_fn(|c| w * uvs[0][c] + u * uvs[1][c] + v * uvs[2][c]);
        let texel = uv.map(|c| c * 4.0);
        if texel.iter().any(|&c| {
            let f = c - c.floor();
            !(0.1..=0.9).contains(&f) || !(0.0..4.0).contains(&c)
        }) {
            return Err(());
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "in [0, 4)"
        )]
        let (x, y) = (texel[0] as usize, texel[1] as usize);
        if opaque_texel(x, y) {
            nearest = Some(nearest.map_or(t, |n| n.min(t)));
        }
    }
    Ok(nearest)
}

/// An alpha-tested skinned figure, captured under the vertex-only
/// (`NO_RASTER`) capture pipeline with its stage-0 texture, sampler and
/// alpha reference bound as `encode_captured_draw` binds them for any
/// alpha-tested draw: the capture runs, stores each vertex's own
/// coordinate, and its non-opaque BLAS lets rays through the texture's
/// holes and stops them at its opaque texels, through the shared
/// intersection function reading the captured coordinates.
#[test]
fn captured_alpha_holes_let_rays_through() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let q = quad();
    let list = [q[0], q[1], q[2], q[1], q[3], q[2]];
    let vb = gpu.buffer(&vertex_bytes(&list));
    let (texture, sampler) = (checker(&gpu), sampler(&gpu));
    let d = figure.capture_tested(
        &mut enc,
        &Figure::snapshot(0, 0),
        (0x91, raw(&vb)),
        (
            CapturedIndex::None {
                start_vertex: 0,
                vertex_count: 6,
            },
            &figure_constants(&BONES),
        ),
        Some(Stage0 {
            texture: mtld3d_core::ids::TextureId::new_unique(),
            handle: core::ptr::from_ref(&*texture).cast::<u8>() as u64,
            sampler: core::ptr::from_ref(&*sampler).cast::<u8>() as u64,
        }),
    );
    assert!(d.alpha.is_some(), "the figure keeps its alpha test");
    prebuild(&mut enc, &d);
    enc.scene.note_boundary(&boundary());
    let (tlas, blits, capture) = build_with_capture(&mut enc, &[d]);
    let tlas = tlas.expect("a TLAS");
    assert!(
        capture
            .iter()
            .any(|c| c.cmd == CommandType::SetFragmentTexture as u32
                && c.param_b == core::ptr::from_ref(&*texture).cast::<u8>() as u64),
        "the capture binds the alpha test's texture"
    );
    gpu.run_descriptors(&[
        pass_to_descriptor(&Pass::scene_vertex(capture), MetalHandle::NULL),
        blit_pass(&blits),
    ]);
    assert_eq!(tlas.instances[0].mask, MASK_ALPHA | MASK_FIGURE);
    assert_eq!(
        blas_builds(&blits)[0].geometry.opaque,
        0,
        "non-opaque geometry"
    );
    let info = tlas.instances[0].info;
    assert_eq!(info.uv_source, UV_CAPTURED);
    let g = tlas.instances[0].geometry.expect("captured geometry");
    // SAFETY: the scene owns the frame's coordinate buffer.
    let uv_buffer =
        unsafe { MetalHandle::<mtld3d_shared::mtl_handle::MTLBufferKind>::new(g.uv_buf) }
            .into_retained()
            .expect("the coordinates");
    let offset = usize::try_from(g.uv_offset).expect("small") / 8;
    let stored = &read::<2>(&uv_buffer, offset + 6)[offset..];
    let uvs: Vec<[f32; 2]> = list.iter().map(|v| v.uv).collect();
    assert_eq!(stored, uvs.as_slice(), "each vertex's own coordinate");

    let triangles: Vec<([[f32; 3]; 3], [[f32; 2]; 3])> = [[0, 1, 2], [3, 4, 5]]
        .iter()
        .map(|t| (t.map(|i| list[i].skinned(&BONES)), t.map(|i| list[i].uv)))
        .collect();
    let mut targets = Vec::new();
    let mut expected = Vec::new();
    for (positions, _) in &triangles {
        for i in 1..12 {
            for j in 1..12 - i {
                #[expect(clippy::cast_precision_loss, reason = "small values")]
                let (u, v) = (i as f32 / 12.0, j as f32 / 12.0);
                let w = 1.0 - u - v;
                let target: [f32; 3] = core::array::from_fn(|c| {
                    w * positions[0][c] + u * positions[1][c] + v * positions[2][c]
                });
                if let Ok(want) = expected_hit(target, &triangles) {
                    targets.push(target);
                    expected.push(want);
                }
            }
        }
    }
    assert!(
        expected.iter().filter(|e| e.is_some()).count() >= 10,
        "{expected:?}"
    );
    assert!(
        expected.iter().filter(|e| e.is_none()).count() >= 10,
        "{expected:?}"
    );
    for ((traced, want), target) in trace(&gpu, &tlas, &targets)
        .iter()
        .zip(&expected)
        .zip(&targets)
    {
        match want {
            Some(distance) => {
                let (got, id) = traced
                    .hit
                    .unwrap_or_else(|| panic!("{target:?} is hit: {traced:?}"));
                assert_eq!(id, 0);
                assert!(
                    (got - distance).abs() < 1e-3,
                    "{target:?}: {got} against {distance}"
                );
            }
            None => assert!(
                traced.hit.is_none(),
                "{target:?} goes through a hole: {traced:?}"
            ),
        }
    }
}

/// The captures of a frame stay within the budget: a figure whose
/// positions and coordinates would take the frame's buffers past it waits
/// for a later frame (counted as deferred, never instanced) and nothing
/// grows for it, while a small one beside it is captured and traced.
#[test]
fn capture_budget_defers_past_it() {
    /// Vertices of the large figure: 640 000 bytes of positions, 320 000 of coordinates.
    const LARGE: u32 = 40_000;
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let snapshot = Figure::snapshot(0, 0);
    let q = quad();
    let small_vb = gpu.buffer(&vertex_bytes(&[q[0], q[1], q[2], q[1], q[3], q[2]]));
    let large_vb = gpu.buffer(&vertex_bytes(&skipped(LARGE as usize)));
    let constants = figure_constants(&BONES);
    let draws = |enc: &mut FrameEncoder| {
        let small = figure.capture(
            enc,
            &snapshot,
            (0x91, raw(&small_vb)),
            CapturedIndex::None {
                start_vertex: 0,
                vertex_count: 6,
            },
            &constants,
        );
        let large = figure.capture(
            enc,
            &snapshot,
            (0x91, raw(&large_vb)),
            CapturedIndex::None {
                start_vertex: 0,
                vertex_count: LARGE - 1,
            },
            &constants,
        );
        [small, large]
    };

    // The plan under a budget the small figure's buffers (256 bytes each) fit.
    let both = draws(&mut enc);
    enc.scene.note_boundary(&boundary());
    let mut rt = enc.rt.take().expect("the scene");
    rt.cache = super::BlasCache::new(4096);
    let slot = rt.free_slot(0).expect("a slot");
    let device = gpu.device.clone();
    let plan = rt.plan_captures(&mut enc, &device, slot, &both);
    assert!(matches!(plan[0], Some(Planned::Target(_))), "{plan:?}");
    assert_eq!(plan[1], Some(Planned::Deferred));
    let frame = &rt.slots[slot];
    assert_eq!((frame.captured_pos.len, frame.captured_uv.len), (256, 256));
    enc.rt = Some(rt);

    // Through a frame under 1 MiB, past which the large figure's positions alone go.
    next_frame(&mut enc, &coherent);
    let both = draws(&mut enc);
    enc.rt.as_mut().expect("the scene").cache = super::BlasCache::new(1 << 20);
    prebuild(&mut enc, &both[0]);
    let (tlas, _) = trace_frame(&mut enc, &gpu, &both);
    let stats = enc.rt_stats();
    assert_eq!((stats.captured, stats.deferred), (1, 1), "{stats:?}");
    assert_eq!(tlas.instances.len(), 1);
    assert_eq!(tlas.draws, vec![0], "the small figure");
}
