//! Alpha-tested geometry and instance materials on the GPU: rays traced
//! through the shared intersection function (`metal::rt_kernels`) against
//! the frame's TLAS and instance table.

use core::ptr::NonNull;
use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    draw_data::{
        AttrSnapshot, DeclarationHeader, FixedPsSource, PsSource, PsSourcePtr, VsSource,
        VsSourcePtr,
    },
    dxso::{FfPsKey, FfStage, FfStageFlags, FfVsFlags, VariantKey},
    ids::TextureId,
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, VertexAttrDesc,
    mtl::{CullMode, IndexType, PixelFormat, PrimitiveType, VertexFormat, VertexStepFunction},
};
use mtld3d_types::{
    D3DCMP_GREATEREQUAL, D3DTA_DIFFUSE, D3DTA_TEXTURE, D3DTOP_DISABLE, D3DTOP_MODULATE,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResourceOptions,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};

use super::{
    MASK_ALPHA, MASK_OPAQUE, TlasHandle,
    tests::{blit_pass, build_with_capture, length, rt_encoder},
};
use crate::{
    encoder::{
        FrameEncoder,
        replay::tests::Gpu,
        scene_capture::{
            BufferContent, CapturedDraw, CapturedIndex, DrawFacts, Stage0, VertexBind, tests::ff_vs,
        },
    },
    metal::{
        rt::BlasBuild,
        rt_kernels::{RtPipeline, bind_scene, create_counting_rt_pipeline, create_rt_pipeline},
    },
};

/// Bytes per vertex: a float3 position, then a float2 coordinate.
const STRIDE: u32 = 20;
/// The side of the test textures, in texels.
const SIDE: usize = 4;
/// The eye depth of the test quads.
const DEPTH: f32 = 5.0;

/// The test kernel: one ray per thread through the scene, its closest
/// accepted hit `(distance, user id, intersection calls, 1)` and albedo, or
/// `(-1, -1, calls, 0)` on a miss.
const TRACE_MSL: &str = r"
kernel void rt_test_trace(instance_acceleration_structure accel [[buffer(0)]],
                          intersection_function_table<triangle_data, instancing> table [[buffer(1)]],
                          device const RtInstanceInfo *instances [[buffer(2)]],
                          device const float4 *rays [[buffer(3)]],
                          device float4 *out [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    ray r;
    r.origin = rays[2u * i].xyz;
    r.direction = rays[2u * i + 1u].xyz;
    r.min_distance = 0.0;
    r.max_distance = 1.0e6;
    intersector<triangle_data, instancing> isect;
    RtPayload payload;
    payload.instances = instances;
    payload.alpha_calls = 0u;
    intersection_result<triangle_data, instancing> hit = isect.intersect(r, accel, 0xFFu, table, payload);
    if (hit.type == intersection_type::triangle) {
        float3 albedo = rt_hit_albedo(instances[hit.user_instance_id], hit.primitive_id,
                                      hit.triangle_barycentric_coord);
        out[2u * i] = float4(hit.distance, float(hit.user_instance_id), float(payload.alpha_calls), 1.0);
        out[2u * i + 1u] = float4(albedo, 0.0);
    } else {
        out[2u * i] = float4(-1.0, -1.0, float(payload.alpha_calls), 0.0);
        out[2u * i + 1u] = float4(0.0);
    }
}
";

/// One traced ray.
#[derive(Clone, Copy, Debug)]
pub(super) struct Traced {
    /// `(distance, user id)` of the closest accepted hit, `None` on a miss.
    pub(super) hit: Option<(f32, u32)>,
    /// How often the alpha intersection function ran.
    pub(super) calls: u32,
    /// The hit's `rt_hit_albedo`.
    pub(super) albedo: [f32; 3],
}

/// The test kernel's pipeline, through the shared kernel setup, its
/// intersection function counting its calls.
fn trace_pipeline(gpu: &Gpu) -> RtPipeline {
    create_counting_rt_pipeline(&gpu.device, TRACE_MSL, "rt_test_trace")
        .expect("the trace kernel builds")
}

/// Trace rays from the eye at `targets` through `tlas` with the shared
/// intersection function, the scene bound and made resident by `bind_scene`.
pub(super) fn trace(gpu: &Gpu, tlas: &TlasHandle, targets: &[[f32; 3]]) -> Vec<Traced> {
    trace_with(gpu, &trace_pipeline(gpu), tlas, targets)
}

/// [`trace`] through `pipeline`.
fn trace_with(
    gpu: &Gpu,
    pipeline: &RtPipeline,
    tlas: &TlasHandle,
    targets: &[[f32; 3]],
) -> Vec<Traced> {
    let rays: Vec<f32> = targets
        .iter()
        .flat_map(|t| {
            let len = length(*t);
            [0.0, 0.0, 0.0, 0.0, t[0] / len, t[1] / len, t[2] / len, 0.0]
        })
        .collect();
    let rays = gpu.buffer(&floats(&rays));
    let out = gpu
        .device
        .newBufferWithLength_options(targets.len() * 32, MTLResourceOptions::StorageModeShared)
        .expect("out");
    let cmd = gpu.queue.commandBuffer().expect("command buffer");
    let compute = cmd.computeCommandEncoder().expect("compute");
    assert!(bind_scene(&compute, pipeline, &tlas.binding()));
    // SAFETY: the kernel's ray and output buffers.
    unsafe {
        compute.setBuffer_offset_atIndex(Some(&rays), 0, 3);
        compute.setBuffer_offset_atIndex(Some(&out), 0, 4);
    }
    let size = |width| MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    compute.dispatchThreads_threadsPerThreadgroup(size(targets.len()), size(1));
    compute.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the completed dispatch wrote two float4 per ray.
    let out = unsafe {
        core::slice::from_raw_parts(
            out.contents().as_ptr().cast::<[f32; 4]>(),
            targets.len() * 2,
        )
    };
    out.chunks(2)
        .map(|pair| {
            let (h, albedo) = (pair[0], pair[1]);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "small counts"
            )]
            Traced {
                hit: (h[3] > 0.5).then(|| (h[0], h[1] as u32)),
                calls: h[2] as u32,
                albedo: [albedo[0], albedo[1], albedo[2]],
            }
        })
        .collect()
}

/// Little-endian bytes of `values`.
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Texel `(x, y)`'s alpha in [`checker`]: opaque on odd `x + y`.
pub(super) const fn opaque_texel(x: usize, y: usize) -> bool {
    (x + y) % 2 == 1
}

/// Texel `(x, y)`'s colour in [`checker`], in `[0, 255]`.
pub(super) fn texel_rgb(x: usize, y: usize) -> [u8; 3] {
    [
        u8::try_from(40 + 60 * x).expect("small"),
        u8::try_from(30 + 70 * y).expect("small"),
        200,
    ]
}

/// A [`SIDE`]-square `RGBA8Unorm` texture: each texel its [`texel_rgb`],
/// its alpha 255 where [`opaque_texel`], 0 elsewhere.
pub(super) fn checker(gpu: &Gpu) -> Retained<ProtocolObject<dyn MTLTexture>> {
    // SAFETY: a single-level 2D descriptor of a sampleable format.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::RGBA8Unorm,
            SIDE,
            SIDE,
            false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Shared);
    desc.setUsage(MTLTextureUsage::ShaderRead);
    let texture = gpu.device.newTextureWithDescriptor(&desc).expect("checker");
    let texels: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let (x, y) = (i % SIDE, i / SIDE);
            let [r, g, b] = texel_rgb(x, y);
            [r, g, b, if opaque_texel(x, y) { 255 } else { 0 }]
        })
        .collect();
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: SIDE,
            height: SIDE,
            depth: 1,
        },
    };
    // SAFETY: `texels` holds the region at 4 bytes per texel.
    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region,
            0,
            NonNull::from(texels.as_slice()).cast(),
            SIDE * 4,
        );
    }
    texture
}

/// A nearest-filtering sampler an argument buffer can name, as the
/// encoder makes them while the ray-traced scene is on, noted by `enc`'s
/// scene as one.
pub(super) fn sampler(
    gpu: &Gpu,
    enc: &mut FrameEncoder,
) -> Retained<ProtocolObject<dyn MTLSamplerState>> {
    let desc = MTLSamplerDescriptor::new();
    desc.setMinFilter(MTLSamplerMinMagFilter::Nearest);
    desc.setMagFilter(MTLSamplerMinMagFilter::Nearest);
    desc.setSupportArgumentBuffers(true);
    let sampler = gpu
        .device
        .newSamplerStateWithDescriptor(&desc)
        .expect("sampler");
    enc.rt
        .as_mut()
        .expect("the ray-traced scene")
        .note_argument_sampler(raw(&*sampler));
    sampler
}

/// The raw handle of an object the test keeps alive.
fn raw<T: ?Sized>(object: &T) -> u64 {
    core::ptr::from_ref(object).cast::<u8>() as u64
}

/// A textured fixed-function draw's sources: the vertex stage passes
/// `TEXCOORD0` (attribute 4, at byte 12) through to stage 0.
struct Textured {
    vs: VsSource,
    ps: PsSource,
    header: Box<DeclarationHeader>,
    attrs: Box<[VertexAttrDesc; 2]>,
    snapshot: PipelineSnapshot,
}

impl Textured {
    /// `ps` over the stock textured vertex stage, with `tt_flags` as stage 0's texture transform.
    fn new(ps: PsSource, tt_flags: u8) -> Self {
        let mut vs = ff_vs(FfVsFlags::empty(), 0);
        if let VsSource::FixedFunction(source) = &mut vs {
            source.key.tt_flags[0] = tt_flags;
            source.key.light_active_mask = 0;
            source.key.light_directional_mask = 0;
        }
        let mut stream_layouts = [StreamLayout::UNUSED; 16];
        stream_layouts[0] = StreamLayout {
            stride: STRIDE,
            step: VertexStepFunction::PerVertex,
            step_rate: 1,
        };
        Self {
            vs,
            ps,
            header: Box::new(DeclarationHeader {
                vdecl_hash: 0x7e57,
                extents: [STRIDE; 16],
                count: 2,
                used_streams: 1,
                reserved: 0,
            }),
            attrs: Box::new([
                VertexAttrDesc {
                    attr_index: 0,
                    buffer_index: 0,
                    offset: 0,
                    format: VertexFormat::Float3,
                },
                VertexAttrDesc {
                    attr_index: 4,
                    buffer_index: 0,
                    offset: 12,
                    format: VertexFormat::Float2,
                },
            ]),
            snapshot: PipelineSnapshot {
                vs_fn: mtld3d_shared::MetalHandle::NULL,
                ps_fn: mtld3d_shared::MetalHandle::NULL,
                vdecl_hash: 0x7e57,
                stream_layouts,
                color_format: PixelFormat::Bgra8Unorm,
                attach: PipelineAttachFlags::HAS_DEPTH | PipelineAttachFlags::HAS_COLOR_OUTPUT,
                rs: PipelineRsBits::default(),
                extra: ExtraColorAttachments::NONE,
                ps_color_out_mask: 1,
                sample_count: 1,
            },
        }
    }

    /// The same sources, the vertex stage lighting the draw (`D3DRS_LIGHTING`).
    fn lit(mut self) -> Self {
        if let VsSource::FixedFunction(source) = &mut self.vs {
            source.key.flags |= FfVsFlags::LIGHTING_ENABLED;
        }
        self
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
}

/// A fixed-function pixel stage: stage 0 `op(arg1 = TEXTURE, arg2 = DIFFUSE)`, the rest disabled.
fn ff_ps(op: u32) -> PsSource {
    let disabled = FfStage {
        color_op: u8::try_from(D3DTOP_DISABLE).expect("a D3DTOP value"),
        ..FfStage::default()
    };
    let mut stages = [disabled; 8];
    stages[0] = FfStage {
        color_op: u8::try_from(op).expect("a D3DTOP value"),
        color_arg1: u8::try_from(D3DTA_TEXTURE).expect("a D3DTA value"),
        color_arg2: u8::try_from(D3DTA_DIFFUSE).expect("a D3DTA value"),
        alpha_op: u8::try_from(op).expect("a D3DTOP value"),
        alpha_arg1: u8::try_from(D3DTA_TEXTURE).expect("a D3DTA value"),
        alpha_arg2: u8::try_from(D3DTA_DIFFUSE).expect("a D3DTA value"),
        flags: FfStageFlags::HAS_TEXTURE,
    };
    PsSource::FixedFunction(FixedPsSource {
        key: FfPsKey {
            stages,
            specular_add: false,
            tt_projected_mask: 0,
        },
        sampled_stage_mask: 1,
        constant_rows: 0,
        bump_env: false,
        reserved: [0; 2],
    })
}

/// One textured world draw.
struct Draw {
    vertex_buffer: u64,
    index: CapturedIndex,
    primitive: PrimitiveType,
    /// Moves the quad by `(x, 0, DEPTH)` in eye space.
    x: f32,
    mat_diffuse: [f32; 3],
    /// `D3DRS_ALPHAFUNC`, 0 for none.
    alpha_func: u8,
    texture: u64,
    sampler: u64,
}

/// The fixed-function constants of `d`: rows 0..3 the world-view moving
/// by `(d.x, 0, DEPTH)`, row 10 the material diffuse.
fn constants(d: &Draw) -> Vec<u8> {
    let mut rows = vec![[0.0f32; 4]; 16];
    rows[0] = [1.0, 0.0, 0.0, d.x];
    rows[1] = [0.0, 1.0, 0.0, 0.0];
    rows[2] = [0.0, 0.0, 1.0, DEPTH];
    rows[3] = [0.0, 0.0, 0.0, 1.0];
    rows[10] = [d.mat_diffuse[0], d.mat_diffuse[1], d.mat_diffuse[2], 1.0];
    floats(rows.as_flattened())
}

/// Capture `d` of `sources` the way `emit_draw_view` offers it; the record kept.
fn capture(enc: &mut FrameEncoder, sources: &Textured, d: &Draw) -> CapturedDraw {
    let constants = constants(d);
    let vs_draw = vec![0u8; 32];
    let pos_fixup = floats(&[1.0 / 64.0, -1.0 / 64.0, 0.0, 1.0, 0.0]);
    let alpha_ref = 0.5f32.to_le_bytes();
    enc.begin_scene_binds();
    enc.stage_scene_bind(VertexBind::Buffer {
        handle: d.vertex_buffer,
        offset: 0,
        slot: 0,
        content: BufferContent {
            id: d.vertex_buffer,
            version: 1,
        },
    });
    // SAFETY: the sources outlive the test's captures and do not move.
    let vs = unsafe { VsSourcePtr::new(NonNull::from(&sources.vs)) };
    // SAFETY: as above.
    let ps = unsafe { PsSourcePtr::new(NonNull::from(&sources.ps)) };
    let before = enc.scene_draws().len();
    enc.capture_scene_draw(&DrawFacts {
        snapshot: &sources.snapshot,
        attrs: sources.token(),
        fetch_attrs: None,
        vs,
        ps,
        variant: VariantKey {
            alpha_func: d.alpha_func,
            ..VariantKey::default()
        },
        pipeline: 0x91,
        depth_stencil: 0xD5,
        depth_write: true,
        blend: false,
        index: d.index,
        primitive: d.primitive,
        instances: 1,
        vs_constants: &constants,
        vs_draw: &vs_draw,
        vs_int: &[],
        vs_bool: &[],
        pos_fixup: &pos_fixup,
        alpha_ref: &alpha_ref,
        cull: CullMode::None,
        slope_scale: 0.0,
        stage0: Some(Stage0 {
            texture: TextureId::new_unique(),
            handle: d.texture,
            sampler: d.sampler,
        }),
    });
    assert_eq!(enc.scene_draws().len(), before + 1, "the draw is captured");
    enc.scene_draws()[before].clone()
}

/// The quad `x, y` in `[-1, 1]` at `z = 0`, `u = (x + 1) / 2`, `v = (1 - y) / 2`,
/// as its four corners: bottom left, bottom right, top left, top right.
fn corners() -> [[f32; 5]; 4] {
    [
        [-1.0, -1.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0, 1.0],
        [-1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
    ]
}

/// The eye position of texel `(x, y)`'s centre on a quad moved by `dx`.
fn texel_centre(x: usize, y: usize, dx: f32) -> [f32; 3] {
    #[expect(clippy::cast_precision_loss, reason = "small values")]
    let (u, v) = (
        (x as f32 + 0.5) / SIDE as f32,
        (y as f32 + 0.5) / SIDE as f32,
    );
    [2.0 * u - 1.0 + dx, 1.0 - 2.0 * v, DEPTH]
}

/// Every texel `(x, y)` of the test textures.
fn texels() -> impl Iterator<Item = (usize, usize)> {
    (0..SIDE * SIDE).map(|i| (i % SIDE, i / SIDE))
}

/// Build the frame's TLAS over `draws` and run its builds: the TLAS and the blits.
fn traced_frame(
    enc: &mut FrameEncoder,
    gpu: &Gpu,
    draws: &[CapturedDraw],
) -> (TlasHandle, Vec<BlitCommand>) {
    let (tlas, blits, capture) = build_with_capture(enc, draws);
    assert!(capture.is_empty(), "static draws capture nothing");
    gpu.run_descriptors(&[blit_pass(&blits)]);
    (tlas.expect("a TLAS"), blits)
}

/// The BLAS builds of `blits`.
fn blas_builds(blits: &[BlitCommand]) -> Vec<BlasBuild> {
    blits
        .iter()
        .filter(|b| b.cmd == BlitCommandType::BuildBlas as u32)
        // SAFETY: each build names its description in the live scratch arena.
        .map(|b| unsafe { core::ptr::read_unaligned(b.byte_size as *const BlasBuild) })
        .collect()
}

/// The bytes of an indexed quad: its four [`corners`].
fn quad_vertices() -> Vec<u8> {
    floats(corners().as_flattened())
}

/// 16-bit indices of the two triangles of [`corners`], after `lead` leading indices the draw skips.
fn quad_indices(lead: usize) -> Vec<u8> {
    let mut indices = vec![9u16; lead];
    indices.extend([0, 1, 2, 1, 3, 2]);
    indices.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// An alpha-tested quad traced at each texel's centre: a ray at an opaque
/// texel hits it at its distance, a ray at a transparent one goes through
/// the hole and misses. The instance and its geometry are non-opaque, its
/// table entry samples the texture at the draw's own `TEXCOORD0` through
/// its own 16-bit indices, and each hit and hole ran the intersection
/// function.
#[test]
fn alpha_holes_let_rays_through() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let (texture, sampler) = (checker(&gpu), sampler(&gpu, &mut enc));
    let sources = Textured::new(ff_ps(D3DTOP_MODULATE), 0);
    let vb = gpu.buffer(&quad_vertices());
    let ib = gpu.buffer(&quad_indices(1));
    let d = capture(
        &mut enc,
        &sources,
        &Draw {
            vertex_buffer: raw(&*vb),
            index: CapturedIndex::Buffer {
                handle: raw(&*ib),
                offset: 2,
                count: 6,
                index_type: IndexType::UInt16,
                base_vertex: 0,
                content: BufferContent { id: 77, version: 1 },
            },
            primitive: PrimitiveType::Triangle,
            x: 0.0,
            mat_diffuse: [1.0; 3],
            alpha_func: u8::try_from(D3DCMP_GREATEREQUAL).expect("a D3DCMP value"),
            texture: raw(&*texture),
            sampler: raw(&*sampler),
        },
    );
    assert!(d.alpha.is_some());
    let (tlas, blits) = traced_frame(&mut enc, &gpu, &[d]);
    assert_eq!(tlas.instances[0].mask, MASK_ALPHA);
    assert_eq!(
        blas_builds(&blits)[0].geometry.opaque,
        0,
        "non-opaque geometry"
    );
    assert_eq!(tlas.textures, vec![raw(&*texture)]);
    let info = tlas.instances[0].info;
    assert_eq!(info.alpha_func, D3DCMP_GREATEREQUAL);
    assert_eq!(info.alpha_ref, 0.5);
    assert_eq!((info.uv_offset, info.uv_stride), (12, STRIDE));
    let targets: Vec<[f32; 3]> = texels().map(|(x, y)| texel_centre(x, y, 0.0)).collect();
    for ((x, y), (traced, target)) in
        texels().zip(trace(&gpu, &tlas, &targets).into_iter().zip(&targets))
    {
        assert!(
            traced.calls >= 1,
            "texel ({x}, {y}) ran the alpha test: {traced:?}"
        );
        if opaque_texel(x, y) {
            let (distance, id) = traced
                .hit
                .unwrap_or_else(|| panic!("opaque texel ({x}, {y}) is hit"));
            assert_eq!(id, 0);
            assert!(
                (distance - length(*target)).abs() < 1e-3,
                "texel ({x}, {y}): {distance}"
            );
        } else {
            assert!(
                traced.hit.is_none(),
                "transparent texel ({x}, {y}) lets the ray through: {traced:?}"
            );
        }
    }
}

/// An opaque quad never runs the intersection function: every ray at it
/// hits with no call, while the alpha-tested quad beside it runs it on each
/// ray. Its geometry is opaque in its BLAS as well as in its instance.
#[test]
fn opaque_geometry_skips_intersection_function() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let (texture, sampler) = (checker(&gpu), sampler(&gpu, &mut enc));
    let sources = Textured::new(ff_ps(D3DTOP_MODULATE), 0);
    let opaque_vb = gpu.buffer(&quad_vertices());
    let alpha_vb = gpu.buffer(&quad_vertices());
    let ib = gpu.buffer(&quad_indices(0));
    let draw = |vb: u64, x: f32, alpha_func: u32| Draw {
        vertex_buffer: vb,
        index: CapturedIndex::Buffer {
            handle: raw(&*ib),
            offset: 0,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 0,
            content: BufferContent { id: 78, version: 1 },
        },
        primitive: PrimitiveType::Triangle,
        x,
        mat_diffuse: [1.0; 3],
        alpha_func: u8::try_from(alpha_func).expect("a D3DCMP value"),
        texture: raw(&*texture),
        sampler: raw(&*sampler),
    };
    let opaque = capture(&mut enc, &sources, &draw(raw(&*opaque_vb), -1.5, 0));
    let tested = capture(
        &mut enc,
        &sources,
        &draw(raw(&*alpha_vb), 1.5, D3DCMP_GREATEREQUAL),
    );
    let (tlas, blits) = traced_frame(&mut enc, &gpu, &[opaque, tested]);
    assert_eq!(tlas.instances[0].mask, MASK_OPAQUE);
    assert_eq!(tlas.instances[1].mask, MASK_ALPHA);
    let opacity: Vec<u32> = blas_builds(&blits)
        .iter()
        .map(|b| b.geometry.opaque)
        .collect();
    assert_eq!(opacity, vec![1, 0]);
    let at_opaque: Vec<[f32; 3]> = texels().map(|(x, y)| texel_centre(x, y, -1.5)).collect();
    for (traced, (x, y)) in trace(&gpu, &tlas, &at_opaque).into_iter().zip(texels()) {
        assert_eq!(
            traced.calls, 0,
            "texel ({x}, {y}) of the opaque quad: {traced:?}"
        );
        assert_eq!(
            traced.hit.map(|h| h.1),
            Some(0),
            "texel ({x}, {y}) hits the opaque quad"
        );
    }
    let at_tested: Vec<[f32; 3]> = texels().map(|(x, y)| texel_centre(x, y, 1.5)).collect();
    let calls: u32 = trace(&gpu, &tlas, &at_tested).iter().map(|t| t.calls).sum();
    assert!(
        calls >= 16,
        "the alpha-tested quad runs the function: {calls}"
    );
}

/// Two strips of material `MAT`, lit by their vertex stage when `lit`: the
/// first's stage 0 modulates the checker, the second's colours from no
/// texture. Each texel of each traced; `None` without ray tracing.
fn shaded_strips(lit: bool) -> Option<(TlasHandle, Vec<Traced>)> {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let (device, mut enc) = rt_encoder(&coherent, &retained)?;
    let gpu = Gpu::new(device);
    let (texture, sampler) = (checker(&gpu), sampler(&gpu, &mut enc));
    let sources = |ps| {
        let sources = Textured::new(ps, 0);
        if lit { sources.lit() } else { sources }
    };
    let modulated = sources(ff_ps(D3DTOP_MODULATE));
    let untextured = sources(ff_ps(D3DTOP_DISABLE));
    // A strip of a skipped vertex then bottom left, bottom right, top left, top right.
    let [a, b, c, e] = corners();
    let strip = gpu.buffer(&floats(&[[9.0; 5], a, b, c, e].concat()));
    let draw = |x: f32| Draw {
        vertex_buffer: raw(&*strip),
        index: CapturedIndex::None {
            start_vertex: 1,
            vertex_count: 4,
        },
        primitive: PrimitiveType::TriangleStrip,
        x,
        mat_diffuse: MAT,
        alpha_func: 0,
        texture: raw(&*texture),
        sampler: raw(&*sampler),
    };
    let textured = capture(&mut enc, &modulated, &draw(-1.5));
    assert!(
        textured.albedo.is_some(),
        "a modulated stage 0 keeps its texture"
    );
    assert_eq!(textured.lit, lit);
    let plain = capture(&mut enc, &untextured, &draw(1.5));
    assert!(plain.albedo.is_none(), "a disabled stage 0 keeps none");
    let (tlas, _) = traced_frame(&mut enc, &gpu, &[textured, plain]);
    let targets: Vec<[f32; 3]> = texels()
        .map(|(x, y)| texel_centre(x, y, -1.5))
        .chain(texels().map(|(x, y)| texel_centre(x, y, 1.5)))
        .collect();
    let traced = trace(&gpu, &tlas, &targets);
    Some((tlas, traced))
}

/// The material of [`shaded_strips`].
const MAT: [f32; 3] = [0.5, 1.0, 0.25];

/// Each hit of [`shaded_strips`] against its texel's colour times `mat`, then the material alone.
fn assert_albedo(traced: &[Traced], mat: [f32; 3]) {
    for ((x, y), t) in texels().zip(&traced[..SIDE * SIDE]) {
        assert_eq!(t.hit.map(|h| h.1), Some(0), "texel ({x}, {y})");
        let want = texel_rgb(x, y).map(|c| f32::from(c) / 255.0);
        for k in 0..3 {
            assert!(
                (t.albedo[k] - want[k] * mat[k]).abs() < 1e-3,
                "texel ({x}, {y}) channel {k}: {:?} against {want:?} * {mat:?}",
                t.albedo
            );
        }
    }
    for t in &traced[SIDE * SIDE..] {
        assert_eq!(t.hit.map(|h| h.1), Some(1));
        assert_eq!(t.albedo, mat, "the material alone");
    }
}

/// A hit's albedo is its texel's colour times the material diffuse (row 10
/// of the fixed-function constants) for a lit opaque draw whose stage 0
/// modulates its texture, here a non-indexed strip; a draw whose stage 0
/// does not colour from its texture shades with its material alone.
#[test]
fn hit_albedo_samples_texture() {
    let Some((tlas, traced)) = shaded_strips(true) else {
        return;
    };
    assert_eq!(tlas.instances[0].info.mat_diffuse, MAT);
    assert_albedo(&traced, MAT);
}

/// An unlit fixed-function draw never reads its material row (the vertex
/// stage outputs its vertex colour or white), so its hits shade as a
/// material diffuse of 1: the texel's colour alone, white without a texture.
#[test]
fn unlit_draw_albedo_is_white() {
    let Some((tlas, traced)) = shaded_strips(false) else {
        return;
    };
    for instance in &tlas.instances {
        assert_eq!(instance.info.mat_diffuse, [1.0; 3]);
    }
    assert_albedo(&traced, [1.0; 3]);
}

/// Past the device's argument-buffer sampler limit (injected low), a
/// sampler is made without argument-buffer support, said once, and an
/// instance sampling with it is untextured: its alpha test cannot run, so
/// its holes stop rays. A sampler made within the limit keeps its texture.
#[test]
fn samplers_past_the_argument_buffer_limit_trace_untextured() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    enc.rt
        .as_mut()
        .expect("the ray-traced scene")
        .set_argument_sampler_limit(1);
    let within = enc.get_or_create_blit_sampler(mtld3d_types::D3DTEXF_POINT);
    let past = enc.get_or_create_blit_sampler(mtld3d_types::D3DTEXF_LINEAR);
    assert_ne!(within, past);
    let rt = enc.rt.as_ref().expect("the ray-traced scene");
    assert!(rt.argument_sampler(within), "within the limit");
    assert!(!rt.argument_sampler(past), "past the limit");
    let texture = checker(&gpu);
    let sources = Textured::new(ff_ps(D3DTOP_MODULATE), 0);
    let vb = gpu.buffer(&quad_vertices());
    let ib = gpu.buffer(&quad_indices(0));
    let draw = |x: f32, sampler: u64| Draw {
        vertex_buffer: raw(&*vb),
        index: CapturedIndex::Buffer {
            handle: raw(&*ib),
            offset: 0,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 0,
            content: BufferContent { id: 81, version: 1 },
        },
        primitive: PrimitiveType::Triangle,
        x,
        mat_diffuse: [1.0; 3],
        alpha_func: u8::try_from(D3DCMP_GREATEREQUAL).expect("a D3DCMP value"),
        texture: raw(&*texture),
        sampler,
    };
    let kept = capture(&mut enc, &sources, &draw(-1.5, within));
    let dropped = capture(&mut enc, &sources, &draw(1.5, past));
    let (tlas, _) = traced_frame(&mut enc, &gpu, &[kept, dropped]);
    assert_ne!(
        tlas.instances[0].info.tex_id, 0,
        "textured within the limit"
    );
    assert_eq!(tlas.instances[1].info.tex_id, 0, "untextured past it");
    assert_eq!(tlas.textures, vec![raw(&*texture)]);
    let targets: Vec<[f32; 3]> = texels()
        .map(|(x, y)| texel_centre(x, y, -1.5))
        .chain(texels().map(|(x, y)| texel_centre(x, y, 1.5)))
        .collect();
    let traced = trace(&gpu, &tlas, &targets);
    for ((x, y), t) in texels().zip(&traced[..SIDE * SIDE]) {
        assert_eq!(
            t.hit.is_some(),
            opaque_texel(x, y),
            "texel ({x}, {y}) within the limit"
        );
    }
    for ((x, y), t) in texels().zip(&traced[SIDE * SIDE..]) {
        assert_eq!(
            t.hit.map(|h| h.1),
            Some(1),
            "texel ({x}, {y}) past the limit"
        );
    }
    enc.shutdown_cleanup();
}

/// The production pipeline compiles the shared intersection function
/// without its call counter: an alpha-tested quad still lets rays through
/// its holes and stops them on its opaque texels, but the counter stays at
/// zero, where the counting pipeline the tests use counts every call.
#[test]
fn production_pipeline_counts_no_alpha_calls() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let (texture, sampler) = (checker(&gpu), sampler(&gpu, &mut enc));
    let sources = Textured::new(ff_ps(D3DTOP_MODULATE), 0);
    let vb = gpu.buffer(&quad_vertices());
    let ib = gpu.buffer(&quad_indices(0));
    let d = capture(
        &mut enc,
        &sources,
        &Draw {
            vertex_buffer: raw(&*vb),
            index: CapturedIndex::Buffer {
                handle: raw(&*ib),
                offset: 0,
                count: 6,
                index_type: IndexType::UInt16,
                base_vertex: 0,
                content: BufferContent { id: 83, version: 1 },
            },
            primitive: PrimitiveType::Triangle,
            x: 0.0,
            mat_diffuse: [1.0; 3],
            alpha_func: u8::try_from(D3DCMP_GREATEREQUAL).expect("a D3DCMP value"),
            texture: raw(&*texture),
            sampler: raw(&*sampler),
        },
    );
    let (tlas, _) = traced_frame(&mut enc, &gpu, &[d]);
    let targets: Vec<[f32; 3]> = texels().map(|(x, y)| texel_centre(x, y, 0.0)).collect();
    let production = create_rt_pipeline(&gpu.device, TRACE_MSL, "rt_test_trace")
        .expect("the production pipeline builds");
    let counted = trace(&gpu, &tlas, &targets);
    let uncounted = trace_with(&gpu, &production, &tlas, &targets);
    for (((x, y), counted), uncounted) in texels().zip(&counted).zip(&uncounted) {
        assert_eq!(
            uncounted.hit.is_some(),
            opaque_texel(x, y),
            "texel ({x}, {y})"
        );
        assert_eq!(uncounted.hit.is_some(), counted.hit.is_some());
        assert_eq!(uncounted.calls, 0, "texel ({x}, {y}): no counter");
        assert!(counted.calls >= 1, "texel ({x}, {y}): counted");
    }
}

/// An alpha-tested draw whose stage-0 coordinate goes through a texture
/// transform is traced opaque, counted: its holes do not let rays through
/// and the intersection function never runs for it.
#[test]
fn transformed_coordinate_traces_opaque() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let (texture, sampler) = (checker(&gpu), sampler(&gpu, &mut enc));
    // D3DTTFF_COUNT2 on stage 0.
    let sources = Textured::new(ff_ps(D3DTOP_MODULATE), 2);
    let vb = gpu.buffer(&quad_vertices());
    let ib = gpu.buffer(&quad_indices(0));
    let d = capture(
        &mut enc,
        &sources,
        &Draw {
            vertex_buffer: raw(&*vb),
            index: CapturedIndex::Buffer {
                handle: raw(&*ib),
                offset: 0,
                count: 6,
                index_type: IndexType::UInt16,
                base_vertex: 0,
                content: BufferContent { id: 79, version: 1 },
            },
            primitive: PrimitiveType::Triangle,
            x: 0.0,
            mat_diffuse: [1.0; 3],
            alpha_func: u8::try_from(D3DCMP_GREATEREQUAL).expect("a D3DCMP value"),
            texture: raw(&*texture),
            sampler: raw(&*sampler),
        },
    );
    let (tlas, blits) = traced_frame(&mut enc, &gpu, &[d]);
    assert_eq!(enc.rt_stats().alpha_opaque, 1);
    assert_eq!(tlas.instances[0].mask, MASK_OPAQUE);
    assert_eq!(tlas.instances[0].info.tex_id, 0, "untextured");
    assert_eq!(blas_builds(&blits)[0].geometry.opaque, 1);
    let targets: Vec<[f32; 3]> = texels().map(|(x, y)| texel_centre(x, y, 0.0)).collect();
    for (traced, (x, y)) in trace(&gpu, &tlas, &targets).into_iter().zip(texels()) {
        assert_eq!(
            (traced.calls, traced.hit.map(|h| h.1)),
            (0, Some(0)),
            "texel ({x}, {y})"
        );
    }
}
