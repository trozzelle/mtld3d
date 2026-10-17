//! Acceleration structures for the captured world: the BLAS keys, the cache, and the TLAS on the GPU.

use core::ptr::NonNull;
use std::sync::{Arc, atomic::AtomicU64};

use mtld3d_core::{
    config::Mtld3dConfig,
    draw_data::{PsSourcePtr, VsSourcePtr},
    dxso::{FfVsFlags, VariantKey},
    guest_completions::CompletionPool,
    guest_pages::GuestOwnedPageLease,
    ids::BufferId,
    page_box::PageBox,
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
    rt::RtConfig,
    shadows::ShadowConfig,
};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, Command, ExtraColorDesc, MetalHandle, PassDescriptor,
    mtl::{
        CullMode, DestroyKind, IndexType, LoadAction, PixelFormat, PrimitiveType, StoreAction,
        VertexStepFunction,
    },
    mtl_handle::{MTLAccelerationStructureKind, MTLBufferKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLLibrary, MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use super::{BlasCache, BlasKey, Skip, TlasHandle, blas_source, build_frame_tlas};
use crate::{
    encoder::{
        BufferGpuState, FrameEncoder,
        replay::tests::Gpu,
        scene::assemble,
        scene_capture::{
            BufferContent, CapturedDraw, CapturedIndex, DrawFacts, VertexBind,
            tests::{Declaration, ff_ps, ff_vs},
        },
        tests::{encoder_with_config, post_reset_frame},
    },
    metal::{
        handle::IntoRetained,
        rt::{BlasBuild, effective_config, rt_supported},
    },
};

/// Bytes per vertex: the declaration's float3 position.
const STRIDE: u32 = 12;

/// A config with `rt.shadows` on.
fn rt_on() -> Mtld3dConfig {
    Mtld3dConfig {
        rt: RtConfig {
            shadows: true,
            ..RtConfig::default()
        },
        ..Mtld3dConfig::default()
    }
}

/// An encoder with `rt.shadows` on a fresh back buffer at submission 1, or `None` without ray tracing.
fn rt_encoder(
    coherent: &AtomicU64,
    retained: &AtomicU64,
) -> Option<(Retained<ProtocolObject<dyn MTLDevice>>, FrameEncoder)> {
    let (device, mut enc) = encoder_with_config(coherent, retained, rt_on())?;
    if !rt_supported(&device) {
        eprintln!("the device does not support ray tracing, skipping");
        return None;
    }
    enc.current_submit_seq = 1;
    post_reset_frame(&mut enc, (64, 64), 1, true);
    enc.scene.begin_submission(false);
    Some((device, enc))
}

/// The next submission: its seq, a fresh capture, and the GPU done with every earlier one.
fn next_frame(enc: &mut FrameEncoder, coherent: &AtomicU64) {
    coherent.store(
        enc.current_submit_seq,
        core::sync::atomic::Ordering::Release,
    );
    enc.current_submit_seq += 1;
    post_reset_frame(enc, (64, 64), 1, true);
    enc.scene.begin_submission(false);
}

/// The world-view `rows` (D3D row-vector, `p_eye = p * M`) as the
/// fixed-function constants carry it: rows 0..4 are its transpose.
fn ff_constants(m: &[[f32; 4]; 4]) -> Vec<u8> {
    let mut rows = vec![[0.0f32; 4]; 16];
    for (r, row) in rows.iter_mut().take(4).enumerate() {
        *row = [m[0][r], m[1][r], m[2][r], m[3][r]];
    }
    rows.as_flattened()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// `p * m` for a row-vector `p = (x, y, z, 1)`.
fn apply(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|c| p[0] * m[0][c] + p[1] * m[1][c] + p[2] * m[2][c] + m[3][c])
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Scale by `s`, rotate about z by 90° (x → y), then translate by `t`.
fn scale_turn_move(s: [f32; 3], t: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [0.0, s[0], 0.0, 0.0],
        [-s[1], 0.0, 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [t[0], t[1], t[2], 1.0],
    ]
}

/// Scale by `s`, then translate by `t`.
fn scale_move(s: [f32; 3], t: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [s[0], 0.0, 0.0, 0.0],
        [0.0, s[1], 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [t[0], t[1], t[2], 1.0],
    ]
}

/// Little-endian bytes of float3 positions.
fn positions(points: &[[f32; 3]]) -> Vec<u8> {
    points
        .as_flattened()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// The object-space quad `x` in `[0, 1]`, `y` in `[0, 2]` at `z = 0`, as a six-vertex list.
fn quad_list() -> Vec<u8> {
    let (a, b, c, d) = (
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
    );
    positions(&[a, b, c, b, d, c])
}

/// The object-space centroid of [`quad_list`].
const QUAD_CENTRE: [f32; 3] = [0.5, 1.0, 0.0];

/// The fixed-function sources of the test's world draws.
struct Sources {
    vs: mtld3d_core::draw_data::VsSource,
    ps: mtld3d_core::draw_data::PsSource,
    decl: Declaration,
    snapshot: PipelineSnapshot,
}

impl Sources {
    fn new() -> Self {
        let mut stream_layouts = [StreamLayout::UNUSED; 16];
        stream_layouts[0] = StreamLayout {
            stride: STRIDE,
            step: VertexStepFunction::PerVertex,
            step_rate: 1,
        };
        Self {
            vs: ff_vs(FfVsFlags::empty(), 0),
            ps: ff_ps(),
            decl: Declaration::new(),
            snapshot: PipelineSnapshot {
                vs_fn: MetalHandle::NULL,
                ps_fn: MetalHandle::NULL,
                vdecl_hash: 0x5eed,
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
}

/// The variable parts of one fixed-function world draw.
#[derive(Clone, Copy)]
struct Draw {
    vertex_buffer: u64,
    content: BufferContent,
    offset: u32,
    index: CapturedIndex,
    primitive: PrimitiveType,
    world_view: [[f32; 4]; 4],
}

impl Draw {
    /// The six-vertex list of `vertex_buffer` (content `content`) under `world_view`.
    const fn list(vertex_buffer: u64, content: BufferContent, world_view: [[f32; 4]; 4]) -> Self {
        Self {
            vertex_buffer,
            content,
            offset: 0,
            index: CapturedIndex::None {
                start_vertex: 0,
                vertex_count: 6,
            },
            primitive: PrimitiveType::Triangle,
            world_view,
        }
    }
}

/// Capture `draw` the way `emit_draw_view` offers it; the record kept.
fn capture(enc: &mut FrameEncoder, sources: &Sources, draw: &Draw) -> CapturedDraw {
    let constants = ff_constants(&draw.world_view);
    let vs_draw = vec![0u8; 32];
    let pos_fixup: Vec<u8> = [1.0f32 / 64.0, -1.0 / 64.0, 0.0, 1.0, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    enc.begin_scene_binds();
    enc.stage_scene_bind(VertexBind::Buffer {
        handle: draw.vertex_buffer,
        offset: draw.offset,
        slot: 0,
        content: draw.content,
    });
    // SAFETY: the sources outlive the test's captures and do not move.
    let vs = unsafe { VsSourcePtr::new(NonNull::from(&sources.vs)) };
    // SAFETY: as above.
    let ps = unsafe { PsSourcePtr::new(NonNull::from(&sources.ps)) };
    let before = enc.scene_draws().len();
    enc.capture_scene_draw(&DrawFacts {
        snapshot: &sources.snapshot,
        attrs: sources.decl.token(),
        fetch_attrs: None,
        vs,
        ps,
        variant: VariantKey::default(),
        pipeline: 0x91,
        depth_stencil: 0xD5,
        depth_write: true,
        blend: false,
        index: draw.index,
        primitive: draw.primitive,
        instances: 1,
        vs_constants: &constants,
        vs_draw: &vs_draw,
        vs_int: &[],
        vs_bool: &[],
        pos_fixup: &pos_fixup,
        alpha_ref: &[],
        cull: CullMode::None,
        slope_scale: 0.0,
        stage0: None,
    });
    assert_eq!(enc.scene_draws().len(), before + 1, "the draw is captured");
    enc.scene_draws()[before].clone()
}

/// Content `version` of the test buffer `id`.
const fn content(id: u64, version: u64) -> BufferContent {
    BufferContent { id, version }
}

/// The frame's acceleration-structure commands for `draws`: the TLAS, if any, and the blits.
fn build(enc: &mut FrameEncoder, draws: &[CapturedDraw]) -> (Option<TlasHandle>, Vec<BlitCommand>) {
    let mut blits = Vec::new();
    let tlas = build_frame_tlas(enc, draws, &mut blits);
    (tlas, blits)
}

/// How many of `blits` are of `kind`.
fn count(blits: &[BlitCommand], kind: BlitCommandType) -> usize {
    blits.iter().filter(|b| b.cmd == kind as u32).count()
}

/// Run `blits` as the leading blits of a blit-only pass through the real command replay.
fn run_blits(gpu: &Gpu, blits: &[BlitCommand]) {
    let pass = PassDescriptor {
        color_texture: MetalHandle::NULL,
        color_resolve_texture: MetalHandle::NULL,
        depth_texture: MetalHandle::NULL,
        commands_ptr: 0,
        visibility_result_buffer: MetalHandle::NULL,
        leading_blits_ptr: blits.as_ptr() as u64,
        color_load_action: LoadAction::DontCare,
        color_store_action: StoreAction::DontCare,
        clear_r: 0,
        clear_g: 0,
        clear_b: 0,
        clear_a: 0,
        depth_load_action: LoadAction::DontCare,
        depth_store_action: StoreAction::DontCare,
        depth_clear_value: 0,
        stencil_load_action: LoadAction::DontCare,
        stencil_store_action: StoreAction::DontCare,
        stencil_clear_value: 0,
        command_count: 0,
        leading_blits_count: u32::try_from(blits.len()).expect("few blits"),
        pass_flags: PassDescriptor::pack_flags(true, 0, 0, 0, 0),
        reserved: 0,
        extra_color: [ExtraColorDesc::NONE; 3],
    };
    gpu.run_descriptors(&[pass]);
}

/// One ray's closest hit: `(distance, user id)`, `None` on a miss.
type Hit = Option<(f32, u32)>;

const PROBE_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

kernel void probe(instance_acceleration_structure tlas [[buffer(0)]],
                  device const float4 *rays [[buffer(1)]],
                  device float4 *hits [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    ray r;
    r.origin = rays[2 * i].xyz;
    r.direction = rays[2 * i + 1].xyz;
    r.min_distance = 0.0;
    r.max_distance = 1.0e6;
    intersection_query<triangle_data, instancing> q(r, tlas, 0xFFu);
    while (q.next()) {
        q.commit_triangle_intersection();
    }
    if (q.get_committed_intersection_type() == intersection_type::triangle) {
        hits[i] = float4(q.get_committed_distance(), float(q.get_committed_user_instance_id()), 0.0, 1.0);
    } else {
        hits[i] = float4(-1.0, -1.0, 0.0, 0.0);
    }
}
";

/// Trace rays from the eye at `targets` through `tlas`: the closest hit of each.
fn probe(gpu: &Gpu, tlas: &TlasHandle, targets: &[[f32; 3]]) -> Vec<Hit> {
    let device = &gpu.device;
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(PROBE_MSL), None)
        .expect("the probe kernel compiles");
    let function = library
        .newFunctionWithName(&NSString::from_str("probe"))
        .expect("probe");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("the probe pipeline");
    let rays: Vec<f32> = targets
        .iter()
        .flat_map(|t| {
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            [0.0, 0.0, 0.0, 0.0, t[0] / len, t[1] / len, t[2] / len, 0.0]
        })
        .collect();
    let rays: Vec<u8> = rays.iter().flat_map(|v| v.to_le_bytes()).collect();
    let rays = gpu.buffer(&rays);
    let hits = device
        .newBufferWithLength_options(targets.len() * 16, MTLResourceOptions::StorageModeShared)
        .expect("hits");
    // SAFETY: the TLAS this frame built is alive while the scene holds it.
    let accel = unsafe { MetalHandle::<MTLAccelerationStructureKind>::new(tlas.tlas) }
        .into_retained()
        .expect("a TLAS");
    let cmd = gpu.queue.commandBuffer().expect("command buffer");
    let compute = cmd.computeCommandEncoder().expect("compute");
    compute.setComputePipelineState(&pipeline);
    // SAFETY: slot 0 is the kernel's acceleration structure, 1 and 2 its buffers.
    unsafe {
        compute.setAccelerationStructure_atBufferIndex(Some(&accel), 0);
        compute.setBuffer_offset_atIndex(Some(&rays), 0, 1);
        compute.setBuffer_offset_atIndex(Some(&hits), 0, 2);
    }
    for &blas in &tlas.resident {
        // SAFETY: the scene holds every BLAS its TLAS instances.
        let blas = unsafe { MetalHandle::<MTLAccelerationStructureKind>::new(blas) }
            .into_retained()
            .expect("a BLAS");
        compute.useResource_usage(ProtocolObject::from_ref(&*blas), MTLResourceUsage::Read);
    }
    compute.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: targets.len(),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    compute.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the completed dispatch wrote one float4 per ray.
    let out = unsafe {
        core::slice::from_raw_parts(hits.contents().as_ptr().cast::<[f32; 4]>(), targets.len())
    };
    out.iter()
        .map(|h| {
            (h[3] > 0.5).then(|| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a small id"
                )]
                let id = h[1] as u32;
                (h[0], id)
            })
        })
        .collect()
}

fn length(p: [f32; 3]) -> f32 {
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}

/// Every `rt.*` key acts as off on a device without ray tracing, whatever
/// the file said, and only there: the scene passes see no ray-tracing key
/// (no capture for `rt.*` alone, the sun's cascades back under
/// `shadows.sun`), and the encoder keeps no ray-tracing state.
#[test]
fn rt_disabled_without_support() {
    let every_key = Mtld3dConfig {
        shadows: ShadowConfig {
            sun: true,
            ..ShadowConfig::default()
        },
        rt: RtConfig {
            shadows: true,
            ao_pct: 100,
            gi_pct: 50,
            point_shadows: true,
            ..RtConfig::default()
        },
        ..Mtld3dConfig::default()
    };
    let config = effective_config(Arc::new(every_key.clone()), false);
    assert_eq!(config.rt, RtConfig::default(), "every rt key off");
    assert!(config.shadows.sun, "the shadow-map keys stay");
    let supported = Arc::new(every_key);
    assert!(
        Arc::ptr_eq(&effective_config(Arc::clone(&supported), true), &supported),
        "a device with ray tracing keeps the file's config"
    );
    let rt_only = effective_config(Arc::new(rt_on()), false);
    assert!(!mtld3d_core::shadows::scene_capture_wanted(&rt_only));
    assert!(!mtld3d_core::shadows::sun_receivers_wanted(&rt_only));

    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, enc)) = encoder_with_config(&coherent, &retained, (*rt_only).clone()) else {
        return;
    };
    assert!(enc.rt.is_none(), "no ray-tracing state without the keys");
    assert!(!enc.scene.wanted());
}

/// With every `rt.*` key absent the encoder holds no ray-tracing state and
/// a frame builds nothing, under `shadows.sun` too.
#[test]
fn rt_off_builds_nothing() {
    let sun = Mtld3dConfig {
        shadows: ShadowConfig {
            sun: true,
            ..ShadowConfig::default()
        },
        ..Mtld3dConfig::default()
    };
    for config in [Mtld3dConfig::default(), sun] {
        let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
        let Some((_device, mut enc)) = encoder_with_config(&coherent, &retained, config) else {
            return;
        };
        assert!(enc.rt.is_none());
        post_reset_frame(&mut enc, (64, 64), 1, true);
        enc.scene.begin_submission(false);
        let sources = Sources::new();
        let draws = if enc.scene.wanted() {
            vec![capture(
                &mut enc,
                &sources,
                &Draw::list(0x100, content(1, 1), IDENTITY),
            )]
        } else {
            Vec::new()
        };
        let (tlas, blits) = build(&mut enc, &draws);
        assert!(tlas.is_none());
        assert!(blits.is_empty());
    }
}

/// The key names the contents a draw read, not the Metal buffer that held
/// them: the same draw gives the same key, another range, count or content
/// version a new one, and the same contents behind another handle the same.
#[test]
fn blas_key_identity() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder_with_config(&coherent, &retained, rt_on()) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, true);
    enc.scene.begin_submission(false);
    let sources = Sources::new();
    let base = Draw::list(0x100, content(7, 3), IDENTITY);
    let mut key = |draw: Draw| -> BlasKey {
        let d = capture(&mut enc, &sources, &draw);
        blas_source(&d).expect("a static fixed-function draw").key
    };
    let first = key(base);
    assert_eq!(key(base), first, "the same draw");
    assert_eq!(
        key(Draw {
            vertex_buffer: 0x200,
            ..base
        }),
        first,
        "the same contents behind another handle"
    );
    assert_eq!(
        key(Draw {
            world_view: scale_move([2.0, 2.0, 2.0], [0.0, 0.0, 5.0]),
            ..base
        }),
        first,
        "the instance transform is not geometry"
    );
    let others = [
        Draw { offset: 24, ..base },
        Draw {
            index: CapturedIndex::None {
                start_vertex: 0,
                vertex_count: 3,
            },
            ..base
        },
        Draw {
            index: CapturedIndex::None {
                start_vertex: 3,
                vertex_count: 3,
            },
            ..base
        },
        Draw {
            content: content(7, 4),
            ..base
        },
        Draw {
            content: content(8, 3),
            ..base
        },
        Draw {
            primitive: PrimitiveType::TriangleStrip,
            ..base
        },
    ];
    let mut keys = vec![first];
    for draw in others {
        let k = key(draw);
        assert!(!keys.contains(&k), "{k:?} is new");
        keys.push(k);
    }
    let indexed = Draw {
        index: CapturedIndex::Buffer {
            handle: 0x300,
            offset: 0,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 0,
            content: content(9, 1),
        },
        ..base
    };
    let k = key(indexed);
    assert!(!keys.contains(&k));
    for other in [
        CapturedIndex::Buffer {
            handle: 0x300,
            offset: 12,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 0,
            content: content(9, 1),
        },
        CapturedIndex::Buffer {
            handle: 0x300,
            offset: 0,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 2,
            content: content(9, 1),
        },
        CapturedIndex::Buffer {
            handle: 0x300,
            offset: 0,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 0,
            content: content(9, 2),
        },
    ] {
        assert_ne!(
            key(Draw {
                index: other,
                ..base
            }),
            k
        );
    }
}

/// Two quads at known eye positions under non-symmetric world-views: a ray
/// from the eye at each one's centre hits that one, by its user id, at the
/// centre's distance, and a ray where a transposed transform would put the
/// first quad hits nothing. Both instance the one BLAS of their shared range.
#[test]
fn tlas_ray_hits_instance() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let (vertices, vb) = staged(&mut enc, &gpu, 0x51, &quad_list());
    let a = scale_turn_move([2.0, 1.0, 1.0], [0.0, 0.0, 5.0]);
    let b = scale_move([1.0, 0.5, 1.0], [3.0, -1.0, 9.0]);
    let draws =
        [a, b].map(|world_view| capture(&mut enc, &sources, &Draw::list(vertices, vb, world_view)));
    let (tlas, blits) = build(&mut enc, &draws);
    let tlas = tlas.expect("a TLAS");
    assert_eq!(
        count(&blits, BlitCommandType::BuildBlas),
        1,
        "one BLAS for both"
    );
    assert_eq!(count(&blits, BlitCommandType::BuildTlas), 1);
    assert_eq!(tlas.instances.len(), 2);
    assert_eq!(tlas.draws, [0, 1]);
    run_blits(&gpu, &blits);

    let (centre_a, centre_b) = (apply(&a, QUAD_CENTRE), apply(&b, QUAD_CENTRE));
    assert_eq!(centre_a, [-1.0, 1.0, 5.0]);
    // Where a transposed rotation would have put the first quad's centre.
    let transposed = [1.0, -1.0, 5.0];
    let hits = probe(&gpu, &tlas, &[centre_a, centre_b, transposed]);
    let (distance, id) = hits[0].expect("the first quad");
    assert_eq!(id, 0);
    assert!((distance - length(centre_a)).abs() < 1e-3, "{distance}");
    let (distance, id) = hits[1].expect("the second quad");
    assert_eq!(id, 1);
    assert!((distance - length(centre_b)).abs() < 1e-3, "{distance}");
    assert_eq!(hits[2], None);
}

/// A frame drawing what the last one drew builds no BLAS: it instances the
/// cached one, compacted once the GPU has finished the build, and its rays
/// hit as the first frame's did.
#[test]
fn blas_reused_across_frames() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let (vertices, vb) = staged(&mut enc, &gpu, 0x52, &quad_list());
    let world_view = scale_move([1.0, 1.0, 1.0], [0.0, 0.0, 4.0]);
    let draw = Draw::list(vertices, vb, world_view);
    let centre = apply(&world_view, QUAD_CENTRE);

    let first = capture(&mut enc, &sources, &draw);
    let (tlas, blits) = build(&mut enc, &[first]);
    assert_eq!(count(&blits, BlitCommandType::BuildBlas), 1);
    run_blits(&gpu, &blits);
    let built = tlas.expect("a TLAS").resident;

    for frame in 2..=3 {
        next_frame(&mut enc, &coherent);
        let again = capture(&mut enc, &sources, &draw);
        let (tlas, blits) = build(&mut enc, &[again]);
        let tlas = tlas.expect("a TLAS");
        assert_eq!(
            count(&blits, BlitCommandType::BuildBlas),
            0,
            "frame {frame} builds no BLAS"
        );
        assert_eq!(count(&blits, BlitCommandType::BuildTlas), 1);
        let stats = enc.rt_stats();
        assert_eq!((stats.built, stats.reused), (0, 1));
        assert_eq!(
            count(&blits, BlitCommandType::CompactBlas),
            usize::from(frame == 2),
            "compacted once, after its first frame"
        );
        if frame == 2 {
            assert_ne!(tlas.resident, built, "the compacted copy replaces it");
            assert!(
                enc.pending_resource_retention
                    .iter()
                    .any(|e| e.kind == DestroyKind::AccelerationStructure && e.handle == built[0]),
                "the uncompacted BLAS retires behind the GPU"
            );
        }
        run_blits(&gpu, &blits);
        let hit = probe(&gpu, &tlas, &[centre])[0].expect("the quad");
        assert!((hit.0 - length(centre)).abs() < 1e-3);
    }
}

/// A staged buffer `id` of `bytes` whose device buffer the encoder owns, as a warmup leaves it.
fn staged_buffer(
    enc: &mut FrameEncoder,
    device: &ProtocolObject<dyn MTLDevice>,
    id: BufferId,
    bytes: &[u8],
) -> u64 {
    // SAFETY: `bytes` is a live slice, copied now.
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::from(bytes).cast(),
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("a device buffer");
    let handle = Retained::into_raw(buffer) as u64;
    let version = enc.next_buffer_version();
    enc.buffer_cache.insert(
        id,
        BufferGpuState {
            mtl_buffer: MetalHandle::NULL,
            // SAFETY: the retain just taken, handed to the cache.
            device_buffer: unsafe { MetalHandle::<MTLBufferKind>::new(handle) },
            is_staged: true,
            backing_ptr: 0,
            length: bytes.len() as u64,
            backing_generation: 0,
            last_submit_seq: 0,
            content_version: version,
        },
    );
    handle
}

/// The device buffer the cache binds for `id` now.
fn device_buffer(enc: &FrameEncoder, id: BufferId) -> u64 {
    enc.buffer_cache[&id].device_buffer.raw()
}

/// [`staged_buffer`] `id` on `gpu`'s device: its handle and its contents.
fn staged(enc: &mut FrameEncoder, gpu: &Gpu, id: u64, bytes: &[u8]) -> (u64, BufferContent) {
    let id = BufferId::from_raw(id);
    let handle = staged_buffer(enc, &gpu.device, id, bytes);
    (handle, enc.buffer_content(id))
}

/// A BLAS is keyed on what its buffer held: a staging upload into the
/// range in place (same handle) and one that renames the buffer (a draw
/// already read it) each make the next frame build anew, and retire the
/// BLAS of the old contents behind the GPU.
#[test]
fn blas_retired_on_buffer_rename() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let sources = Sources::new();
    let id = BufferId::from_raw(0x5100);
    let bytes = quad_list();
    let handle = staged_buffer(&mut enc, &device, id, &bytes);
    let world_view = scale_move([1.0, 1.0, 1.0], [0.0, 0.0, 4.0]);
    let frame = |enc: &mut FrameEncoder| {
        let handle = device_buffer(enc, id);
        let draw = Draw::list(handle, enc.buffer_content(id), world_view);
        let d = capture(enc, &sources, &draw);
        let (tlas, blits) = build(enc, &[d]);
        (tlas.expect("a TLAS").resident[0], blits)
    };
    let retired = |enc: &FrameEncoder, blas: u64| {
        enc.pending_resource_retention
            .iter()
            .any(|e| e.kind == DestroyKind::AccelerationStructure && e.handle == blas)
    };
    let pool = CompletionPool::new();
    let mut leases = Vec::new();
    let mut upload = |enc: &mut FrameEncoder| {
        let mut page = PageBox::new_zeroed(bytes.len());
        page.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
        let lease = GuestOwnedPageLease::new(page, &pool, None);
        // SAFETY: the lease outlives the encoder's adoption (dropped after the test's encoder).
        let page = unsafe { lease.descriptor().adopt() }.expect("a valid lease");
        leases.push(lease);
        let size = u32::try_from(bytes.len()).expect("small");
        enc.apply_stage_upload(id, page, 0, size);
    };

    let (first, blits) = frame(&mut enc);
    assert_eq!(count(&blits, BlitCommandType::BuildBlas), 1);

    // In place: no draw read the range this frame.
    next_frame(&mut enc, &coherent);
    upload(&mut enc);
    assert_eq!(device_buffer(&enc, id), handle, "uploaded in place");
    let (second, blits) = frame(&mut enc);
    assert_eq!(
        count(&blits, BlitCommandType::BuildBlas),
        1,
        "new contents, new BLAS"
    );
    assert_ne!(second, first);
    assert!(retired(&enc, first), "the old contents' BLAS retires");

    // Renamed: a draw this frame already read the range.
    next_frame(&mut enc, &coherent);
    let len = u32::try_from(bytes.len()).expect("small");
    enc.note_buffer_draw_range(id.raw(), 0, len, len);
    upload(&mut enc);
    assert_ne!(device_buffer(&enc, id), handle, "renamed");
    let (third, blits) = frame(&mut enc);
    assert_eq!(count(&blits, BlitCommandType::BuildBlas), 1);
    assert_ne!(third, second);
    assert!(retired(&enc, second));
    drop(enc);
    drop(leases);
}

/// Strips become lists through the index rewrite: an indexed strip (read
/// past an index-buffer offset) and a non-indexed one (its start folded into
/// the vertex offset) each build from a cache-owned list with every odd
/// triangle's winding turned back, and a ray hits the strip's second triangle.
#[test]
fn strip_converted_to_list() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    // A ribbon two units long, one high, in z = 0.
    let ribbon = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ];
    let (vertices, vb) = staged(&mut enc, &gpu, 0x60, &positions(&ribbon));
    // A leading index the draw skips by its offset, then the ribbon reversed.
    let indices: Vec<u8> = [99u16, 5, 4, 3, 2, 1, 0]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    let (index_buffer, ib) = staged(&mut enc, &gpu, 0x61, &indices);
    let world_view = scale_move([1.0, 1.0, 1.0], [0.0, 0.0, 4.0]);
    let indexed = Draw {
        index: CapturedIndex::Buffer {
            handle: index_buffer,
            offset: 2,
            count: 6,
            index_type: IndexType::UInt16,
            base_vertex: 0,
            content: ib,
        },
        primitive: PrimitiveType::TriangleStrip,
        ..Draw::list(vertices, vb, world_view)
    };
    let plain = Draw {
        index: CapturedIndex::None {
            start_vertex: 1,
            vertex_count: 4,
        },
        primitive: PrimitiveType::TriangleStrip,
        world_view: scale_move([1.0, 1.0, 1.0], [10.0, 0.0, 4.0]),
        ..Draw::list(vertices, vb, world_view)
    };
    let draws = [indexed, plain].map(|d| capture(&mut enc, &sources, &d));
    let (tlas, blits) = build(&mut enc, &draws);
    let tlas = tlas.expect("a TLAS");
    run_blits(&gpu, &blits);

    let builds: Vec<BlasBuild> = blits
        .iter()
        .filter(|b| b.cmd == BlitCommandType::BuildBlas as u32)
        // SAFETY: a BuildBlas names its `BlasBuild` in the live encoder's scratch arena.
        .map(|b| unsafe { core::ptr::read_unaligned(b.byte_size as *const BlasBuild) })
        .collect();
    assert_eq!(builds.len(), 2);
    let list = |build: &BlasBuild| -> Vec<u32> {
        assert_ne!(build.strip.pipeline, 0, "a strip converts");
        assert_eq!(build.geometry.index_buffer, build.strip.dest);
        assert_eq!(build.geometry.index_offset, build.strip.dest_offset);
        assert_eq!(build.geometry.index_type, IndexType::UInt32 as u32);
        // SAFETY: the converting scene owns this Shared buffer until it retires.
        let buffer = unsafe { MetalHandle::<MTLBufferKind>::new(build.strip.dest) }
            .into_retained()
            .expect("the list buffer");
        let count = build.geometry.triangle_count as usize * 3;
        // SAFETY: the completed conversion wrote `count` indices at the offset.
        unsafe {
            core::slice::from_raw_parts(
                buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(build.strip.dest_offset as usize)
                    .cast::<u32>(),
                count,
            )
        }
        .to_vec()
    };
    assert_eq!(list(&builds[0]), [5, 4, 3, 3, 4, 2, 3, 2, 1, 1, 2, 0]);
    assert_eq!(builds[0].geometry.vertex_offset, 0);
    assert_eq!(list(&builds[1]), [0, 1, 2, 2, 1, 3]);
    assert_eq!(builds[1].geometry.vertex_offset, u64::from(STRIDE));

    // The indexed strip's second triangle is ribbon vertices 3, 4, 2.
    let second = [3, 4, 2].map(|i| ribbon[i]);
    let centre = apply(
        &world_view,
        core::array::from_fn(|c| (second[0][c] + second[1][c] + second[2][c]) / 3.0),
    );
    let beyond = apply(&world_view, [2.5, 0.5, 0.0]);
    let hits = probe(&gpu, &tlas, &[centre, beyond]);
    let (distance, id) = hits[0].expect("the strip");
    assert_eq!(id, 0);
    assert!((distance - length(centre)).abs() < 1e-3);
    assert_eq!(hits[1], None, "past the ribbon's end");
}

/// The cache stays within its budget by retiring the least recently used
/// BLAS first, never one this frame uses, and says when it cannot.
#[test]
fn budget_evicts_lru() {
    let key = |vb: u64| BlasKey {
        vb,
        ..BlasKey::default()
    };
    let mut cache = BlasCache::new(300);
    cache.insert(key(1), 0xA1, 100, 1);
    cache.insert(key(2), 0xA2, 100, 2);
    cache.insert(key(3), 0xA3, 100, 3);
    assert_eq!(cache.bytes(), 300);
    assert_eq!(cache.get(&key(1), 4), Some(0xA1), "used again");
    let evicted = cache.make_room(100, 5).expect("room");
    assert_eq!(evicted, [0xA2], "the least recently used goes");
    assert!(cache.get(&key(2), 5).is_none());
    cache.insert(key(4), 0xA4, 100, 5);
    assert_eq!(cache.bytes(), 300);
    assert_eq!(cache.make_room(150, 6).expect("room").len(), 2);
    assert_eq!(cache.bytes(), 100);
    // Everything left is this frame's: nothing goes, and there is no room.
    assert!(cache.get(&key(4), 7).is_some());
    assert!(cache.make_room(250, 7).is_none());
    assert_eq!(cache.bytes(), 100, "a refusal evicts nothing");
    assert_eq!(cache.make_room(200, 7), Some(Vec::new()), "already room");
}

/// Under `rt.shadows` the scene stages lead the world's first pass with the
/// frame's acceleration-structure builds, ahead of everything else it runs,
/// and keep the TLAS for the kernels that trace it; a split world builds
/// none.
#[test]
fn assemble_leads_the_world_with_the_builds() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    enc.prepare_scene_textures((64, 64), 1);
    let sources = Sources::new();
    let (vertices, vb) = staged(&mut enc, &gpu, 0x70, &quad_list());
    enc.pass_state
        .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 6));
    capture(&mut enc, &sources, &Draw::list(vertices, vb, IDENTITY));
    enc.pass_state.end_current_pass("test");
    assemble(&mut enc, false);
    let world = enc
        .pass_state
        .passes()
        .iter()
        .find(|p| p.color_texture().raw() == 0xB000)
        .expect("the world pass");
    let kinds: Vec<u32> = world.leading_blits().iter().map(|b| b.cmd).collect();
    assert_eq!(
        kinds[..2],
        [
            BlitCommandType::BuildBlas as u32,
            BlitCommandType::BuildTlas as u32
        ]
    );
    let tlas = enc.rt_frame_tlas().expect("the frame's TLAS");
    assert_eq!(world.leading_blits()[1].dst_handle, tlas.tlas);

    // A world a mid-frame flush cut builds nothing and keeps no TLAS.
    next_frame(&mut enc, &coherent);
    enc.prepare_scene_textures((64, 64), 1);
    enc.pass_state
        .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 6));
    capture(&mut enc, &sources, &Draw::list(vertices, vb, IDENTITY));
    enc.pass_state.end_current_pass("test");
    assemble(&mut enc, true);
    assert!(enc.rt_frame_tlas().is_none());
    assert!(
        enc.pass_state
            .passes()
            .iter()
            .flat_map(|p| p.leading_blits())
            .all(|b| !BlitCommandType::from_repr(b.cmd)
                .is_some_and(BlitCommandType::is_acceleration_structure))
    );
}

/// A stream bound at an offset that is no multiple of its stride (but
/// 4-byte aligned) still builds, from the bound position on; one off the
/// 4-byte alignment Metal needs is left out.
#[test]
fn blas_built_from_an_offset_off_the_stride() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let mut bytes = vec![0u8; 4];
    bytes.extend(quad_list());
    let (vertices, vb) = staged(&mut enc, &gpu, 0x90, &bytes);
    let world_view = scale_move([1.0, 1.0, 1.0], [0.0, 0.0, 4.0]);
    let shifted = Draw {
        offset: 4,
        ..Draw::list(vertices, vb, world_view)
    };
    let d = capture(&mut enc, &sources, &shifted);
    let (tlas, blits) = build(&mut enc, &[d]);
    let tlas = tlas.expect("a TLAS");
    run_blits(&gpu, &blits);
    let centre = apply(&world_view, QUAD_CENTRE);
    let (distance, _) = probe(&gpu, &tlas, &[centre])[0].expect("the quad");
    assert!((distance - length(centre)).abs() < 1e-3);

    let unaligned = capture(
        &mut enc,
        &sources,
        &Draw {
            offset: 2,
            ..shifted
        },
    );
    assert_eq!(blas_source(&unaligned).err(), Some(Skip::Offset));
}

/// The acceleration-structure commands of `enc`'s world pass, `None` when no pass leads with any.
fn world_rt_blits(enc: &FrameEncoder) -> Vec<BlitCommand> {
    enc.pass_state
        .passes()
        .iter()
        .flat_map(|p| p.leading_blits())
        .filter(|b| {
            BlitCommandType::from_repr(b.cmd)
                .is_some_and(BlitCommandType::is_acceleration_structure)
        })
        .copied()
        .collect()
}

/// A submission whose scene commands would be dropped (no back-buffer
/// pass to lead: a flush with no world draw, or world draws whose pass is
/// not there) commits nothing to the cache: a compaction the GPU is ready
/// for waits, its BLAS stays the written one and stays alive, and the next
/// world frame compacts it.
#[test]
fn compaction_waits_for_a_world_pass() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let (vertices, vb) = staged(&mut enc, &gpu, 0x80, &quad_list());
    let draw = Draw::list(vertices, vb, scale_move([1.0; 3], [0.0, 0.0, 4.0]));
    let world_frame = |enc: &mut FrameEncoder| {
        enc.prepare_scene_textures((64, 64), 1);
        enc.pass_state
            .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 6));
        capture(enc, &sources, &draw);
        enc.pass_state.end_current_pass("test");
        assemble(enc, false);
        world_rt_blits(enc)
    };
    let blits = world_frame(&mut enc);
    assert_eq!(count(&blits, BlitCommandType::BuildBlas), 1);
    let built = enc.rt_frame_tlas().expect("a TLAS").resident[0];
    run_blits(&gpu, &blits);
    let alive = |enc: &FrameEncoder| {
        !enc.pending_resource_retention
            .iter()
            .any(|e| e.kind == DestroyKind::AccelerationStructure && e.handle == built)
    };

    // No world draw and no back-buffer pass.
    next_frame(&mut enc, &coherent);
    enc.prepare_scene_textures((64, 64), 1);
    assemble(&mut enc, false);
    assert!(world_rt_blits(&enc).is_empty());
    assert!(enc.rt_frame_tlas().is_none());
    assert_eq!(enc.rt_stats().compacted, 0);
    assert!(alive(&enc), "the written BLAS is not retired");

    // World draws captured, but no back-buffer pass to lead.
    next_frame(&mut enc, &coherent);
    enc.prepare_scene_textures((64, 64), 1);
    capture(&mut enc, &sources, &draw);
    assemble(&mut enc, false);
    assert!(world_rt_blits(&enc).is_empty());
    assert_eq!(enc.rt_stats().compacted, 0);
    assert!(alive(&enc));

    // The next world frame instances the written BLAS and compacts it.
    next_frame(&mut enc, &coherent);
    let blits = world_frame(&mut enc);
    let compactions: Vec<&BlitCommand> = blits
        .iter()
        .filter(|b| b.cmd == BlitCommandType::CompactBlas as u32)
        .collect();
    assert_eq!(compactions.len(), 1);
    assert_eq!(
        compactions[0].src_handle, built,
        "compacted from the written BLAS"
    );
    assert_eq!(count(&blits, BlitCommandType::BuildBlas), 0);
    assert!(!alive(&enc), "retired once its copy is queued");
    run_blits(&gpu, &blits);
    let tlas = enc.rt_frame_tlas().expect("a TLAS").clone();
    let centre = apply(&scale_move([1.0; 3], [0.0, 0.0, 4.0]), QUAD_CENTRE);
    assert!(probe(&gpu, &tlas, &[centre])[0].is_some());
}
