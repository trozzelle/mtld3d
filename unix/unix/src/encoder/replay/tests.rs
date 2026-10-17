//! Scene replay on the GPU: a captured draw drawn again through the real command replay.

use core::ptr::NonNull;
use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    async_compile::Resolution,
    config::Mtld3dConfig,
    depth_stencil_state::DepthStencilSnapshot,
    draw_data::{
        AttrSnapshot, DeclarationHeader, FixedPsSource, FixedVsSource, PsSource, PsSourcePtr,
        ShaderRef, VsSource, VsSourcePtr,
    },
    dxso::{FfPsKey, FfStage, FfVsFlags, FfVsKey, VariantKey, tci_entry},
    ids::TextureId,
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
    shadows::ShadowConfig,
    vs_draw::VS_DRAW_BYTES,
};
use mtld3d_shared::{
    Command, ExtraColorDesc, MetalHandle, PassDescriptor, VertexAttrDesc,
    mtl::{
        CullMode, LoadAction, PixelFormat, PrimitiveType, StoreAction, VertexFormat,
        VertexStepFunction,
    },
};
use mtld3d_types::{D3DCMP_GREATEREQUAL, D3DCMP_LESSEQUAL, D3DTOP_DISABLE};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResourceOptions,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};

use super::{IDENTITY, Replayed, neutral_pos_fixup, replay_draw, reproject_bytes};
use crate::encoder::{
    FrameEncoder,
    scene_capture::{
        CapturedDraw, CapturedIndex, DrawFacts, ReplayBinding, Stage0, VertexBind,
        encode_captured_draw,
    },
    tests::{encoder_with_config, post_reset_frame},
};

/// The side of every target, in pixels.
pub(in crate::encoder) const SIZE: u32 = 16;
/// Bytes per vertex: a float3 position, then a float2 coordinate.
const STRIDE: u32 = 20;
/// The near and far planes of the test projection.
pub(in crate::encoder) const NEAR: f32 = 1.0;
pub(in crate::encoder) const FAR: f32 = 10.0;

/// The textured fixed-function vertex stage of a world draw, no lighting, no clip planes.
fn ff_vs() -> VsSource {
    VsSource::FixedFunction(FixedVsSource {
        key: FfVsKey {
            ppl_knee: 0,
            color_style: 0,
            scene: 0,
            ppl_mask: 0,
            ppl_model: 0,
            flags: FfVsFlags::empty(),
            input_tex_coord_count: 1,
            tex_coord_count: 1,
            light_active_mask: 0,
            light_directional_mask: 0,
            light_spot_mask: 0,
            diffuse_source: 0,
            ambient_source: 0,
            specular_source: 0,
            emissive_source: 0,
            fog_mode: 0,
            tci: [tci_entry(0, 0); 8],
            tex_coord_dims: [2, 0, 0, 0, 0, 0, 0, 0],
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

/// A fixed-function pixel stage with every stage disabled: it outputs the diffuse colour.
fn ff_ps() -> PsSource {
    let disabled = FfStage {
        color_op: u8::try_from(D3DTOP_DISABLE).expect("a D3DTOP value"),
        ..FfStage::default()
    };
    PsSource::FixedFunction(FixedPsSource {
        key: FfPsKey {
            stages: [disabled; 8],
            specular_add: false,
            tt_projected_mask: 0,
        },
        sampled_stage_mask: 0,
        constant_rows: 0,
        bump_env: false,
        reserved: [0; 2],
    })
}

/// The declaration of [`STRIDE`]-byte vertices on stream 0, as the
/// fixed-function stage reads them: position at attribute 0, `TEXCOORD0` at 4.
struct Declaration {
    header: Box<DeclarationHeader>,
    attrs: Box<[VertexAttrDesc; 2]>,
}

impl Declaration {
    fn new() -> Self {
        Self {
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
}

/// The world pass's pipeline snapshot: `Bgra8Unorm` colour over `Depth32Float`.
pub(in crate::encoder) fn snapshot(vs_fn: u64, ps_fn: u64) -> PipelineSnapshot {
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
        vdecl_hash: 0x7e57,
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

/// Little-endian bytes of `values`.
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The fixed-function constants: identity world-view, a perspective
/// projection whose `w` is the eye `z`, zero elsewhere.
fn vs_constants() -> Vec<u8> {
    let a = FAR / (FAR - NEAR);
    let mut rows = vec![[0.0f32; 4]; 96];
    rows[..8].copy_from_slice(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, a, -NEAR * a],
        [0.0, 0.0, 1.0, 0.0],
    ]);
    bytes(rows.as_flattened())
}

/// The draw-local bytes a draw binds, kept alive by the test.
struct DrawBytes {
    vs_constants: Vec<u8>,
    vs_draw: Vec<u8>,
    pos_fixup: Vec<u8>,
    alpha_ref: Vec<u8>,
}

impl DrawBytes {
    fn new() -> Self {
        let mut vs_draw = vec![0u8; VS_DRAW_BYTES];
        // Point size 1 with an open clamp, as `vs_draw` packs a stock draw.
        vs_draw[..16].copy_from_slice(&bytes(&[1.0, 1.0, 64.0, 0.0]));
        let pixel = 1.0 / f32::from(u16::try_from(SIZE).expect("small"));
        Self {
            vs_constants: vs_constants(),
            vs_draw,
            // The camera's half-pixel offset, no depth bias.
            pos_fixup: bytes(&[pixel, -pixel, 0.0, 1.0, 0.0]),
            alpha_ref: 0.5f32.to_le_bytes().to_vec(),
        }
    }
}

/// The GPU objects a test renders with.
pub(in crate::encoder) struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Gpu {
    pub(in crate::encoder) fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let queue = device.newCommandQueue().expect("command queue");
        Self { device, queue }
    }

    /// A Shared buffer holding `data`.
    pub(in crate::encoder) fn buffer(
        &self,
        data: &[u8],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        // SAFETY: `data` is a live slice of `data.len()` bytes, copied now.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::from(data).cast(),
                data.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("vertex buffer")
    }

    /// A Private render target of `format`, [`SIZE`] square.
    fn target(&self, format: MTLPixelFormat) -> Retained<ProtocolObject<dyn MTLTexture>> {
        // SAFETY: a single-level 2D descriptor of a renderable format.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                SIZE as usize,
                SIZE as usize,
                false,
            )
        };
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setUsage(MTLTextureUsage::RenderTarget);
        self.device
            .newTextureWithDescriptor(&desc)
            .expect("render target")
    }

    /// A 4x4 `RGBA8Unorm` checkerboard whose alpha is 0 and 255 by turns.
    fn checker(&self) -> Retained<ProtocolObject<dyn MTLTexture>> {
        // SAFETY: a single-level 2D descriptor of a sampleable format.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                4,
                4,
                false,
            )
        };
        desc.setStorageMode(MTLStorageMode::Shared);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        let texture = self
            .device
            .newTextureWithDescriptor(&desc)
            .expect("checker");
        let texels: Vec<u8> = (0..16)
            .flat_map(|i| {
                let alpha = if (i % 4 + i / 4) % 2 == 0 { 0 } else { 255 };
                [255, 255, 255, alpha]
            })
            .collect();
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: 4,
                height: 4,
                depth: 1,
            },
        };
        // SAFETY: `texels` holds the 4x4 region at 16 bytes per row.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::from(texels.as_slice()).cast(),
                16,
            );
        }
        texture
    }

    /// A nearest-filtering sampler.
    fn sampler(&self) -> Retained<ProtocolObject<dyn MTLSamplerState>> {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Nearest);
        desc.setMagFilter(MTLSamplerMinMagFilter::Nearest);
        self.device
            .newSamplerStateWithDescriptor(&desc)
            .expect("sampler")
    }

    /// Run one pass of `commands` into `depth` (cleared to 1), with `color` when given.
    fn run(
        &self,
        color: Option<&ProtocolObject<dyn MTLTexture>>,
        depth: &ProtocolObject<dyn MTLTexture>,
        commands: &[Command],
    ) {
        let handle = |texture: &ProtocolObject<dyn MTLTexture>| {
            // SAFETY: the test owns the texture until the pass completes.
            unsafe { MetalHandle::new(core::ptr::from_ref(texture) as u64) }
        };
        let pass = PassDescriptor {
            color_texture: color.map_or(MetalHandle::NULL, handle),
            color_resolve_texture: MetalHandle::NULL,
            depth_texture: handle(depth),
            commands_ptr: commands.as_ptr() as u64,
            visibility_result_buffer: MetalHandle::NULL,
            leading_blits_ptr: 0,
            color_load_action: LoadAction::Clear,
            color_store_action: StoreAction::Store,
            clear_r: 0,
            clear_g: 0,
            clear_b: 0,
            clear_a: 0,
            depth_load_action: LoadAction::Clear,
            depth_store_action: StoreAction::Store,
            depth_clear_value: 1.0f32.to_bits(),
            stencil_load_action: LoadAction::DontCare,
            stencil_store_action: StoreAction::DontCare,
            stencil_clear_value: 0,
            command_count: u32::try_from(commands.len()).expect("few commands"),
            leading_blits_count: 0,
            pass_flags: PassDescriptor::pack_flags(false, 0, 0, 0, 0),
            reserved: 0,
            extra_color: [ExtraColorDesc::NONE; 3],
        };
        assert!(crate::metal::run_passes_for_test(&self.queue, &[pass]));
    }

    /// Run `passes` through the real command replay in one command buffer.
    pub(in crate::encoder) fn run_descriptors(&self, passes: &[PassDescriptor]) {
        assert!(crate::metal::run_passes_for_test(&self.queue, passes));
    }

    /// Every depth of a `Depth32Float` target, rows packed, as bits.
    fn depths(&self, depth: &ProtocolObject<dyn MTLTexture>) -> Vec<u32> {
        self.depths_at(depth, SIZE, 0)
    }

    /// Every depth of slice `slice` of the `side`-square `Depth32Float` texture `depth`, rows packed, as bits.
    pub(in crate::encoder) fn depths_at(
        &self,
        depth: &ProtocolObject<dyn MTLTexture>,
        side: u32,
        slice: usize,
    ) -> Vec<u32> {
        let row = side as usize * 4;
        let stride = row.next_multiple_of(256);
        let buffer = self
            .device
            .newBufferWithLength_options(
                stride * side as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("readback buffer");
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let blit = cmd.blitCommandEncoder().expect("blit");
        // SAFETY: the buffer holds every row of the slice at `stride`, and
        // both stay alive until the copy completes.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                depth, slice, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: side as usize, height: side as usize, depth: 1 },
                &buffer, 0, stride, stride * side as usize,
            );
        }
        blit.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: the completed copy initialized `stride * side` bytes.
        let all = unsafe {
            core::slice::from_raw_parts(
                buffer.contents().as_ptr().cast::<u8>(),
                stride * side as usize,
            )
        };
        all.chunks_exact(stride)
            .flat_map(|line| line[..row].as_chunks::<4>().0.iter().copied())
            .map(u32::from_le_bytes)
            .collect()
    }
}

/// An encoder with scene capture on (`shadows.sun`), on a fresh back buffer.
fn encoder(
    coherent: &AtomicU64,
    retained: &AtomicU64,
) -> Option<(Retained<ProtocolObject<dyn MTLDevice>>, FrameEncoder)> {
    let config = Mtld3dConfig {
        shadows: ShadowConfig {
            sun: true,
            ..ShadowConfig::default()
        },
        ..Mtld3dConfig::default()
    };
    let (device, mut enc) = encoder_with_config(coherent, retained, config)?;
    post_reset_frame(&mut enc, (SIZE, SIZE), 1, true);
    enc.scene.begin_submission(false);
    Some((device, enc))
}

/// What `resolve` answers once every build it waits for has landed.
pub(in crate::encoder) fn built<T>(
    enc: &mut FrameEncoder,
    mut resolve: impl FnMut(&mut FrameEncoder) -> Resolution<T>,
) -> T {
    loop {
        match resolve(enc) {
            Resolution::Ready(value) => return value,
            Resolution::Pending(ticket) => enc.wait_for_compiles(&[ticket]),
            Resolution::Failed => panic!("a build failed"),
        }
    }
}

/// [`replay_draw`] once its builds have landed: the commands it encoded.
pub(in crate::encoder) fn replayed(
    enc: &mut FrameEncoder,
    d: &CapturedDraw,
    pos_fixup: super::BytesRef,
) -> Vec<Command> {
    replayed_through(enc, d, &IDENTITY, pos_fixup)
}

/// [`replay_draw`] re-projected by `reproject` once its builds have landed: the commands it encoded.
fn replayed_through(
    enc: &mut FrameEncoder,
    d: &CapturedDraw,
    reproject: &[[f32; 4]; 4],
    pos_fixup: super::BytesRef,
) -> Vec<Command> {
    loop {
        let mut out = vec![Command::set_viewport(0, 0, SIZE, SIZE, 0.0, 1.0)];
        let matrix = reproject_bytes(enc, reproject);
        match replay_draw(enc, &mut out, d, matrix, pos_fixup, (0.0, 0.0)) {
            Replayed::Drawn => return out,
            Replayed::Pending(ticket) => {
                assert_eq!(out.len(), 1, "nothing is encoded while a build is pending");
                enc.wait_for_compiles(&[ticket]);
            }
            Replayed::Failed => panic!("the replay failed"),
        }
    }
}

/// The sources, declaration and draw-local bytes of one scripted draw.
pub(in crate::encoder) struct Fixture {
    vs: VsSource,
    ps: PsSource,
    decl: Declaration,
    bytes: DrawBytes,
}

impl Fixture {
    pub(in crate::encoder) fn new() -> Self {
        Self {
            vs: ff_vs(),
            ps: ff_ps(),
            decl: Declaration::new(),
            bytes: DrawBytes::new(),
        }
    }

    /// Capture its draw of two triangles from `vertex_buffer` the way
    /// `emit_draw_view` offers it, alpha-tested by `alpha` when given.
    pub(in crate::encoder) fn capture(
        &self,
        enc: &mut FrameEncoder,
        snapshot: &PipelineSnapshot,
        (pipeline, vertex_buffer): (u64, u64),
        alpha: Option<(u8, Stage0)>,
    ) -> CapturedDraw {
        enc.begin_scene_binds();
        enc.stage_scene_bind(VertexBind::Buffer {
            handle: vertex_buffer,
            offset: 0,
            slot: 0,
        });
        // SAFETY: the sources outlive the test's captures and do not move.
        let vs = unsafe { VsSourcePtr::new(NonNull::from(&self.vs)) };
        // SAFETY: as above.
        let ps = unsafe { PsSourcePtr::new(NonNull::from(&self.ps)) };
        enc.capture_scene_draw(&DrawFacts {
            snapshot,
            attrs: self.decl.token(),
            fetch_attrs: None,
            vs,
            ps,
            variant: VariantKey {
                alpha_func: alpha.map_or(0, |(func, _)| func),
                ..VariantKey::default()
            },
            pipeline,
            depth_stencil: 0xD5,
            depth_write: true,
            blend: false,
            index: CapturedIndex::None {
                start_vertex: 0,
                vertex_count: 6,
            },
            primitive: PrimitiveType::Triangle,
            instances: 1,
            vs_constants: &self.bytes.vs_constants,
            vs_draw: &self.bytes.vs_draw,
            vs_int: &[],
            vs_bool: &[],
            pos_fixup: &self.bytes.pos_fixup,
            alpha_ref: &self.bytes.alpha_ref,
            cull: CullMode::None,
            slope_scale: 0.0,
            stage0: alpha.map(|(_, stage0)| stage0),
        });
        enc.scene_draws()
            .last()
            .expect("the draw is captured")
            .clone()
    }
}

/// Two triangles of `(x, y, z)` corners, each with its `(u, v)`.
pub(in crate::encoder) fn quad(corners: [[f32; 5]; 4]) -> Vec<u8> {
    let [a, b, c, d] = corners;
    bytes([a, b, c, b, d, c].as_flattened())
}

/// Replaying a captured fixed-function draw with the identity re-projection
/// and its own camera `pos_fixup` into a `Depth32Float` target writes the
/// depths the world pass wrote, bit for bit, at the pixels it covered.
#[test]
fn reproject_identity_reproduces_camera_depth() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let fixture = Fixture::new();
    let variant = VariantKey::default();
    let vs_lib = built(&mut enc, |enc| enc.resolve_vs_library(fixture.vs.as_view()));
    let ps_lib = built(&mut enc, |enc| {
        enc.resolve_ps_library(fixture.ps.as_view(), variant)
    });
    let snapshot = snapshot(vs_lib.func.raw(), ps_lib.func.raw());
    let shaders = ShaderRef {
        vs: fixture.vs.as_view(),
        ps: fixture.ps.as_view(),
        variant,
    };
    let pipeline = built(&mut enc, |enc| {
        enc.get_or_create_pipeline(&snapshot, &fixture.decl.attrs[..], &shaders)
    });
    // A trapezoid receding from z 2 at the bottom to z 7 at the top, its
    // edges inside the target, so depth varies down the target and some
    // pixels stay clear.
    let vertices = gpu.buffer(&quad([
        [-1.6, -1.6, 2.0, 0.0, 1.0],
        [1.6, -1.6, 2.0, 1.0, 1.0],
        [-5.0, 5.0, 7.0, 0.0, 0.0],
        [5.0, 5.0, 7.0, 1.0, 0.0],
    ]));
    let d = fixture.capture(
        &mut enc,
        &snapshot,
        (pipeline, Retained::as_ptr(&vertices) as u64),
        None,
    );
    assert!(d.alpha.is_none());

    // The world pass: the draw with its own pipeline, depth state and camera `pos_fixup`.
    let world_depth = enc.get_or_create_depth_stencil(
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
            depth_stencil: world_depth,
            cull: d.cull,
            depth_bias: (0.0, d.slope_scale),
            pos_fixup: d.pos_fixup,
            extra_vs: &[],
        },
    );
    let color = gpu.target(MTLPixelFormat::BGRA8Unorm);
    let camera = gpu.target(MTLPixelFormat::Depth32Float);
    gpu.run(Some(&color), &camera, &world);

    let replay = replayed(&mut enc, &d, d.pos_fixup);
    assert!(
        replay.iter().any(|command| {
            command.cmd == mtld3d_shared::CommandType::SetVertexBytesAt as u32
                && command.param_a == mtld3d_shared::mtl::VS_REPROJECT_SLOT
        }),
        "the matrix is bound at its slot"
    );
    let predepth = gpu.target(MTLPixelFormat::Depth32Float);
    gpu.run(None, &predepth, &replay);

    let (camera, predepth) = (gpu.depths(&camera), gpu.depths(&predepth));
    let clear = 1.0f32.to_bits();
    let covered = camera.iter().filter(|&&depth| depth != clear).count();
    assert!(
        covered > 0 && covered < camera.len(),
        "{covered} pixels covered"
    );
    let distinct: std::collections::BTreeSet<u32> = camera
        .iter()
        .copied()
        .filter(|&depth| depth != clear)
        .collect();
    assert!(distinct.len() > 4, "the depth varies across the quad");
    for (index, (world, replay)) in camera.iter().zip(&predepth).enumerate() {
        assert_eq!(
            world,
            replay,
            "pixel {index}: world {} replay {}",
            f32::from_bits(*world),
            f32::from_bits(*replay)
        );
    }
    enc.shutdown_cleanup();
}

/// An alpha-tested draw replays with a caster that tests stage 0's alpha:
/// a checkerboard of transparent and opaque texels leaves a checkerboard
/// of holes (the cleared depth) in the caster depth.
#[test]
fn caster_alpha_discards_holes() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let fixture = Fixture::new();
    // The caster replaces both functions, so the world's are never built here.
    let snapshot = snapshot(0, 0);
    // A flat quad at z 2 covering the whole target, its coordinate 0..1 across it.
    let vertices = gpu.buffer(&quad([
        [-2.0, -2.0, 2.0, 0.0, 0.0],
        [2.0, -2.0, 2.0, 1.0, 0.0],
        [-2.0, 2.0, 2.0, 0.0, 1.0],
        [2.0, 2.0, 2.0, 1.0, 1.0],
    ]));
    let (checker, sampler) = (gpu.checker(), gpu.sampler());
    let d = fixture.capture(
        &mut enc,
        &snapshot,
        (0x91, Retained::as_ptr(&vertices) as u64),
        Some((
            u8::try_from(D3DCMP_GREATEREQUAL).expect("a D3DCMP value"),
            Stage0 {
                texture: TextureId::new_unique(),
                handle: Retained::as_ptr(&checker) as u64,
                sampler: Retained::as_ptr(&sampler) as u64,
            },
        )),
    );
    assert!(d.alpha.is_some(), "the alpha-tested draw is captured");

    let replay = replayed(&mut enc, &d, neutral_pos_fixup());
    let caster = gpu.target(MTLPixelFormat::Depth32Float);
    gpu.run(None, &caster, &replay);
    let depths = gpu.depths(&caster);

    let clear = 1.0f32.to_bits();
    // Each texel covers 4x4 pixels; which parity is transparent depends on
    // the coordinate's orientation, the checkerboard on nothing.
    let even_is_hole = depths[0] == clear;
    let side = SIZE as usize;
    let mut solid = std::collections::BTreeSet::new();
    for (index, &depth) in depths.iter().enumerate() {
        let (x, y) = (index % side, index / side);
        let hole = ((x / 4 + y / 4) % 2 == 0) == even_is_hole;
        if hole {
            assert_eq!(
                depth,
                clear,
                "pixel ({x}, {y}) is a hole: {}",
                f32::from_bits(depth)
            );
        } else {
            assert_ne!(depth, clear, "pixel ({x}, {y}) is solid");
            solid.insert(depth);
        }
    }
    // The flat quad's depth at z 2, under no depth bias.
    let a = FAR / (FAR - NEAR);
    let expected = a - NEAR * a / 2.0;
    assert_eq!(solid.len(), 1, "one depth across a flat quad: {solid:?}");
    let depth = f32::from_bits(*solid.first().expect("solid pixels"));
    assert!((depth - expected).abs() < 1e-6, "{depth} vs {expected}");
    let holes = depths.iter().filter(|&&depth| depth == clear).count();
    assert_eq!(holes, depths.len() / 2, "half the texels are transparent");
    enc.shutdown_cleanup();
}

/// The re-projection bytes are read as the row-vector matrix they hold: a
/// translation-only matrix with `m[3][0] = 0.5` (`x' = x + 0.5·w`) moves the
/// caster a quarter of the target to the right and leaves every depth as it
/// was. Read transposed, the same bytes would add `0.5·x` to `w` instead and
/// warp the quad in place.
#[test]
fn reproject_translation_moves_the_caster() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let fixture = Fixture::new();
    let snapshot = snapshot(0, 0);
    let vertices = gpu.buffer(&quad([
        [-1.6, -1.6, 2.0, 0.0, 1.0],
        [1.6, -1.6, 2.0, 1.0, 1.0],
        [-5.0, 5.0, 7.0, 0.0, 0.0],
        [5.0, 5.0, 7.0, 1.0, 0.0],
    ]));
    let d = fixture.capture(
        &mut enc,
        &snapshot,
        (0x91, Retained::as_ptr(&vertices) as u64),
        None,
    );
    let camera = gpu.target(MTLPixelFormat::Depth32Float);
    gpu.run(None, &camera, &replayed(&mut enc, &d, d.pos_fixup));
    let mut shift = IDENTITY;
    shift[3][0] = 0.5;
    let moved = gpu.target(MTLPixelFormat::Depth32Float);
    gpu.run(
        None,
        &moved,
        &replayed_through(&mut enc, &d, &shift, d.pos_fixup),
    );
    let (camera, moved) = (gpu.depths(&camera), gpu.depths(&moved));
    assert_ne!(camera, moved, "the translation moved something");

    let clear = 1.0f32.to_bits();
    let side = SIZE as usize;
    let offset = side / 4;
    let at = |depths: &[u32], x: usize, y: usize| depths[y * side + x];
    let mut compared = 0;
    for y in 1..side - 1 {
        for x in 1..side - 1 - offset {
            // Inner pixels only: an edge pixel's coverage may round either way.
            let inner = [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .iter()
                .all(|&(x, y)| at(&camera, x, y) != clear);
            if !inner {
                continue;
            }
            let (before, after) = (
                f32::from_bits(at(&camera, x, y)),
                f32::from_bits(at(&moved, x + offset, y)),
            );
            assert!(
                (before - after).abs() < 1e-6,
                "pixel ({x}, {y}) moved to ({}, {y}): {before} vs {after}",
                x + offset
            );
            compared += 1;
        }
    }
    assert!(compared > 8, "{compared} inner pixels compared");
    enc.shutdown_cleanup();
}
