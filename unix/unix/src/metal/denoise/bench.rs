//! A GPU timing harness of the sun's denoiser at 1920×1080, ignored by default:
//! `cargo test -p mtld3d-unix --lib denoise_bench -- --ignored --nocapture`.
//!
//! The scene is synthetic: a floor from a horizon at 35 % of the height down
//! to the camera, sky above it, a few boxes standing on it. The raw
//! visibility is lit, with the boxes' shadows at 0, their edges a penumbra
//! of random samples six pixels wide, and a patch of dappled light (random
//! samples) under trees. The camera holds still, so every history merges.
//!
//! Each frame runs the denoise stage as the frame encodes it, through
//! [`encode`]: the geometry kernel, the temporal pass and the à-trous
//! passes, each on its own encoder, with medium quality's passes and with
//! high quality's (one more). The frames go back to back, as a game keeps
//! the GPU busy (an idle GPU runs slower clocks); every other frame puts
//! each pass in a command buffer of its own. It prints the median GPU time
//! of a frame and of each pass alone.
//!
//! With `DENOISE_BENCH_DUMP` naming a directory, it writes this revision's
//! sun kernels there first (`kernel_source`): `geometry.msl`,
//! `temporal.msl` and `atrous.msl`. With `DENOISE_BENCH_BASELINE` naming
//! such directories (`:`-separated), it times their kernels too, at the
//! steps [`ATROUS_STEPS`], frame for frame interleaved with this
//! revision's, so all see the same clocks.

use core::ptr::NonNull;

use mtld3d_core::rt::RtQuality;
use mtld3d_shared::{
    BlitCommand, MetalHandle, SceneKernelInfo,
    mtl_handle::{MTLComputePipelineStateKind, MTLTextureKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLMathMode, MTLOrigin, MTLPixelFormat, MTLResourceOptions, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};

use super::{
    ATROUS_STEPS, DenoiseKind, FrameCamera, HistoryFrame, HistoryTextures, IDENTITY, atrous_steps,
    passes,
};
use crate::metal::scene_vis::{
    Kernel, RtConstants, SceneConstants, SceneDispatch, SceneTrace, create_pipeline, encode,
    kernel_source,
};

const W: usize = 1920;
const H: usize = 1080;
/// Frames run before the timed ones: every history fills, the clocks rise.
const WARM: usize = 60;
/// Frames timed per variant.
const TIMED: usize = 200;

type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Pipelines = [MetalHandle<MTLComputePipelineStateKind>; 3];

fn texture(device: &ProtocolObject<dyn MTLDevice>, format: MTLPixelFormat) -> Texture {
    // SAFETY: a single-level 2D descriptor of a sampleable format.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format, W, H, false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
    device.newTextureWithDescriptor(&desc).expect("texture")
}

fn raw(texture: &Texture) -> MetalHandle<MTLTextureKind> {
    // SAFETY: the bench keeps every texture alive past its dispatches.
    unsafe { MetalHandle::new(Retained::as_ptr(texture) as u64) }
}

/// A deterministic generator of the random samples.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[expect(clippy::cast_precision_loss, reason = "24 random bits")]
        let bits = (self.0 >> 40) as f32;
        bits / 16_777_216.0
    }
}

/// The eye depth of each pixel (0 for the sky) and its raw visibility.
fn scene() -> (Vec<f32>, Vec<u16>) {
    let horizon = H * 35 / 100;
    let boxes: [(usize, usize, usize, usize, f32); 4] = [
        (200, horizon - 120, 520, horizon + 300, 60.0),
        (800, horizon - 60, 1000, horizon + 80, 300.0),
        (1300, horizon - 200, 1750, horizon + 420, 40.0),
        (1100, horizon - 20, 1180, horizon + 30, 900.0),
    ];
    let shadow = |x: usize, y: usize| -> f32 {
        // Each box's shadow falls down and right of it on the floor.
        boxes
            .iter()
            .map(|&(x0, _, x1, y1, _)| {
                let (sx0, sx1, sy0, sy1) = (x0 + 80, x1 + 160, y1, (y1 + 180).min(H));
                let inside_x = (x as f32 - sx0 as f32).min(sx1 as f32 - x as f32);
                let inside_y = (y as f32 - sy0 as f32).min(sy1 as f32 - y as f32);
                inside_x.min(inside_y)
            })
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut rng = Lcg(7);
    let mut depth = vec![0.0f32; W * H];
    let mut vis = vec![0u16; W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let on_box = boxes
                .iter()
                .find(|&&(x0, y0, x1, y1, _)| (x0..x1).contains(&x) && (y0..y1).contains(&y));
            let z = match on_box {
                Some(&(.., z)) => z,
                None if y > horizon => 10.0 * 1.732 * 540.0 / (y - horizon) as f32,
                None => 0.0,
            };
            depth[i] = z;
            let lit = if z == 0.0 {
                1.0
            } else if on_box.is_none() && (1400..1900).contains(&x) && y > H - 300 {
                // Dappled light under trees: half the samples lit.
                if rng.unit() < 0.5 { 1.0 } else { 0.0 }
            } else {
                // Inside a shadow (positive) or out of it, a penumbra of six pixels.
                let s = if on_box.is_some() {
                    -10.0
                } else {
                    shadow(x, y)
                };
                let p = (0.5 - s / 6.0).clamp(0.0, 1.0);
                if p == 0.0 || p == 1.0 {
                    p
                } else if rng.unit() < p {
                    1.0
                } else {
                    0.0
                }
            };
            vis[i] = if lit > 0.5 { 0x3C00 } else { 0 };
        }
    }
    (depth, vis)
}

/// Copy `bytes` (rows of `row` bytes) into `texture` through a Shared buffer.
fn upload(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &Texture,
    bytes: &[u8],
    row: usize,
) {
    // SAFETY: `bytes` is a live slice of its length.
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::from(bytes).cast(),
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("staging");
    let cb = queue.commandBuffer().expect("command buffer");
    let blit = cb.blitCommandEncoder().expect("blit");
    // SAFETY: the buffer holds the whole texture at `row` bytes a row.
    unsafe {
        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            &buffer,
            0,
            row,
            row * H,
            MTLSize { width: W, height: H, depth: 1 },
            texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
    }
    blit.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
}

/// The baseline's pipelines from the MSL in `dir`, compiled as `create_pipeline` compiles them.
fn baseline(device: &ProtocolObject<dyn MTLDevice>, dir: &str) -> Pipelines {
    [
        ("geometry", "scene_geometry"),
        ("temporal", "scene_temporal"),
        ("atrous", "scene_atrous"),
    ]
    .map(|(file, entry)| {
        let source = std::fs::read_to_string(format!("{dir}/{file}.msl")).expect("baseline MSL");
        let options = MTLCompileOptions::new();
        if file == "geometry" {
            options.setMathMode(MTLMathMode::Safe);
        }
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .expect("baseline library");
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .expect("baseline function");
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .expect("baseline pipeline");
        // SAFETY: the canonical retain is the bench's, released at its end.
        unsafe { MetalHandle::new(Retained::into_raw(pipeline) as u64) }
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "a GPU timing harness: run with --ignored --nocapture"]
fn denoise_bench_1080p() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let queue = device.newCommandQueue().expect("queue");
    let kind = DenoiseKind::Sun;
    let kernels = [Kernel::Geometry, kind.temporal(), kind.atrous()];
    if let Ok(dir) = std::env::var("DENOISE_BENCH_DUMP") {
        for (kernel, file) in kernels.into_iter().zip(["geometry", "temporal", "atrous"]) {
            std::fs::write(format!("{dir}/{file}.msl"), kernel_source(kernel)).expect("dump");
        }
    }
    // Each variant's name, pipelines and à-trous steps.
    let mine = || kernels.map(|k| create_pipeline(&device, k).expect("pipeline"));
    let mut variants: Vec<(String, Pipelines, &[u32])> = vec![
        (
            "this revision, medium".to_owned(),
            mine(),
            atrous_steps(RtQuality::Medium),
        ),
        (
            "this revision, high".to_owned(),
            mine(),
            atrous_steps(RtQuality::High),
        ),
    ];
    if let Ok(dirs) = std::env::var("DENOISE_BENCH_BASELINE") {
        for dir in dirs.split(':') {
            variants.push((dir.to_owned(), baseline(&device, dir), &ATROUS_STEPS));
        }
    }

    let (eye, vis) = scene();
    let (n, f) = (1.0f32, 10_000.0f32);
    let proj = [1.732 * 540.0 / 960.0, 1.732, f / (f - n), -n * f / (f - n)];
    let depths: Vec<u8> = eye
        .iter()
        .map(|&z| if z > 0.0 { proj[2] + proj[3] / z } else { 1.0 })
        .flat_map(f32::to_le_bytes)
        .collect();
    let halves: Vec<u8> = vis.iter().flat_map(|h| h.to_le_bytes()).collect();
    let lit = vis.iter().filter(|&&h| h != 0).count();
    let sky = eye.iter().filter(|&&z| z == 0.0).count();
    println!(
        "scene {W}x{H}: sky {:.0}%, lit samples {:.0}%",
        100.0 * sky as f64 / (W * H) as f64,
        100.0 * lit as f64 / (W * H) as f64
    );

    let depth = texture(&device, MTLPixelFormat::Depth32Float);
    upload(&device, &queue, &depth, &depths, W * 4);
    // What the rays write afresh each frame, restored into the raw buffer
    // the à-trous passes ping-pong through.
    let pristine = texture(&device, MTLPixelFormat::R16Float);
    upload(&device, &queue, &pristine, &halves, W * 2);
    let raw_vis = texture(&device, MTLPixelFormat::R16Float);
    let geometry = [0, 1].map(|_| texture(&device, MTLPixelFormat::RGBA16Float));
    let values = [0, 1].map(|_| texture(&device, MTLPixelFormat::R16Float));
    let lengths = [0, 1].map(|_| texture(&device, MTLPixelFormat::RGBA16Float));
    let out = texture(&device, MTLPixelFormat::R16Float);
    let history = HistoryTextures {
        value: [raw(&values[0]), raw(&values[1])],
        length: [raw(&lengths[0]), raw(&lengths[1])],
    };
    let viewport = [0.0, 0.0, W as f32, H as f32];
    let camera = FrameCamera {
        view: IDENTITY,
        proj,
        viewport,
    };
    let scene_constants = SceneConstants {
        proj,
        viewport,
        extent: [W as u32, H as u32, 0, 0],
        sun: [0.0, -1.0, 0.0, 1.0],
        split_far: [0.0; 4],
        blend_start: [0.0; 4],
        texel: [0.0; 4],
        pcf_radius: [0.0; 4],
        cascades: [[[0.0; 4]; 4]; 4],
    };

    // One frame's commands through `pipelines`: the geometry, then the
    // denoise passes at `steps`; each dispatch's description boxed so its
    // address holds.
    let frame_commands = |frame: usize,
                          (pipelines, steps): (&Pipelines, &[u32])|
     -> (Vec<BlitCommand>, Vec<Box<SceneDispatch>>) {
        let history_frame = HistoryFrame {
            current: frame & 1,
            previous: (frame & 1) ^ 1,
            frame: frame as u32,
            reproject: IDENTITY,
            previous_camera: camera,
        };
        let constants = RtConstants {
            scene: scene_constants,
            trace: [frame as u32, 1, 1, kind.cap()],
            params: [0.01, f, if frame == 0 { 0.0 } else { 1.0 }, 0.0],
            prev_proj: proj,
            prev_viewport: viewport,
            reproject: IDENTITY,
            light: [0.0; 4],
            ambient: [0.0; 4],
            sky: [0.0; 4],
            points: [[0.0; 4]; 4],
        };
        let mut boxes = Vec::new();
        let mut commands = Vec::new();
        let mut push = |kernel: Kernel, pipeline: u64, textures: [u64; 8], step: u32| {
            let mut constants = constants;
            constants.params[3] = step as f32;
            let dispatch = Box::new(SceneDispatch {
                constants,
                textures: [textures[4], textures[5], textures[6], textures[7]],
                trace: SceneTrace::default(),
            });
            commands.push(BlitCommand::scene_kernel(&SceneKernelInfo {
                kernel: kernel as u32,
                pipeline,
                textures: [textures[0], textures[1], textures[2], textures[3]],
                constants: core::ptr::from_ref(dispatch.as_ref()) as u64,
                width: W as u32,
                height: H as u32,
            }));
            boxes.push(dispatch);
        };
        let into = raw(&geometry[frame & 1]).raw();
        push(
            Kernel::Geometry,
            pipelines[0].raw(),
            [raw(&depth).raw(), into, 0, 0, 0, 0, 0, 0],
            0,
        );
        for (i, pass) in passes(
            kind,
            raw(&raw_vis),
            &history,
            [raw(&geometry[0]), raw(&geometry[1])],
            raw(&out),
            &history_frame,
            steps,
        )
        .enumerate()
        {
            push(
                pass.kernel,
                pipelines[1 + usize::from(i > 0)].raw(),
                pass.textures.map(MetalHandle::raw),
                pass.step,
            );
        }
        (commands, boxes)
    };

    // The command buffers to time: their variant and pass (`None` for a whole frame).
    let mut timed: Vec<(
        usize,
        Option<usize>,
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    )> = Vec::new();
    let rounds = WARM + 2 * TIMED;
    for frame in 0..rounds * variants.len() {
        let variant = frame % variants.len();
        let round = frame / variants.len();
        let restore = queue.commandBuffer().expect("command buffer");
        let blit = restore.blitCommandEncoder().expect("blit");
        // SAFETY: both textures are R16Float of the raster's size and alive until the copy completes.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &pristine,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: W, height: H, depth: 1 },
                &raw_vis,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();
        restore.commit();
        let (_, pipelines, steps) = &variants[variant];
        let (commands, _boxes) = frame_commands(round, (pipelines, steps));
        let split = round % 2 == 1;
        let mut cb = queue.commandBuffer().expect("command buffer");
        for (i, command) in commands.iter().enumerate() {
            assert!(encode(&cb, command, None), "pass {i} encodes");
            if split {
                cb.commit();
                timed.push((variant, Some(i), cb));
                cb = queue.commandBuffer().expect("command buffer");
            }
        }
        cb.commit();
        if !split {
            timed.push((variant, None, cb.clone()));
        }
        if frame + 1 == WARM * variants.len() {
            cb.waitUntilCompleted();
            timed.clear();
        }
    }
    let mut frame_ms = vec![Vec::new(); variants.len()];
    let mut pass_ms: Vec<Vec<Vec<f64>>> = vec![Vec::new(); variants.len()];
    for (variant, pass, cb) in timed {
        cb.waitUntilCompleted();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        let ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0;
        match pass {
            None => frame_ms[variant].push(ms),
            Some(i) => {
                let passes = &mut pass_ms[variant];
                if passes.len() <= i {
                    passes.resize(i + 1, Vec::new());
                }
                passes[i].push(ms);
            }
        }
    }
    for (variant, (name, pipelines, steps)) in variants.iter().enumerate() {
        println!(
            "{name}: denoise frame (geometry, temporal, à-trous) {:.3} ms",
            median(core::mem::take(&mut frame_ms[variant]))
        );
        let names = ["geometry".to_owned(), "temporal".to_owned()]
            .into_iter()
            .chain(steps.iter().map(|s| format!("à-trous step {s}")));
        for (pass, all) in names.zip(core::mem::take(&mut pass_ms[variant])) {
            println!("  {pass}: {:.3} ms alone", median(all));
        }
        for pipeline in pipelines {
            crate::metal::post::destroy_pipeline(pipeline.raw());
        }
    }
}
