//! The shared ray-tracing MSL on the GPU: the alpha comparison and the instance table's layout.

use core::{mem::offset_of, ptr::NonNull};

use mtld3d_types::{
    D3DCMP_ALWAYS, D3DCMP_EQUAL, D3DCMP_GREATER, D3DCMP_GREATEREQUAL, D3DCMP_LESS,
    D3DCMP_LESSEQUAL, D3DCMP_NEVER, D3DCMP_NOTEQUAL,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions, MTLSize,
};

use super::{RtInstanceInfo, create_rt_pipeline};

/// The system device when it traces rays.
fn device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    let device = MTLCreateSystemDefaultDevice()?;
    if !device.supportsRaytracing() {
        eprintln!("the device does not support ray tracing, skipping");
        return None;
    }
    Some(device)
}

fn shared(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[u8],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    // SAFETY: `data` is a live slice, copied now.
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::from(data).cast(),
            data.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("a buffer")
}

/// Run kernel `entry` of `kernels` over `threads` threads with `input` at
/// slot 0 and a zeroed `out_words`-word output at slot 1: the output.
fn run(
    kernels: &str,
    entry: &str,
    input: &[u8],
    threads: usize,
    out_words: usize,
) -> Option<Vec<u32>> {
    let device = device()?;
    let pipeline = create_rt_pipeline(&device, kernels, entry).expect("the kernel compiles");
    let input = shared(&device, input);
    let out = shared(&device, &vec![0u8; out_words * 4]);
    let queue = device.newCommandQueue().expect("queue");
    let cmd = queue.commandBuffer().expect("command buffer");
    let compute = cmd.computeCommandEncoder().expect("compute");
    compute.setComputePipelineState(&pipeline.pipeline);
    // SAFETY: the kernel's two buffers.
    unsafe {
        compute.setBuffer_offset_atIndex(Some(&input), 0, 0);
        compute.setBuffer_offset_atIndex(Some(&out), 0, 1);
    }
    let size = |width| MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    compute.dispatchThreads_threadsPerThreadgroup(size(threads), size(1));
    compute.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the completed dispatch wrote `out_words` words.
    let words =
        unsafe { core::slice::from_raw_parts(out.contents().as_ptr().cast::<u32>(), out_words) };
    Some(words.to_vec())
}

/// `rt_alpha_passes` compares as the raster's alpha test does for every
/// `D3DCMP_*` value: `NEVER` never passes, `0`, `ALWAYS` and an unknown
/// value always do, and the six comparisons test the texel's alpha against
/// the reference, equality exactly.
#[test]
fn alpha_func_table_matches_d3dcmp() {
    const KERNEL: &str = r"
kernel void cmp_table(device const float4 *cases [[buffer(0)]],
                      device uint *out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float4 c = cases[i];
    out[i] = rt_alpha_passes(uint(c.x), c.y, c.z) ? 1u : 0u;
}
";
    let reference = 128.0 / 255.0;
    let alphas = [0.0, 0.25, reference, 0.75, 1.0];
    let expect = |func: u32, a: f32| -> bool {
        match func {
            D3DCMP_NEVER => false,
            D3DCMP_LESS => a < reference,
            D3DCMP_EQUAL => a == reference,
            D3DCMP_LESSEQUAL => a <= reference,
            D3DCMP_GREATER => a > reference,
            D3DCMP_NOTEQUAL => a != reference,
            D3DCMP_GREATEREQUAL => a >= reference,
            _ => true,
        }
    };
    assert_eq!(D3DCMP_ALWAYS, 8);
    let mut cases = Vec::new();
    let mut expected = Vec::new();
    for func in 0..=9u32 {
        for &a in &alphas {
            #[expect(clippy::cast_precision_loss, reason = "small values")]
            cases.extend([func as f32, a, reference, 0.0]);
            expected.push(u32::from(expect(func, a)));
        }
    }
    let input: Vec<u8> = cases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let Some(out) = run(KERNEL, "cmp_table", &input, expected.len(), expected.len()) else {
        return;
    };
    for (i, (got, want)) in out.iter().zip(&expected).enumerate() {
        let (func, a) = (i / alphas.len(), alphas[i % alphas.len()]);
        assert_eq!(got, want, "func {func}, alpha {a} against {reference}");
    }
    // Every comparison is exercised both ways.
    for func in 2..=7 {
        let row = &out[func * alphas.len()..(func + 1) * alphas.len()];
        assert!(row.contains(&0) && row.contains(&1), "func {func}: {row:?}");
    }
}

/// The MSL `RtInstanceInfo` lays out field for field as the Rust one the scene writes.
#[test]
fn instance_info_layout_matches_msl() {
    const KERNEL: &str = r"
kernel void info_layout(device const RtInstanceInfo *info [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    device const uchar *base = (device const uchar *)info;
    out[0] = uint(sizeof(RtInstanceInfo));
    out[1] = uint((device const uchar *)&info->alpha_ref - base);
    out[2] = uint((device const uchar *)&info->alpha_func - base);
    out[3] = uint((device const uchar *)&info->uv_source - base);
    out[4] = uint((device const uchar *)&info->uv_buf - base);
    out[5] = uint((device const uchar *)&info->uv_offset - base);
    out[6] = uint((device const uchar *)&info->uv_stride - base);
    out[7] = uint((device const uchar *)&info->ib - base);
    out[8] = uint((device const uchar *)&info->index_type - base);
    out[9] = uint((device const uchar *)&info->mat_diffuse - base);
    out[10] = uint((device const uchar *)&info->pos_buf - base);
    out[11] = uint((device const uchar *)&info->pos_offset - base);
    out[12] = uint((device const uchar *)&info->pos_stride - base);
    out[13] = uint((device const uchar *)&info->to_eye - base);
}
";
    let rust = [
        size_of::<RtInstanceInfo>(),
        offset_of!(RtInstanceInfo, alpha_ref),
        offset_of!(RtInstanceInfo, alpha_func),
        offset_of!(RtInstanceInfo, uv_source),
        offset_of!(RtInstanceInfo, uv_buf),
        offset_of!(RtInstanceInfo, uv_offset),
        offset_of!(RtInstanceInfo, uv_stride),
        offset_of!(RtInstanceInfo, ib),
        offset_of!(RtInstanceInfo, index_type),
        offset_of!(RtInstanceInfo, mat_diffuse),
        offset_of!(RtInstanceInfo, pos_buf),
        offset_of!(RtInstanceInfo, pos_offset),
        offset_of!(RtInstanceInfo, pos_stride),
        offset_of!(RtInstanceInfo, to_eye),
    ];
    assert_eq!(size_of::<RtInstanceInfo>(), 136);
    let Some(out) = run(KERNEL, "info_layout", &[0u8; 136], 1, rust.len()) else {
        return;
    };
    let out: Vec<usize> = out.iter().map(|&w| w as usize).collect();
    assert_eq!(out, rust);
}
