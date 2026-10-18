use mtld3d_shared::perf::GpuStage;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLAccelerationStructureCommandEncoder, MTLAccelerationStructureTriangleGeometryDescriptor,
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCounterErrorValue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDispatchType, MTLLibrary, MTLPrimitiveAccelerationStructureDescriptor, MTLResourceOptions,
    MTLSize,
};

use super::{STAGE_SAMPLE_PAIRS, StageTimer, accel_encoder, compute_encoder, span};
use crate::metal::gpu_time::GpuTime;

/// A pair the GPU did not sample, or whose end precedes its start, is no span.
#[test]
fn unsampled_pairs_are_no_span() {
    assert_eq!(span(100, 350), Some(250));
    assert_eq!(span(MTLCounterErrorValue, 350), None);
    assert_eq!(span(100, MTLCounterErrorValue), None);
    assert_eq!(span(0, 350), None);
    assert_eq!(span(400, 350), None);
}

const SPIN: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void spin(device float *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {
    float x = float(t);
    for (int i = 0; i < 2000; ++i) { x = metal::sin(x) * 1.0001 + 0.5; }
    out[t] = x;
}
";

/// The encoders a timer makes sample their own start and end: a compute
/// encoder's span lands under its stage, an acceleration-structure
/// encoder's under its own, the untimed stages stay at zero, and the
/// finished timer parks its buffer for the next submission. Past the
/// buffer's pairs an encoder is made untimed.
#[test]
fn timed_encoders_land_under_their_stages() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("no Metal device, skipping");
        return;
    };
    let time = GpuTime::new();
    let Some(mut timer) = StageTimer::make(&device, &time) else {
        eprintln!("the device samples no counters at stage boundaries, skipping");
        return;
    };
    let queue = device.newCommandQueue().expect("a queue");
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SPIN), None)
        .expect("the kernel compiles");
    let function = library
        .newFunctionWithName(&NSString::from_str("spin"))
        .expect("spin");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("the pipeline");
    let out = device
        .newBufferWithLength_options(4 << 16, MTLResourceOptions::StorageModePrivate)
        .expect("a buffer");
    let cb = queue.commandBuffer().expect("a command buffer");

    let compute = compute_encoder(
        &cb,
        Some(&mut timer),
        GpuStage::RtRays,
        MTLDispatchType::Serial,
    )
    .expect("a timed compute encoder");
    compute.setComputePipelineState(&pipeline);
    // SAFETY: slot 0 is the kernel's output, sized for the grid.
    unsafe { compute.setBuffer_offset_atIndex(Some(&out), 0, 0) };
    compute.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: 1 << 16,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    compute.endEncoding();

    let points: Vec<u8> = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    // SAFETY: `points` is a live slice, copied now.
    let vertices = unsafe {
        device.newBufferWithBytes_length_options(
            core::ptr::NonNull::from(points.as_slice()).cast(),
            points.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("vertices");
    let triangle = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
    triangle.setVertexBuffer(Some(&vertices));
    triangle.setVertexStride(12);
    triangle.setTriangleCount(1);
    let descriptor = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    descriptor.setGeometryDescriptors(Some(&objc2_foundation::NSArray::from_retained_slice(&[
        objc2::rc::Retained::into_super(triangle),
    ])));
    let sizes = device.accelerationStructureSizesWithDescriptor(&descriptor);
    let blas = device
        .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
        .expect("a BLAS");
    let scratch = device
        .newBufferWithLength_options(
            sizes.buildScratchBufferSize.max(16),
            MTLResourceOptions::StorageModePrivate,
        )
        .expect("scratch");
    let accel = accel_encoder(&cb, Some(&mut timer), GpuStage::RtBuild).expect("an AS encoder");
    accel.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
        &blas,
        &descriptor,
        &scratch,
        0,
    );
    accel.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);

    let stages = timer.resolve();
    let rays = &stages[GpuStage::RtRays as usize];
    let build = &stages[GpuStage::RtBuild as usize];
    assert_eq!(rays.buffers, 1);
    assert!(rays.ns > 0, "the dispatch took time");
    assert_eq!(build.buffers, 1);
    for stage in [GpuStage::RtCapture, GpuStage::RtTlas, GpuStage::RtDenoise] {
        assert_eq!(stages[stage as usize].buffers, 0);
    }

    for _ in 2..STAGE_SAMPLE_PAIRS {
        assert!(timer.pair(GpuStage::RtDenoise).is_some());
    }
    assert_eq!(timer.pair(GpuStage::RtDenoise), None, "every pair is taken");
    let untimed = queue.commandBuffer().expect("a command buffer");
    let encoder = accel_encoder(&untimed, Some(&mut timer), GpuStage::RtTlas)
        .expect("made untimed past the pairs");
    encoder.endEncoding();

    timer.finish(&time);
    assert!(time.take_samples().is_some(), "the buffer is parked");
    assert!(time.take_samples().is_none());
}
