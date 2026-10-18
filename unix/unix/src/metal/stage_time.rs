//! GPU time of the ray-traced scene's stages, for the perf grid.
//!
//! A submission's [`StageTimer`] holds one counter sample buffer from its
//! device's pool ([`GpuTime`]). Every encoder of a timed stage
//! ([`GpuStage`]) is made with a pass descriptor that samples the GPU
//! timestamp where the encoder starts and ends; the frame buffer's
//! completion handler resolves the samples, adds each pair's span to its
//! stage and puts the buffer back. Apple GPUs report the timestamps in
//! nanoseconds. Encoders without a dependency between them can overlap, so
//! a stage's sum is the busy time of its encoders, not wall time.
//!
//! Nothing is made unless the perf target is on and the device samples at
//! stage boundaries; outside a `PERF=1` build the gate is a constant, and
//! every encoder is made exactly as without a timer.

use core::ptr::NonNull;

use mtld3d_shared::perf::{GpuBusy, GpuStage, perf_enabled};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLAccelerationStructureCommandEncoder, MTLAccelerationStructurePassDescriptor,
    MTLCommandBuffer, MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder,
    MTLComputePassDescriptor, MTLCounterDontSample, MTLCounterErrorValue, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice,
    MTLDispatchType, MTLRenderPassDescriptor, MTLStorageMode,
};
use strum::EnumCount;

use super::gpu_time::GpuTime;
use crate::LOG_TARGET;

#[cfg(test)]
mod tests;

/// Encoders one submission can time; later ones go untimed.
pub const STAGE_SAMPLE_PAIRS: usize = 32;

/// A counter sample buffer's canonical retain, parked in a device's pool between submissions.
pub struct PooledSamples(NonNull<ProtocolObject<dyn MTLCounterSampleBuffer>>);

// SAFETY: the pool's lock, or the one submission that took the buffer, is
// its only user at a time; Metal objects may move between threads.
unsafe impl Send for PooledSamples {}

impl PooledSamples {
    fn new(buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>) -> Self {
        Self(NonNull::new(Retained::into_raw(buffer)).expect("a retained object is never null"))
    }

    fn into_retained(self) -> Retained<ProtocolObject<dyn MTLCounterSampleBuffer>> {
        let raw = self.0.as_ptr();
        core::mem::forget(self);
        // SAFETY: the canonical retain `new` took, handed back once.
        unsafe { Retained::from_raw(raw) }.expect("a pooled buffer is never null")
    }
}

impl Drop for PooledSamples {
    fn drop(&mut self) {
        // SAFETY: the canonical retain `new` took; nothing else releases it.
        drop(unsafe { Retained::from_raw(self.0.as_ptr()) });
    }
}

/// One submission's stage timestamps: the buffer they land in, and the stage of each pair.
pub struct StageTimer {
    buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    /// Pair `i` (samples `2i` and `2i + 1`) times an encoder of `stages[i]`.
    stages: Vec<GpuStage>,
}

impl StageTimer {
    /// A timer for one submission on `device`, its buffer from `time`'s pool;
    /// `None` unless the perf target is on and the device samples at stage
    /// boundaries.
    #[must_use]
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, time: &GpuTime) -> Option<Self> {
        if !perf_enabled() {
            return None;
        }
        Self::make(device, time)
    }

    /// [`Self::new`] whatever the perf target says.
    fn make(device: &ProtocolObject<dyn MTLDevice>, time: &GpuTime) -> Option<Self> {
        let buffer = time
            .take_samples()
            .map(PooledSamples::into_retained)
            .or_else(|| sample_buffer(device))?;
        Some(Self {
            buffer,
            stages: Vec::with_capacity(STAGE_SAMPLE_PAIRS),
        })
    }

    /// The sample indices of the next encoder of `stage`, `None` once every pair is taken.
    fn pair(&mut self, stage: GpuStage) -> Option<(usize, usize)> {
        if self.stages.len() >= STAGE_SAMPLE_PAIRS {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "perf: a submission has more than {STAGE_SAMPLE_PAIRS} timed scene encoders; \
                 later ones go untimed");
            return None;
        }
        let first = 2 * self.stages.len();
        self.stages.push(stage);
        Some((first, first + 1))
    }

    /// Sample the vertex stage of the render pass `descriptor` describes under `stage`.
    pub fn time_render_pass(&mut self, descriptor: &MTLRenderPassDescriptor, stage: GpuStage) {
        let Some((start, end)) = self.pair(stage) else {
            return;
        };
        // SAFETY: index 0 of the descriptor's attachment array always exists.
        let attachment = unsafe {
            descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0)
        };
        attachment.setSampleBuffer(Some(&self.buffer));
        // SAFETY: both indices lie within the buffer's samples, the others are unsampled.
        unsafe {
            attachment.setStartOfVertexSampleIndex(start);
            attachment.setEndOfVertexSampleIndex(end);
            attachment.setStartOfFragmentSampleIndex(MTLCounterDontSample);
            attachment.setEndOfFragmentSampleIndex(MTLCounterDontSample);
        }
    }

    /// Each pair's span, added to its stage: `(ns, encoders)` per stage.
    #[must_use]
    pub fn resolve(&self) -> [GpuBusy; GpuStage::COUNT] {
        let mut out = [const { GpuBusy::new() }; GpuStage::COUNT];
        if self.stages.is_empty() {
            return out;
        }
        let range = NSRange::new(0, 2 * self.stages.len());
        // SAFETY: the range lies within the buffer's samples.
        let Some(data) = (unsafe { self.buffer.resolveCounterRange(range) }) else {
            return out;
        };
        let bytes = data.to_vec();
        let samples: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")))
            .collect();
        for (stage, pair) in self.stages.iter().zip(samples.chunks_exact(2)) {
            if let Some(ns) = span(pair[0], pair[1]) {
                let busy = &mut out[*stage as usize];
                busy.ns = busy.ns.saturating_add(ns);
                busy.buffers += 1;
            }
        }
        out
    }

    /// Add the spans to `time` and put the buffer back in its pool; from the frame buffer's completion handler.
    pub fn finish(self, time: &GpuTime) {
        time.add_stages(&self.resolve());
        time.park_samples(PooledSamples::new(self.buffer));
    }
}

/// The span between two timestamps, `None` for a sample the GPU did not take.
const fn span(start: u64, end: u64) -> Option<u64> {
    if start == MTLCounterErrorValue || end == MTLCounterErrorValue || start == 0 || end < start {
        return None;
    }
    Some(end - start)
}

/// A Shared sample buffer of the device's timestamp counter set, `None` where it samples none at stage boundaries.
fn sample_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> {
    if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
        mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "perf: the device samples no counters at stage boundaries; the scene's stages go \
             untimed");
        return None;
    }
    // SAFETY: a static string constant of the framework.
    let wanted: &NSString = unsafe { MTLCommonCounterSetTimestamp };
    let sets = device.counterSets()?;
    let set = sets.iter().find(|set| set.name().isEqualToString(wanted))?;
    let descriptor = MTLCounterSampleBufferDescriptor::new();
    descriptor.setCounterSet(Some(&set));
    descriptor.setStorageMode(MTLStorageMode::Shared);
    descriptor.setLabel(&NSString::from_str("mtld3d-scene-stage-times"));
    // SAFETY: a count the device's sample buffers allow.
    unsafe { descriptor.setSampleCount(2 * STAGE_SAMPLE_PAIRS) };
    device
        .newCounterSampleBufferWithDescriptor_error(&descriptor)
        .map_err(|e| {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "perf: the scene's stage timer could not be made: {e}");
        })
        .ok()
}

/// A compute encoder on `cb` dispatching as `dispatch`, timed under `stage` when `timer` is.
#[must_use]
pub fn compute_encoder(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    timer: Option<&mut StageTimer>,
    stage: GpuStage,
    dispatch: MTLDispatchType,
) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    let Some((timer, (start, end))) = timer.and_then(|t| t.pair(stage).map(|pair| (t, pair)))
    else {
        return cb.computeCommandEncoderWithDispatchType(dispatch);
    };
    let descriptor = MTLComputePassDescriptor::computePassDescriptor();
    descriptor.setDispatchType(dispatch);
    // SAFETY: index 0 of the descriptor's attachment array always exists.
    let attachment = unsafe {
        descriptor
            .sampleBufferAttachments()
            .objectAtIndexedSubscript(0)
    };
    attachment.setSampleBuffer(Some(&timer.buffer));
    // SAFETY: both indices lie within the buffer's samples.
    unsafe {
        attachment.setStartOfEncoderSampleIndex(start);
        attachment.setEndOfEncoderSampleIndex(end);
    }
    cb.computeCommandEncoderWithDescriptor(&descriptor)
}

/// An acceleration-structure encoder on `cb`, timed under `stage` when `timer` is.
#[must_use]
pub fn accel_encoder(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    timer: Option<&mut StageTimer>,
    stage: GpuStage,
) -> Option<Retained<ProtocolObject<dyn MTLAccelerationStructureCommandEncoder>>> {
    let Some((timer, (start, end))) = timer.and_then(|t| t.pair(stage).map(|pair| (t, pair)))
    else {
        return cb.accelerationStructureCommandEncoder();
    };
    let descriptor = MTLAccelerationStructurePassDescriptor::accelerationStructurePassDescriptor();
    // SAFETY: index 0 of the descriptor's attachment array always exists.
    let attachment = unsafe {
        descriptor
            .sampleBufferAttachments()
            .objectAtIndexedSubscript(0)
    };
    attachment.setSampleBuffer(Some(&timer.buffer));
    // SAFETY: both indices lie within the buffer's samples.
    unsafe {
        attachment.setStartOfEncoderSampleIndex(start);
        attachment.setEndOfEncoderSampleIndex(end);
    }
    Some(cb.accelerationStructureCommandEncoderWithDescriptor(&descriptor))
}
