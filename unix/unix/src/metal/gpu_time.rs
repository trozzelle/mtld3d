//! GPU execution time of one device's finished command buffers, for the perf grid.
//!
//! Each completion handler adds its buffer's `GPUEndTime - GPUStartTime`
//! under the buffer's role; the next `SubmitFrame` of the device moves the
//! sums into its output and leaves zero behind, so every buffer is reported
//! once, by whichever submission follows its completion. Nothing is read or
//! added while the perf target is off, and outside a `PERF=1` build the gate
//! is a constant, so the handlers carry no extra work.

use std::{
    sync::{
        Mutex,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
    time::Duration,
};

use mtld3d_shared::perf::{CommandBufferRole, GpuBusy, GpuStage, SubmitTimings, perf_enabled};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;
use strum::EnumCount;

use super::stage_time::PooledSamples;

/// One role's sums since the last drain.
struct RoleSums {
    ns: AtomicU64,
    buffers: AtomicU32,
}

/// A device's GPU busy time per command-buffer role, summed until a submission takes it.
///
/// Lives on the device record: the completion handlers that add to it
/// already hold the record, and the submissions of that device are what
/// drain it. Atomics because the handlers run on a Metal thread while the
/// submit thread drains. The two sums of a role are taken one after the
/// other, so a buffer finishing between them can land its time in one
/// report and its count in the next; the window totals still agree. The
/// ray-traced scene's stages ([`GpuStage`], `metal::stage_time`) add and
/// drain the same way, and their sample buffers wait here between
/// submissions.
pub struct GpuTime {
    roles: [RoleSums; CommandBufferRole::COUNT],
    stages: [RoleSums; GpuStage::COUNT],
    samples: Mutex<Vec<PooledSamples>>,
}

impl GpuTime {
    pub const fn new() -> Self {
        Self {
            roles: [const {
                RoleSums {
                    ns: AtomicU64::new(0),
                    buffers: AtomicU32::new(0),
                }
            }; CommandBufferRole::COUNT],
            stages: [const {
                RoleSums {
                    ns: AtomicU64::new(0),
                    buffers: AtomicU32::new(0),
                }
            }; GpuStage::COUNT],
            samples: Mutex::new(Vec::new()),
        }
    }

    /// Add a completed buffer's execution time under `role`; called from its completion handler.
    pub fn record(&self, role: CommandBufferRole, cb: &ProtocolObject<dyn MTLCommandBuffer>) {
        if !perf_enabled() {
            return;
        }
        if let Some(ns) = busy_ns(cb.GPUStartTime(), cb.GPUEndTime()) {
            self.add(role, ns);
        }
    }

    fn add(&self, role: CommandBufferRole, ns: u64) {
        let sums = &self.roles[role as usize];
        sums.ns.fetch_add(ns, Ordering::Relaxed);
        sums.buffers.fetch_add(1, Ordering::Relaxed);
    }

    /// Add one submission's stage times, each with the encoders behind it.
    pub fn add_stages(&self, stages: &[GpuBusy; GpuStage::COUNT]) {
        for (sums, busy) in self.stages.iter().zip(stages) {
            sums.ns.fetch_add(busy.ns, Ordering::Relaxed);
            sums.buffers.fetch_add(busy.buffers, Ordering::Relaxed);
        }
    }

    /// A parked stage sample buffer, if one is free.
    pub fn take_samples(&self) -> Option<PooledSamples> {
        self.samples.lock().ok()?.pop()
    }

    /// Park a stage sample buffer for a later submission.
    pub fn park_samples(&self, samples: PooledSamples) {
        if let Ok(mut pool) = self.samples.lock() {
            pool.push(samples);
        }
    }

    /// Move every role's and stage's sums into a submission's output, leaving zero behind.
    pub fn drain(&self, out: &mut SubmitTimings) {
        if perf_enabled() {
            self.take(out);
        }
    }

    fn take(&self, out: &mut SubmitTimings) {
        let sums = self.roles.iter().zip(out.gpu.iter_mut());
        for (sums, busy) in sums.chain(self.stages.iter().zip(out.stages.iter_mut())) {
            busy.ns = sums.ns.swap(0, Ordering::Relaxed);
            busy.buffers = sums.buffers.swap(0, Ordering::Relaxed);
        }
    }
}

impl Default for GpuTime {
    fn default() -> Self {
        Self::new()
    }
}

/// Nanoseconds between two host times in seconds, `None` for a buffer the GPU never ran.
///
/// Metal reports zero for a start it never reached, which is what an
/// aborted buffer can carry, and an end before the start is not a duration.
fn busy_ns(start: f64, end: f64) -> Option<u64> {
    if start <= 0.0 || end < start {
        return None;
    }
    let busy = Duration::try_from_secs_f64(end - start).ok()?;
    u64::try_from(busy.as_nanos()).ok()
}

#[cfg(test)]
mod tests;
