use mtld3d_shared::perf::{CommandBufferRole, GpuBusy, GpuStage, SubmitTimings};
use strum::EnumCount;

use super::{GpuTime, busy_ns};

/// A buffer's host times become whole nanoseconds of GPU execution.
#[test]
fn busy_time_is_end_minus_start_in_nanoseconds() {
    assert_eq!(busy_ns(10.0, 10.25), Some(250_000_000));
    assert_eq!(busy_ns(1.5, 1.5), Some(0));
}

/// A buffer that never started, or whose end precedes its start, reports nothing.
#[test]
fn unstarted_or_inverted_buffers_report_nothing() {
    assert_eq!(busy_ns(0.0, 4.0), None);
    assert_eq!(busy_ns(5.0, 4.0), None);
    assert_eq!(busy_ns(f64::NAN, 4.0), None);
}

/// Sums accumulate per role, and a take hands them over once and leaves zero behind.
#[test]
fn take_hands_each_sum_over_once() {
    let time = GpuTime::new();
    time.add(CommandBufferRole::Frame, 3_000);
    time.add(CommandBufferRole::Frame, 4_000);
    time.add(CommandBufferRole::Present, 500);

    let mut first = SubmitTimings::new();
    time.take(&mut first);
    let mut second = SubmitTimings::new();
    time.take(&mut second);

    let frame = &first.gpu[CommandBufferRole::Frame as usize];
    let upload = &first.gpu[CommandBufferRole::Upload as usize];
    let present = &first.gpu[CommandBufferRole::Present as usize];
    assert_eq!((frame.ns, frame.buffers), (7_000, 2));
    assert_eq!((upload.ns, upload.buffers), (0, 0));
    assert_eq!((present.ns, present.buffers), (500, 1));
    assert!(
        second
            .gpu
            .iter()
            .all(|busy| busy.ns == 0 && busy.buffers == 0)
    );
}

/// Stage times add per stage with their encoders, and a take hands them over once beside the roles.
#[test]
fn stage_times_add_and_hand_over_once() {
    let time = GpuTime::new();
    let mut frame = [const { GpuBusy::new() }; GpuStage::COUNT];
    frame[GpuStage::RtRays as usize] = GpuBusy {
        ns: 2_000,
        buffers: 1,
        pad0: 0,
    };
    frame[GpuStage::RtBuild as usize] = GpuBusy {
        ns: 500,
        buffers: 2,
        pad0: 0,
    };
    time.add_stages(&frame);
    time.add_stages(&frame);
    time.add(CommandBufferRole::Frame, 9_000);

    let mut first = SubmitTimings::new();
    time.take(&mut first);
    let rays = &first.stages[GpuStage::RtRays as usize];
    let build = &first.stages[GpuStage::RtBuild as usize];
    let tlas = &first.stages[GpuStage::RtTlas as usize];
    assert_eq!((rays.ns, rays.buffers), (4_000, 2));
    assert_eq!((build.ns, build.buffers), (1_000, 4));
    assert_eq!((tlas.ns, tlas.buffers), (0, 0));
    assert_eq!(first.gpu[CommandBufferRole::Frame as usize].ns, 9_000);
    let mut second = SubmitTimings::new();
    time.take(&mut second);
    assert!(second.stages.iter().all(|s| s.ns == 0 && s.buffers == 0));
}
