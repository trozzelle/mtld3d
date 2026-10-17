use std::sync::Arc;

use mtld3d_shared::command_header::COMMAND_HEADER_BYTES;

use super::*;
use crate::{
    encoder_data::{BeginVisibilityOp, SetViewportOp},
    visibility::VisibilityQueryCore,
};

fn first_payload<T: CommandRecord>(scratch: &mut ScratchArena) -> &T {
    let (address, length) = scratch.command_ranges().next().unwrap();
    assert!(length >= COMMAND_HEADER_BYTES as u64 + size_of::<T>() as u64);
    // SAFETY: the test retains the recorder's scratch arena and this exact initialized
    // command region. Its fixed header precedes a payload with the asserted size.
    let payload = unsafe {
        core::slice::from_raw_parts(
            (address + COMMAND_HEADER_BYTES as u64) as *const u8,
            size_of::<T>(),
        )
    };
    crate::encoder_records::borrow(payload).unwrap()
}

#[test]
fn typed_viewport_is_constructed_in_final_command_storage() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    recorder
        .record_typed(
            &mut scratch,
            SetViewportOp {
                x: 7,
                y: 9,
                width: 640,
                height: 480,
                min_z: 0.25,
                max_z: 0.75,
            },
        )
        .unwrap();
    let record = first_payload::<SetViewportRecord>(&mut scratch);
    assert_eq!(
        (record.x, record.y, record.width, record.height),
        (7, 9, 640, 480)
    );
    assert_eq!((record.min_z, record.max_z), (0.25, 0.75));
    assert_eq!(recorder.len(), 1);
    assert!(recorder.rejected_ops.is_empty());
}

#[test]
fn typed_query_retains_original_owner_and_captured_generation() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    recorder
        .record_typed(
            &mut scratch,
            BeginVisibilityOp {
                generation: 37,
                c: query,
            },
        )
        .unwrap();
    assert_eq!(first_payload::<QueryRecord>(&mut scratch).generation, 37);
    assert_eq!(recorder.queries.len(), 1);
    assert!(weak.upgrade().is_some());
    assert!(recorder.rejected_ops.is_empty());
    // SAFETY: this fixture was never published and no native reader can adopt the lease.
    unsafe { recorder.queries[0].cancel_unadopted() };
    drop(recorder);
    assert!(weak.upgrade().is_none());
}

#[test]
fn latched_failure_retains_typed_owner_without_publishing_command() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    recorder.error = Some(WireError::AllocationFailed);
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    assert!(
        recorder
            .record_typed(
                &mut scratch,
                BeginVisibilityOp {
                    generation: 9,
                    c: query
                }
            )
            .is_err()
    );
    assert!(scratch.command_ranges().next().is_none());
    assert!(recorder.queries.is_empty());
    assert_eq!(recorder.rejected_ops.len(), 1);
    assert!(weak.upgrade().is_some());
    drop(recorder);
    assert!(weak.upgrade().is_none());
}

#[test]
fn post_boundary_codec_roundtrip() {
    use crate::{encoder_data::PostBoundary, encoder_records::PostBoundaryRecord};
    for sun in [Some([0.0, 0.6, -0.8]), None] {
        let mut scratch = ScratchArena::new();
        let mut recorder = FrameRecorder::new();
        let op = PostBoundary {
            proj: [1.5, 2.0, 1.001, -0.1],
            sun_dir_eye: sun,
            sun_color: [0.9, 0.8, 0.7],
            sun_above_horizon: sun.is_some(),
        };
        recorder.record_typed(&mut scratch, op).unwrap();
        let record = first_payload::<PostBoundaryRecord>(&mut scratch);
        assert_eq!(record.proj, op.proj);
        assert_eq!(record.sun_color, op.sun_color);
        assert_eq!(record.has_sun, u32::from(sun.is_some()));
        assert_eq!(record.sun_dir_eye, sun.unwrap_or([0.0; 3]));
        assert_eq!(record.sun_above_horizon, u32::from(sun.is_some()));
        assert_eq!(PostBoundary::from_record(record), Ok(op));
        assert_eq!(recorder.len(), 1);
        assert!(recorder.rejected_ops.is_empty());
    }
}

#[test]
fn post_boundary_record_rejects_a_bad_sun_flag() {
    use crate::{encoder_data::PostBoundary, encoder_records::PostBoundaryRecord};
    let record = PostBoundaryRecord {
        proj: [1.0; 4],
        sun_dir_eye: [0.0; 3],
        has_sun: 2,
        sun_color: [0.0; 3],
        sun_above_horizon: 0,
    };
    assert_eq!(
        PostBoundary::from_record(&record),
        Err(mtld3d_shared::encoder_wire::WireError::InvalidValue)
    );
    let record = PostBoundaryRecord {
        has_sun: 1,
        sun_above_horizon: 2,
        ..record
    };
    assert_eq!(
        PostBoundary::from_record(&record),
        Err(mtld3d_shared::encoder_wire::WireError::InvalidValue)
    );
}
