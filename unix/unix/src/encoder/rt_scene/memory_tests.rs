//! What the scene allocates: nothing in a steady frame, frame slots trimmed back to their frames,
//! refit tracks retired once idle.

use std::sync::atomic::AtomicU64;

use mtld3d_core::ids::BufferId;
use mtld3d_shared::mtl::DestroyKind;

use super::{
    TRIM_FRAMES,
    capture_tests::{
        BONES, Figure, figure_constants, moved, prebuild, quad, raw, trace_frame, vertex_bytes,
    },
    tests::{
        Draw, Sources, apply, build, build_with_capture, capture, device_buffer, length,
        next_frame, probe, quad_list, rt_encoder, run_blits, scale_move, staged_buffer,
    },
    tracks::TRACK_IDLE_FRAMES,
};
use crate::encoder::{
    FrameEncoder, replay::tests::Gpu, scene_capture::CapturedIndex, tests::post_reset_frame,
};

/// A figure no frame draws for [`TRACK_IDLE_FRAMES`] submissions retires
/// its track's acceleration structures behind the GPU, and its bytes leave
/// the frames' count; until then they stay for its return.
#[test]
fn idle_tracks_retire() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let figure = Figure::new(&mut enc);
    let q = quad();
    let vb = gpu.buffer(&vertex_bytes(&[q[0], q[1], q[2], q[1], q[3], q[2]]));
    let d = figure.capture(
        &mut enc,
        &Figure::snapshot(0, 0),
        (0x91, raw(&vb)),
        CapturedIndex::None {
            start_vertex: 0,
            vertex_count: 6,
        },
        &figure_constants(&BONES),
    );
    prebuild(&mut enc, &d);
    let (tlas, _) = trace_frame(&mut enc, &gpu, &[d]);
    let track = tlas.resident[0];
    let with_track = enc.rt_perf().slot_bytes;
    let id = BufferId::from_raw(0x7500);
    staged_buffer(&mut enc, &gpu.device, id, &quad_list());
    let world_view = scale_move([1.0, 1.0, 1.0], [0.0, 0.0, 4.0]);
    let retired = |enc: &FrameEncoder| {
        enc.pending_resource_retention
            .iter()
            .any(|e| e.kind == DestroyKind::AccelerationStructure && e.handle == track)
    };
    let mut bytes = 0;
    for n in 1..=TRACK_IDLE_FRAMES + 1 {
        next_frame(&mut enc, &coherent);
        let draw = Draw::list(device_buffer(&enc, id), enc.buffer_content(id), world_view);
        let still = capture(&mut enc, &sources, &draw);
        build(&mut enc, &[still]);
        let now = enc.rt_perf().slot_bytes;
        if n <= TRACK_IDLE_FRAMES {
            assert!(!retired(&enc), "frame {n}: kept for the figure's return");
            bytes = now;
        } else {
            assert!(retired(&enc), "idle for {TRACK_IDLE_FRAMES} frames");
            assert!(now < bytes, "its bytes leave the count: {now} of {bytes}");
        }
    }
    assert!(with_track > 0);
    drop(enc);
}

/// The next submission with the GPU still on the earlier ones: its frame takes another slot.
fn next_frame_in_flight(enc: &mut FrameEncoder) {
    enc.current_submit_seq += 1;
    post_reset_frame(enc, (64, 64), 1, true);
    enc.scene.begin_submission(false);
}

/// Once each frame slot has been used, a still scene with a cached static
/// draw and a moving captured figure makes no acceleration structure and
/// no slot grows, whether the GPU keeps up (one slot) or frames are in
/// flight (four slots in turn).
#[test]
fn steady_frames_make_no_acceleration_structures() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let figure = Figure::new(&mut enc);
    let q = quad();
    let vb = gpu.buffer(&vertex_bytes(&[q[0], q[1], q[2], q[1], q[3], q[2]]));
    let id = BufferId::from_raw(0x7300);
    staged_buffer(&mut enc, &gpu.device, id, &quad_list());
    let world_view = scale_move([1.0, 1.0, 1.0], [-6.0, 0.0, 6.0]);
    let mut n = 0.0f32;
    let mut frame = |enc: &mut FrameEncoder| {
        n += 1.0;
        let draw = Draw::list(device_buffer(enc, id), enc.buffer_content(id), world_view);
        let still = capture(enc, &sources, &draw);
        let figure = figure.capture(
            enc,
            &Figure::snapshot(0, 0),
            (0x91, raw(&vb)),
            CapturedIndex::None {
                start_vertex: 0,
                vertex_count: 6,
            },
            &figure_constants(&moved(BONES, [0.01 * n, 0.0, 0.0])),
        );
        prebuild(enc, &figure);
        enc.scene.note_boundary(&super::capture_tests::boundary());
        let (tlas, blits, _) = build_with_capture(enc, &[still, figure]);
        run_blits(&gpu, &blits);
        let perf = enc.rt_perf();
        (tlas.expect("a TLAS"), perf.as_allocs, perf.slot_bytes)
    };

    let (_, made, _) = frame(&mut enc);
    assert!(
        made >= 3,
        "the first frame makes its BLASes and TLAS: {made}"
    );
    // Three frames in flight: the other three slots, each made once.
    for _ in 0..3 {
        next_frame_in_flight(&mut enc);
        frame(&mut enc);
    }
    // One frame more lets the static BLAS's compaction land.
    next_frame(&mut enc, &coherent);
    frame(&mut enc);
    let (_, _, bytes) = {
        next_frame(&mut enc, &coherent);
        frame(&mut enc)
    };
    // Ten frames: the figure's chain refits throughout (it builds again after `REFIT_FRAMES`).
    for step in 0..10 {
        if step % 3 == 0 {
            next_frame(&mut enc, &coherent);
        } else {
            next_frame_in_flight(&mut enc);
        }
        let (tlas, made, now) = frame(&mut enc);
        assert_eq!(made, 0, "step {step} makes no acceleration structure");
        assert_eq!(now, bytes, "step {step}: the slots hold what they held");
        assert_eq!(tlas.instances.len(), 2);
    }
    let centre = apply(&world_view, super::tests::QUAD_CENTRE);
    let (tlas, _, _) = {
        next_frame(&mut enc, &coherent);
        frame(&mut enc)
    };
    let (distance, _) = probe(&gpu, &tlas, &[centre])[0].expect("the static quad");
    assert!((distance - length(centre)).abs() < 1e-3);
    drop(enc);
}

/// A frame slot keeps the length its busiest frame needed until
/// [`TRIM_FRAMES`] submissions have passed, then shrinks to what its frames
/// since asked of it; slots no frame used since are released. The scene
/// still traces after.
#[test]
fn slot_objects_trim_to_their_frames() {
    const CROWD: usize = 600;
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let id = BufferId::from_raw(0x7400);
    staged_buffer(&mut enc, &gpu.device, id, &quad_list());
    let at = |i: usize| {
        #[expect(clippy::cast_precision_loss, reason = "a small index")]
        let x = i as f32;
        scale_move([1.0, 1.0, 1.0], [x * 1.5, 0.0, 8.0])
    };
    let frame = |enc: &mut FrameEncoder, count: usize| {
        let draws: Vec<_> = (0..count)
            .map(|i| {
                let draw = Draw::list(device_buffer(enc, id), enc.buffer_content(id), at(i));
                capture(enc, &sources, &draw)
            })
            .collect();
        let (tlas, blits, _) = build_with_capture(enc, &draws);
        run_blits(&gpu, &blits);
        (tlas.expect("a TLAS"), enc.rt_perf().slot_bytes)
    };

    // A crowd in one slot, a second slot used once while it is in flight.
    let (_, crowded) = frame(&mut enc, CROWD);
    next_frame_in_flight(&mut enc);
    frame(&mut enc, 1);
    next_frame(&mut enc, &coherent);
    let (_, held) = frame(&mut enc, 1);
    assert!(held >= crowded, "nothing shrinks before the window ends");
    // The first window holds the crowd: its trim keeps what it needed.
    while enc.current_submit_seq < 2 * TRIM_FRAMES - 1 {
        next_frame(&mut enc, &coherent);
        let (_, now) = frame(&mut enc, 1);
        assert_eq!(now, held, "seq {}: still held", enc.current_submit_seq);
    }
    // The second holds one instance a frame, and nothing at all in the second slot.
    next_frame(&mut enc, &coherent);
    let (tlas, trimmed) = frame(&mut enc, 1);
    assert!(
        trimmed * 4 <= held,
        "the window's frames needed one instance: {trimmed} of {held} bytes"
    );
    let centre = apply(&at(0), super::tests::QUAD_CENTRE);
    let (distance, _) = probe(&gpu, &tlas, &[centre])[0].expect("the quad");
    assert!((distance - length(centre)).abs() < 1e-3);
    // Steady again: the trimmed slot holds.
    next_frame(&mut enc, &coherent);
    let (_, again) = frame(&mut enc, 1);
    assert_eq!(again, trimmed);
    drop(enc);
}

/// A trim that remakes the TLAS the chain refits from ends the chain, so the
/// next frame builds its TLAS in full — even when Metal hands the new
/// object the old one's address, which the chain's handle check alone
/// cannot tell apart (refitting from it would read an object that never
/// held a TLAS: a GPU fault).
#[test]
fn trimming_the_chains_tlas_ends_the_chain() {
    const CROWD: usize = 600;
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let id = BufferId::from_raw(0x7600);
    staged_buffer(&mut enc, &gpu.device, id, &quad_list());
    let at = |i: usize| {
        #[expect(clippy::cast_precision_loss, reason = "a small index")]
        let x = i as f32;
        scale_move([1.0, 1.0, 1.0], [x * 1.5, 0.0, 8.0])
    };
    let frame = |enc: &mut FrameEncoder, count: usize| {
        let draws: Vec<_> = (0..count)
            .map(|i| {
                let draw = Draw::list(device_buffer(enc, id), enc.buffer_content(id), at(i));
                capture(enc, &sources, &draw)
            })
            .collect();
        let (tlas, blits, _) = build_with_capture(enc, &draws);
        run_blits(&gpu, &blits);
        tlas.expect("a TLAS")
    };

    // A crowd grows slot 0's TLAS; its window's peaks start over, so the
    // next frame's single instance is all the slot is asked for.
    frame(&mut enc, CROWD);
    for (grown, ..) in enc.rt.as_mut().expect("the scene").slots[0].objects() {
        grown.peak = 0;
    }
    next_frame(&mut enc, &coherent);
    let chained = frame(&mut enc, 1).tlas;
    {
        let rt = enc.rt.as_ref().expect("the scene");
        assert_eq!(rt.tlas_chain.slot, Some(0));
        assert_eq!(rt.tlas_chain.handle, chained);
    }

    // The trim remakes that TLAS smaller.
    next_frame(&mut enc, &coherent);
    let mut rt = enc.rt.take().expect("the scene");
    let device = enc.device.clone();
    let (settled, _) = enc.upload_gate().expect("the GPU's progress");
    let seq = enc.current_submit_seq;
    rt.trim_slots(&mut enc, &device, (seq, settled));
    let remade = rt.slots[0].tlas.handle;
    assert_ne!(remade, chained, "the trim remade the chain's TLAS");
    assert_eq!(rt.tlas_chain.slot, None, "the trim ends the chain");
    // Metal may hand the new TLAS the old one's address.
    rt.tlas_chain.handle = remade;
    enc.rt = Some(rt);

    let tlas = frame(&mut enc, 1);
    assert_eq!(tlas.tlas, remade, "the frame keeps the trimmed TLAS");
    let rt = enc.rt.as_ref().expect("the scene");
    assert_eq!(
        (rt.tlas_chain.built, rt.tlas_chain.refits),
        (enc.current_submit_seq, 0),
        "the frame after the trim builds its TLAS in full"
    );
    let centre = apply(&at(0), super::tests::QUAD_CENTRE);
    let (distance, _) = probe(&gpu, &tlas, &[centre])[0].expect("the quad");
    assert!((distance - length(centre)).abs() < 1e-3);
    drop(enc);
}
