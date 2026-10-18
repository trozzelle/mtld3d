//! Refit tracks: captured figures and animated static geometry refit from the previous frame's BLAS.

use std::sync::atomic::{AtomicU64, Ordering};

use mtld3d_core::ids::BufferId;
use mtld3d_shared::{BlitCommand, BlitCommandType};

use super::{
    TlasBuild, TlasHandle,
    capture_tests::{
        BONES, Bone, Figure, blas_builds, boundary, centroid, figure_constants, moved, prebuild,
        quad, raw, trace_frame, vertex_bytes,
    },
    reuse_tests::Uploads,
    tests::{
        Draw, Sources, apply, build, build_with_capture, capture, device_buffer, length,
        next_frame, positions, probe, rt_encoder, run_blits, scale_move, staged_buffer,
    },
    tracks::REFIT_FRAMES,
};
use crate::encoder::{
    FrameEncoder, replay::tests::Gpu, scene_capture::CapturedIndex, tests::post_reset_frame,
};

/// The next submission with the GPU still on the earlier ones: its frame takes another slot.
fn next_frame_in_flight(enc: &mut FrameEncoder) {
    enc.current_submit_seq += 1;
    post_reset_frame(enc, (64, 64), 1, true);
    enc.scene.begin_submission(false);
}

/// The figure's six-vertex list of the test quad.
fn figure_list() -> [super::capture_tests::Vertex; 6] {
    let q = quad();
    [q[0], q[1], q[2], q[1], q[3], q[2]]
}

/// A grid of eye-space targets over the figure's neighbourhood under `bones`.
fn grid(bones: &[Bone]) -> Vec<[f32; 3]> {
    let list = figure_list();
    let centre = centroid([0, 1, 4].map(|i| list[i].skinned(bones)));
    let mut targets = Vec::new();
    for j in -6..=6 {
        for i in -6..=6 {
            #[expect(clippy::cast_precision_loss, reason = "small grid offsets")]
            let (x, y) = (i as f32 * 0.35, j as f32 * 0.35);
            targets.push([centre[0] + x, centre[1] + y, centre[2]]);
        }
    }
    targets
}

/// The captured figure under `bones` in `enc`'s frame: its draw, built and traced.
fn figure_frame(
    enc: &mut FrameEncoder,
    gpu: &Gpu,
    figure: &Figure,
    vb: u64,
    bones: &[Bone],
) -> (TlasHandle, Vec<BlitCommand>) {
    let d = figure.capture(
        enc,
        &Figure::snapshot(0, 0),
        (0x91, vb),
        CapturedIndex::None {
            start_vertex: 0,
            vertex_count: 6,
        },
        &figure_constants(bones),
    );
    prebuild(enc, &d);
    trace_frame(enc, gpu, &[d])
}

/// Every probe of `targets` through `a` and `b` agrees: both miss, or both
/// hit within `1e-4` of each other.
fn same_hits(gpu: &Gpu, a: &TlasHandle, b: &TlasHandle, targets: &[[f32; 3]]) -> usize {
    let (hits_a, hits_b) = (probe(gpu, a, targets), probe(gpu, b, targets));
    let mut hit = 0;
    for (i, (x, y)) in hits_a.iter().zip(&hits_b).enumerate() {
        match (x, y) {
            (None, None) => {}
            (Some((da, _)), Some((db, _))) => {
                assert!((da - db).abs() < 1e-4, "target {i}: {da} against {db}");
                hit += 1;
            }
            _ => panic!("target {i}: {x:?} against {y:?}"),
        }
    }
    hit
}

/// The figure under `bones` built anew by a scene that never saw an earlier frame.
struct Rebuilt {
    _enc: FrameEncoder,
    _figure: Figure,
    tlas: TlasHandle,
    // The encoder reads both through raw pointers: boxed, they never move.
    _counters: Box<(AtomicU64, AtomicU64)>,
}

impl Rebuilt {
    fn new(gpu: &Gpu, vb: u64, bones: &[Bone]) -> Option<Self> {
        let counters = Box::new((AtomicU64::new(0), AtomicU64::new(0)));
        let (_, mut enc) = rt_encoder(&counters.0, &counters.1)?;
        let figure = Figure::new(&mut enc);
        let (tlas, blits) = figure_frame(&mut enc, gpu, &figure, vb, bones);
        assert_eq!(blas_builds(&blits)[0].refit, 0, "built anew");
        Some(Self {
            _enc: enc,
            _figure: figure,
            tlas,
            _counters: counters,
        })
    }
}

/// A captured figure's BLAS is refit from the previous frame's once its
/// bones move: out of place while that frame may still be in flight (the
/// next frame takes another slot), in place once its own slot comes back;
/// the refit BLAS hits exactly where a BLAS built anew from the same
/// vertices does, and misses where the figure was.
#[test]
fn refit_hits_where_a_rebuild_does() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let vb = gpu.buffer(&vertex_bytes(&figure_list()));

    let (first, blits) = figure_frame(&mut enc, &gpu, &figure, raw(&vb), &BONES);
    let stats = enc.rt_stats();
    assert_eq!((stats.captured, stats.refit), (1, 0), "{stats:?}");
    assert_eq!(
        blas_builds(&blits)[0].refit,
        0,
        "a full build starts the chain"
    );
    let built = first.resident[0];

    // The bones move while the first frame is in flight: another slot, out of place.
    next_frame_in_flight(&mut enc);
    let bones = moved(BONES, [1.5, 0.5, 0.0]);
    let (second, blits) = figure_frame(&mut enc, &gpu, &figure, raw(&vb), &bones);
    let stats = enc.rt_stats();
    assert_eq!((stats.captured, stats.refit), (0, 1), "{stats:?}");
    let refit = blas_builds(&blits)[0];
    assert_eq!(refit.refit, built, "refit from the first frame's BLAS");
    assert_ne!(second.resident[0], built, "into the second slot's own");

    // The same vertices built anew, by a scene that never saw the first frame.
    let Some(rebuilt) = Rebuilt::new(&gpu, raw(&vb), &bones) else {
        return;
    };
    let targets = grid(&bones);
    let hit = same_hits(&gpu, &second, &rebuilt.tlas, &targets);
    assert!(
        hit >= 20,
        "the grid hits the figure: {hit} of {}",
        targets.len()
    );
    let list = figure_list();
    let was = centroid([0, 1, 2].map(|i| list[i].skinned(&BONES)));
    let now = centroid([0, 1, 2].map(|i| list[i].skinned(&bones)));
    let hits = probe(&gpu, &second, &[was, now]);
    assert_eq!(hits[0], None, "where the figure was");
    let (distance, _) = hits[1].expect("where it is");
    assert!((distance - length(now)).abs() < 1e-3, "{distance}");

    // The GPU passes both: the first slot comes back, refit from the second's.
    next_frame(&mut enc, &coherent);
    let bones = moved(BONES, [-1.0, 0.0, 0.5]);
    let (third, blits) = figure_frame(&mut enc, &gpu, &figure, raw(&vb), &bones);
    assert_eq!(blas_builds(&blits)[0].refit, second.resident[0]);
    assert_eq!(third.resident[0], built, "back in the first slot's");
    // Again: the newest is the first slot's own, refit in place.
    next_frame(&mut enc, &coherent);
    let bones = moved(BONES, [0.0, -1.0, 0.0]);
    let (fourth, blits) = figure_frame(&mut enc, &gpu, &figure, raw(&vb), &bones);
    assert_eq!(blas_builds(&blits)[0].refit, built, "in place");
    assert_eq!(fourth.resident[0], built);
    let Some(rebuilt) = Rebuilt::new(&gpu, raw(&vb), &bones) else {
        return;
    };
    same_hits(&gpu, &fourth, &rebuilt.tlas, &grid(&bones));
    drop(enc);
}

/// A chain refits [`REFIT_FRAMES`] times after its build, then builds
/// again; a failed submission at or past the chain's build makes the next
/// frame build too.
#[test]
fn refit_chain_rebuilds() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let failed = AtomicU64::new(0);
    enc.failed_seq_ptr = core::ptr::from_ref(&failed) as u64;
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let vb = gpu.buffer(&vertex_bytes(&figure_list()));
    let frame = |enc: &mut FrameEncoder| {
        let d = figure.capture(
            enc,
            &Figure::snapshot(0, 0),
            (0x91, raw(&vb)),
            CapturedIndex::None {
                start_vertex: 0,
                vertex_count: 6,
            },
            &figure_constants(&BONES),
        );
        prebuild(enc, &d);
        enc.scene.note_boundary(&boundary());
        let (tlas, _, _) = build_with_capture(enc, &[d]);
        assert!(tlas.is_some());
        let stats = enc.rt_stats();
        (stats.captured, stats.refit)
    };
    assert_eq!(frame(&mut enc), (1, 0), "the chain's build");
    for n in 0..REFIT_FRAMES {
        next_frame(&mut enc, &coherent);
        assert_eq!(frame(&mut enc), (0, 1), "refit {n}");
    }
    next_frame(&mut enc, &coherent);
    assert_eq!(
        frame(&mut enc),
        (1, 0),
        "built again after {REFIT_FRAMES} refits"
    );
    next_frame(&mut enc, &coherent);
    assert_eq!(frame(&mut enc), (0, 1));
    failed.store(enc.current_submit_seq, Ordering::Release);
    next_frame(&mut enc, &coherent);
    assert_eq!(
        frame(&mut enc),
        (1, 0),
        "a failed submission past the build"
    );
    next_frame(&mut enc, &coherent);
    assert_eq!(frame(&mut enc), (0, 1), "and the new chain refits");
    drop(enc);
}

/// The TLAS builds of `blits`.
fn tlas_builds(blits: &[BlitCommand]) -> Vec<TlasBuild> {
    blits
        .iter()
        .filter(|b| b.cmd == BlitCommandType::BuildTlas as u32)
        // SAFETY: a BuildTlas names its `TlasBuild` in the live encoder's scratch arena.
        .map(|b| unsafe { core::ptr::read_unaligned(b.byte_size as *const TlasBuild) })
        .collect()
}

/// A frame whose instances are the previous frame's (their count, BLAS
/// slots and masks; only their transforms and BLASes' contents moved)
/// refits that frame's TLAS: out of place while it may be in flight, in
/// place once its own slot comes back, hitting where a TLAS built anew
/// does. A frame with other instances builds, as does the frame after
/// [`REFIT_FRAMES`] refits or after a failed submission.
#[test]
fn tlas_refit_follows_its_instances() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let failed = AtomicU64::new(0);
    enc.failed_seq_ptr = core::ptr::from_ref(&failed) as u64;
    let gpu = Gpu::new(device);
    let figure = Figure::new(&mut enc);
    let vb = gpu.buffer(&vertex_bytes(&figure_list()));
    let refit_of = |blits: &[BlitCommand]| {
        let builds = tlas_builds(blits);
        assert_eq!(builds.len(), 1, "one TLAS a frame");
        builds[0].refit
    };

    let (first, blits) = figure_frame(&mut enc, &gpu, &figure, raw(&vb), &BONES);
    assert_eq!(refit_of(&blits), 0, "a full build starts the chain");

    // The bones move while the first frame is in flight: another slot, out of place.
    next_frame_in_flight(&mut enc);
    let bones = moved(BONES, [1.5, 0.5, 0.0]);
    let (second, blits) = figure_frame(&mut enc, &gpu, &figure, raw(&vb), &bones);
    assert_eq!(refit_of(&blits), first.tlas, "refit from the first frame's");
    assert_ne!(second.tlas, first.tlas, "into the second slot's own");
    assert_ne!(second.resident, first.resident, "over another slot's BLAS");
    let Some(rebuilt) = Rebuilt::new(&gpu, raw(&vb), &bones) else {
        return;
    };
    let hit = same_hits(&gpu, &second, &rebuilt.tlas, &grid(&bones));
    assert!(hit >= 20, "the grid hits the figure: {hit}");
    let list = figure_list();
    let was = centroid([0, 1, 2].map(|i| list[i].skinned(&BONES)));
    assert_eq!(
        probe(&gpu, &second, &[was])[0],
        None,
        "where the figure was"
    );

    // The first slot comes back, refit from the second's; then in place.
    next_frame(&mut enc, &coherent);
    let bones = moved(BONES, [-1.0, 0.0, 0.5]);
    let (third, blits) = figure_frame(&mut enc, &gpu, &figure, raw(&vb), &bones);
    assert_eq!(refit_of(&blits), second.tlas);
    assert_eq!(third.tlas, first.tlas, "back in the first slot's");
    next_frame(&mut enc, &coherent);
    let bones = moved(BONES, [0.0, -1.0, 0.0]);
    let (fourth, blits) = figure_frame(&mut enc, &gpu, &figure, raw(&vb), &bones);
    assert_eq!(refit_of(&blits), first.tlas, "in place");
    assert_eq!(fourth.tlas, first.tlas);
    let Some(rebuilt) = Rebuilt::new(&gpu, raw(&vb), &bones) else {
        return;
    };
    same_hits(&gpu, &fourth, &rebuilt.tlas, &grid(&bones));

    // The figure drawn twice: other instances, built; then a chain of refits.
    let twice = |enc: &mut FrameEncoder| {
        let draws: Vec<_> = (0..2)
            .map(|_| {
                let d = figure.capture(
                    enc,
                    &Figure::snapshot(0, 0),
                    (0x91, raw(&vb)),
                    CapturedIndex::None {
                        start_vertex: 0,
                        vertex_count: 6,
                    },
                    &figure_constants(&BONES),
                );
                prebuild(enc, &d);
                d
            })
            .collect();
        let (tlas, blits) = trace_frame(enc, &gpu, &draws);
        assert_eq!(tlas.instances.len(), 2);
        refit_of(&blits)
    };
    next_frame(&mut enc, &coherent);
    assert_eq!(twice(&mut enc), 0, "two instances where there was one");
    for n in 0..REFIT_FRAMES {
        next_frame(&mut enc, &coherent);
        assert_ne!(twice(&mut enc), 0, "refit {n}");
    }
    next_frame(&mut enc, &coherent);
    assert_eq!(
        twice(&mut enc),
        0,
        "built again after {REFIT_FRAMES} refits"
    );
    next_frame(&mut enc, &coherent);
    assert_ne!(twice(&mut enc), 0);
    failed.store(enc.current_submit_seq, Ordering::Release);
    next_frame(&mut enc, &coherent);
    assert_eq!(twice(&mut enc), 0, "a failed submission past the build");
    drop(enc);
}

/// A static draw whose range takes an upload every frame goes stale twice
/// in the cache, then is refit frame after frame, never cached again; the
/// refit BLAS follows the uploaded vertices.
#[test]
fn animated_static_draw_is_refit() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let gpu = Gpu::new(device);
    let sources = Sources::new();
    let id = BufferId::from_raw(0x7200);
    let at = |x: f32| {
        let (a, b, c, d) = (
            [x, 0.0, 0.0],
            [x + 1.0, 0.0, 0.0],
            [x, 2.0, 0.0],
            [x + 1.0, 2.0, 0.0],
        );
        positions(&[a, b, c, b, d, c])
    };
    staged_buffer(&mut enc, &gpu.device, id, &at(0.0));
    let world_view = scale_move([1.0, 1.0, 1.0], [0.0, 0.0, 4.0]);
    let mut uploads = Uploads::new();
    let frame = |enc: &mut FrameEncoder| {
        let draw = Draw::list(device_buffer(enc, id), enc.buffer_content(id), world_view);
        let d = capture(enc, &sources, &draw);
        let (tlas, blits) = build(enc, &[d]);
        let stats = enc.rt_stats();
        (
            tlas.expect("a TLAS"),
            blits,
            (stats.built, stats.refit, stats.stale),
        )
    };

    let (_, blits, churn) = frame(&mut enc);
    assert_eq!(churn, (1, 0, 0), "cached");
    // Run as a frame's would be: the next frame's TLAS is refit from this one.
    let mut all = core::mem::take(&mut enc.frame_blit_commands);
    all.extend(blits);
    run_blits(&gpu, &all);
    let mut x = 0.0;
    for (n, expected) in [
        (1, (1, 0, 1)),
        (2, (1, 0, 1)),
        (3, (0, 1, 0)),
        (4, (0, 1, 0)),
    ] {
        next_frame(&mut enc, &coherent);
        x += 0.25;
        uploads.write(&mut enc, id, 0, &at(x));
        let (tlas, blits, churn) = frame(&mut enc);
        assert_eq!(churn, expected, "frame {n} after the upload");
        // The uploads land ahead of the builds, as the frame's leading blits do.
        let mut all = core::mem::take(&mut enc.frame_blit_commands);
        all.extend(blits);
        run_blits(&gpu, &all);
        let centre = apply(&world_view, [x + 0.5, 1.0, 0.0]);
        let (distance, _) = probe(&gpu, &tlas, &[centre])[0].expect("the moved quad");
        assert!(
            (distance - length(centre)).abs() < 1e-3,
            "frame {n}: {distance}"
        );
        let old = apply(&world_view, [x - 0.2, 1.0, 0.0]);
        assert_eq!(
            probe(&gpu, &tlas, &[old])[0],
            None,
            "frame {n}: behind its left edge"
        );
    }
    drop(enc);
}
