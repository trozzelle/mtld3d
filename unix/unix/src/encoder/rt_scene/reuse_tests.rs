//! What a cached BLAS survives: staging uploads into the ranges other draws of its buffer read.

use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    guest_completions::CompletionPool, guest_pages::GuestOwnedPageLease, ids::BufferId,
    page_box::PageBox,
};
use mtld3d_shared::mtl::IndexType;

use super::tests::{
    Draw, Sources, build, capture, device_buffer, next_frame, positions, rt_encoder, scale_move,
    staged_buffer,
};
use crate::encoder::{FrameEncoder, UploadLog, scene_capture::CapturedIndex};

/// Writes staging uploads into a test's buffers the way the PE's unlocks reach the encoder.
pub(super) struct Uploads {
    pool: CompletionPool,
    leases: Vec<GuestOwnedPageLease>,
}

impl Uploads {
    pub(super) fn new() -> Self {
        Self {
            pool: CompletionPool::new(),
            leases: Vec::new(),
        }
    }

    /// Upload `bytes` into buffer `id` at `offset`.
    pub(super) fn write(
        &mut self,
        enc: &mut FrameEncoder,
        id: BufferId,
        offset: u32,
        bytes: &[u8],
    ) {
        let mut page = PageBox::new_zeroed(bytes.len());
        page.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        let lease = GuestOwnedPageLease::new(page, &self.pool, None);
        // SAFETY: the lease outlives the encoder's adoption (the test drops the encoder first).
        let page = unsafe { lease.descriptor().adopt() }.expect("a valid lease");
        self.leases.push(lease);
        let size = u32::try_from(bytes.len()).expect("small");
        enc.apply_stage_upload(id, page, offset, size);
    }
}

/// Two quads, six vertices each, the second 2 units right of the first.
fn two_quads() -> Vec<u8> {
    let quad = |x: f32| {
        let (a, b, c, d) = (
            [x, 0.0, 0.0],
            [x + 1.0, 0.0, 0.0],
            [x, 2.0, 0.0],
            [x + 1.0, 2.0, 0.0],
        );
        [a, b, c, b, d, c]
    };
    let mut points = quad(0.0).to_vec();
    points.extend(quad(2.0));
    positions(&points)
}

/// A frame's static builds, reuses and stale retirements.
fn churn(enc: &FrameEncoder) -> (u32, u32, u32) {
    let stats = enc.rt_stats();
    (stats.built, stats.reused, stats.stale)
}

/// A staging upload in place rebuilds only the BLASes whose ranges it
/// wrote: one into the second quad of a shared vertex buffer leaves the
/// first quad's BLAS cached, frame after frame; one into the first quad
/// rebuilds it. Indexed draws keep theirs likewise when an upload writes
/// another range of their index buffer, or vertices below their base vertex.
#[test]
fn upload_elsewhere_in_the_buffer_keeps_the_blas() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((device, mut enc)) = rt_encoder(&coherent, &retained) else {
        return;
    };
    let sources = Sources::new();
    let (vb, ib) = (BufferId::from_raw(0x7100), BufferId::from_raw(0x7101));
    let vertices = two_quads();
    staged_buffer(&mut enc, &device, vb, &vertices);
    // Each quad's six indices, from 0 under its base vertex.
    let indices: Vec<u8> = [0u16, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    staged_buffer(&mut enc, &device, ib, &indices);
    let world_view = scale_move([1.0, 1.0, 1.0], [0.0, 0.0, 4.0]);
    let quad_bytes = vertices.len() / 2;
    let mut uploads = Uploads::new();

    // Draws: quad 0 and quad 1 non-indexed, then quad 1 indexed from the
    // index buffer's second half at base vertex 6.
    let frame = |enc: &mut FrameEncoder| {
        let handle = device_buffer(enc, vb);
        let content = enc.buffer_content(vb);
        let list = |start_vertex| Draw {
            index: CapturedIndex::None {
                start_vertex,
                vertex_count: 6,
            },
            ..Draw::list(handle, content, world_view)
        };
        let indexed = Draw {
            index: CapturedIndex::Buffer {
                handle: device_buffer(enc, ib),
                offset: 12,
                count: 6,
                index_type: IndexType::UInt16,
                base_vertex: 6,
                content: enc.buffer_content(ib),
            },
            ..Draw::list(handle, content, world_view)
        };
        let draws = [list(0), list(6), indexed].map(|d| capture(enc, &sources, &d));
        let (tlas, _) = build(enc, &draws);
        assert_eq!(tlas.expect("a TLAS").instances.len(), 3);
        churn(enc)
    };

    assert_eq!(frame(&mut enc), (3, 0, 0), "first frame builds all three");
    for n in 0..3 {
        next_frame(&mut enc, &coherent);
        assert_eq!(
            frame(&mut enc),
            (0, 3, 0),
            "still frame {n} rebuilds nothing"
        );
    }

    // Quad 1's vertices: its list and its indexed draw rebuild, quad 0's list stays.
    next_frame(&mut enc, &coherent);
    uploads.write(
        &mut enc,
        vb,
        u32::try_from(quad_bytes).expect("small"),
        &vertices[quad_bytes..],
    );
    assert_eq!(
        frame(&mut enc),
        (2, 1, 2),
        "an upload into quad 1 rebuilds its two draws"
    );
    next_frame(&mut enc, &coherent);
    assert_eq!(frame(&mut enc), (0, 3, 0), "and nothing the frame after");

    // Quad 0's vertices: below the indexed draw's base vertex, so only quad 0's list.
    next_frame(&mut enc, &coherent);
    uploads.write(&mut enc, vb, 0, &vertices[..quad_bytes]);
    assert_eq!(
        frame(&mut enc),
        (1, 2, 1),
        "an upload into quad 0 rebuilds its list alone"
    );

    // The index buffer's first half, which no draw reads; then the second half.
    next_frame(&mut enc, &coherent);
    uploads.write(&mut enc, ib, 0, &indices[..12]);
    assert_eq!(
        frame(&mut enc),
        (0, 3, 0),
        "indices no draw reads rebuild nothing"
    );
    next_frame(&mut enc, &coherent);
    uploads.write(&mut enc, ib, 12, &indices[12..]);
    assert_eq!(
        frame(&mut enc),
        (1, 2, 1),
        "the indexed draw's own indices rebuild it"
    );
    // Its indices again, twice: rebuilt each time, never refit (a refit
    // moves vertices only), so uploads into indices strike nothing.
    for _ in 0..2 {
        next_frame(&mut enc, &coherent);
        uploads.write(&mut enc, ib, 12, &indices[12..]);
        assert_eq!(frame(&mut enc), (1, 2, 1), "the indices again");
    }
    assert_eq!(
        enc.rt.as_ref().expect("the scene").tracks.bytes(),
        0,
        "no draw went to a refit track"
    );
    drop(enc);
}

/// The upload log answers for the uploads it holds and, once it forgot
/// some, conservatively for anything checked before the newest it forgot.
#[test]
fn upload_log_forgets_conservatively() {
    let mut log = UploadLog::default();
    log.note(10, 0, 64);
    assert!(
        log.written_since(9, 32, 48),
        "an overlapping upload after the check"
    );
    assert!(!log.written_since(10, 32, 48), "one at or before the check");
    assert!(
        !log.written_since(9, 64, 128),
        "one that ends where the range starts"
    );
    for stamp in 11..=17 {
        log.note(stamp, 1000, 1004);
    }
    assert!(log.written_since(9, 0, 4), "all eight remembered");
    log.note(18, 1000, 1004);
    assert!(
        log.written_since(9, 2000, 2004),
        "the ring forgot stamp 10, so a check from 9 cannot rule it out"
    );
    assert!(
        !log.written_since(10, 2000, 2004),
        "a check from the forgotten stamp itself can"
    );
}
