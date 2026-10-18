//! BLASes refit frame after frame: captured draws, and static draws whose range keeps taking uploads.
//!
//! Each identity ([`TrackKey`]) keeps one acceleration structure per frame
//! slot, all of the size its triangles take with [`REFIT_USAGE`]
//! (`crate::metal::rt::REFIT_USAGE`). A frame in slot `s` refits the newest
//! of them into slot `s`'s from this frame's vertices (in place when the
//! newest is slot `s`'s own): the frames in flight keep tracing theirs,
//! which nothing writes. A full build starts each chain, and again after
//! [`REFIT_FRAMES`] refits (staggered across identities, so the rebuilds of
//! a scene spread over frames) or once a submission at or past the chain's
//! build failed. An identity no frame instanced for [`TRACK_IDLE_FRAMES`]
//! retires its acceleration structures behind the GPU.
//!
//! A captured draw is one identity per frame position: its pipeline, its
//! streams and index range, and its rank among the frame's draws of that
//! same identity, so the same figure drawn twice keeps two. A static draw
//! joins after [`ANIMATED_STRIKES`] cached BLASes of its key went stale
//! for uploads into their range (CPU-animated vertices in a static
//! buffer), and from then on is refit too, never cached.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};

use super::{BlasKey, Grown, MAX_FRAME_SLOTS, NO_INDEX};
use crate::encoder::scene_capture::{BufferContent, CapturedDraw, CapturedIndex, VertexBind};

/// Refits after which a track builds again, its quality restored.
pub const REFIT_FRAMES: u32 = 16;
/// Submissions a track may go uninstanced before its acceleration structures retire.
pub const TRACK_IDLE_FRAMES: u64 = 60;
/// Stale retirements for uploads into its range after which a static key is refit, not cached.
pub const ANIMATED_STRIKES: u8 = 2;

/// The identity of a captured draw across frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CaptureKey {
    /// The pipeline the draw bound: its vertex shader and vertex layout.
    pipeline: u64,
    /// The draw's per-vertex streams: their buffers and bind offsets, hashed.
    streams: u64,
    /// Its index buffer, offset, count, type and base vertex; a
    /// non-indexed draw's first vertex and count under [`NO_INDEX`].
    index: (u64, u32, u32, u8, i32),
    /// The vertex ids its capture holds.
    vertices: u32,
    strip: bool,
    alpha: bool,
    /// Its rank among the frame's draws of the same identity.
    occurrence: u32,
}

impl CaptureKey {
    /// The identity of `d`, captured over `vertices` ids, alpha-tested when
    /// `alpha`; its rank still 0 ([`CaptureRanks::rank`]).
    pub fn new(d: &CapturedDraw, vertices: u32, alpha: bool) -> Self {
        let mut streams = FxHasher::default();
        for bind in d.vertex_binds() {
            if let VertexBind::Buffer {
                offset,
                slot,
                content,
                ..
            } = *bind
            {
                (content.id, offset, slot).hash(&mut streams);
            }
        }
        let index = match d.index {
            CapturedIndex::None {
                start_vertex,
                vertex_count,
            } => (0, start_vertex, vertex_count, NO_INDEX, 0),
            CapturedIndex::Buffer {
                offset,
                count,
                index_type,
                base_vertex,
                content,
                ..
            } => (content.id, offset, count, index_type as u8, base_vertex),
            CapturedIndex::Inline {
                count, index_type, ..
            } => (u64::MAX, 0, count, index_type as u8, 0),
        };
        Self {
            pipeline: d.pipeline,
            streams: streams.finish(),
            index,
            vertices,
            strip: d.primitive == mtld3d_shared::mtl::PrimitiveType::TriangleStrip,
            alpha,
            occurrence: 0,
        }
    }
}

/// The ranks one frame gave its captured draws' identities so far.
#[derive(Default)]
pub struct CaptureRanks(FxHashMap<CaptureKey, u32>);

impl CaptureRanks {
    /// `key` ranked after every earlier draw of the frame with the same identity.
    pub fn rank(&mut self, mut key: CaptureKey) -> CaptureKey {
        let next = self.0.entry(key).or_insert(0);
        key.occurrence = *next;
        *next += 1;
        key
    }

    /// Forget every rank, for the next frame; the map keeps its room.
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

/// Geometry refit frame after frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrackKey {
    /// A captured draw.
    Captured(CaptureKey),
    /// A static draw whose range keeps taking uploads.
    Animated(BlasKey),
}

/// One identity's acceleration structures and where its refit chain stands.
#[derive(Default)]
struct Track {
    /// One per frame slot, made on the slot's first use.
    accels: [Grown; MAX_FRAME_SLOTS],
    /// The slot whose acceleration structure holds the newest geometry.
    latest: Option<usize>,
    /// The submission of the chain's full build.
    built: u64,
    /// Refits since the chain's build, counting from the track's stagger.
    refits: u32,
    /// Where the count starts again after a build: `REFIT_FRAMES` spread over identities.
    stagger: u32,
    /// The submission that last instanced it.
    last_used: u64,
    /// The index buffer's contents the chain was built from and the
    /// content clock of its last placement; `None` for non-indexed geometry.
    index: Option<(BufferContent, u64)>,
}

/// What the index buffer of a placement holds: its contents and byte
/// range (`BlasSource::index_read`), and whether an upload in place wrote
/// into that range since a content clock.
pub struct IndexState<'a> {
    pub read: Option<(BufferContent, (u32, u32))>,
    pub written_since: &'a dyn Fn(u64, (u32, u32)) -> bool,
}

/// The refit tracks of the scene, and the static keys going stale for uploads.
#[derive(Default)]
pub struct Tracks {
    map: FxHashMap<TrackKey, Track>,
    /// Tracks made so far, which staggers their rebuilds.
    made: u32,
    /// Static keys retired for uploads into their range: the strikes and the submission of the last.
    strikes: FxHashMap<BlasKey, (u8, u64)>,
}

impl Tracks {
    /// The bytes of every track's acceleration structures.
    pub fn bytes(&self) -> u64 {
        self.map
            .values()
            .flat_map(|t| t.accels.iter())
            .map(|g| g.len)
            .sum()
    }

    /// Note that `key`'s cached BLAS was retired in submission `seq` for an upload into its range.
    pub fn strike(&mut self, key: BlasKey, seq: u64) {
        let entry = self.strikes.entry(key).or_insert((0, seq));
        *entry = (entry.0.saturating_add(1), seq);
    }

    /// Whether static `key` is refit rather than cached: it went stale for uploads often enough.
    pub fn animated(&self, key: &BlasKey) -> bool {
        self.strikes
            .get(key)
            .is_some_and(|&(strikes, _)| strikes >= ANIMATED_STRIKES)
    }

    /// `key`'s acceleration structure of `slot` when submission `seq` already placed it there.
    pub fn placed_now(&self, key: &TrackKey, slot: usize, seq: u64) -> Option<u64> {
        self.map
            .get(key)
            .filter(|t| t.last_used == seq && t.latest == Some(slot))
            .map(|t| t.accels[slot].handle)
    }

    /// What `key` refits from, `None` to build: its newest acceleration
    /// structure of at least `size` bytes while its chain is short of
    /// [`REFIT_FRAMES`] refits, no submission at or past the chain's build
    /// failed (`failed`, the newest that did) and its indices (`index`) are
    /// the ones the chain was built from: the same contents, nothing
    /// uploaded in place into the range since the last placement. A refit
    /// moves vertices only.
    pub fn refit_source(
        &self,
        key: &TrackKey,
        size: u64,
        failed: u64,
        index: &IndexState<'_>,
    ) -> Option<u64> {
        let track = self.map.get(key)?;
        let latest = &track.accels[track.latest?];
        let same_indices = match (track.index, index.read) {
            (None, None) => true,
            (Some((was, clock)), Some((now, range))) => {
                was == now && !(index.written_since)(clock, range)
            }
            _ => false,
        };
        (track.refits < REFIT_FRAMES
            && track.built > failed
            && same_indices
            && latest.len >= size
            && latest.handle != 0)
            .then_some(latest.handle)
    }

    /// `key`'s acceleration structure of `slot`, the track made on first use.
    pub fn accel(&mut self, key: TrackKey, slot: usize) -> &mut Grown {
        let made = &mut self.made;
        let track = self.map.entry(key).or_insert_with(|| {
            let stagger = *made % REFIT_FRAMES;
            *made = made.wrapping_add(1);
            Track {
                stagger,
                ..Track::default()
            }
        });
        &mut track.accels[slot]
    }

    /// The bytes `key` holds in `slot` now.
    pub fn held(&self, key: &TrackKey, slot: usize) -> u64 {
        self.map.get(key).map_or(0, |t| t.accels[slot].len)
    }

    /// Record that submission `seq` refit (`refit`) or built `key` into
    /// `slot`, from the indices `index` of content clock `clock`.
    pub fn placed(
        &mut self,
        key: &TrackKey,
        (slot, seq, clock): (usize, u64, u64),
        refit: bool,
        index: Option<BufferContent>,
    ) {
        if let TrackKey::Animated(blas) = key
            && let Some(strike) = self.strikes.get_mut(blas)
        {
            // Still animated while it is refit: its strikes age from its last use.
            strike.1 = seq;
        }
        let Some(track) = self.map.get_mut(key) else {
            return;
        };
        track.latest = Some(slot);
        track.last_used = seq;
        track.index = index.map(|content| (content, clock));
        if refit {
            track.refits += 1;
        } else {
            track.built = seq;
            track.refits = track.stagger;
            track.stagger = 0;
        }
    }

    /// Forget what `key` holds: a placement whose build was never queued.
    pub fn forget(&mut self, key: &TrackKey) {
        if let Some(track) = self.map.get_mut(key) {
            track.latest = None;
        }
    }

    /// Take out the tracks no submission instanced since `seq - TRACK_IDLE_FRAMES`,
    /// and the strikes as old: their acceleration structures and bytes onto `retired`, for retirement.
    pub fn take_idle(&mut self, seq: u64, retired: &mut Vec<(u64, u64)>) {
        let idle = |last: u64| last.saturating_add(TRACK_IDLE_FRAMES) < seq;
        self.strikes.retain(|_, &mut (_, last)| !idle(last));
        self.map.retain(|_, track| {
            if !idle(track.last_used) {
                return true;
            }
            retired.extend(
                track
                    .accels
                    .iter()
                    .filter(|g| g.handle != 0)
                    .map(|g| (g.handle, g.len)),
            );
            false
        });
    }

    /// Every acceleration structure the tracks hold, the tracks gone: the encoder's teardown.
    pub fn take_all(&mut self) -> Vec<u64> {
        self.strikes.clear();
        self.map
            .drain()
            .flat_map(|(_, t)| t.accels)
            .map(|g| g.handle)
            .filter(|&h| h != 0)
            .collect()
    }
}

#[cfg(test)]
mod tests;
