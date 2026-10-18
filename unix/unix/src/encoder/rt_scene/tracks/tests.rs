//! Refit tracks: when a chain may refit, host-side.

use super::{IndexState, TrackKey, Tracks};
use crate::encoder::{rt_scene::BlasKey, scene_capture::BufferContent};

const SIZE: u64 = 4096;

/// A track of one slot holding a structure of [`SIZE`] bytes, built in
/// submission 1 at content clock 10 from `index`.
fn built(index: Option<BufferContent>) -> (Tracks, TrackKey) {
    let mut tracks = Tracks::default();
    let key = TrackKey::Animated(BlasKey::default());
    let accel = tracks.accel(key, 0);
    accel.handle = 0xA5;
    accel.len = SIZE;
    tracks.placed(&key, (0, 1, 10), false, index);
    (tracks, key)
}

/// A chain refits only from the indices it was built from: the same
/// contents (the buffer's id and version), and no upload in place into the
/// range the draw reads since the last placement. Another buffer, another
/// version, an upload into the range, or indices where there were none
/// (or none where there were) build anew; an upload elsewhere refits.
#[test]
fn refit_needs_the_chains_indices() {
    let ib = BufferContent { id: 7, version: 3 };
    let range = (64, 96);
    let never = |_: u64, _: (u32, u32)| false;
    let into_range = |since: u64, (offset, size): (u32, u32)| {
        // An upload at clock 12 into bytes 80..90.
        since < 12 && offset < 90 && 80 < offset + size
    };
    fn state<'a>(
        read: Option<(BufferContent, (u32, u32))>,
        written_since: &'a dyn Fn(u64, (u32, u32)) -> bool,
    ) -> IndexState<'a> {
        IndexState {
            read,
            written_since,
        }
    }
    let (tracks, key) = built(Some(ib));
    let refits = |index: &IndexState<'_>| tracks.refit_source(&key, SIZE, 0, index);
    assert_eq!(refits(&state(Some((ib, range)), &never)), Some(0xA5));
    assert_eq!(
        refits(&state(Some((ib, range)), &into_range)),
        None,
        "an upload into the indices read"
    );
    assert_eq!(
        refits(&state(Some((ib, (0, 64))), &into_range)),
        Some(0xA5),
        "an upload past the indices read"
    );
    assert_eq!(
        refits(&state(
            Some((BufferContent { version: 4, ..ib }, range)),
            &never
        )),
        None,
        "a new version"
    );
    assert_eq!(
        refits(&state(Some((BufferContent { id: 8, ..ib }, range)), &never)),
        None,
        "another buffer"
    );
    assert_eq!(refits(&state(None, &never)), None, "no indices now");
    let (plain, key) = built(None);
    assert_eq!(
        plain.refit_source(&key, SIZE, 0, &state(None, &never)),
        Some(0xA5)
    );
    assert_eq!(
        plain.refit_source(&key, SIZE, 0, &state(Some((ib, range)), &never)),
        None,
        "indices where there were none"
    );
}

/// A refit records the clock of its placement: an upload into the indices
/// before it no longer stops the next refit.
#[test]
fn placement_moves_the_index_clock() {
    let ib = BufferContent { id: 7, version: 3 };
    let (mut tracks, key) = built(Some(ib));
    let before_15 = |since: u64, _: (u32, u32)| since < 15;
    let index = IndexState {
        read: Some((ib, (0, 32))),
        written_since: &before_15,
    };
    assert_eq!(tracks.refit_source(&key, SIZE, 0, &index), None);
    tracks.placed(&key, (0, 2, 20), false, Some(ib));
    assert_eq!(tracks.refit_source(&key, SIZE, 0, &index), Some(0xA5));
}
