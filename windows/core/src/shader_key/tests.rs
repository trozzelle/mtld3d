//! Unit tests for the shader record kind and the content keys.
//!
//! `CachedKind` survives its wire byte and maps shader-model majors, and `ff_key_hash`
//! gives equal keys equal hashes.

use super::*;

#[test]
fn cached_kind_round_trips_via_byte() {
    for k in [
        CachedKind::FfVs,
        CachedKind::FfPs,
        CachedKind::Sm1Vs,
        CachedKind::Sm1Ps,
        CachedKind::Sm2Vs,
        CachedKind::Sm2Ps,
        CachedKind::Sm3Vs,
        CachedKind::Sm3Ps,
    ] {
        assert_eq!(CachedKind::from_byte(k as u8), Some(k));
    }
}

#[test]
fn from_programmable_maps_supported_majors() {
    assert_eq!(
        CachedKind::from_programmable(1, false),
        Some(CachedKind::Sm1Vs)
    );
    assert_eq!(
        CachedKind::from_programmable(2, true),
        Some(CachedKind::Sm2Ps)
    );
    assert_eq!(
        CachedKind::from_programmable(3, false),
        Some(CachedKind::Sm3Vs)
    );
    assert_eq!(CachedKind::from_programmable(0, false), None);
    assert_eq!(CachedKind::from_programmable(4, true), None);
}

#[test]
fn ff_key_hash_is_stable() {
    let a = (1u32, 2u32, 3u32);
    let b = (1u32, 2u32, 3u32);
    assert_eq!(ff_key_hash(&a), ff_key_hash(&b));
}

/// The figure flag splits the programmable-VS disk key only when set: stock
/// (flag clear) keys keep their pre-figure hashes, figure-on keys differ from
/// their stock twins, and equal figure-on inputs hash equally.
#[test]
fn figure_flag_splits_vs_disk_key_only_when_set() {
    let id = ProgramId::from_tokens(&[0xFFFE_0101, 0x0000_FFFF]);
    let kinds = VsSamplerKinds {
        volume_mask: 1,
        cube_mask: 2,
        lod_table: true,
    };
    let stock = vs_source_disk_key_programmable(id, 0xA55A, 3, kinds, false, false);
    assert_eq!(
        stock,
        ff_key_hash(&(id.raw(), 0xA55Au16, 3u8, kinds)),
        "stock key must keep its pre-figure hash"
    );
    let figure = vs_source_disk_key_programmable(id, 0xA55A, 3, kinds, true, false);
    assert_ne!(figure, stock);
    assert_eq!(
        figure,
        vs_source_disk_key_programmable(id, 0xA55A, 3, kinds, true, false)
    );
    // A different stock shader keeps its own key, unaffected by the flag path.
    assert_ne!(
        stock,
        vs_source_disk_key_programmable(id, 0xA55A, 2, kinds, false, false)
    );
}

/// The figure-fog flag (`fog.figures`) keys its own VS libraries independently
/// of the per-pixel figure flag: both clear keeps the stock hash, and every
/// combination of the two flags hashes distinctly.
#[test]
fn figure_fog_flag_splits_vs_disk_key_only_when_set() {
    let id = ProgramId::from_tokens(&[0xFFFE_0101, 0x0000_FFFF]);
    let kinds = VsSamplerKinds::default();
    let key = |figure, fog| vs_source_disk_key_programmable(id, 0xA55A, 3, kinds, figure, fog);
    assert_eq!(
        key(false, false),
        ff_key_hash(&(id.raw(), 0xA55Au16, 3u8, kinds))
    );
    assert_eq!(
        key(true, false),
        ff_key_hash(&((id.raw(), 0xA55Au16, 3u8, kinds), true))
    );
    let all = [
        key(false, false),
        key(true, false),
        key(false, true),
        key(true, true),
    ];
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(a, b);
        }
    }
    assert_eq!(key(false, true), key(false, true));
}
