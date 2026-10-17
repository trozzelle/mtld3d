use mtld3d_core::post::PostConfig;
use mtld3d_shared::{MetalHandle, mtl::PixelFormat};

use super::{Effect, PostScratch, Rekey, enabled_effects};

fn tex(raw: u64) -> MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind> {
    // SAFETY: tests; opaque values never dereferenced.
    unsafe { MetalHandle::new(raw) }
}

#[test]
fn post_scratch_rekeyed_on_resize() {
    let small = (64, 64, PixelFormat::Bgra8Unorm);
    let wide = (128, 64, PixelFormat::Bgra8Unorm);
    let mut scratch = PostScratch::default();
    assert!(matches!(scratch.rekey(small), Rekey::Miss { stale: None }));
    scratch.install(small, [tex(1), tex(2)]);
    assert!(matches!(scratch.rekey(small), Rekey::Hit([a, b]) if (a.raw(), b.raw()) == (1, 2)));
    // A resize hands the old pair back for retirement and empties the slot.
    let Rekey::Miss {
        stale: Some([a, b]),
    } = scratch.rekey(wide)
    else {
        panic!("a new extent misses and retires the old pair");
    };
    assert_eq!((a.raw(), b.raw()), (1, 2));
    assert!(matches!(scratch.rekey(wide), Rekey::Miss { stale: None }));
    assert!(scratch.handles().is_none());
    scratch.install(wide, [tex(3), tex(4)]);
    assert!(matches!(scratch.rekey(wide), Rekey::Hit([a, _]) if a.raw() == 3));
    // A format change rekeys too.
    let float = (128, 64, PixelFormat::Rgba16Float);
    assert!(matches!(
        scratch.rekey(float),
        Rekey::Miss { stale: Some([a, _]) } if a.raw() == 3
    ));
    scratch.install(float, [tex(5), tex(6)]);
    let taken = scratch.take().expect("shutdown takes the pair");
    assert_eq!((taken[0].raw(), taken[1].raw()), (5, 6));
    assert!(scratch.take().is_none());
}

#[test]
fn the_chain_follows_the_config() {
    assert_eq!(enabled_effects(PostConfig::default()).as_slice(), []);
    let chain = enabled_effects(PostConfig { debug: true });
    assert_eq!(chain.as_slice(), [Effect::Invert]);
    assert!(!Effect::Invert.reads_depth());
    assert_eq!(Effect::from_id(Effect::Invert as u32), Some(Effect::Invert));
    assert_eq!(Effect::from_id(u32::MAX), None);
}

#[test]
fn constants_match_the_kernel_layout() {
    assert_eq!(size_of::<super::PostConstants>(), 64);
    let b = mtld3d_core::encoder_data::PostBoundary {
        proj: [1.5, 2.0, 1.001, -0.1],
        sun_dir_eye: Some([0.0, 0.6, -0.8]),
        sun_color: [0.9, 0.8, 0.7],
    };
    let k = super::PostConstants::new(&b, 640, 480, false);
    assert_eq!(k.proj, b.proj);
    assert_eq!(k.sun_dir, [0.0, 0.6, -0.8, 1.0]);
    assert_eq!(k.sun_color, [0.9, 0.8, 0.7, 0.0]);
    assert_eq!(k.extent, [640, 480, 0, 0]);
    let k = super::PostConstants::new(
        &mtld3d_core::encoder_data::PostBoundary {
            sun_dir_eye: None,
            ..b
        },
        8,
        8,
        true,
    );
    assert_eq!(k.sun_dir, [0.0; 4]);
    assert_eq!(k.extent, [8, 8, 1, 0]);
}
