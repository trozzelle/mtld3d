use mtld3d_core::post::PostConfig;
use mtld3d_shared::{MetalHandle, mtl::PixelFormat};

use super::{Effect, Extent, Input, Kernel, Output, PostScratch, Rekey, Slot, enabled_effects};

fn tex(raw: u64) -> MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind> {
    // SAFETY: tests; opaque values never dereferenced.
    unsafe { MetalHandle::new(raw) }
}

#[test]
fn post_scratch_rekeyed_on_resize() {
    let small = (64, 64, PixelFormat::Bgra8Unorm);
    let wide = (128, 64, PixelFormat::Bgra8Unorm);
    let mut scratch = PostScratch::default();
    assert!(matches!(
        scratch.rekey(Slot::ColourA, small),
        Rekey::Miss { stale: None }
    ));
    scratch.install(Slot::ColourA, small, tex(1));
    assert!(matches!(scratch.rekey(Slot::ColourA, small), Rekey::Hit(a) if a.raw() == 1));
    // Slots are independent: another slot of the same key misses.
    assert!(matches!(
        scratch.rekey(Slot::ColourB, small),
        Rekey::Miss { stale: None }
    ));
    scratch.install(Slot::ColourB, small, tex(2));
    // A resize hands the slot's old texture back for retirement and empties it.
    let Rekey::Miss { stale: Some(a) } = scratch.rekey(Slot::ColourA, wide) else {
        panic!("a new extent misses and retires the old texture");
    };
    assert_eq!(a.raw(), 1);
    assert!(matches!(
        scratch.rekey(Slot::ColourA, wide),
        Rekey::Miss { stale: None }
    ));
    assert!(scratch.handle(Slot::ColourA).is_none());
    assert_eq!(scratch.handle(Slot::ColourB).map(MetalHandle::raw), Some(2));
    scratch.install(Slot::ColourA, wide, tex(3));
    assert!(matches!(scratch.rekey(Slot::ColourA, wide), Rekey::Hit(a) if a.raw() == 3));
    // A format change rekeys too.
    let float = (128, 64, PixelFormat::Rgba16Float);
    assert!(matches!(
        scratch.rekey(Slot::ColourA, float),
        Rekey::Miss { stale: Some(a) } if a.raw() == 3
    ));
    scratch.install(Slot::ColourA, float, tex(5));
    let mut taken: Vec<u64> = scratch.take_all().map(MetalHandle::raw).collect();
    taken.sort_unstable();
    assert_eq!(taken, [2, 5], "shutdown takes every slot");
    assert_eq!(scratch.take_all().count(), 0);
}

#[test]
fn the_chain_follows_the_config() {
    assert_eq!(enabled_effects(PostConfig::default()).as_slice(), []);
    let chain = enabled_effects(PostConfig {
        debug: true,
        ..PostConfig::default()
    });
    assert_eq!(chain.as_slice(), [Effect::Invert]);
    assert!(!chain.reads_depth());
    assert!(!Effect::Invert.reads_depth());
    // `post.ao`: any strength above zero runs AO, first; zero is off.
    let ao = PostConfig {
        ao_pct: 1,
        ..PostConfig::default()
    };
    assert_eq!(enabled_effects(ao).as_slice(), [Effect::Ao]);
    assert!(Effect::Ao.reads_depth());
    let both = PostConfig { debug: true, ..ao };
    let chain = enabled_effects(both);
    assert_eq!(chain.as_slice(), [Effect::Ao, Effect::Invert]);
    assert!(chain.reads_depth());
    assert_eq!(chain.without_depth().as_slice(), [Effect::Invert]);
    assert_eq!(Kernel::from_id(Kernel::Invert as u32), Some(Kernel::Invert));
    assert_eq!(Kernel::from_id(u32::MAX), None);
    for kernel in Kernel::ALL {
        assert_eq!(Kernel::from_id(kernel as u32), Some(kernel));
    }
}

/// An effect's passes read the colour as the effect found it, and only its
/// last pass writes the new colour, so "the colour before the effect" stays
/// readable by every pass and the colour slots can ping-pong per effect.
#[test]
fn every_effect_writes_the_colour_once_last() {
    for effect in Effect::CHAIN {
        let passes = effect.passes();
        let (last, earlier) = passes.split_last().expect("an effect has a pass");
        assert_eq!(last.output, Output::Colour, "{effect:?}");
        assert_eq!(last.extent(), Extent::FULL, "{effect:?}");
        for pass in passes {
            for input in [Some(pass.input), pass.aux].into_iter().flatten() {
                if let Input::Slot(slot) = input {
                    assert!(!slot.is_colour(), "{effect:?} reads a colour slot");
                }
            }
            if let Output::Slot(slot) = pass.output {
                assert!(!slot.is_colour(), "{effect:?} writes a colour slot");
            }
        }
        for pass in earlier {
            assert_ne!(pass.output, Output::Colour, "{effect:?}");
        }
        assert_eq!(
            effect.reads_depth(),
            passes.iter().any(|p| p.kernel.reads_depth())
        );
    }
}

#[test]
fn extents_round_up_and_slots_key_by_extent_and_format() {
    assert_eq!(Extent::FULL.of((5, 3)), (5, 3));
    assert_eq!(Extent::HALF.of((5, 3)), (3, 2));
    assert_eq!(Extent::HALF.of((1, 1)), (1, 1));
    assert_eq!(Extent::HALF.of((1920, 1080)), (960, 540));
    for slot in [Slot::ColourA, Slot::ColourB] {
        assert!(slot.is_colour());
        assert_eq!(
            slot.key((640, 480), PixelFormat::Bgra8Unorm),
            (640, 480, PixelFormat::Bgra8Unorm)
        );
    }
    // AO's intermediates are half-extent single-channel floats, whatever the colour.
    for slot in [Slot::AoRaw, Slot::AoBlurred] {
        assert!(!slot.is_colour());
        assert_eq!(
            slot.key((641, 480), PixelFormat::Rgba16Float),
            (321, 240, PixelFormat::R16Float)
        );
    }
}

#[test]
fn constants_match_the_kernel_layout() {
    assert_eq!(size_of::<super::PostConstants>(), 80);
    let b = mtld3d_core::encoder_data::PostBoundary {
        proj: [1.5, 2.0, 1.001, -0.1],
        sun_dir_eye: Some([0.0, 0.6, -0.8]),
        sun_color: [0.9, 0.8, 0.7],
    };
    let cfg = PostConfig {
        ao_pct: 125,
        ..PostConfig::default()
    };
    let k = super::PostConstants::new(&b, cfg, 640, 480, false);
    assert_eq!(k.proj, b.proj);
    assert_eq!(k.sun_dir, [0.0, 0.6, -0.8, 1.0]);
    assert_eq!(k.sun_color, [0.9, 0.8, 0.7, 0.0]);
    assert_eq!(k.extent, [640, 480, 0, 0]);
    assert_eq!(k.strength, [1.25, 0.0, 0.0, 0.0]);
    let k = super::PostConstants::new(
        &mtld3d_core::encoder_data::PostBoundary {
            sun_dir_eye: None,
            ..b
        },
        PostConfig::default(),
        8,
        8,
        true,
    );
    assert_eq!(k.sun_dir, [0.0; 4]);
    assert_eq!(k.extent, [8, 8, 1, 0]);
    assert_eq!(k.strength, [0.0; 4]);
}
