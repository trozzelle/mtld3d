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
    assert_eq!(size_of::<super::PostConstants>(), 96);
    let b = mtld3d_core::encoder_data::PostBoundary {
        proj: [1.5, 2.0, 1.001, -0.1],
        sun_dir_eye: Some([-0.3, -0.4, -0.8]),
        sun_color: [0.9, 0.8, 0.7],
        sun_above_horizon: true,
    };
    let cfg = PostConfig {
        ao_pct: 125,
        god_rays_pct: 50,
        ..PostConfig::default()
    };
    let k = super::PostConstants::new(&b, cfg, 640, 480, false);
    assert_eq!(k.proj, b.proj);
    assert_eq!(k.sun_dir, [-0.3, -0.4, -0.8, 1.0]);
    assert_eq!(k.sun_color, [0.9, 0.8, 0.7, 0.0]);
    assert_eq!(k.extent, [640, 480, 0, 0]);
    assert_eq!(k.strength, [1.25, 0.5, 0.0, 0.0]);
    let [u, v, visible, zero] = k.sun_screen;
    assert!((u - 0.781_25).abs() < 1e-6 && v.abs() < 1e-6, "{u} {v}");
    assert_eq!((visible, zero), (1.0, 0.0));
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
    assert_eq!(k.sun_screen, [0.0; 4]);
}

/// God rays need a sun above the horizon, in front of the camera, that
/// projects near the screen; otherwise the chain drops them, says why, and
/// nothing of theirs is dispatched.
#[test]
fn godrays_off_without_visible_sun() {
    use mtld3d_core::encoder_data::PostBoundary;

    use super::{SunSkip, sun_screen};
    let cfg = PostConfig {
        debug: true,
        god_rays_pct: 100,
        ..PostConfig::default()
    };
    let chain = enabled_effects(cfg);
    assert_eq!(chain.as_slice(), [Effect::GodRays, Effect::Invert]);
    assert!(Effect::GodRays.reads_depth());
    let sun = |dir: Option<[f32; 3]>| PostBoundary {
        proj: [1.5, 2.0, 1.001, -0.1],
        sun_dir_eye: dir,
        sun_color: [1.0, 0.9, 0.8],
        sun_above_horizon: dir.is_some(),
    };
    // The light travels from the sun toward the scene, so the sun lies along
    // −direction: here at eye (0.3, 0.4, 0.8), NDC (0.5625, 1.0).
    let visible = sun(Some([-0.3, -0.4, -0.8]));
    let [u, v] = sun_screen(&visible).expect("a sun ahead, on screen");
    assert!((u - 0.781_25).abs() < 1e-6 && v.abs() < 1e-6, "{u} {v}");
    let (kept, skip) = chain.for_sun(&visible);
    assert_eq!((kept.as_slice(), skip), (chain.as_slice(), None));
    // Off screen but inside the 1.5× margin still runs: NDC x 1.4.
    let margin = sun(Some([-1.4, 0.0, -1.5]));
    let [u, _] = sun_screen(&margin).expect("inside the margin");
    assert!((u - 1.2).abs() < 1e-5, "{u}");
    let hidden = [
        ("no sun", sun(None), SunSkip::NoSun),
        // The same on-screen sun after it set: a camera pitched down sees
        // the open sky streak toward a point in the terrain otherwise.
        (
            "below the horizon",
            PostBoundary {
                sun_above_horizon: false,
                ..visible
            },
            SunSkip::BelowHorizon,
        ),
        // Straight behind the camera: NDC (0, 0) if the sign were ignored.
        ("behind", sun(Some([0.0, 0.0, 1.0])), SunSkip::BehindCamera),
        (
            "behind and up",
            sun(Some([0.0, -0.2, 0.9])),
            SunSkip::BehindCamera,
        ),
        (
            "in the eye plane",
            sun(Some([0.0, -1.0, 0.0])),
            SunSkip::BehindCamera,
        ),
        (
            "right of the margin",
            sun(Some([-2.0, 0.0, -1.5])),
            SunSkip::OffScreen { ndc: [2.0, 0.0] },
        ),
        (
            "below the margin",
            sun(Some([0.0, 0.8, -1.0])),
            SunSkip::OffScreen { ndc: [0.0, -1.6] },
        ),
    ];
    for (what, boundary, reason) in hidden {
        assert_eq!(sun_screen(&boundary), Err(reason), "{what}");
        let (kept, skip) = chain.for_sun(&boundary);
        assert_eq!(kept.as_slice(), [Effect::Invert], "{what}");
        assert_eq!(skip, Some(reason), "{what}");
        let k = super::PostConstants::new(&boundary, cfg, 8, 8, true);
        assert_eq!(k.sun_screen, [0.0; 4], "{what}");
    }
    // Without god rays the sun changes nothing and gives no reason.
    let ao = enabled_effects(PostConfig {
        ao_pct: 100,
        ..PostConfig::default()
    });
    let (kept, skip) = ao.for_sun(&sun(None));
    assert_eq!((kept.as_slice(), skip), ([Effect::Ao].as_slice(), None));
}
