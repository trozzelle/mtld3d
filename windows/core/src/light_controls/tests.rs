//! Ports of every case in `dev/lighting/relight/test_core.c` (the parse tests
//! have no native equivalent — settings arrive as config keys, not text).

use mtld3d_types::{D3DCOLORVALUE, D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DLIGHT9};

use super::{
    LightClass, LightControls, Scaled, apply_ambient, apply_ambient_rgb, apply_figure_rows,
    apply_light_rows, classify, figure_map_from_ctab, luma, rows_to_store,
};

const FIXTURE_VS: &[u8] = include_bytes!("fixtures/vs_2759a290.bin");
const FIXTURE_FIGURE: &[u8] = include_bytes!("fixtures/vs_figure_2772.bin");
const FIXTURE_SPEEDTREE: &[u8] = include_bytes!("fixtures/vs_speedtree_388.bin");

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5 * (1.0 + b.abs())
}

fn rgb(r: f32, g: f32, b: f32) -> D3DCOLORVALUE {
    D3DCOLORVALUE { r, g, b, a: 0.0 }
}

/// `point(d, a0, a1, a2)` from test_core.c.
fn point(d: f32, a0: f32, a1: f32, a2: f32) -> D3DLIGHT9 {
    D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        diffuse: rgb(d, d, d),
        range: 1.84467e19,
        attenuation0: a0,
        attenuation1: a1,
        attenuation2: a2,
        ..D3DLIGHT9::default()
    }
}

/// Probe r3/a1: slot 315 sun, 317/318 torch pair, 329 figure head light.
fn sun() -> D3DLIGHT9 {
    D3DLIGHT9 {
        type_: 3,
        diffuse: rgb(0.5, 0.5, 0.5),
        direction: mtld3d_types::D3DVECTOR {
            x: 0.685933,
            y: 0.469672,
            z: 0.555792,
        },
        ..D3DLIGHT9::default()
    }
}

fn torch_a() -> D3DLIGHT9 {
    point(2.0, 0.0, 0.006, 6.78168e-06)
}

fn torch_b() -> D3DLIGHT9 {
    point(0.5, 0.0, 0.006, 6.78168e-06)
}

fn head() -> D3DLIGHT9 {
    point(1.5, 0.0, 0.0236537, 0.0)
}

fn placed() -> D3DLIGHT9 {
    point(1.0, 0.0, 0.0, 1.0 / (512.0 * 512.0))
}

/// Pack a `D3DLIGHT9` into the six `ff_state` rows (`fill_lights_section`
/// layout: +2 diffuse, +4 attenuation and range).
fn pack_rows(l: &D3DLIGHT9) -> [[f32; 4]; 6] {
    let mut rows = [[0.0; 4]; 6];
    rows[2] = [l.diffuse.r, l.diffuse.g, l.diffuse.b, l.diffuse.a];
    rows[4] = [l.attenuation0, l.attenuation1, l.attenuation2, l.range];
    rows
}

fn unpack_rows(rows: &[[f32; 4]; 6], l: &mut D3DLIGHT9) {
    l.diffuse.r = rows[2][0];
    l.diffuse.g = rows[2][1];
    l.diffuse.b = rows[2][2];
    l.attenuation0 = rows[4][0];
    l.attenuation1 = rows[4][1];
    l.attenuation2 = rows[4][2];
    l.range = rows[4][3];
}

fn tokens(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn classify_matches_relight_vectors() {
    let mut spot = placed();
    spot.type_ = D3DLIGHT_SPOT;
    assert_eq!(classify(&sun()), LightClass::Sun);
    assert_eq!(classify(&torch_a()), LightClass::Torch);
    assert_eq!(classify(&torch_b()), LightClass::Torch);
    assert_eq!(classify(&head()), LightClass::Figure);
    assert_eq!(classify(&placed()), LightClass::Placed);
    assert_eq!(classify(&spot), LightClass::Other);
}

#[test]
fn identity_changes_nothing() {
    let c = LightControls::default();
    assert!(c.is_identity());
    let ls = [sun(), torch_a(), torch_b(), head(), placed()];
    for l in ls {
        let orig = l;
        let mut rows = pack_rows(&l);
        apply_light_rows(&c, classify(&l), &mut rows);
        let mut out = l;
        unpack_rows(&rows, &mut out);
        assert_eq!(out.diffuse.r, orig.diffuse.r);
        assert_eq!(out.diffuse.g, orig.diffuse.g);
        assert_eq!(out.diffuse.b, orig.diffuse.b);
        assert_eq!(out.attenuation0, orig.attenuation0);
        assert_eq!(out.attenuation1, orig.attenuation1);
        assert_eq!(out.attenuation2, orig.attenuation2);
        assert_eq!(out.range, orig.range);
    }
    assert_eq!(apply_ambient(&c, 0xff6a_7c99, 0.5), 0xff6a_7c99);
    let mut not_default = LightControls::default();
    not_default.sun.scale_pct = 101;
    assert!(!not_default.is_identity());
}

#[test]
fn torch_scale_and_tint() {
    let c = LightControls {
        torch: Scaled {
            scale_pct: 50,
            tint_pct: [100, 80, 60],
        },
        ..LightControls::default()
    };
    let l = torch_a();
    let mut rows = pack_rows(&l);
    apply_light_rows(&c, LightClass::Torch, &mut rows);
    assert!(near(rows[2][0], 1.0) && near(rows[2][1], 0.8) && near(rows[2][2], 0.6));
    assert!(near(rows[4][1], 0.006) && near(rows[4][2], 6.78168e-06));
}

#[test]
fn placed_range_scales_attenuation_and_range() {
    let c = LightControls {
        placed_range_pct: 200,
        ..LightControls::default()
    };
    // att1/k, att2/k², range·k.
    let l = point(1.0, 0.0, 0.006, 1.0 / (256.0 * 256.0));
    let mut l2 = l;
    l2.range = 100.0;
    let mut rows = pack_rows(&l2);
    apply_light_rows(&c, LightClass::Placed, &mut rows);
    assert!(near(rows[4][1], 0.006 / 2.0));
    assert!(near(rows[4][2], 1.0 / (512.0 * 512.0)));
    assert!(near(rows[4][3], 200.0));
    assert!(near(rows[2][0], 1.0));
    // The C case: att1 = 0, range at the 1.84467e19 sentinel stays.
    let mut rows = pack_rows(&l);
    apply_light_rows(&c, LightClass::Placed, &mut rows);
    assert!(near(rows[4][2], 1.0 / (512.0 * 512.0)));
    assert!(near(rows[2][0], 1.0));
    assert_eq!(rows[4][3], 1.84467e19);
}

#[test]
fn sun_scale_and_tint() {
    // test_core.c apply_sun: scale 1.6, tint (1.0, 0.9, 0.8) on (0.5, 0.5, 0.5).
    let c = LightControls {
        sun: Scaled {
            scale_pct: 160,
            tint_pct: [100, 90, 80],
        },
        ..LightControls::default()
    };
    let l = sun();
    let mut rows = pack_rows(&l);
    rows[1][0] = l.direction.x; // the direction row, untouched by the module
    apply_light_rows(&c, LightClass::Sun, &mut rows);
    assert!(near(rows[2][0], 0.8) && near(rows[2][1], 0.72) && near(rows[2][2], 0.64));
    // The direction row (+1) is untouched.
    assert!(near(rows[1][0], 0.685933));
}

#[test]
fn identity_leaves_out_of_range_ambient_untouched() {
    // Identity is "off": no clamp, no mutation — an out-of-range ambient row
    // (as figure shaders can hold) must come back exactly as it went in.
    let c = LightControls::default();
    let mut a = [1.5, 1.5, 1.5];
    apply_ambient_rgb(&c, &mut a, 0.1);
    assert_eq!(a, [1.5, 1.5, 1.5]);
    let mut dark = [0.0, 0.0, 0.0];
    apply_ambient_rgb(&c, &mut dark, 0.1);
    assert_eq!(dark, [0.0, 0.0, 0.0]);
    assert_eq!(apply_ambient(&c, 0xff6a_7c99, 0.1), 0xff6a_7c99);
}

#[test]
fn ambient_scale_tint_and_black_untouched() {
    // Black ambient (shadow casters, sky) is never changed, even with a floor.
    let c = LightControls {
        ambient: Scaled {
            scale_pct: 200,
            tint_pct: [100; 3],
        },
        night_floor_pct: 30,
        ..LightControls::default()
    };
    assert_eq!(apply_ambient(&c, 0xff00_0000, 0.05), 0xff00_0000);
    // 0x6a7c99 = 106,124,153 -> 53,62,76 (rounded).
    let half = LightControls {
        ambient: Scaled {
            scale_pct: 50,
            tint_pct: [100; 3],
        },
        ..LightControls::default()
    };
    let scaled = apply_ambient(&half, 0xff6a_7c99, 0.5);
    assert!(scaled == 0xff35_3e4d || scaled == 0xff35_3e4c);
    // Tint on top of the scale: (1.0, 0.8, 0.6).
    let tinted = LightControls {
        ambient: Scaled {
            scale_pct: 50,
            tint_pct: [100, 80, 60],
        },
        ..LightControls::default()
    };
    assert_eq!(apply_ambient(&tinted, 0xff6a_7c99, 0.5), 0xff35_322e);
}

#[test]
fn night_floor_only_when_sun_weak() {
    let c = LightControls {
        night_floor_pct: 25,
        ..LightControls::default()
    };
    let mut a = [12.0 / 255.0, 12.0 / 255.0, 38.0 / 255.0];
    apply_ambient_rgb(&c, &mut a, 0.1); // night: sun weak
    assert!(luma(&a) >= 0.25 - 1e-4);
    assert!(near(a[2] / a[0], 38.0 / 12.0)); // hue kept
    let mut d = [12.0 / 255.0, 12.0 / 255.0, 38.0 / 255.0];
    apply_ambient_rgb(&c, &mut d, 0.5); // day: untouched
    assert!(near(d[0], 12.0 / 255.0) && near(d[2], 38.0 / 255.0));
}

#[test]
fn night_floor_uses_unscaled_sun() {
    // The floor decision reads the scene sun luma argument as given; the sun
    // scale never feeds it (relight passes `scene_sun` through unscaled).
    let c = LightControls {
        sun: Scaled {
            scale_pct: 160,
            tint_pct: [100; 3],
        },
        night_floor_pct: 25,
        ..LightControls::default()
    };
    let mut night = [0.2, 0.2, 0.2];
    apply_ambient_rgb(&c, &mut night, 0.1);
    assert!(luma(&night) >= 0.25 - 1e-4);
    let mut day = [0.2, 0.2, 0.2];
    apply_ambient_rgb(&c, &mut day, 0.5);
    assert!(near(day[0], 0.2) && near(day[1], 0.2) && near(day[2], 0.2));
}

#[test]
fn figure_map_from_ctab_fixture() {
    let vs = tokens(FIXTURE_VS);
    let vs2 = tokens(FIXTURE_FIGURE);
    let st = tokens(FIXTURE_SPEEDTREE);
    assert_eq!(super::ctab_register(&vs, "$sunlight_Direction"), Some(12));
    assert_eq!(super::ctab_register(&vs, "$sunlight_Diffuse"), Some(13));
    assert_eq!(super::ctab_register(&vs, "$sunlight_Ambient"), Some(14));
    assert_eq!(super::ctab_register(&vs, "$plight3_Factors"), Some(23));
    assert_eq!(super::ctab_register(&vs, "$nope"), None);
    let m = figure_map_from_ctab(&vs).unwrap();
    assert_eq!(m.sun_diffuse, 13);
    assert_eq!(m.sun_ambient, 14);
    assert_eq!(m.plight_color[0], 16);
    assert_eq!(m.plight_factors[2], 23);
    assert!(figure_map_from_ctab(&vs2).is_some());
    assert!(figure_map_from_ctab(&st).is_none()); // SpeedTree leaves: no CTAB
}

/// r3 vsconst write (reg 0): c12 sun dir, c13 sun 0.5, c14 ambient, c15-17
/// torch, c18-23 two unused lights.
#[test]
fn figure_rows_classify_by_factors() {
    let vs = tokens(FIXTURE_VS);
    let m = figure_map_from_ctab(&vs).unwrap();
    let c = LightControls::default();
    let mut v = [[0.0; 4]; 24];
    let block: [f32; 48] = [
        -0.700409,
        -0.482137,
        -0.52628,
        1.0,
        0.5,
        0.5,
        0.5,
        1.0,
        0.418824,
        0.489412,
        0.6,
        0.0,
        43846.0,
        25553.0,
        -4916.8,
        1.0,
        2.0,
        2.0,
        2.0,
        1.0,
        0.0,
        0.006,
        6.78168e-06,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ];
    v[12..24].copy_from_slice(
        &block
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect::<Vec<_>>(),
    );
    let orig = v;
    apply_figure_rows(&c, &m, 0, 0.5, &mut v);
    assert_eq!(v, orig); // identity
    let c = LightControls {
        sun: Scaled {
            scale_pct: 160,
            tint_pct: [100; 3],
        },
        ambient: Scaled {
            scale_pct: 50,
            tint_pct: [100; 3],
        },
        torch: Scaled {
            scale_pct: 50,
            tint_pct: [100; 3],
        },
        ..LightControls::default()
    };
    let mut v = orig;
    apply_figure_rows(&c, &m, 0, 0.5, &mut v);
    assert!(near(v[13][0], 0.8) && near(v[13][3], 1.0)); // sun diffuse, w kept
    assert!(near(v[14][1], 0.489412 * 0.5)); // ambient
    assert!(near(v[16][0], 1.0) && near(v[16][3], 1.0)); // torch colour
    assert!(near(v[12][0], -0.700409)); // direction untouched
    assert!(near(v[19][0], 0.0) && near(v[20][0], 1.0)); // unused light untouched
    // A write that starts past the lights changes nothing.
    let mut past = [[0.0; 4]; 28];
    past[..24].copy_from_slice(&orig);
    apply_figure_rows(&c, &m, 24, 0.5, &mut past[24..]);
    assert_eq!(&past[..24], &orig[..]);
}

fn rows_to_store_controls() -> LightControls {
    LightControls {
        sun: Scaled {
            scale_pct: 160,
            tint_pct: [100; 3],
        },
        torch: Scaled {
            scale_pct: 50,
            tint_pct: [100; 3],
        },
        ..LightControls::default()
    }
}

/// The game's r3 write (reg 0, 24 rows): c13 sun 0.5, c16 torch 2.0 with its
/// factors at c17 — a point light classifies by its factors, which must be in
/// the same write.
fn rows_to_store_game() -> [[f32; 4]; 24] {
    let mut game = [[0.0; 4]; 24];
    game[3] = [9.0, 8.0, 7.0, 6.0];
    game[13] = [0.5, 0.5, 0.5, 1.0];
    game[16] = [2.0, 2.0, 2.0, 1.0];
    game[17] = [0.0, 0.006, 6.78168e-06, 1.0];
    game
}

/// Identity controls or no figure map: the stored rows are the game's slice
/// itself and the scratch is never touched.
#[test]
fn rows_to_store_stock_path_returns_game_rows() {
    let vs = tokens(FIXTURE_VS);
    let m = figure_map_from_ctab(&vs).unwrap();
    let game = rows_to_store_game();
    let sentinel = [-7.0f32; 4];
    let mut scratch = [sentinel; 32];
    let out = rows_to_store(
        &LightControls::default(),
        Some(&m),
        0,
        &game,
        0.5,
        &mut scratch,
    );
    assert!(std::ptr::eq(out, &game[..]));
    assert_eq!(out.len(), game.len());
    assert!(scratch.iter().all(|r| *r == sentinel));
    let c = rows_to_store_controls();
    let out = rows_to_store(&c, None, 0, &game, 0.5, &mut scratch);
    assert!(std::ptr::eq(out, &game[..]));
    assert_eq!(out.len(), game.len());
    assert!(scratch.iter().all(|r| *r == sentinel));
}

/// The record-time figure patch the `SetVertexShaderConstantF` thunk stores,
/// records and propagates: the game's rows patched once through the map,
/// computed from the game's rows — so storing the same game rows again yields
/// identical rows (no compounding); Apply replays them verbatim.
#[test]
fn rows_to_store_figure_path_patches_once() {
    let vs = tokens(FIXTURE_VS);
    let m = figure_map_from_ctab(&vs).unwrap();
    let c = rows_to_store_controls();
    let game = rows_to_store_game();
    let sentinel = [-7.0f32; 4];
    let mut scratch = [sentinel; 32];
    let first: Vec<[f32; 4]> = rows_to_store(&c, Some(&m), 0, &game, 0.5, &mut scratch).to_vec();
    assert_eq!(first.len(), game.len());
    let mut expected = game;
    apply_figure_rows(&c, &m, 0, 0.5, &mut expected);
    assert_eq!(first, expected);
    assert!(near(first[13][0], 0.8) && near(first[13][3], 1.0));
    assert!(near(first[16][0], 1.0));
    // Rows outside the mapped registers are the game's.
    assert_eq!(first[3], game[3]);
    assert_eq!(first[0], game[0]);
    assert!(scratch[game.len()..].iter().all(|r| *r == sentinel));
    // A second store from the same game rows gives identical output.
    let second = rows_to_store(&c, Some(&m), 0, &game, 0.5, &mut scratch);
    assert_eq!(second, &first[..]);
    assert_eq!(game, rows_to_store_game());
}
