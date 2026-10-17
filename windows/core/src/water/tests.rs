//! The reflective-water match rule, its config gate and the water uniform.

use mtld3d_types::{
    D3DCOLORVALUE, D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT9, D3DMATRIX, D3DRS_LIGHTING,
    D3DTA_CURRENT, D3DTA_DIFFUSE, D3DTA_TEXTURE, D3DTA_TFACTOR, D3DTOP_BUMPENVMAP, D3DTOP_MODULATE,
    D3DTOP_SELECTARG1, D3DTS_VIEW, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP,
    D3DTSS_TCI_CAMERASPACEPOSITION, D3DTSS_TEXCOORDINDEX, D3DTSS_TEXTURETRANSFORMFLAGS,
    D3DTTFF_COUNT3, D3DTTFF_PROJECTED, D3DVECTOR, RENDER_STATE_COUNT, render_state_defaults,
};

use super::{
    WATER_BYTES, WaterConfig, WaterSun, build_water_bytes, is_reflective_water, water_flag,
};
use crate::{
    convert::{FfVsLayout, FfVsLayoutFlags},
    dxso::{FfPsKey, FfVsKey},
    ff_state::FfState,
};
/// The water mesh's layout: position, diffuse and one 2D coordinate set.
const WATER_LAYOUT: FfVsLayout = FfVsLayout {
    flags: FfVsLayoutFlags::HAS_COLOR0,
    tex_coord_count: 1,
    tex_coord_dims: [2, 0, 0, 0, 0, 0, 0, 0],
    declared_weights_count: 0,
};

/// Stages 0 and 1 carry textures: the V8U8 ripple map and the reflection target.
const WATER_TEXTURES: u8 = 0b011;

fn unlit() -> [u32; RENDER_STATE_COUNT] {
    let mut rs = render_state_defaults();
    rs[D3DRS_LIGHTING as usize] = 0;
    rs
}

fn set(state: &mut FfState, stage: usize, ty: u32, value: u32) {
    state.set_texture_stage_state(stage, ty as usize, value);
}

/// The FF state of the a1 reflective water draw: stage 0
/// `BUMPENVMAP(TEXTURE, DIFFUSE)` on the ripple map, stage 1
/// `MODULATE(TEXTURE, CURRENT)` on the reflection target through
/// `CAMERASPACEPOSITION` and `COUNT3 | PROJECTED`, stage 2
/// `MODULATE(CURRENT, TFACTOR)`.
pub(crate) fn a1_water_state() -> FfState {
    let mut state = FfState::new();
    set(&mut state, 0, D3DTSS_COLOROP, D3DTOP_BUMPENVMAP);
    set(&mut state, 0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    set(&mut state, 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
    set(&mut state, 1, D3DTSS_COLOROP, D3DTOP_MODULATE);
    set(&mut state, 1, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    set(&mut state, 1, D3DTSS_COLORARG2, D3DTA_CURRENT);
    set(
        &mut state,
        1,
        D3DTSS_TEXCOORDINDEX,
        D3DTSS_TCI_CAMERASPACEPOSITION | 1,
    );
    set(
        &mut state,
        1,
        D3DTSS_TEXTURETRANSFORMFLAGS,
        D3DTTFF_COUNT3 | D3DTTFF_PROJECTED,
    );
    set(&mut state, 2, D3DTSS_COLOROP, D3DTOP_MODULATE);
    set(&mut state, 2, D3DTSS_COLORARG1, D3DTA_CURRENT);
    set(&mut state, 2, D3DTSS_COLORARG2, D3DTA_TFACTOR);
    state
}

fn keys(state: &FfState, rs: &[u32; RENDER_STATE_COUNT]) -> (FfVsKey, FfPsKey) {
    (
        state.build_vs_key(rs, WATER_LAYOUT, WATER_TEXTURES, [0; 8]),
        state.build_ps_key(rs, WATER_TEXTURES),
    )
}

/// The a1 water draw's FF VS and PS keys.
pub(crate) fn a1_water_keys() -> (FfVsKey, FfPsKey) {
    keys(&a1_water_state(), &unlit())
}

const FRESNEL: WaterConfig = WaterConfig {
    fresnel: true,
    glint_pct: 0,
};

#[test]
fn water_rule_matches_reflective_water() {
    let (vs, ps) = a1_water_keys();
    assert!(is_reflective_water(&vs, &ps));
    // The config gates the key bit: off draws stay stock.
    assert!(!water_flag(&WaterConfig::default(), &vs, &ps));
    assert!(water_flag(&FRESNEL, &vs, &ps));
    let glint = WaterConfig {
        fresnel: false,
        glint_pct: 50,
    };
    assert!(water_flag(&glint, &vs, &ps));
}

#[test]
fn water_rule_rejects_classic_and_lava() {
    // Classic water: the Rosetta ripple path samples a 256² target with
    // SELECTARG1 and no bump-environment stage.
    let mut classic = FfState::new();
    set(&mut classic, 0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
    set(&mut classic, 0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    let (vs, ps) = keys(&classic, &unlit());
    assert!(!is_reflective_water(&vs, &ps), "classic water");
    assert!(!water_flag(&FRESNEL, &vs, &ps), "classic water");

    // Lava: the same bump-environment cascade, but stage 1 samples its own
    // coordinate set instead of the projective reflection lookup.
    let mut lava = a1_water_state();
    set(&mut lava, 1, D3DTSS_TEXCOORDINDEX, 1);
    set(&mut lava, 1, D3DTSS_TEXTURETRANSFORMFLAGS, 0);
    let (vs, ps) = keys(&lava, &unlit());
    assert!(!is_reflective_water(&vs, &ps), "lava");

    // Each half of the projective lookup is required on its own.
    let mut no_tci = a1_water_state();
    set(&mut no_tci, 1, D3DTSS_TEXCOORDINDEX, 1);
    let (vs, ps) = keys(&no_tci, &unlit());
    assert!(
        !is_reflective_water(&vs, &ps),
        "PROJECTED without CAMERASPACEPOSITION"
    );
    let mut unprojected = a1_water_state();
    set(
        &mut unprojected,
        1,
        D3DTSS_TEXTURETRANSFORMFLAGS,
        D3DTTFF_COUNT3,
    );
    let (vs, ps) = keys(&unprojected, &unlit());
    assert!(!is_reflective_water(&vs, &ps), "COUNT3 without PROJECTED");

    // A lit draw with the water stages is not the unlit water draw.
    let (vs, ps) = keys(&a1_water_state(), &render_state_defaults());
    assert!(vs.lighting_enabled());
    assert!(!is_reflective_water(&vs, &ps), "lit");

    // The opacity stage and both textures belong to the cascade.
    let mut no_factor = a1_water_state();
    set(&mut no_factor, 2, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
    let (vs, ps) = keys(&no_factor, &unlit());
    assert!(!is_reflective_water(&vs, &ps), "stage 2 without TFACTOR");
    let state = a1_water_state();
    let rs = unlit();
    let vs = state.build_vs_key(&rs, WATER_LAYOUT, 0b001, [0; 8]);
    let ps = state.build_ps_key(&rs, 0b001);
    assert!(!is_reflective_water(&vs, &ps), "no reflection target");
}

fn lanes(bytes: &[u8; WATER_BYTES]) -> [f32; 8] {
    let mut out = [0.0; 8];
    for (lane, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *lane = f32::from_le_bytes(*chunk);
    }
    out
}

fn light(type_: u32, direction: [f32; 3], diffuse: [f32; 3]) -> D3DLIGHT9 {
    D3DLIGHT9 {
        type_,
        diffuse: D3DCOLORVALUE {
            r: diffuse[0],
            g: diffuse[1],
            b: diffuse[2],
            a: 1.0,
        },
        direction: D3DVECTOR {
            x: direction[0],
            y: direction[1],
            z: direction[2],
        },
        ..D3DLIGHT9::default()
    }
}

#[test]
fn water_uniform_bytes() {
    let cfg = WaterConfig {
        fresnel: true,
        glint_pct: 150,
    };
    // No sun: the sun row and colour are zero, the strengths stay.
    assert_eq!(
        lanes(&build_water_bytes(&cfg, None)),
        [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        lanes(&build_water_bytes(&WaterConfig::default(), None)),
        [0.0; 8]
    );
    let sun = WaterSun {
        dir_eye: [0.0, 0.6, 0.8],
        diffuse: [0.9, 0.8, 0.7],
    };
    assert_eq!(
        lanes(&build_water_bytes(&cfg, Some(sun))),
        [0.0, 0.6, 0.8, 1.5, 0.9, 0.8, 0.7, 1.0]
    );

    // The stored state's sun: the lowest-index enabled directional light,
    // its direction through the view and normalized like the FF light rows.
    let mut state = FfState::new();
    assert_eq!(state.scene_sun_eye(), None);
    // A view swapping world Y and Z (row-vector convention).
    let mut view = D3DMATRIX::IDENTITY;
    view.m[5] = 0.0;
    view.m[6] = 1.0;
    view.m[9] = 1.0;
    view.m[10] = 0.0;
    assert!(state.set_transform(D3DTS_VIEW, &view));
    state.set_light(0, &light(D3DLIGHT_POINT, [0.0, 0.0, 1.0], [1.0; 3]));
    state.set_light_enabled(0, true);
    state.set_light(
        2,
        &light(D3DLIGHT_DIRECTIONAL, [0.0, 0.0, 2.0], [0.5, 0.4, 0.3]),
    );
    state.set_light(4, &light(D3DLIGHT_DIRECTIONAL, [1.0, 0.0, 0.0], [1.0; 3]));
    state.set_light_enabled(4, true);
    // Light 2 is defined but not enabled: light 4 is the sun.
    assert_eq!(
        state.scene_sun_eye(),
        Some(WaterSun {
            dir_eye: [1.0, 0.0, 0.0],
            diffuse: [1.0; 3],
        })
    );
    state.set_light_enabled(2, true);
    let sun = state.scene_sun_eye().expect("light 2 is the sun");
    assert_eq!(sun.dir_eye, [0.0, 1.0, 0.0]);
    assert_eq!(sun.diffuse, [0.5, 0.4, 0.3]);
    assert_eq!(
        lanes(&build_water_bytes(&cfg, Some(sun))),
        [0.0, 1.0, 0.0, 1.5, 0.5, 0.4, 0.3, 1.0]
    );
}
