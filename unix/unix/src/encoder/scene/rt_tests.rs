//! Ray-traced sun visibility through the scene passes, and its denoiser, on the GPU.

use std::sync::atomic::{AtomicU64, Ordering};

use mtld3d_core::{
    config::Mtld3dConfig,
    encoder_data::PostBoundary,
    passes::Pass,
    rt::{RtConfig, RtQuality},
    shadows::ShadowConfig,
};
use mtld3d_shared::{BlitCommand, BlitCommandType, MetalHandle};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLBuffer;

use super::{
    assemble,
    tests::{
        CAMERA_HEIGHT, Quad, Seen, VIS_RASTER, VisScene, begin, daoc_proj, plate_scene, sun,
        where_seen,
    },
};
use crate::{
    encoder::{
        FrameEncoder,
        replay::{
            neutral_pos_fixup,
            tests::{Fixture, Gpu, half_to_f32, replayed, snapshot},
        },
        scene_capture::BufferContent,
        tests::encoder_with_config,
    },
    metal::{
        denoise::DenoiseKind,
        handle::IntoRetained,
        rt::rt_supported,
        scene_vis::{Kernel, SceneDispatch},
    },
};

/// `rt.shadows` with the sun `sun_size_pct` hundredths of a degree across, at `quality`.
fn rt_sun(sun_size_pct: u16, quality: RtQuality) -> Mtld3dConfig {
    Mtld3dConfig {
        rt: RtConfig {
            shadows: true,
            sun_size_pct,
            quality,
            ..RtConfig::default()
        },
        ..Mtld3dConfig::default()
    }
}

/// The default `rt.shadows`: a sun one degree across, medium quality.
fn rt_default() -> Mtld3dConfig {
    rt_sun(100, RtQuality::Medium)
}

/// A plate 60 units square 100 units over the floor, the sun overhead: a
/// penumbra a few pixels wide under a large sun.
fn high_plate_scene() -> VisScene {
    VisScene {
        quads: vec![
            Quad::floor(),
            Quad::level((-10.0, 50.0), (120.0, 180.0), CAMERA_HEIGHT - 100.0),
        ],
        ..plate_scene()
    }
}

/// `scene` with its plate (quad 1) moved `dx` world units east.
fn moved(scene: &VisScene, dx: f64) -> VisScene {
    let mut quads = scene.quads.clone();
    quads[1].origin[0] += dx;
    VisScene {
        quads,
        camera: super::tests::Camera::new(),
        sun: scene.sun,
    }
}

/// An encoder tracing `VisScene`s frame after frame, and the GPU that runs them.
struct Session {
    enc: FrameEncoder,
    gpu: Gpu,
    fixture: Fixture,
    /// Quad `i`'s own fixture when given (a coloured material), else `fixture`.
    materials: Vec<Option<Fixture>>,
    /// What the boundary carries beside the scene's sun.
    light: Light,
    /// The scene's vertex buffers, made once per scene (by its address).
    buffers: Vec<(usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>)>,
    // The encoder reads both through raw pointers: boxed, they never move.
    coherent: Box<AtomicU64>,
    _retained: Box<AtomicU64>,
}

/// The lights of a session's boundary.
#[derive(Clone)]
struct Light {
    /// The scene's sun shines (white); `false`: no sun at all.
    sun: bool,
    ambient: [f32; 3],
    fog: [f32; 3],
    /// Point lights: world position and range.
    points: Vec<(super::tests::V3, f32)>,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            sun: true,
            ambient: [0.2; 3],
            fog: [0.0; 3],
            points: Vec::new(),
        }
    }
}

/// How one submission of [`Session::frame`] ends.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Submission {
    /// A whole world under the boundary.
    Whole,
    /// A world cut by a mid-frame flush before the boundary.
    Split,
    /// No world draw at all, the boundary seen (a submission of the UI alone).
    Empty,
}

impl Session {
    /// A session under `config`; `None` without a device that traces rays.
    fn new(config: Mtld3dConfig) -> Option<Self> {
        let coherent = Box::new(AtomicU64::new(0));
        let retained = Box::new(AtomicU64::new(0));
        let (device, mut enc) = encoder_with_config(&coherent, &retained, config)?;
        if !rt_supported(&device) {
            eprintln!("the device does not support ray tracing, skipping");
            return None;
        }
        enc.current_submit_seq = 1;
        Some(Self {
            enc,
            gpu: Gpu::new(device),
            fixture: Fixture::with_camera(
                super::tests::DAOC_NEAR,
                super::tests::DAOC_FAR,
                VIS_RASTER,
            ),
            materials: Vec::new(),
            light: Light::default(),
            buffers: Vec::new(),
            coherent,
            _retained: retained,
        })
    }

    /// Quad `quad` of every scene lit with the material diffuse `rgb`.
    fn material(&mut self, quad: usize, rgb: [f32; 3]) {
        if self.materials.len() <= quad {
            self.materials.resize_with(quad + 1, || None);
        }
        self.materials[quad] = Some(
            Fixture::with_camera(super::tests::DAOC_NEAR, super::tests::DAOC_FAR, VIS_RASTER)
                .with_diffuse(rgb),
        );
    }

    /// The vertex buffers of `scene`, made on first use.
    fn buffers_of(&mut self, scene: &VisScene) -> Vec<u64> {
        let key = core::ptr::from_ref(scene) as usize;
        if !self.buffers.iter().any(|(k, _)| *k == key) {
            let made = scene
                .quads
                .iter()
                .map(|q| self.gpu.buffer(&q.vertices(&scene.camera)))
                .collect();
            self.buffers.push((key, made));
        }
        let (_, buffers) = self
            .buffers
            .iter()
            .find(|(k, _)| *k == key)
            .expect("made above");
        buffers.iter().map(|b| Retained::as_ptr(b) as u64).collect()
    }

    /// One submission of `scene` seen through `view` (the camera's own when
    /// `None`), assembled and run on the GPU; the world pass's leading blits.
    fn frame(
        &mut self,
        scene: &VisScene,
        view: Option<[[f32; 4]; 4]>,
        submission: Submission,
    ) -> Vec<BlitCommand> {
        // The GPU finished every earlier submission.
        self.coherent
            .store(self.enc.current_submit_seq, Ordering::Release);
        self.enc.current_submit_seq += 1;
        begin(&mut self.enc, (VIS_RASTER, VIS_RASTER), 1, false);
        let buffers = if submission == Submission::Empty {
            Vec::new()
        } else {
            self.buffers_of(scene)
        };
        let snapshot = snapshot(0, 0);
        self.enc
            .pass_state
            .emit_command(mtld3d_shared::Command::draw_primitives(
                mtld3d_shared::mtl::PrimitiveType::Triangle,
                0,
                3,
            ));
        let draws: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, &buffer)| {
                // A quad of its own material draws with its own pipeline.
                let (fixture, pipeline) = match self.materials.get(i).and_then(Option::as_ref) {
                    Some(fixture) => (fixture, 0xA0 + i as u64),
                    None => (&self.fixture, 0x91),
                };
                fixture.capture_with(
                    &mut self.enc,
                    &snapshot,
                    (pipeline, buffer),
                    None,
                    BufferContent {
                        id: buffer,
                        version: 1,
                    },
                )
            })
            .collect();
        self.enc.pass_state.end_current_pass("test");
        for d in &draws {
            replayed(&mut self.enc, d, neutral_pos_fixup());
        }
        if submission != Submission::Split {
            let sun = scene.camera.to_eye(scene.sun).map(|v| v as f32);
            let light = &self.light;
            let mut points = [mtld3d_core::rt::ScenePoint::default(); 4];
            for (point, &(at, range)) in points.iter_mut().zip(&light.points) {
                point.position = scene.camera.to_eye(at).map(|v| v as f32);
                point.range = range;
                point.diffuse = [1.0; 3];
                point.attenuation = [1.0, 0.0, 0.0];
            }
            self.enc.post_boundary(&PostBoundary {
                proj: daoc_proj(),
                sun_dir_eye: light.sun.then_some(sun),
                sun_color: if light.sun { [1.0; 3] } else { [0.0; 3] },
                sun_above_horizon: light.sun,
                view: view.unwrap_or_else(|| scene.camera.view()),
                ambient: light.ambient,
                frame: 0,
                points,
                fog_color: light.fog,
            });
        }
        assemble(&mut self.enc, submission == Submission::Split);
        let passes = self.enc.pass_state.passes();
        let (world, before) = passes.split_last().expect("the world pass");
        assert!(before.iter().all(Pass::is_scene));
        let mut descriptors: Vec<_> = before
            .iter()
            .map(|pass| crate::encoder::pass_to_descriptor(pass, MetalHandle::NULL))
            .collect();
        let leading = world.leading_blits().to_vec();
        descriptors.push(crate::encoder::trailing_blit_descriptor(&leading));
        self.gpu.run_descriptors(&descriptors);
        leading
    }

    /// `frames` whole frames of `scene` under its own camera.
    fn frames(&mut self, scene: &VisScene, frames: usize) {
        for _ in 0..frames {
            self.frame(scene, None, Submission::Whole);
        }
    }

    /// The `R16Float` scene texture `texture`, every texel.
    fn read(&self, texture: MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind>) -> Vec<f32> {
        let texture = texture.into_retained().expect("a live scene texture");
        self.gpu.halves(&texture, VIS_RASTER)
    }

    /// The sun visibility the receivers read.
    fn vis(&self) -> Vec<f32> {
        self.read(self.enc.scene_textures().expect("scene textures").vis)
    }

    /// The ambient occlusion the receivers read.
    fn ao(&self) -> Vec<f32> {
        self.read(self.enc.scene_textures().expect("scene textures").ao)
    }

    /// The `RGBA16Float` scene texture `texture`, every texel.
    fn read4(
        &self,
        texture: MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind>,
    ) -> Vec<[f32; 4]> {
        let texture = texture.into_retained().expect("a live scene texture");
        self.gpu
            .texels_at(&texture, VIS_RASTER, 0, 8)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|t| {
                let lane = |i: usize| half_to_f32(u16::from_le_bytes([t[2 * i], t[2 * i + 1]]));
                [lane(0), lane(1), lane(2), lane(3)]
            })
            .collect()
    }

    /// The one-bounce sunlight the receivers read.
    fn gi(&self) -> Vec<[f32; 4]> {
        self.read4(self.enc.scene_textures().expect("scene textures").gi)
    }

    /// The point-light visibility the receivers read.
    fn points(&self) -> Vec<[f32; 4]> {
        self.read4(self.enc.scene_textures().expect("scene textures").points)
    }

    /// The sun history's length (frames merged) at every pixel, this frame's.
    fn history_length(&self) -> Vec<f32> {
        let rt = self
            .enc
            .scene_textures()
            .and_then(|t| t.rt)
            .expect("ray-traced targets");
        let texture = rt
            .kind(DenoiseKind::Sun)
            .expect("the sun's targets")
            .history
            .length[self.enc.scene_targets.history.current()]
        .into_retained()
        .expect("a live scene texture");
        // `RGBA16Float`: the length first.
        self.gpu
            .texels_at(&texture, VIS_RASTER, 0, 8)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|texel| half_to_f32(u16::from_le_bytes([texel[0], texel[1]])))
            .collect()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.enc.shutdown_cleanup();
    }
}

/// The described dispatches of `blits` in order: `(kernel, extent, dispatch)`.
fn dispatches(blits: &[BlitCommand]) -> Vec<(Kernel, (u32, u32), SceneDispatch)> {
    blits
        .iter()
        .filter(|b| b.cmd == BlitCommandType::SceneKernel as u32)
        .map(|b| {
            let kernel = Kernel::from_id(b.mip_level).expect("a scene kernel");
            assert!(
                kernel.described(),
                "{kernel:?} reads a dispatch description"
            );
            // SAFETY: the description the encoder wrote in its live scratch arena.
            let dispatch =
                unsafe { core::ptr::read_unaligned(b.byte_size as *const SceneDispatch) };
            (kernel, (b.region_w, b.region_h), dispatch)
        })
        .collect()
}

/// Pixels whose visibility is partly lit: inside `(0.1, 0.9)`, on the floor.
fn penumbra(scene: &VisScene, vis: &[f32]) -> usize {
    (0..VIS_RASTER * VIS_RASTER)
        .filter(|&i| {
            let (x, y) = (i % VIS_RASTER, i / VIS_RASTER);
            let v = vis[i as usize];
            scene.seen_at(x, y, 0.0).quad() == Some(0) && v > 0.1 && v < 0.9
        })
        .count()
}

/// The Task 6 scene traced: with the sun straight down, the floor under the
/// plate is in shadow (≈ 0), the rest of the floor and the plate's top in
/// full sun, on the very first frame (no history). The penumbra widens with
/// `rt.sunSize`: a sun 4° across leaves many more partly lit floor pixels
/// than one a quarter degree across, once the history has settled.
#[test]
fn rt_vis_occluder_shadows_floor() {
    let scene = plate_scene();
    let Some(mut session) = Session::new(rt_default()) else {
        return;
    };
    session.frames(&scene, 1);
    let vis = session.vis();
    let seen = scene.seen(6.0);
    let shadowed = where_seen(&vis, &seen, |s| s == Seen::Shadowed(0));
    assert!(shadowed.len() >= 20, "{} shadowed pixels", shadowed.len());
    let darkest = shadowed.iter().copied().fold(0.0f32, f32::max);
    assert!(darkest <= 0.05, "under the plate: up to {darkest}");
    let lit = where_seen(&vis, &seen, |s| matches!(s, Seen::Lit(_)));
    assert!(lit.len() >= 2000, "{} lit pixels", lit.len());
    let dimmest = lit.iter().copied().fold(1.0f32, f32::min);
    assert!(dimmest >= 0.99, "in the sun: down to {dimmest}");
    let plate = where_seen(&vis, &seen, |s| s == Seen::Lit(1));
    assert!(plate.len() >= 20, "{} plate pixels", plate.len());
    let sky = where_seen(&vis, &scene.seen(0.0), |s| s == Seen::Sky);
    assert!(
        sky.len() >= 500 && sky.iter().all(|&v| v == 1.0),
        "the sky is in full sun"
    );
    drop(session);

    let scene = high_plate_scene();
    let partial = |sun_size_pct| {
        let mut session = Session::new(rt_sun(sun_size_pct, RtQuality::Medium))?;
        session.frames(&scene, 24);
        Some(penumbra(&scene, &session.vis()))
    };
    let (Some(small), Some(large)) = (partial(25), partial(400)) else {
        return;
    };
    assert!(
        large >= 40 && large >= 3 * small,
        "a 4° sun: {large} penumbra pixels, a 0.25° sun: {small}"
    );
}

/// Half resolution (`rt.quality = low`) traces a quarter of the pixels and
/// upsamples by depth: the same shadow and sun, no sky bleeding into the
/// plate's edge. Medium and high trace every pixel, with one and two rays.
#[test]
fn rt_quality_sets_resolution_and_rays() {
    let scene = plate_scene();
    for (quality, extent, rays) in [
        (RtQuality::Low, VIS_RASTER / 2, 1),
        (RtQuality::Medium, VIS_RASTER, 1),
        (RtQuality::High, VIS_RASTER, 2),
    ] {
        let Some(mut session) = Session::new(rt_sun(100, quality)) else {
            return;
        };
        let blits = session.frame(&scene, None, Submission::Whole);
        let rays_dispatch = dispatches(&blits)
            .into_iter()
            .find(|(kernel, ..)| *kernel == Kernel::SunVisRays)
            .expect("the rays");
        assert_eq!(rays_dispatch.1, (extent, extent), "{quality:?}");
        assert_eq!(rays_dispatch.2.constants.trace[1], rays, "{quality:?}");
        assert_eq!(rays_dispatch.2.constants.trace[2], VIS_RASTER / extent);
        let vis = session.vis();
        // Half resolution resolves edges to two pixels.
        let seen = scene.seen(9.0);
        let shadowed = where_seen(&vis, &seen, |s| s == Seen::Shadowed(0));
        assert!(
            shadowed.len() >= 20,
            "{quality:?}: {} shadowed",
            shadowed.len()
        );
        let darkest = shadowed.iter().copied().fold(0.0f32, f32::max);
        assert!(
            darkest <= 0.05,
            "{quality:?}: under the plate up to {darkest}"
        );
        let lit = where_seen(&vis, &seen, |s| matches!(s, Seen::Lit(_)));
        let dimmest = lit.iter().copied().fold(1.0f32, f32::min);
        assert!(dimmest >= 0.99, "{quality:?}: in the sun down to {dimmest}");
    }
}

/// A static view of a large sun's penumbra settles: over 16 frames after
/// the history filled, every pixel's visibility varies by less than 1e-3,
/// the penumbra included.
#[test]
fn denoise_converges_static() {
    let scene = high_plate_scene();
    let Some(mut session) = Session::new(rt_sun(400, RtQuality::Medium)) else {
        return;
    };
    session.frames(&scene, 16);
    let frames: Vec<Vec<f32>> = (0..16)
        .map(|_| {
            session.frames(&scene, 1);
            session.vis()
        })
        .collect();
    let pixels = frames[0].len();
    let mut worst = (0.0f32, 0usize);
    let mut mean_of = vec![0.0f32; pixels];
    for (i, mean) in mean_of.iter_mut().enumerate() {
        *mean = frames.iter().map(|f| f[i]).sum::<f32>() / 16.0;
        let variance = frames.iter().map(|f| (f[i] - *mean).powi(2)).sum::<f32>() / 16.0;
        if variance > worst.0 {
            worst = (variance, i);
        }
    }
    assert!(
        penumbra(&scene, &mean_of) >= 40,
        "the scene has a penumbra: {}",
        penumbra(&scene, &mean_of)
    );
    assert!(
        worst.0 < 1e-3,
        "pixel {} varies by {} over 16 frames",
        worst.1,
        worst.0
    );
}

/// The lengths are the frames merged: they grow while the view holds, and
/// a camera cut (the view moving 600 units, or turning 40°) starts every
/// pixel over at 0. Moving 400 units or turning 20° is no cut.
#[test]
fn camera_cut_resets_history() {
    let scene = plate_scene();
    let Some(mut session) = Session::new(rt_default()) else {
        return;
    };
    let view = scene.camera.view();
    let moved_by = |d: f32| {
        let mut v = view;
        v[3][0] += d;
        v
    };
    let turned_by = |degrees: f32| {
        // Turn about the eye's z axis: the rows' first two columns rotate.
        let (s, c) = degrees.to_radians().sin_cos();
        let mut v = view;
        for row in &mut v[..3] {
            let (x, y) = (row[0], row[1]);
            row[0] = c * x - s * y;
            row[1] = s * x + c * y;
        }
        v
    };
    for (cut, label) in [
        (moved_by(600.0), "moved 600"),
        (turned_by(40.0), "turned 40°"),
    ] {
        session.frames(&scene, 4);
        let before = session.history_length();
        let longest = before.iter().copied().fold(0.0f32, f32::max);
        assert!(longest >= 3.0, "{label}: the history grew to {longest}");
        session.frame(&scene, Some(cut), Submission::Whole);
        let after = session.history_length();
        assert!(
            after.iter().all(|&l| l == 0.0),
            "{label}: every pixel starts over"
        );
        assert!(!session.enc.scene_targets.history.valid(DenoiseKind::Sun));
        // Back to the camera's own view: another cut.
        session.frame(&scene, None, Submission::Whole);
    }
    for (near, label) in [
        (moved_by(400.0), "moved 400"),
        (turned_by(20.0), "turned 20°"),
    ] {
        session.frames(&scene, 2);
        session.frame(&scene, Some(near), Submission::Whole);
        assert!(
            session.enc.scene_targets.history.valid(DenoiseKind::Sun),
            "{label} is no cut"
        );
        session.frame(&scene, None, Submission::Whole);
    }
}

/// A world cut by a mid-frame flush clears the history: the next whole
/// frame merges nothing. A submission with no world draw at all (the UI
/// alone) leaves the history and the camera it continues from untouched.
#[test]
fn split_frame_resets_history() {
    let scene = plate_scene();
    let Some(mut session) = Session::new(rt_default()) else {
        return;
    };
    session.frames(&scene, 4);
    session.frame(&scene, None, Submission::Empty);
    assert!(
        session.enc.scene_targets.history.valid(DenoiseKind::Sun),
        "a submission with no world keeps the history"
    );
    session.frames(&scene, 1);
    let longest = session
        .history_length()
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(longest >= 4.0, "the history went on growing: {longest}");

    session.frame(&scene, None, Submission::Split);
    assert!(session.enc.scene_split());
    assert!(!session.enc.scene_targets.history.valid(DenoiseKind::Sun));
    session.frames(&scene, 1);
    let lengths = session.history_length();
    assert!(
        lengths.iter().all(|&l| l == 0.0),
        "the frame after a split merges nothing"
    );
}

/// A static view, the plate moved 35 units (10 pixels at the floor) after
/// the history filled: one frame later the floor it uncovered is lit (no
/// ghost of the old shadow darker than 0.1) and the floor it now covers is
/// dark, the history rejected or clamped where the scene changed.
#[test]
fn disocclusion_rejects_history() {
    let before = plate_scene();
    let after = moved(&before, 35.0);
    let Some(mut session) = Session::new(rt_default()) else {
        return;
    };
    session.frames(&before, 16);
    session.frames(&after, 1);
    let vis = session.vis();
    let (old, new) = (before.seen(6.0), after.seen(6.0));
    let uncovered: Vec<f32> = vis
        .iter()
        .zip(old.iter().zip(&new))
        .filter(|(_, (o, n))| **o == Some(Seen::Shadowed(0)) && **n == Some(Seen::Lit(0)))
        .map(|(&v, _)| v)
        .collect();
    assert!(
        uncovered.len() >= 20,
        "{} uncovered pixels",
        uncovered.len()
    );
    let ghost = uncovered.iter().copied().fold(1.0f32, f32::min);
    assert!(ghost >= 0.9, "the old shadow lingers down to {ghost}");
    let covered = where_seen(&vis, &new, |s| s == Seen::Shadowed(0));
    let lightest = covered.iter().copied().fold(0.0f32, f32::max);
    assert!(lightest <= 0.1, "the new shadow is up to {lightest}");
}

/// Under `rt.shadows` the sun's cascades never render, even with
/// `shadows.sun` on: the world pass leads with the acceleration-structure
/// builds and then, in order, the geometry kernel, the rays, the temporal
/// pass and, at medium quality, two à-trous passes (steps 1 and 2) ending in `vis`; the one
/// scene pass ahead of the world is the depth pre-pass.
#[test]
fn rt_shadows_replace_cascades() {
    let scene = plate_scene();
    let config = Mtld3dConfig {
        shadows: ShadowConfig {
            sun: true,
            ..ShadowConfig::default()
        },
        ..rt_default()
    };
    assert!(sun().shadows.sun);
    let Some(mut session) = Session::new(config) else {
        return;
    };
    let blits = session.frame(&scene, None, Submission::Whole);
    let textures = *session.enc.scene_textures().expect("scene textures");
    assert!(textures.cascades.is_none());
    assert!(session.enc.scene_cascades().is_none());
    let passes = session.enc.pass_state.passes();
    assert_eq!(passes.len(), 2, "the pre-pass, then the world");
    assert_eq!(passes[0].depth_texture(), textures.predepth);
    let tlas = blits
        .iter()
        .position(|b| b.cmd == BlitCommandType::BuildTlas as u32)
        .expect("the TLAS build");
    let first_kernel = blits
        .iter()
        .position(|b| b.cmd == BlitCommandType::SceneKernel as u32)
        .expect("the kernels");
    assert!(tlas < first_kernel, "the TLAS is built before it is traced");
    let rt = textures.rt.expect("ray-traced targets");
    let current = session.enc.scene_targets.history.current();
    let previous = 1 - current;
    let kernels = dispatches(&blits);
    let order: Vec<Kernel> = kernels.iter().map(|(k, ..)| *k).collect();
    assert_eq!(
        order,
        [
            Kernel::Geometry,
            Kernel::SunVisRays,
            Kernel::TemporalSun,
            Kernel::AtrousSun,
            Kernel::AtrousSun,
        ]
    );
    let scene_kernels: Vec<&BlitCommand> = blits
        .iter()
        .filter(|b| b.cmd == BlitCommandType::SceneKernel as u32)
        .collect();
    let slots = |i: usize| {
        let b = scene_kernels[i];
        let extra = kernels[i].2.textures;
        [
            b.src_handle,
            b.dst_handle,
            b.dst_offset,
            b.bytes_per_row,
            extra[0],
            extra[1],
            extra[2],
            extra[3],
        ]
    };
    let raw = |h: MetalHandle<_>| h.raw();
    let sun = rt.kind(DenoiseKind::Sun).expect("the sun's targets");
    let (sun_raw, sun_history) = (sun.raw, sun.history);
    assert_eq!(
        slots(0),
        [
            raw(textures.predepth),
            raw(rt.geometry[current]),
            0,
            0,
            0,
            0,
            0,
            0
        ]
    );
    assert_eq!(
        slots(1),
        [
            raw(textures.predepth),
            raw(sun_raw),
            raw(rt.geometry[current]),
            0,
            0,
            0,
            0,
            0
        ]
    );
    assert_ne!(
        kernels[1].2.trace.tlas, 0,
        "the rays trace this frame's TLAS"
    );
    assert_eq!(
        slots(2),
        [
            raw(sun_raw),
            raw(sun_history.value[current]),
            raw(rt.geometry[current]),
            raw(sun_history.value[previous]),
            raw(rt.geometry[previous]),
            raw(sun_history.length[previous]),
            raw(sun_history.length[current]),
            0
        ]
    );
    assert_eq!(kernels[2].2.constants.trace[3], 16, "the sun's history cap");
    let atrous: Vec<([u64; 2], f32)> = (3..5)
        .map(|i| ([slots(i)[0], slots(i)[1]], kernels[i].2.constants.params[3]))
        .collect();
    assert_eq!(
        atrous,
        [
            ([raw(sun_history.value[current]), raw(sun_raw)], 1.0),
            ([raw(sun_raw), raw(textures.vis)], 2.0),
        ]
    );
    for i in 3..5 {
        assert_eq!(slots(i)[2], raw(rt.geometry[current]));
        assert_eq!(slots(i)[3], raw(sun_history.length[current]));
    }
    assert!(
        !blits
            .iter()
            .any(|b| b.cmd == BlitCommandType::SceneKernel as u32
                && b.mip_level == Kernel::SunVisCascades as u32)
    );
}

/// A world with no TLAS this submission (no sun up) clears `vis` to full
/// sun and resets the history; no ray is traced.
#[test]
fn no_sun_traces_nothing_and_resets() {
    let scene = plate_scene();
    let Some(mut session) = Session::new(rt_default()) else {
        return;
    };
    session.frames(&scene, 2);
    assert!(session.enc.scene_targets.history.valid(DenoiseKind::Sun));
    // The next frame's boundary has the sun below the horizon.
    session
        .coherent
        .store(session.enc.current_submit_seq, Ordering::Release);
    session.enc.current_submit_seq += 1;
    begin(&mut session.enc, (VIS_RASTER, VIS_RASTER), 1, false);
    let buffers = session.buffers_of(&scene);
    let snapshot = snapshot(0, 0);
    session
        .enc
        .pass_state
        .emit_command(mtld3d_shared::Command::draw_primitives(
            mtld3d_shared::mtl::PrimitiveType::Triangle,
            0,
            3,
        ));
    for buffer in buffers {
        session.fixture.capture_with(
            &mut session.enc,
            &snapshot,
            (0x91, buffer),
            None,
            BufferContent {
                id: buffer,
                version: 1,
            },
        );
    }
    session.enc.pass_state.end_current_pass("test");
    session.enc.post_boundary(&PostBoundary {
        proj: daoc_proj(),
        sun_dir_eye: Some([0.0, 0.0, 1.0]),
        sun_color: [1.0; 3],
        sun_above_horizon: false,
        view: scene.camera.view(),
        ambient: [0.2; 3],
        frame: 0,
        ..PostBoundary::default()
    });
    assemble(&mut session.enc, false);
    let passes = session.enc.pass_state.passes();
    let (world, before) = passes.split_last().expect("the world pass");
    let vis = session.enc.scene_textures().expect("scene textures").vis;
    assert!(
        before.iter().any(|p| p.color_texture() == vis),
        "vis is cleared"
    );
    assert!(
        !world
            .leading_blits()
            .iter()
            .any(|b| b.cmd == BlitCommandType::SceneKernel as u32),
        "no kernel runs"
    );
    assert!(!session.enc.scene_targets.history.valid(DenoiseKind::Sun));
}

// ── Ambient occlusion, bounce light, point lights ──

/// The quad and world point raster pixel `(x, y)` sees, `None` for the sky
/// and where a neighbour sees another quad (the raster and the ray cast
/// may part there).
fn surface(scene: &VisScene, x: u32, y: u32) -> Option<(usize, super::tests::V3)> {
    let at = |x: u32, y: u32| {
        let w = scene.camera.ray(x, y);
        scene
            .quads
            .iter()
            .enumerate()
            .filter_map(|(i, q)| q.hit([0.0; 3], w, 0.0).map(|t| (t, i)))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(t, i)| (i, super::tests::add([0.0; 3], w, t)))
    };
    let centre = at(x, y)?;
    let steady = [(-1, 0), (1, 0), (0, -1), (0, 1)].iter().all(|&(dx, dy)| {
        let (nx, ny) = (x.saturating_add_signed(dx), y.saturating_add_signed(dy));
        nx >= VIS_RASTER || ny >= VIS_RASTER || at(nx, ny).map(|(i, _)| i) == Some(centre.0)
    });
    steady.then_some(centre)
}

/// The values of the floor pixels (quad 0) whose world point satisfies `class`.
fn on_floor<T: Copy>(scene: &VisScene, values: &[T], class: impl Fn([f64; 3]) -> bool) -> Vec<T> {
    (0..VIS_RASTER * VIS_RASTER)
        .filter(|&i| {
            surface(scene, i % VIS_RASTER, i / VIS_RASTER)
                .is_some_and(|(quad, p)| quad == 0 && class(p))
        })
        .map(|i| values[i as usize])
        .collect()
}

fn mean(values: &[f32]) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "a few thousand pixels")]
    let n = values.len() as f32;
    values.iter().sum::<f32>() / n
}

/// The floor and a wall 100 units high across it 260 units north of the
/// camera, facing it.
fn wall_scene() -> VisScene {
    VisScene {
        quads: vec![
            Quad::floor(),
            Quad {
                origin: [-300.0, 260.0, CAMERA_HEIGHT - 100.0],
                u: [600.0, 0.0, 0.0],
                v: [0.0, 0.0, 100.0],
            },
        ],
        // Shining from the south, behind the camera, onto the wall's face.
        sun: VisScene::sun(45.0, 180.0),
        ..plate_scene()
    }
}

fn rt_keys(edit: impl FnOnce(&mut RtConfig)) -> Mtld3dConfig {
    let mut rt = RtConfig::default();
    edit(&mut rt);
    Mtld3dConfig {
        rt,
        ..Mtld3dConfig::default()
    }
}

/// `rt.ao` alone, no sun: the floor at the wall's foot, within its reach,
/// is occluded; the floor beyond `rt.aoRadius` of every surface is open.
#[test]
fn ao_corner_darker_than_open_floor() {
    let scene = wall_scene();
    let Some(mut session) = Session::new(rt_keys(|rt| rt.ao_pct = 100)) else {
        return;
    };
    session.light.sun = false;
    session.frames(&scene, 12);
    let ao = session.ao();
    let corner = on_floor(&scene, &ao, |p| {
        (225.0..255.0).contains(&p[1]) && p[0].abs() < 200.0
    });
    let open = on_floor(&scene, &ao, |p| p[1] < 80.0);
    assert!(
        corner.len() >= 20 && open.len() >= 200,
        "{} corner, {} open",
        corner.len(),
        open.len()
    );
    let (corner, open) = (mean(&corner), mean(&open));
    assert!(open >= 0.97, "the open floor: {open}");
    assert!(
        corner <= 0.8 && corner < open - 0.15,
        "the corner {corner} against open {open}"
    );
}

/// `rt.gi`: a red wall in the sun tints the floor at its foot red; the
/// floor far from it, open to a white sky, gathers the stock ambient (0.2
/// in every channel), which the receivers put in place of the ambient at
/// `rt.gi = 1` (R44).
#[test]
fn gi_bounces_wall_colour_onto_floor() {
    let scene = wall_scene();
    let Some(mut session) = Session::new(rt_keys(|rt| rt.gi_pct = 100)) else {
        return;
    };
    session.material(1, [1.0, 0.0, 0.0]);
    session.light.fog = [1.0; 3];
    session.frames(&scene, 16);
    let gi = session.gi();
    let red = |v: &[f32; 4]| v[0] - v[1];
    let near = on_floor(&scene, &gi, |p| {
        (215.0..255.0).contains(&p[1]) && p[0].abs() < 200.0
    });
    let far = on_floor(&scene, &gi, |p| p[1] < 80.0);
    assert!(near.len() >= 20 && far.len() >= 200);
    let near_red = mean(&near.iter().map(red).collect::<Vec<_>>());
    let far_red = mean(&far.iter().map(red).collect::<Vec<_>>());
    let far_mean = |c: usize| mean(&far.iter().map(|v| v[c]).collect::<Vec<_>>());
    assert!(
        near_red > 0.05 && near_red > far_red + 0.04,
        "red over green: {near_red} near the wall, {far_red} far"
    );
    assert!(far_red.abs() < 0.03, "the far floor is grey: {far_red}");
    for c in 0..3 {
        assert!(
            (far_mean(c) - 0.2).abs() < 0.03,
            "the far floor gathers the stock ambient: channel {c} {}",
            far_mean(c)
        );
    }
}

/// `rt.pointShadows` with no sun at all: one point light over the plate
/// shadows the floor under the plate (channel 0), lights the floor beside
/// it, and the empty entries read 1.
#[test]
fn rt_points_without_sun() {
    let scene = plate_scene();
    let Some(mut session) = Session::new(rt_keys(|rt| rt.point_shadows = true)) else {
        return;
    };
    session.light.sun = false;
    let light = [20.0, 150.0, CAMERA_HEIGHT - 80.0];
    session.light.points = vec![(light, 600.0)];
    let blits = session.frame(&scene, None, Submission::Whole);
    assert!(
        dispatches(&blits)
            .iter()
            .all(|(kernel, ..)| *kernel != Kernel::SunVisRays),
        "no sun, no sun rays"
    );
    session.frames(&scene, 11);
    let points = session.points();
    let plate = scene.quads[1];
    let toward = |p: [f64; 3]| super::tests::add(light, p, -1.0);
    let blocked = |p: [f64; 3], grow: f64| plate.hit(p, toward(p), grow).is_some_and(|t| t < 1.0);
    let shadowed = on_floor(&scene, &points, |p| blocked(p, -20.0));
    let lit = on_floor(&scene, &points, |p| !blocked(p, 20.0));
    assert!(
        shadowed.len() >= 20 && lit.len() >= 500,
        "{} shadowed, {} lit",
        shadowed.len(),
        lit.len()
    );
    let darkest = shadowed.iter().map(|v| v[0]).fold(0.0f32, f32::max);
    assert!(darkest <= 0.1, "under the plate: up to {darkest}");
    let dimmest = lit.iter().map(|v| v[0]).fold(1.0f32, f32::min);
    assert!(dimmest >= 0.9, "in the light: down to {dimmest}");
    // Each à-trous tap weighs every channel alike: a channel of ones stays
    // one up to the half float's rounding.
    let empty = shadowed
        .iter()
        .chain(&lit)
        .flat_map(|v| v[1..].iter().copied())
        .fold(1.0f32, f32::min);
    assert!(
        empty >= 0.999,
        "the empty entries are unshadowed: down to {empty}"
    );
}

/// A point light 5 units over an open floor does not shadow the floor it
/// lights: its rays aim at a disc facing each surface point, never through
/// the floor under it.
#[test]
fn low_point_light_lights_the_floor_under_it() {
    let scene = VisScene {
        quads: vec![Quad::floor()],
        ..plate_scene()
    };
    let Some(mut session) = Session::new(rt_keys(|rt| rt.point_shadows = true)) else {
        return;
    };
    session.light.sun = false;
    let light = [0.0, 200.0, CAMERA_HEIGHT - 5.0];
    session.light.points = vec![(light, 400.0)];
    session.frames(&scene, 12);
    let points = session.points();
    let near = on_floor(&scene, &points, |p| {
        (p[0] - light[0]).hypot(p[1] - light[1]) < 60.0
    });
    assert!(near.len() >= 20, "{} pixels near the light", near.len());
    let dimmest = near.iter().map(|v| v[0]).fold(1.0f32, f32::min);
    assert!(
        dimmest >= 0.97,
        "the floor near the light: down to {dimmest}"
    );
}

// ── The denoiser's history ──

/// A moving occluder's shadow clears: the plate moves 70 units east after
/// the history filled, and two frames later the floor it no longer shades
/// (seen in both frames) is within 0.03 of full sun, not held at the
/// clamp's margin.
#[test]
fn moving_occluder_leaves_no_trail() {
    let scene = plate_scene();
    let Some(mut session) = Session::new(rt_default()) else {
        return;
    };
    session.frames(&scene, 16);
    let after = moved(&scene, 70.0);
    session.frames(&after, 2);
    let vis = session.vis();
    let (before, now) = (scene.seen(6.0), after.seen(6.0));
    let uncovered: Vec<f32> = (0..vis.len())
        .filter(|&i| before[i] == Some(Seen::Shadowed(0)) && now[i] == Some(Seen::Lit(0)))
        .map(|i| vis[i])
        .collect();
    assert!(
        uncovered.len() >= 20,
        "{} uncovered pixels",
        uncovered.len()
    );
    let dimmest = uncovered.iter().copied().fold(1.0f32, f32::min);
    assert!(dimmest >= 0.97, "the old shadow trails at {dimmest}");
}

/// The first frame after a cut is filtered: over 16 first frames of a large
/// sun's penumbra (the history reset before each), every penumbra pixel's
/// visibility varies less than half as much as one ray a pixel does
/// (`p · (1 − p)`).
#[test]
fn first_frame_after_a_cut_is_filtered() {
    let scene = high_plate_scene();
    let Some(mut session) = Session::new(rt_sun(400, RtQuality::Medium)) else {
        return;
    };
    let frames: Vec<Vec<f32>> = (0..16)
        .map(|_| {
            session.enc.scene_targets.history.reset();
            session.frames(&scene, 1);
            session.vis()
        })
        .collect();
    let (mut spread, mut raw, mut count) = (0.0f32, 0.0f32, 0);
    for i in 0..frames[0].len() {
        let (x, y) = (i as u32 % VIS_RASTER, i as u32 / VIS_RASTER);
        if scene.seen_at(x, y, 0.0).quad() != Some(0) {
            continue;
        }
        let p = frames.iter().map(|f| f[i]).sum::<f32>() / 16.0;
        if !(0.2..0.8).contains(&p) {
            continue;
        }
        spread += frames.iter().map(|f| (f[i] - p).powi(2)).sum::<f32>() / 16.0;
        raw += p * (1.0 - p);
        count += 1;
    }
    assert!(count >= 40, "{count} penumbra pixels");
    assert!(
        spread < 0.5 * raw,
        "first-frame variance {spread} against one ray's {raw} over {count} pixels"
    );
}
