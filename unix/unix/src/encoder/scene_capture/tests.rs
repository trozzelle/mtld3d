//! Scene capture: which draws a scripted frame keeps, and the replay of one.

use core::ptr::NonNull;
use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    config::Mtld3dConfig,
    draw_data::{
        AttrSnapshot, DeclarationHeader, FixedPsSource, FixedVsSource, ProgrammablePsSource,
        ProgrammableVsSource, PsSource, PsSourcePtr, ShaderSourceFlags, VsSource, VsSourcePtr,
    },
    dxso::{
        FfPsKey, FfStage, FfStageFlags, FfVsFlags, FfVsKey, VariantKey, VsSamplerKinds, tci_entry,
    },
    ids::{ProgramId, TextureId},
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
    render_scale::RenderScale,
    shadows::ShadowConfig,
};
use mtld3d_shared::{
    Command, CommandType, MetalHandle, VertexAttrDesc,
    mtl::{CullMode, IndexType, PixelFormat, PrimitiveType, VertexFormat},
};
use mtld3d_types::{
    D3DCMP_ALWAYS, D3DCMP_GREATEREQUAL, D3DCMP_NEVER, D3DTA_COMPLEMENT, D3DTA_DIFFUSE,
    D3DTA_TEXTURE, D3DTOP_MODULATE, D3DTOP_SELECTARG1,
};

use super::{
    BufferContent, BytesRef, CapturedDraw, CapturedIndex, CapturedTexture, DrawClass, DrawFacts,
    ReplayBinding, Stage0, VertexBind, encode_captured_draw,
};
use crate::encoder::{
    FrameEncoder, TextureGpuState,
    tests::{encoder_with_config, post_boundary_op, post_reset_frame},
};

const BACK_BUFFER: u64 = 0xB000;

/// An encoder with `shadows.sun` set to `sun`, on a fresh 64x64 back buffer with depth.
fn encoder(
    coherent: &AtomicU64,
    retained: &AtomicU64,
    sun: bool,
) -> Option<(impl Sized, FrameEncoder)> {
    let config = Mtld3dConfig {
        shadows: ShadowConfig {
            sun,
            ..ShadowConfig::default()
        },
        ..Mtld3dConfig::default()
    };
    let (device, mut enc) = encoder_with_config(coherent, retained, config)?;
    post_reset_frame(&mut enc, (64, 64), 1, true);
    enc.scene.begin_submission(false);
    Some((device, enc))
}

pub(in crate::encoder) fn ff_vs(flags: FfVsFlags, vertex_blend_count: u8) -> VsSource {
    VsSource::FixedFunction(FixedVsSource {
        key: FfVsKey {
            ppl_knee: 0,
            color_style: 0,
            scene: 0,
            ppl_mask: 0,
            ppl_model: 0,
            flags,
            input_tex_coord_count: 1,
            tex_coord_count: 1,
            light_active_mask: 1,
            light_directional_mask: 1,
            light_spot_mask: 0,
            diffuse_source: 0,
            ambient_source: 0,
            specular_source: 0,
            emissive_source: 0,
            fog_mode: 0,
            tci: [tci_entry(0, 0); 8],
            tex_coord_dims: [2; 8],
            tt_flags: [0; 8],
            vertex_blend_count,
            declared_weights_count: 0,
            clip_plane_count: 0,
            passthrough: [0; 8],
        },
        max_row_count: 16,
        reserved: [0; 2],
    })
}

pub(in crate::encoder) fn programmable_vs(vs_id: ProgramId) -> VsSource {
    VsSource::Programmable(ProgrammableVsSource {
        vs_id,
        max_const_used: 96,
        provided_input_mask: 0,
        flags: ShaderSourceFlags::empty(),
        clip_plane_count: 0,
        sampler_kinds: VsSamplerKinds::default(),
        scene: 0,
        reserved: [0; 6],
    })
}

/// The a1 `SpeedTree` leaf VS id, from its fixture.
pub(in crate::encoder) fn speedtree_id() -> ProgramId {
    let bytes: &[u8] = include_bytes!(
        "../../../../../windows/core/src/light_controls/fixtures/vs_speedtree_388.bin"
    );
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    ProgramId::from_tokens(&words)
}

pub(in crate::encoder) fn ff_ps() -> PsSource {
    PsSource::FixedFunction(FixedPsSource {
        key: FfPsKey {
            stages: [FfStage::default(); 8],
            specular_add: false,
            tt_projected_mask: 0,
        },
        sampled_stage_mask: 1,
        constant_rows: 0,
        bump_env: false,
        reserved: [0; 2],
    })
}

/// One declaration record (a float3 position on stream 0) and its token.
pub(in crate::encoder) struct Declaration {
    header: Box<DeclarationHeader>,
    attrs: Box<[VertexAttrDesc; 1]>,
}

impl Declaration {
    pub(in crate::encoder) fn new() -> Self {
        Self {
            header: Box::new(DeclarationHeader {
                vdecl_hash: 0x5eed,
                extents: [12; 16],
                count: 1,
                used_streams: 0b111,
                reserved: 0,
            }),
            attrs: Box::new([VertexAttrDesc {
                attr_index: 0,
                buffer_index: 0,
                offset: 0,
                format: VertexFormat::Float3,
            }]),
        }
    }

    pub(in crate::encoder) fn token(&self) -> AttrSnapshot {
        // SAFETY: both boxes outlive every capture the test makes.
        unsafe {
            AttrSnapshot::new(
                NonNull::from(&self.attrs[0]),
                NonNull::from(self.header.as_ref()),
            )
        }
    }
}

fn snapshot() -> PipelineSnapshot {
    PipelineSnapshot {
        vs_fn: MetalHandle::NULL,
        ps_fn: MetalHandle::NULL,
        vdecl_hash: 0x5eed,
        stream_layouts: [StreamLayout::UNUSED; 16],
        color_format: PixelFormat::Bgra8Unorm,
        attach: PipelineAttachFlags::HAS_DEPTH | PipelineAttachFlags::HAS_COLOR_OUTPUT,
        rs: PipelineRsBits::default(),
        extra: ExtraColorAttachments::NONE,
        ps_color_out_mask: 1,
        sample_count: 1,
    }
}

/// The bytes a draw binds: its own in the encoder's scratch arena, as a
/// packet's are in this submission's arenas, but its camera `pos_fixup`,
/// which a draw builds on its stack.
pub(in crate::encoder) struct Bytes {
    vs_constants: BytesRef,
    vs_draw: BytesRef,
    vs_int: BytesRef,
    vs_bool: BytesRef,
    pos_fixup: Vec<u8>,
    alpha_ref: BytesRef,
    up_vertices: BytesRef,
    up_indices: BytesRef,
}

impl Bytes {
    pub(in crate::encoder) fn new(enc: &mut FrameEncoder) -> Self {
        let up_indices: Vec<u8> = [0u16, 1, 2].iter().flat_map(|i| i.to_le_bytes()).collect();
        Self {
            vs_constants: arena(enc, &(0..=255).collect::<Vec<u8>>()),
            vs_draw: arena(enc, &[7; 32]),
            vs_int: arena(enc, &[3; 256]),
            vs_bool: arena(enc, &[1, 0, 0, 0]),
            pos_fixup: [1.0f32 / 64.0, -1.0 / 64.0, 0.0, 1.0, 0.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
            alpha_ref: arena(enc, &0.5f32.to_le_bytes()),
            up_vertices: arena(enc, &(0..36).collect::<Vec<u8>>()),
            up_indices: arena(enc, &up_indices),
        }
    }
}

/// `bytes` placed in `enc`'s scratch arena, where a draw's own bytes are.
pub(in crate::encoder) fn arena(enc: &mut FrameEncoder, bytes: &[u8]) -> BytesRef {
    if bytes.is_empty() {
        return BytesRef::EMPTY;
    }
    BytesRef {
        ptr: enc.alloc_scratch(bytes),
        len: u32::try_from(bytes.len()).expect("small"),
    }
}

/// The variable parts of one scripted draw.
#[derive(Clone, Copy)]
pub(in crate::encoder) struct Draw {
    depth_write: bool,
    blend: bool,
    alpha_func: u8,
    stage0: Option<Stage0>,
    pipeline: u64,
    index: CapturedIndex,
}

impl Draw {
    pub(in crate::encoder) const OPAQUE: Self = Self {
        depth_write: true,
        blend: false,
        alpha_func: 0,
        stage0: None,
        pipeline: 0x91,
        index: CapturedIndex::None {
            start_vertex: 0,
            vertex_count: 3,
        },
    };
}

/// Emit one scripted draw the way `emit_draw_view` does: stage the
/// bindings as they resolve, then offer the draw to capture.
///
/// Staging and the offer run whether or not capture is armed, so a capture
/// that ignored its own gate would show.
pub(in crate::encoder) fn run(
    enc: &mut FrameEncoder,
    vs: &VsSource,
    ps: &PsSource,
    decl: &Declaration,
    bytes: &Bytes,
    binds: &[VertexBind],
    draw: Draw,
) {
    let snapshot = snapshot();
    enc.begin_scene_binds();
    for &bind in binds {
        enc.stage_scene_bind(bind);
    }
    // SAFETY: the sources outlive the test's captures and do not move.
    let vs = unsafe { VsSourcePtr::new(NonNull::from(vs)) };
    // SAFETY: as above.
    let ps = unsafe { PsSourcePtr::new(NonNull::from(ps)) };
    enc.capture_scene_draw(&DrawFacts {
        snapshot: &snapshot,
        attrs: decl.token(),
        fetch_attrs: None,
        vs,
        ps,
        variant: VariantKey {
            alpha_func: draw.alpha_func,
            ..VariantKey::default()
        },
        pipeline: draw.pipeline,
        depth_stencil: 0xD5,
        depth_write: draw.depth_write,
        blend: draw.blend,
        index: draw.index,
        primitive: PrimitiveType::Triangle,
        instances: 1,
        vs_constants: bytes.vs_constants,
        vs_draw: bytes.vs_draw,
        vs_int: bytes.vs_int,
        vs_bool: bytes.vs_bool,
        pos_fixup: &bytes.pos_fixup,
        alpha_ref: bytes.alpha_ref,
        cull: CullMode::Back,
        slope_scale: 0.25,
        stage0: draw.stage0,
    });
}

pub(in crate::encoder) fn buffer_binds() -> [VertexBind; 2] {
    [
        VertexBind::Buffer {
            handle: 0x100,
            offset: 16,
            slot: 0,
            content: BufferContent { id: 1, version: 1 },
        },
        VertexBind::Buffer {
            handle: 0x200,
            offset: 0,
            slot: 1,
            content: BufferContent { id: 2, version: 1 },
        },
    ]
}

/// The bytes a capture's copy names.
fn read(bytes: BytesRef) -> Vec<u8> {
    // SAFETY: a copy in the live encoder's scratch arena.
    unsafe { core::slice::from_raw_parts(bytes.ptr as *const u8, bytes.len as usize) }.to_vec()
}

/// With every `shadows.*`/`rt.*` key absent a whole frame of world draws,
/// the boundary and UI draws records nothing and copies nothing.
#[test]
fn capture_off_records_nothing() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(&coherent, &retained, false) else {
        return;
    };
    assert!(!enc.scene_capture_armed());
    let (decl, bytes) = (Declaration::new(), Bytes::new(&mut enc));
    let (vs, rhw, ps) = (
        ff_vs(FfVsFlags::LIGHTING_ENABLED, 0),
        ff_vs(FfVsFlags::HAS_RHW, 0),
        ff_ps(),
    );
    let up = VertexBind::Inline {
        bytes: bytes.up_vertices,
        slot: 0,
    };
    let up_index = CapturedIndex::Inline {
        bytes: bytes.up_indices,
        count: 3,
        index_type: IndexType::UInt16,
    };
    let scratch = enc.scratch.bytes_used();
    for _ in 0..3 {
        run(
            &mut enc,
            &vs,
            &ps,
            &decl,
            &bytes,
            &buffer_binds(),
            Draw::OPAQUE,
        );
    }
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &[up],
        Draw {
            index: up_index,
            ..Draw::OPAQUE
        },
    );
    enc.post_boundary(&post_boundary_op());
    run(
        &mut enc,
        &rhw,
        &ps,
        &decl,
        &bytes,
        &[up],
        Draw {
            depth_write: false,
            blend: true,
            ..Draw::OPAQUE
        },
    );
    assert!(enc.scene_draws().is_empty());
    assert_eq!(enc.scratch.bytes_used(), scratch, "nothing copied");
    enc.shutdown_cleanup();
}

/// Opaque depth-writing perspective back-buffer draws are kept, in order;
/// the sky (no depth write), a blended draw, a pre-transformed draw, a draw
/// whose pipeline is still building and every draw after the boundary are
/// not. The boundary holds through a mid-frame flush and resets at the next
/// frame.
#[test]
fn captures_opaque_zwrite_world_draws() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(&coherent, &retained, true) else {
        return;
    };
    assert!(enc.scene_capture_armed());
    let (decl, bytes) = (Declaration::new(), Bytes::new(&mut enc));
    let (lit, blended_vs, rhw, ps) = (
        ff_vs(FfVsFlags::LIGHTING_ENABLED | FfVsFlags::HAS_NORMAL, 0),
        ff_vs(FfVsFlags::empty(), 2),
        ff_vs(FfVsFlags::HAS_RHW, 0),
        ff_ps(),
    );
    let sky = Draw {
        depth_write: false,
        pipeline: 0x51,
        ..Draw::OPAQUE
    };
    run(&mut enc, &lit, &ps, &decl, &bytes, &buffer_binds(), sky);
    let scratch = enc.scratch.bytes_used();
    run(
        &mut enc,
        &lit,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw::OPAQUE,
    );
    let grown = enc.scratch.bytes_used() - scratch;
    assert!(
        (bytes.pos_fixup.len() as u64..64).contains(&grown),
        "the capture copies its pos_fixup alone: {grown} bytes"
    );
    run(
        &mut enc,
        &lit,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            blend: true,
            pipeline: 0x52,
            ..Draw::OPAQUE
        },
    );
    run(
        &mut enc,
        &rhw,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            pipeline: 0x53,
            ..Draw::OPAQUE
        },
    );
    // A draw whose pipeline is still building binds a placeholder that only
    // this submission's finalize resolves.
    run(
        &mut enc,
        &lit,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            pipeline: (1 << 63) | 7,
            ..Draw::OPAQUE
        },
    );
    let up = VertexBind::Inline {
        bytes: bytes.up_vertices,
        slot: 0,
    };
    run(
        &mut enc,
        &blended_vs,
        &ps,
        &decl,
        &bytes,
        &[up],
        Draw {
            pipeline: 0x92,
            index: CapturedIndex::Inline {
                bytes: bytes.up_indices,
                count: 3,
                index_type: IndexType::UInt16,
            },
            ..Draw::OPAQUE
        },
    );
    enc.post_boundary(&post_boundary_op());
    assert!(!enc.scene_capture_armed());
    run(
        &mut enc,
        &lit,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            pipeline: 0x54,
            ..Draw::OPAQUE
        },
    );

    let draws = enc.scene_draws();
    let pipelines: Vec<u64> = draws.iter().map(|d| d.pipeline).collect();
    assert_eq!(pipelines, [0x91, 0x92], "the two world draws, in order");
    let first = &draws[0];
    assert_eq!((first.class, first.lit), (DrawClass::Ff, true));
    assert_eq!(first.vertex_binds(), buffer_binds());
    assert_eq!(first.depth_stencil, 0xD5);
    assert_eq!((first.cull, first.slope_scale), (CullMode::Back, 0.25));
    assert_eq!(first.alpha, None);
    assert!(first.alpha_ref_bytes.is_empty());
    assert_eq!(first.vertex_attrs().len(), 1);
    // The pipeline inputs a caster sibling is built from.
    assert_eq!(first.snapshot.vdecl_hash, 0x5eed);
    assert_eq!(first.variant, VariantKey::default());
    assert!(matches!(
        first.vs.as_ref(),
        mtld3d_core::draw_data::VsSourceView::FixedFunction(source)
            if source.key.flags.contains(FfVsFlags::LIGHTING_ENABLED)
    ));
    assert!(matches!(
        first.ps.as_ref(),
        mtld3d_core::draw_data::PsSourceView::FixedFunction(_)
    ));
    // The draw's own bytes, already in the submission's arena, are kept by
    // address; the pos_fixup it built on its stack is copied.
    for (kept, own) in [
        (first.vs_constants, bytes.vs_constants),
        (first.vs_draw, bytes.vs_draw),
        (first.vs_int, bytes.vs_int),
        (first.vs_bool, bytes.vs_bool),
    ] {
        assert_eq!(kept, own);
    }
    assert_ne!(first.pos_fixup.ptr, bytes.pos_fixup.as_ptr() as u64);
    assert_eq!(read(first.pos_fixup), bytes.pos_fixup);
    let second = &draws[1];
    assert_eq!((second.class, second.lit), (DrawClass::FfBlend, false));
    let [
        VertexBind::Inline {
            bytes: vertices,
            slot: 0,
        },
    ] = second.vertex_binds()
    else {
        panic!("one inline vertex bind: {:?}", second.vertex_binds());
    };
    assert_eq!(*vertices, bytes.up_vertices);
    let CapturedIndex::Inline {
        bytes: indices,
        count: 3,
        index_type: IndexType::UInt16,
    } = second.index
    else {
        panic!("inline indices: {:?}", second.index);
    };
    assert_eq!(indices, bytes.up_indices);

    // A mid-frame flush keeps the boundary: the UI continues.
    enc.scene.begin_submission(true);
    assert!(
        enc.scene_draws().is_empty(),
        "records die with their submission"
    );
    run(
        &mut enc,
        &lit,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw::OPAQUE,
    );
    assert!(enc.scene_draws().is_empty());
    // The next frame captures again.
    enc.scene.begin_submission(false);
    run(
        &mut enc,
        &lit,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw::OPAQUE,
    );
    assert_eq!(enc.scene_draws().len(), 1);
    enc.shutdown_cleanup();
}

/// An alpha-tested draw is kept with its stage-0 texture, sampler, reference
/// and function; `ALWAYS` keeps it opaque and `NEVER` drops it. A stage-0
/// texture that is a render target or an ordered upload of this submission
/// is not ready before the world's passes, so its draw is not kept, and
/// neither is one that bound no alpha reference for its caster to test.
#[test]
fn captures_alpha_tested_with_stage0() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(&coherent, &retained, true) else {
        return;
    };
    let (decl, bytes) = (Declaration::new(), Bytes::new(&mut enc));
    let (vs, ps) = (ff_vs(FfVsFlags::LIGHTING_ENABLED, 0), ff_ps());
    let leaf = TextureId::new_unique();
    let stage0 = |texture| Stage0 {
        texture,
        handle: 0x7700,
        sampler: 0x5500,
    };
    let tested = Draw {
        alpha_func: u8::try_from(D3DCMP_GREATEREQUAL).unwrap(),
        stage0: Some(stage0(leaf)),
        ..Draw::OPAQUE
    };
    run(&mut enc, &vs, &ps, &decl, &bytes, &buffer_binds(), tested);
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            alpha_func: u8::try_from(D3DCMP_ALWAYS).unwrap(),
            pipeline: 0x93,
            ..Draw::OPAQUE
        },
    );
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            alpha_func: u8::try_from(D3DCMP_NEVER).unwrap(),
            stage0: Some(stage0(leaf)),
            pipeline: 0x94,
            ..Draw::OPAQUE
        },
    );
    // Alpha-tested without a stage-0 texture.
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            stage0: None,
            pipeline: 0x95,
            ..tested
        },
    );
    let target = TextureId::new_unique();
    enc.texture_cache.insert(
        target,
        TextureGpuState {
            views: mtld3d_shared::texture_views::TextureViews::EMPTY,
            mip_staging_buffers: Vec::new(),
            render_target: true,
        },
    );
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            stage0: Some(stage0(target)),
            pipeline: 0x96,
            ..tested
        },
    );
    let uploaded = TextureId::new_unique();
    enc.note_scene_ordered_upload(uploaded);
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            stage0: Some(stage0(uploaded)),
            pipeline: 0x97,
            ..tested
        },
    );
    // No alpha reference bound: a replay would bind empty bytes at slot 14.
    let unreferenced = Bytes {
        alpha_ref: BytesRef::EMPTY,
        ..Bytes::new(&mut enc)
    };
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &unreferenced,
        &buffer_binds(),
        Draw {
            pipeline: 0x98,
            ..tested
        },
    );

    let draws = enc.scene_draws();
    let pipelines: Vec<u64> = draws.iter().map(|d| d.pipeline).collect();
    assert_eq!(pipelines, [0x91, 0x93]);
    let alpha = draws[0]
        .alpha
        .expect("the alpha-tested draw keeps its test");
    assert_eq!((alpha.tex, alpha.sampler), (0x7700, 0x5500));
    assert_eq!(alpha.alpha_ref, 0.5);
    assert_eq!(u32::from(alpha.func), D3DCMP_GREATEREQUAL);
    assert_eq!(
        alpha.uv_attr, 0,
        "the FF pixel stage samples stage 0 at texcoord0"
    );
    assert_eq!(draws[0].alpha_ref_bytes, bytes.alpha_ref, "kept by address");
    assert_eq!(draws[1].alpha, None, "ALWAYS is opaque");
    // An ordered upload is forgotten with its submission.
    enc.scene.begin_submission(false);
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            stage0: Some(stage0(uploaded)),
            ..tested
        },
    );
    assert_eq!(enc.scene_draws().len(), 1);
    enc.texture_cache.remove(&target);
    enc.shutdown_cleanup();
}

/// An opaque draw keeps its stage-0 texture for the ray-traced scene's hit
/// albedo only when its fixed-function pixel stage colours from it (a
/// modulate or select of the texture): not when stage 0 does not, not under
/// a programmable pixel shader (whose stage 0 need not be colour), and not
/// when the texture is a render target or an ordered upload. An
/// alpha-tested draw names its texture through its alpha test alone.
#[test]
fn keeps_albedo_only_for_ff_colour_stage() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(&coherent, &retained, true) else {
        return;
    };
    let (decl, bytes) = (Declaration::new(), Bytes::new(&mut enc));
    let vs = ff_vs(FfVsFlags::empty(), 0);
    let stage = |op: u32, arg1: u32| FfStage {
        color_op: u8::try_from(op).unwrap(),
        color_arg1: u8::try_from(arg1).unwrap(),
        color_arg2: u8::try_from(D3DTA_DIFFUSE).unwrap(),
        flags: FfStageFlags::HAS_TEXTURE,
        ..FfStage::default()
    };
    let ff = |stage0: FfStage| {
        let mut stages = [FfStage::default(); 8];
        stages[0] = stage0;
        PsSource::FixedFunction(FixedPsSource {
            key: FfPsKey {
                stages,
                specular_add: false,
                tt_projected_mask: 0,
            },
            sampled_stage_mask: 1,
            constant_rows: 0,
            bump_env: false,
            reserved: [0; 2],
        })
    };
    let modulate = ff(stage(D3DTOP_MODULATE, D3DTA_TEXTURE));
    let select = ff(stage(D3DTOP_SELECTARG1, D3DTA_TEXTURE));
    let complemented = ff(stage(D3DTOP_MODULATE, D3DTA_TEXTURE | D3DTA_COMPLEMENT));
    let diffuse_only = ff(stage(D3DTOP_SELECTARG1, D3DTA_DIFFUSE));
    let splat = PsSource::Programmable(ProgrammablePsSource {
        ps_id: ProgramId::from_tokens(&[0xFFFF_0200, 0x0000_FFFF]),
        max_const_used: 0,
        flags: ShaderSourceFlags::empty(),
        color_out_mask: 1,
        reserved: [0; 4],
    });
    let stage0 = |texture| Stage0 {
        texture,
        handle: 0x7700,
        sampler: 0x5500,
    };
    let target = TextureId::new_unique();
    enc.texture_cache.insert(
        target,
        TextureGpuState {
            views: mtld3d_shared::texture_views::TextureViews::EMPTY,
            mip_staging_buffers: Vec::new(),
            render_target: true,
        },
    );
    let uploaded = TextureId::new_unique();
    enc.note_scene_ordered_upload(uploaded);
    let cases: [(&PsSource, TextureId, u8, bool); 8] = [
        (&modulate, TextureId::new_unique(), 0, true),
        (&select, TextureId::new_unique(), 0, true),
        (&complemented, TextureId::new_unique(), 0, false),
        (&diffuse_only, TextureId::new_unique(), 0, false),
        (&splat, TextureId::new_unique(), 0, false),
        (&modulate, target, 0, false),
        (&modulate, uploaded, 0, false),
        (
            &modulate,
            TextureId::new_unique(),
            u8::try_from(D3DCMP_GREATEREQUAL).unwrap(),
            false,
        ),
    ];
    for (i, &(ps, texture, alpha_func, _)) in (0u64..).zip(&cases) {
        run(
            &mut enc,
            &vs,
            ps,
            &decl,
            &bytes,
            &buffer_binds(),
            Draw {
                alpha_func,
                stage0: Some(stage0(texture)),
                pipeline: 0x100 + i,
                ..Draw::OPAQUE
            },
        );
    }
    let draws = enc.scene_draws();
    assert_eq!(draws.len(), cases.len(), "every draw is kept");
    for (i, (d, &(_, _, _, kept))) in draws.iter().zip(&cases).enumerate() {
        let want = kept.then_some(CapturedTexture {
            tex: 0x7700,
            sampler: 0x5500,
            uv_attr: 0,
        });
        assert_eq!(d.albedo, want, "case {i}");
    }
    assert!(draws[7].alpha.is_some(), "the alpha test names the texture");
    enc.texture_cache.remove(&target);
    enc.shutdown_cleanup();
}

/// A draw into a target other than the back buffer is not kept, even with
/// the `SpeedTree` VS that also draws the world: only render target 0 decides.
#[test]
fn offscreen_draws_not_captured() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(&coherent, &retained, true) else {
        return;
    };
    let (decl, bytes) = (Declaration::new(), Bytes::new(&mut enc));
    let (leaf, ff, ps) = (
        programmable_vs(speedtree_id()),
        ff_vs(FfVsFlags::LIGHTING_ENABLED, 0),
        ff_ps(),
    );
    // SAFETY: an opaque handle the pass state only names.
    let reflection = unsafe { MetalHandle::new(0xC000) };
    enc.pass_state.set_color_render_target(
        reflection,
        32,
        32,
        PixelFormat::Bgra8Unorm,
        RenderScale::IDENTITY,
    );
    run(
        &mut enc,
        &leaf,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw::OPAQUE,
    );
    run(
        &mut enc,
        &ff,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw::OPAQUE,
    );
    assert!(
        enc.scene_draws().is_empty(),
        "the reflection target is not the world"
    );
    // SAFETY: as above, the back buffer `post_reset_frame` named.
    let back_buffer = unsafe { MetalHandle::new(BACK_BUFFER) };
    enc.pass_state.set_color_render_target(
        back_buffer,
        64,
        64,
        PixelFormat::Bgra8Unorm,
        RenderScale::IDENTITY,
    );
    run(
        &mut enc,
        &leaf,
        &ps,
        &decl,
        &bytes,
        &buffer_binds(),
        Draw {
            pipeline: 0x98,
            ..Draw::OPAQUE
        },
    );
    let draws = enc.scene_draws();
    assert_eq!(draws.len(), 1);
    assert_eq!(
        (draws[0].pipeline, draws[0].class, draws[0].lit),
        (0x98, DrawClass::SpeedTree, true)
    );
    enc.shutdown_cleanup();
}

/// The commands `encode_captured_draw` writes for `draw` alone, as `(type, a, b, c)`.
fn encoded(draw: &CapturedDraw, binding: &ReplayBinding<'_>) -> Vec<Command> {
    let mut out = Vec::new();
    encode_captured_draw(&mut out, draw, binding);
    out
}

/// Re-encoding a captured draw writes every binding it reads, so it draws
/// right as the first draw of a fresh pass and right after any other draw.
#[test]
fn captured_draw_reencodes_alone() {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = encoder(&coherent, &retained, true) else {
        return;
    };
    let (decl, bytes) = (Declaration::new(), Bytes::new(&mut enc));
    let (vs, ps) = (ff_vs(FfVsFlags::LIGHTING_ENABLED, 0), ff_ps());
    let zeros = VertexBind::Zeros {
        bytes: BytesRef {
            ptr: mtld3d_core::draw_data::NULL_STREAM_ZEROS.as_ptr() as u64,
            len: 12,
        },
        slot: 2,
    };
    let [vb0, vb1] = buffer_binds();
    let indexed = CapturedIndex::Buffer {
        handle: 0x300,
        offset: 64,
        count: 36,
        index_type: IndexType::UInt16,
        base_vertex: 5,
        content: BufferContent { id: 3, version: 1 },
    };
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &[vb0, vb1, zeros],
        Draw {
            index: indexed,
            ..Draw::OPAQUE
        },
    );
    run(
        &mut enc,
        &vs,
        &ps,
        &decl,
        &bytes,
        &[VertexBind::Buffer {
            handle: 0x400,
            offset: 0,
            slot: 0,
            content: BufferContent::default(),
        }],
        Draw {
            alpha_func: u8::try_from(D3DCMP_GREATEREQUAL).unwrap(),
            stage0: Some(Stage0 {
                texture: TextureId::new_unique(),
                handle: 0x7700,
                sampler: 0x5500,
            }),
            pipeline: 0x99,
            ..Draw::OPAQUE
        },
    );
    let draws = enc.scene_draws().to_vec();
    assert_eq!(draws.len(), 2);
    let extra = BytesRef {
        ptr: enc.alloc_scratch(&[9; 64]),
        len: 64,
    };
    let own = |draw: &CapturedDraw| ReplayBinding {
        pipeline: draw.pipeline,
        depth_stencil: draw.depth_stencil,
        cull: draw.cull,
        depth_bias: (0.0, draw.slope_scale),
        pos_fixup: draw.pos_fixup,
        extra_vs: &[],
    };
    let first = &draws[0];
    let bytes_at = |b: BytesRef, slot| Command::set_vertex_bytes_at(b.ptr, b.len, slot);
    let expected = [
        Command::set_render_pipeline_state(0x91),
        Command::set_depth_stencil_state(0xD5),
        Command::set_cull_mode(CullMode::Back),
        Command::set_depth_bias(0.0, 0.25),
        Command::set_vertex_buffer(0x100, 16, 0),
        Command::set_vertex_buffer(0x200, 0, 1),
        bytes_at(
            BytesRef {
                ptr: mtld3d_core::draw_data::NULL_STREAM_ZEROS.as_ptr() as u64,
                len: 12,
            },
            2,
        ),
        bytes_at(first.vs_constants, 30),
        bytes_at(first.vs_draw, 27),
        bytes_at(first.vs_int, 29),
        bytes_at(first.vs_bool, 26),
        bytes_at(first.pos_fixup, 28),
        Command::draw_indexed_primitives(
            PrimitiveType::Triangle,
            36,
            IndexType::UInt16,
            0x300,
            64,
            5,
            1,
        ),
    ];
    let key = |c: &Command| (c.cmd, c.param_a, c.param_b, c.param_c, c.param_d);
    let keys = |cs: &[Command]| cs.iter().map(key).collect::<Vec<_>>();
    assert_eq!(keys(&encoded(first, &own(first))), keys(&expected));
    // The pos_fixup and extra vertex bytes are the caller's: a light view
    // binds its own after the draw's.
    let light = ReplayBinding {
        pipeline: 0xCA,
        depth_stencil: 0,
        cull: CullMode::None,
        depth_bias: (1.0, 2.0),
        pos_fixup: extra,
        extra_vs: &[(16, extra)],
    };
    let commands = encoded(first, &light);
    assert_eq!(
        key(&commands[0]),
        key(&Command::set_render_pipeline_state(0xCA))
    );
    assert!(
        commands
            .iter()
            .all(|c| c.cmd != CommandType::SetDepthStencilState as u32),
        "no depth-stencil state was asked for"
    );
    assert!(commands.iter().any(|c| key(c) == key(&bytes_at(extra, 16))));
    assert!(commands.iter().any(|c| key(c) == key(&bytes_at(extra, 28))));

    // Encoded after the first, the alpha-tested draw still binds its whole
    // set: its own slot 0 and every constant, its stage-0 texture, sampler
    // and reference, and nothing it reads comes from the draw before.
    let second = &draws[1];
    let mut out = Vec::new();
    encode_captured_draw(&mut out, first, &own(first));
    let start = out.len();
    encode_captured_draw(&mut out, second, &own(second));
    let alone = encoded(second, &own(second));
    assert_eq!(keys(&out[start..]), keys(&alone));
    for needed in [
        Command::set_render_pipeline_state(0x99),
        Command::set_vertex_buffer(0x400, 0, 0),
        bytes_at(second.vs_constants, 30),
        bytes_at(second.vs_draw, 27),
        bytes_at(second.pos_fixup, 28),
        Command::set_fragment_texture(0x7700, 0),
        Command::set_fragment_sampler_state(0x5500, 0),
        Command::set_fragment_bytes_at(second.alpha_ref_bytes.ptr, 4, 14),
        Command::draw_primitives(PrimitiveType::Triangle, 0, 3),
    ] {
        assert!(
            alone.iter().any(|c| key(c) == key(&needed)),
            "{:?}",
            key(&needed)
        );
    }
    assert_eq!(
        key(alone.last().unwrap()),
        key(&Command::draw_primitives(PrimitiveType::Triangle, 0, 3))
    );
    enc.shutdown_cleanup();
}
