//! World-draw capture for scene replay (`shadows.*`, `rt.*`).
//!
//! While a key that replays the world is on
//! ([`mtld3d_core::shadows::scene_capture_wanted`]), every back-buffer world
//! draw recorded before this frame's 3D/UI boundary is kept as a
//! [`CapturedDraw`]: everything [`encode_captured_draw`] needs to encode it
//! again into another pass with no reliance on earlier commands there. The
//! scene passes replay the list from the sun (cascades), from the camera
//! (depth pre-pass) and into acceleration structures.
//!
//! A recorded draw is a set of changes against earlier draws of its pass
//! (`LastBoundCache`), so capture keeps the resolved handles and copies of
//! the bytes each draw bound, taken after the draw's own binds resolved
//! them; nothing is re-resolved later, since a rename can swap a handle.
//! The copies live in the frame's scratch arena, so a record is valid only
//! until this submission's finalize.
//!
//! Only draws whose render target 0 is the back buffer are kept: the same
//! `SpeedTree` and figure shaders also draw the water reflection and the
//! game's shadow-caster targets. A kept draw writes depth with the test on,
//! is perspective (no pre-transformed position) and does not blend; an
//! alpha-tested draw is kept with its stage-0 texture ([`CapturedAlpha`])
//! unless that texture is a render target or an ordered upload of this
//! submission, whose contents are not in place before the world's passes.

use core::ptr::NonNull;

use mtld3d_core::{
    draw_data::{AttrSnapshot, PsSourcePtr, PsSourceView, VsSourcePtr, VsSourceView},
    dxso::{FfStage, FfVsFlags, VariantKey},
    encoder_data::PostBoundary,
    ids::TextureId,
    pipeline_state::PipelineSnapshot,
};
use mtld3d_shared::{
    Command, VertexAttrDesc,
    mtl::{
        CullMode, IndexType, PrimitiveType, VS_BOOL_CONST_SLOT, VS_DRAW_SLOT, VS_FLOAT_CONST_SLOT,
        VS_INT_CONST_SLOT, VS_POS_FIXUP_SLOT,
    },
};
use mtld3d_types::{
    D3DCMP_ALWAYS, D3DCMP_NEVER, D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_SELECTMASK,
    D3DTA_TEXTURE, D3DTOP_MODULATE, D3DTOP_MODULATE2X, D3DTOP_MODULATE4X, D3DTOP_SELECTARG1,
    D3DTOP_SELECTARG2,
};
use rustc_hash::FxHashSet;

use super::{FrameEncoder, LOG_TARGET};

/// The fragment slot of the `alpha_ref` uniform (`emit_draw_view` binds it there too).
const ALPHA_REF_SLOT: u32 = 14;

#[cfg(test)]
pub(super) mod tests;

/// Bytes in the frame's scratch arena (or static storage) a command names by address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BytesRef {
    pub ptr: u64,
    pub len: u32,
}

impl BytesRef {
    pub const EMPTY: Self = Self { ptr: 0, len: 0 };

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Which contents of a vertex or index buffer a draw read: the buffer and its whole-content version.
///
/// A Metal handle names storage, not contents: a staging upload into the
/// range a draw did not read rewrites it in place, a rename gives the same
/// contents a new handle, and a destroyed handle's value can come back for
/// another buffer. The encoder gives every buffer a content version that
/// each change of its contents as a whole (a rename, a recreate, a `Direct`
/// rebind) replaces with one never used before
/// (`FrameEncoder::buffer_content`); an upload in place is stamped with its
/// range instead (`FrameEncoder::buffer_range_written`), so what read
/// another range of the same contents still holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BufferContent {
    /// The D3D9 buffer (`BufferId`), or [`Self::FAN_PATTERN`]'s id.
    pub id: u64,
    pub version: u64,
}

impl BufferContent {
    /// The encoder's shared triangle-fan index pattern, whose contents never change.
    pub const FAN_PATTERN: Self = Self {
        id: u64::MAX,
        version: 0,
    };
}

/// One vertex-stage buffer binding of a captured draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexBind {
    /// A Metal buffer at an offset (`setVertexBuffer`), resolved by the draw.
    Buffer {
        handle: u64,
        offset: u32,
        slot: u32,
        /// The contents the handle held for the draw.
        content: BufferContent,
    },
    /// `DrawPrimitiveUP` vertices (`setVertexBytes`, which the submit side
    /// moves to its upload ring past the inline limit), copied at capture.
    Inline { bytes: BytesRef, slot: u32 },
    /// The constant zero feed of a stream the declaration reads with nothing
    /// bound (`setVertexBytes` at a slot).
    Zeros { bytes: BytesRef, slot: u32 },
}

/// The most vertex bindings one draw makes: every crossing-fetch slot stays below `MAX_STREAMS`.
pub const MAX_VERTEX_BINDS: usize = mtld3d_types::MAX_STREAMS as usize;

/// Where a captured draw's vertices or indices come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturedIndex {
    /// Non-indexed.
    None {
        start_vertex: u32,
        vertex_count: u32,
    },
    /// An index buffer: the application's, or the shared triangle-fan pattern
    /// with the fan's start as the base vertex.
    Buffer {
        handle: u64,
        offset: u32,
        count: u32,
        index_type: IndexType,
        base_vertex: i32,
        /// The contents the handle held for the draw.
        content: BufferContent,
    },
    /// Inline indices (`DrawIndexedPrimitiveUP`, a generated fan list),
    /// copied at capture; drawn from the upload ring.
    Inline {
        bytes: BytesRef,
        count: u32,
        index_type: IndexType,
    },
}

/// A captured draw's viewport, in the bound target's pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapturedViewport {
    /// x, y, width, height.
    pub rect: (u32, u32, u32, u32),
    /// `MinZ`, `MaxZ`.
    pub depth: (f32, f32),
}

impl CapturedViewport {
    /// `setViewport` of this viewport.
    #[must_use]
    pub const fn command(&self) -> Command {
        let (x, y, width, height) = self.rect;
        Command::set_viewport(x, y, width, height, self.depth.0, self.depth.1)
    }
}

/// The stage-0 alpha test of a captured draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapturedAlpha {
    /// The stage-0 texture handle the draw sampled.
    pub tex: u64,
    /// The stage-0 sampler state handle.
    pub sampler: u64,
    /// `D3DRS_ALPHAREF` as the shader compares it, in `[0, 1]`.
    pub alpha_ref: f32,
    /// `D3DRS_ALPHAFUNC` (`D3DCMP_*`), never `ALWAYS` or `NEVER`.
    pub func: u8,
    /// The texture-coordinate varying (`texcoord<n>`) stage 0 samples at.
    pub uv_attr: u8,
}

/// The stage-0 colour texture of an opaque captured draw, for the ray-traced scene's hit albedo.
///
/// Kept only for a fixed-function pixel stage whose stage 0 colours from
/// its texture ([`colours_from_texture`]). A programmable pixel shader's
/// stage 0 need not be colour (`DAoC`'s terrain splat samples a weight mask
/// there), so its draws keep none and their hits are shaded by the
/// material alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapturedTexture {
    /// The stage-0 texture handle the draw sampled.
    pub tex: u64,
    /// The stage-0 sampler state handle.
    pub sampler: u64,
    /// The texture-coordinate varying (`texcoord<n>`) stage 0 samples at.
    pub uv_attr: u8,
}

/// What kind of shader drew a captured draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawClass {
    /// Fixed-function vertex processing.
    Ff,
    /// Fixed-function vertex blending (`D3DRS_VERTEXBLEND`).
    FfBlend,
    /// One of the game's skinned figure VS (`ppl::figures::figure_vs`).
    Figure,
    /// The game's `SpeedTree` leaf VS (`ppl::figures::speedtree_vs`).
    SpeedTree,
    /// Any other programmable VS.
    OtherProgrammable,
}

/// One back-buffer world draw, re-encodable on its own.
///
/// Every handle is the one the draw bound; every byte range is a copy in this
/// submission's scratch arena (the zero feed excepted, which is static). Valid
/// until this submission's finalize.
#[derive(Clone)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the replay stages (cascades, the depth pre-pass, ray tracing) read the records"
    )
)]
pub struct CapturedDraw {
    /// The pipeline snapshot the draw resolved its pipeline from.
    pub snapshot: PipelineSnapshot,
    /// The vertex declaration record.
    pub attrs: AttrSnapshot,
    /// A crossing fetch's remapped attributes (scratch copy), `None` for the declaration's own.
    fetch_attrs: Option<(NonNull<VertexAttrDesc>, u32)>,
    /// The VS source record.
    pub vs: VsSourcePtr,
    /// The PS source record.
    pub ps: PsSourcePtr,
    /// The pixel variant key the draw's pipeline was built with.
    pub variant: VariantKey,
    /// The pipeline the draw bound.
    pub pipeline: u64,
    /// The depth-stencil state the draw bound.
    pub depth_stencil: u64,
    binds: [VertexBind; MAX_VERTEX_BINDS],
    bind_count: u8,
    pub index: CapturedIndex,
    pub primitive: PrimitiveType,
    pub instances: u32,
    /// VS float constants (slot 30).
    pub vs_constants: BytesRef,
    /// `VsDraw` (slot 27).
    pub vs_draw: BytesRef,
    /// VS integer constants (slot 29), empty when the VS reads none.
    pub vs_int: BytesRef,
    /// VS boolean constants (slot 26), empty when the VS reads none.
    pub vs_bool: BytesRef,
    /// The draw's camera `pos_fixup` (slot 28).
    pub pos_fixup: BytesRef,
    /// The viewport the draw rasterized through, as Metal held it.
    pub viewport: CapturedViewport,
    pub cull: CullMode,
    /// `D3DRS_SLOPESCALEDEPTHBIAS` as the draw passed it to `setDepthBias`.
    pub slope_scale: f32,
    pub alpha: Option<CapturedAlpha>,
    /// An opaque draw's stage-0 colour texture ([`CapturedTexture`]); `None`
    /// for an alpha-tested draw, whose `alpha` names its texture.
    pub albedo: Option<CapturedTexture>,
    /// The `alpha_ref` uniform bytes (fragment slot 14); empty without `alpha`.
    pub alpha_ref_bytes: BytesRef,
    pub class: DrawClass,
    /// The VS lights the draw.
    pub lit: bool,
}

impl CapturedDraw {
    /// The vertex buffer bindings, crossing-fetch slots and zero feeds included.
    #[must_use]
    pub fn vertex_binds(&self) -> &[VertexBind] {
        &self.binds[..usize::from(self.bind_count)]
    }

    /// Whether a crossing fetch remapped the draw's attributes onto slots of its own.
    #[must_use]
    pub const fn crossing_fetch(&self) -> bool {
        self.fetch_attrs.is_some()
    }

    /// The vertex attributes the pipeline was built over.
    #[must_use]
    pub const fn vertex_attrs(&self) -> &[VertexAttrDesc] {
        match self.fetch_attrs {
            // SAFETY: a scratch copy of this submission, valid until its
            // finalize, which no record outlives.
            Some((ptr, len)) => unsafe { core::slice::from_raw_parts(ptr.as_ptr(), len as usize) },
            None => self.attrs.as_slice(),
        }
    }
}

/// What a replay binds in place of the draw's own pass state.
pub struct ReplayBinding<'a> {
    /// The pipeline to draw with: the draw's own, or a caster built from its snapshot.
    pub pipeline: u64,
    /// The depth-stencil state; 0 binds none.
    pub depth_stencil: u64,
    pub cull: CullMode,
    /// `setDepthBias(constant, slope)`.
    pub depth_bias: (f32, f32),
    /// The `pos_fixup` bytes (slot 28): the draw's own for the camera, a
    /// neutral one for a light view.
    pub pos_fixup: BytesRef,
    /// Extra vertex-stage bytes `(slot, bytes)`, such as a re-projection.
    pub extra_vs: &'a [(u32, BytesRef)],
}

/// Encode `draw` into `out` with every binding it reads, then the draw.
///
/// Nothing relies on commands earlier in `out`: the pipeline, depth-stencil
/// state, cull, depth bias, every vertex buffer slot, the VS constants,
/// `VsDraw`, the integer and boolean constants, `pos_fixup` and the extra
/// vertex bytes are all bound again. The fragment stage gets only what a
/// caster pixel function (`replay.rs`) reads: for an alpha-tested draw the
/// stage-0 texture, its sampler and `alpha_ref`, nothing else, so
/// `binding.pipeline` must be a caster's, not one with the world draw's own
/// pixel shader. The pass (viewport, scissor, attachments) is the caller's.
pub fn encode_captured_draw(
    out: &mut Vec<Command>,
    draw: &CapturedDraw,
    binding: &ReplayBinding<'_>,
) {
    out.push(Command::set_render_pipeline_state(binding.pipeline));
    if binding.depth_stencil != 0 {
        out.push(Command::set_depth_stencil_state(binding.depth_stencil));
    }
    out.push(Command::set_cull_mode(binding.cull));
    out.push(Command::set_depth_bias(
        binding.depth_bias.0,
        binding.depth_bias.1,
    ));
    for bind in draw.vertex_binds() {
        out.push(match *bind {
            VertexBind::Buffer {
                handle,
                offset,
                slot,
                ..
            } => Command::set_vertex_buffer(handle, offset, slot),
            VertexBind::Inline { bytes, slot } => {
                Command::set_vertex_bytes(bytes.ptr, bytes.len, slot)
            }
            VertexBind::Zeros { bytes, slot } => {
                Command::set_vertex_bytes_at(bytes.ptr, bytes.len, slot)
            }
        });
    }
    let mut bytes_at = |bytes: BytesRef, slot: u32| {
        if !bytes.is_empty() {
            out.push(Command::set_vertex_bytes_at(bytes.ptr, bytes.len, slot));
        }
    };
    bytes_at(draw.vs_constants, VS_FLOAT_CONST_SLOT);
    bytes_at(draw.vs_draw, VS_DRAW_SLOT);
    bytes_at(draw.vs_int, VS_INT_CONST_SLOT);
    bytes_at(draw.vs_bool, VS_BOOL_CONST_SLOT);
    bytes_at(binding.pos_fixup, VS_POS_FIXUP_SLOT);
    for &(slot, bytes) in binding.extra_vs {
        bytes_at(bytes, slot);
    }
    if let Some(alpha) = draw.alpha {
        out.push(Command::set_fragment_texture(alpha.tex, 0));
        out.push(Command::set_fragment_sampler_state(alpha.sampler, 0));
        out.push(Command::set_fragment_bytes_at(
            draw.alpha_ref_bytes.ptr,
            draw.alpha_ref_bytes.len,
            ALPHA_REF_SLOT,
        ));
    }
    out.push(match draw.index {
        CapturedIndex::None {
            start_vertex,
            vertex_count,
        } => Command::draw_primitives(draw.primitive, start_vertex, vertex_count),
        CapturedIndex::Buffer {
            handle,
            offset,
            count,
            index_type,
            base_vertex,
            ..
        } => Command::draw_indexed_primitives(
            draw.primitive,
            count,
            index_type,
            handle,
            offset,
            base_vertex,
            draw.instances,
        ),
        CapturedIndex::Inline {
            bytes,
            count,
            index_type,
        } => Command::draw_indexed_primitives_up(
            draw.primitive,
            count,
            index_type,
            bytes.ptr,
            bytes.len,
            draw.instances,
        ),
    });
}

/// The stage-0 texture of an alpha-tested draw, as the draw bound it.
#[derive(Clone, Copy)]
pub struct Stage0 {
    pub texture: TextureId,
    pub handle: u64,
    pub sampler: u64,
}

/// One just-emitted draw, as `emit_draw_view` resolved it.
///
/// Byte slices and the source index bytes are the draw's own; capture copies
/// what it keeps. The vertex bindings were staged with
/// [`FrameEncoder::stage_scene_bind`] while the draw bound them.
pub struct DrawFacts<'a> {
    pub snapshot: &'a PipelineSnapshot,
    pub attrs: AttrSnapshot,
    /// A crossing fetch's attributes, `None` for the declaration's own.
    pub fetch_attrs: Option<&'a [VertexAttrDesc]>,
    pub vs: VsSourcePtr,
    pub ps: PsSourcePtr,
    pub variant: VariantKey,
    pub pipeline: u64,
    pub depth_stencil: u64,
    /// The pass binds depth, the test is on and the draw writes it.
    pub depth_write: bool,
    pub blend: bool,
    /// The index source; `Inline` names the draw's own index bytes.
    pub index: CapturedIndex,
    pub primitive: PrimitiveType,
    pub instances: u32,
    pub vs_constants: &'a [u8],
    pub vs_draw: &'a [u8],
    pub vs_int: &'a [u8],
    pub vs_bool: &'a [u8],
    pub pos_fixup: &'a [u8],
    pub alpha_ref: &'a [u8],
    pub cull: CullMode,
    pub slope_scale: f32,
    /// Stage 0 as sampled, `None` when the PS samples no texture there.
    pub stage0: Option<Stage0>,
}

/// One copy of a byte range, reused while the next draw names the same source.
#[derive(Clone, Copy, Default)]
struct CopyMemo {
    src: u64,
    len: u32,
    copy: u64,
}

/// The encoder's capture state, see the module docs.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent per-submission facts, not a state machine"
)]
pub struct SceneCapture {
    /// `scene_capture_wanted` for this encoder's configuration.
    wanted: bool,
    /// The boundary fired this frame; nothing more is captured until `Present`.
    boundary_fired: bool,
    /// Set by the scene passes when this submission's world is split.
    split: bool,
    /// An earlier submission of this frame, ended by a mid-frame flush,
    /// captured world draws: the rest of the world is elsewhere.
    earlier_world: bool,
    /// The boundary's constants when it fired this submission.
    boundary: Option<PostBoundary>,
    draws: Vec<CapturedDraw>,
    /// The bindings of the draw being emitted.
    pending: Vec<VertexBind>,
    /// Textures written by an ordered upload this submission.
    ordered_uploads: FxHashSet<TextureId>,
    /// The copies of the last vertex bytes, keyed by their source.
    vs_constants: CopyMemo,
    vs_draw: CopyMemo,
}

impl SceneCapture {
    #[must_use]
    pub fn new(wanted: bool) -> Self {
        Self {
            wanted,
            ..Self::default()
        }
    }

    /// Start a submission: the records of the last one died with its arena.
    ///
    /// The boundary stays fired across a mid-frame flush (`continues_frame`):
    /// the D3D9 frame did not end there. Whether the flushed part of the
    /// frame held world draws is carried over, since the records themselves
    /// are not.
    pub fn begin_submission(&mut self, continues_frame: bool) {
        self.earlier_world = continues_frame && (self.earlier_world || !self.draws.is_empty());
        self.draws.clear();
        self.pending.clear();
        self.ordered_uploads.clear();
        self.vs_constants = CopyMemo::default();
        self.vs_draw = CopyMemo::default();
        self.split = false;
        self.boundary = None;
        if !continues_frame {
            self.boundary_fired = false;
        }
    }

    /// `scene_capture_wanted` for this encoder's configuration.
    #[must_use]
    pub const fn wanted(&self) -> bool {
        self.wanted
    }

    /// Why this submission holds only part of the frame's world, `None` when it holds all of it or none.
    ///
    /// The world is the captured draws, so a submission that captured none
    /// is never split. One that did is split when it ends in a mid-frame
    /// flush (`no_present`) before the boundary, or when an earlier
    /// submission of its frame, cut by a flush, captured world draws too:
    /// that part of the world was encoded with the other submission, whose
    /// records died with it. A flush that cut the frame before any world
    /// draw (a readback, a query) leaves the world whole.
    #[must_use]
    pub const fn world_split(&self, no_present: bool) -> Option<WorldSplit> {
        if self.draws.is_empty() {
            None
        } else if no_present && !self.boundary_fired {
            Some(WorldSplit::Flushed)
        } else if self.earlier_world {
            Some(WorldSplit::Continued)
        } else {
            None
        }
    }

    /// The scene passes found this submission's world split.
    pub const fn mark_split(&mut self) {
        self.split = true;
    }

    /// The boundary op ran: the world of this frame is complete, under `boundary`'s constants.
    pub const fn note_boundary(&mut self, boundary: &PostBoundary) {
        self.boundary_fired = true;
        if self.wanted {
            self.boundary = Some(*boundary);
        }
    }

    /// The boundary's constants, when it fired this submission.
    #[must_use]
    pub const fn boundary(&self) -> Option<&PostBoundary> {
        self.boundary.as_ref()
    }

    /// Move this submission's records out, for a replay that also needs the encoder; [`Self::put_draws`] returns them.
    pub fn take_draws(&mut self) -> Vec<CapturedDraw> {
        core::mem::take(&mut self.draws)
    }

    /// Return the records [`Self::take_draws`] moved out.
    pub fn put_draws(&mut self, draws: Vec<CapturedDraw>) {
        debug_assert!(
            self.draws.is_empty(),
            "nothing is captured while they are out"
        );
        self.draws = draws;
    }

    /// Capture is on and the boundary has not fired this frame.
    #[must_use]
    pub const fn armed(&self) -> bool {
        self.wanted && !self.boundary_fired
    }
}

/// How a submission's world is split, see [`SceneCapture::world_split`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldSplit {
    /// A mid-frame flush ends this submission before the boundary.
    Flushed,
    /// This submission continues after a mid-frame flush.
    Continued,
}

impl FrameEncoder {
    /// The world draws captured this submission, in draw order.
    #[must_use]
    pub fn scene_draws(&self) -> &[CapturedDraw] {
        &self.scene.draws
    }

    /// Whether this submission's world is split across submissions (set by the scene passes).
    #[cfg_attr(not(test), expect(dead_code, reason = "the receivers read it"))]
    #[must_use]
    pub const fn scene_split(&self) -> bool {
        self.scene.split
    }

    /// Whether the draw being emitted may be captured: capture is on and the boundary is ahead.
    #[must_use]
    pub const fn scene_capture_armed(&self) -> bool {
        self.scene.armed()
    }

    /// Forget bindings staged by a draw that was dropped before its capture.
    pub fn begin_scene_binds(&mut self) {
        self.scene.pending.clear();
    }

    /// Stage one vertex binding of the draw being emitted, after the draw resolved it.
    pub fn stage_scene_bind(&mut self, bind: VertexBind) {
        self.scene.pending.push(bind);
    }

    /// Note a texture an ordered upload wrote this submission.
    pub fn note_scene_ordered_upload(&mut self, texture: TextureId) {
        if self.scene.wanted {
            self.scene.ordered_uploads.insert(texture);
        }
    }

    /// Keep the draw just emitted if it is a back-buffer world draw, see the module docs.
    ///
    /// Takes the staged vertex bindings either way.
    pub fn capture_scene_draw(&mut self, facts: &DrawFacts<'_>) {
        let binds = core::mem::take(&mut self.scene.pending);
        if self.scene.armed()
            && let Some(admitted) = self.admit(facts)
        {
            self.record(facts, &binds, admitted);
        }
        let mut binds = binds;
        binds.clear();
        self.scene.pending = binds;
    }

    /// What of the draw is kept, `None` when it is not.
    fn admit(&self, facts: &DrawFacts<'_>) -> Option<Admitted> {
        let back_buffer = self.pass_state.backbuffer_texture();
        if back_buffer.is_null() || self.pass_state.current_color_texture() != back_buffer {
            return None;
        }
        if !facts.depth_write || facts.blend {
            return None;
        }
        if mtld3d_core::async_compile::DeferredPipelineId::is_placeholder(facts.pipeline) {
            // Its pipeline (and maybe its libraries) is still building; the
            // submission resolves or drops the draw only at finalize.
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "scene: a world draw's pipeline is still building; not captured this frame");
            return None;
        }
        let (class, lit) = match facts.vs.as_ref() {
            VsSourceView::FixedFunction(source) => {
                if source.key.flags.contains(FfVsFlags::HAS_RHW) {
                    return None;
                }
                let class = if source.key.vertex_blend_count > 0 {
                    DrawClass::FfBlend
                } else {
                    DrawClass::Ff
                };
                (
                    class,
                    source.key.flags.contains(FfVsFlags::LIGHTING_ENABLED),
                )
            }
            VsSourceView::Programmable(source) => {
                // A replay binds no vertex textures.
                if self.ps_declared_samplers(source.vs_id).mask() != 0 {
                    mtld3d_shared::log_once_info!(target: LOG_TARGET,
                        "scene: a world draw's vertex shader samples textures; not captured");
                    return None;
                }
                let id = source.vs_id.raw();
                if mtld3d_core::ppl::figures::figure_vs(id).is_some() {
                    (DrawClass::Figure, true)
                } else if mtld3d_core::ppl::figures::speedtree_vs(id) {
                    (DrawClass::SpeedTree, true)
                } else {
                    (DrawClass::OtherProgrammable, false)
                }
            }
        };
        let func = u32::from(facts.variant.alpha_func);
        if func == D3DCMP_NEVER {
            return None;
        }
        if func == 0 || func == D3DCMP_ALWAYS {
            return Some(Admitted {
                alpha: None,
                albedo: self.albedo_texture(facts),
                class,
                lit,
            });
        }
        let Some(stage0) = facts.stage0.filter(|stage0| stage0.handle != 0) else {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "scene: an alpha-tested world draw samples no stage-0 texture; not captured");
            return None;
        };
        if self.texture_is_render_target(stage0.texture)
            || self.scene.ordered_uploads.contains(&stage0.texture)
        {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "scene: an alpha-tested world draw samples a render target or an ordered upload \
                 at stage 0; not captured");
            return None;
        }
        let Some(uv_attr) = self.stage0_varying(facts.ps) else {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "scene: an alpha-tested world draw's pixel shader picks its stage-0 coordinate \
                 itself; not captured");
            return None;
        };
        let Ok(alpha_ref) = <[u8; 4]>::try_from(facts.alpha_ref).map(f32::from_le_bytes) else {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "scene: an alpha-tested world draw bound no alpha reference; not captured");
            return None;
        };
        Some(Admitted {
            alpha: Some(CapturedAlpha {
                tex: stage0.handle,
                sampler: stage0.sampler,
                alpha_ref,
                func: facts.variant.alpha_func,
                uv_attr,
            }),
            albedo: None,
            class,
            lit,
        })
    }

    /// An opaque draw's stage-0 colour texture, see [`CapturedTexture`]:
    /// a plain 2D texture the fixed-function pixel stage colours from, and
    /// neither a render target nor an ordered upload of this submission.
    fn albedo_texture(&self, facts: &DrawFacts<'_>) -> Option<CapturedTexture> {
        let PsSourceView::FixedFunction(source) = facts.ps.as_ref() else {
            return None;
        };
        let v = &facts.variant;
        if !colours_from_texture(&source.key.stages[0])
            || (v.volume_sampler_mask | v.cube_sampler_mask | v.depth_sampler_mask | v.fetch4_mask)
                & 1
                != 0
        {
            return None;
        }
        let stage0 = facts.stage0.filter(|stage0| stage0.handle != 0)?;
        if self.texture_is_render_target(stage0.texture)
            || self.scene.ordered_uploads.contains(&stage0.texture)
        {
            return None;
        }
        Some(CapturedTexture {
            tex: stage0.handle,
            sampler: stage0.sampler,
            uv_attr: 0,
        })
    }

    /// The `texcoord<n>` varying stage 0 samples at, `None` when the PS chooses it.
    ///
    /// The fixed-function pixel stage samples stage `n` at `texcoord<n>`, and
    /// so does `tex t0` of `ps_1_0`..`ps_1_3`; `ps_1_4` and later name any
    /// coordinate register.
    fn stage0_varying(&self, ps: PsSourcePtr) -> Option<u8> {
        match ps.as_ref() {
            PsSourceView::FixedFunction(_) => Some(0),
            PsSourceView::Programmable(source) => self
                .program_cache
                .get(&source.ps_id)
                .filter(|program| program.major == 1 && program.minor < 4)
                .map(|_| 0),
        }
    }

    fn record(&mut self, facts: &DrawFacts<'_>, binds: &[VertexBind], admitted: Admitted) {
        let Admitted {
            alpha,
            albedo,
            class,
            lit,
        } = admitted;
        if binds.len() > MAX_VERTEX_BINDS {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "scene: a world draw binds more vertex slots than a capture holds; not captured");
            return;
        }
        let mut kept = [VertexBind::Buffer {
            handle: 0,
            offset: 0,
            slot: 0,
            content: BufferContent::default(),
        }; MAX_VERTEX_BINDS];
        for (slot, bind) in kept.iter_mut().zip(binds) {
            *slot = match *bind {
                VertexBind::Inline { bytes, slot } => VertexBind::Inline {
                    bytes: self.copy_bytes(bytes),
                    slot,
                },
                other => other,
            };
        }
        let fetch_attrs = facts.fetch_attrs.map(|attrs| {
            let (ptr, len) = self.scratch.alloc_slice(attrs);
            (
                NonNull::new(ptr).expect("the scratch arena never hands out null"),
                len,
            )
        });
        let index = match facts.index {
            CapturedIndex::Inline {
                bytes,
                count,
                index_type,
            } => CapturedIndex::Inline {
                bytes: self.copy_bytes(bytes),
                count,
                index_type,
            },
            other => other,
        };
        let vs_constants = copy_memo(
            &mut self.scratch,
            &mut self.scene.vs_constants,
            facts.vs_constants,
        );
        let vs_draw = copy_memo(&mut self.scratch, &mut self.scene.vs_draw, facts.vs_draw);
        let vs_int = self.copy_slice(facts.vs_int);
        let vs_bool = self.copy_slice(facts.vs_bool);
        let pos_fixup = self.copy_slice(facts.pos_fixup);
        let alpha_ref_bytes = if alpha.is_some() {
            self.copy_slice(facts.alpha_ref)
        } else {
            BytesRef::EMPTY
        };
        self.scene.draws.push(CapturedDraw {
            snapshot: facts.snapshot.clone(),
            attrs: facts.attrs,
            fetch_attrs,
            vs: facts.vs,
            ps: facts.ps,
            variant: facts.variant,
            pipeline: facts.pipeline,
            depth_stencil: facts.depth_stencil,
            binds: kept,
            bind_count: u8::try_from(binds.len()).expect("at most MAX_VERTEX_BINDS"),
            index,
            primitive: facts.primitive,
            instances: facts.instances,
            vs_constants,
            vs_draw,
            vs_int,
            vs_bool,
            pos_fixup,
            viewport: CapturedViewport {
                rect: self.pass_state.effective_viewport(),
                depth: self.pass_state.viewport_depth_range(),
            },
            cull: facts.cull,
            slope_scale: facts.slope_scale,
            alpha,
            albedo,
            alpha_ref_bytes,
            class,
            lit,
        });
    }

    /// Copy `bytes` (live for this submission) into the scratch arena.
    fn copy_bytes(&mut self, bytes: BytesRef) -> BytesRef {
        if bytes.is_empty() {
            return BytesRef::EMPTY;
        }
        // SAFETY: `bytes` names the draw's own retained bytes, live through
        // this submission's replay.
        let slice =
            unsafe { core::slice::from_raw_parts(bytes.ptr as *const u8, bytes.len as usize) };
        self.copy_slice(slice)
    }

    fn copy_slice(&mut self, bytes: &[u8]) -> BytesRef {
        if bytes.is_empty() {
            return BytesRef::EMPTY;
        }
        BytesRef {
            ptr: self.scratch.alloc(bytes),
            len: u32::try_from(bytes.len()).expect("a draw's bound bytes fit u32"),
        }
    }
}

/// What [`FrameEncoder::admit`] keeps of a draw.
struct Admitted {
    alpha: Option<CapturedAlpha>,
    albedo: Option<CapturedTexture>,
    class: DrawClass,
    lit: bool,
}

/// Whether fixed-function stage `stage` colours from its texture: a
/// modulate of it or a select of it, the texture argument unmodified.
#[must_use]
pub fn colours_from_texture(stage: &FfStage) -> bool {
    let texture = |arg: u8| {
        let arg = u32::from(arg);
        arg & D3DTA_SELECTMASK == D3DTA_TEXTURE
            && arg & (D3DTA_COMPLEMENT | D3DTA_ALPHAREPLICATE) == 0
    };
    stage.has_texture()
        && match u32::from(stage.color_op) {
            D3DTOP_MODULATE | D3DTOP_MODULATE2X | D3DTOP_MODULATE4X => {
                texture(stage.color_arg1) || texture(stage.color_arg2)
            }
            D3DTOP_SELECTARG1 => texture(stage.color_arg1),
            D3DTOP_SELECTARG2 => texture(stage.color_arg2),
            _ => false,
        }
}

/// Copy `bytes` into `scratch`, or reuse the last copy when it names the same source.
///
/// Consecutive draws that share their constants share one snapshot (the
/// encoder dedups by snapshot token), so the copy is taken once per run.
fn copy_memo(
    scratch: &mut mtld3d_core::scratch::ScratchArena,
    memo: &mut CopyMemo,
    bytes: &[u8],
) -> BytesRef {
    if bytes.is_empty() {
        return BytesRef::EMPTY;
    }
    let len = u32::try_from(bytes.len()).expect("a draw's bound bytes fit u32");
    let src = bytes.as_ptr() as u64;
    if memo.copy == 0 || memo.src != src || memo.len != len {
        *memo = CopyMemo {
            src,
            len,
            copy: scratch.alloc(bytes),
        };
    }
    BytesRef {
        ptr: memo.copy,
        len,
    }
}
