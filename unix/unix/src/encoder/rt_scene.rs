//! Acceleration structures for the captured world: the BLAS cache and the per-frame instance list.
//!
//! While an `rt.*` key traces rays on a device that can
//! ([`crate::metal::rt::effective_config`]), each submission whose world
//! the scene capture holds whole builds an instance acceleration structure
//! (TLAS) of it in eye space, ahead of the world ([`build_frame_tlas`],
//! called by `scene::assemble`). Its instances are the captured world
//! draws. A static fixed-function draw (no vertex blending) is instanced
//! from object-space positions read straight from the bound vertex buffer,
//! under its own world-view (rows 0..3 of its VS constants). A draw whose
//! vertex shader moves its vertices every frame (a skinned figure, a
//! `SpeedTree` leaf's wind, fixed-function vertex blending) is captured
//! instead ([`capture_draw`]): a vertex-only pass ahead of the world runs
//! its own vertex shader keyed for the capture, which stores each vertex's
//! eye position and stage-0 coordinate, and its BLAS is built from those,
//! under the identity.
//!
//! The geometry of a static draw is one primitive acceleration structure
//! (BLAS), cached across frames under a [`BlasKey`]: the contents (buffer and
//! content version, `scene_capture::BufferContent`) and the range the draw
//! read, never a Metal handle, so a rename or an in-place upload builds anew
//! and a reused handle never matches. A frame retires every cached BLAS
//! whose contents its buffer no longer holds (an upload, a rename, a
//! release) behind the GPU. Strips become lists through an index rewrite
//! into a list buffer of the frame; fans already are (the shared pattern,
//! its first vertex folded into the vertex offset). A BLAS is compacted once
//! the GPU has finished its build: a size query rides the build, a later
//! frame reads it and copies the BLAS into one that size. A captured draw's
//! BLAS is never cached: it is built every frame into the frame slot's own
//! acceleration structures, and counted apart ([`FrameStats::captured`]).
//! The cache and the frames' buffers stay within [`RT_BUDGET_BYTES`], the
//! least recently used BLAS going first; a frame builds at most
//! [`MAX_BUILDS_PER_FRAME`] BLASes with at most [`MAX_FRAME_SCRATCH`] of
//! scratch, leaving the rest to later frames.
//!
//! Draws a BLAS cannot describe are left out and counted, each cause said
//! once: the inline (`DrawPrimitiveUP`, generated fan) vertices or indices
//! of a submission's arena are not copied, a position that is not `float3`
//! or `float4`, an offset Metal cannot build from, an instanced or
//! crossing-fetch draw, a primitive other than triangles, another
//! programmable vertex shader, a capture with no camera projection.

use mtld3d_core::{
    async_compile::Resolution,
    dxso::VsScene,
    shadows::cascades::{camera_near_far, projection, reproject_for},
};
use mtld3d_shared::{
    BlitCommand, Command, MetalHandle,
    mtl::{
        CullMode, DestroyKind, IndexType, PrimitiveType, VS_CAPTURE_COUNT_SLOT,
        VS_CAPTURE_POS_SLOT, VS_CAPTURE_UV_SLOT, VS_REPROJECT_SLOT, VertexFormat,
        VertexStepFunction,
    },
    mtl_handle::{MTLAccelerationStructureKind, MTLBufferKind, MTLComputePipelineStateKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice};
use rustc_hash::FxHashMap;

use super::{
    FrameEncoder, LOG_TARGET, PendingResourceRetention,
    replay::{IDENTITY, neutral_pos_fixup, reproject_bytes},
    scene_capture::{
        BufferContent, BytesRef, CapturedDraw, CapturedIndex, DrawClass, ReplayBinding, VertexBind,
        encode_captured_draw,
    },
};
use crate::metal::{
    handle::IntoRetained,
    rt::{
        BlasBuild, BlasGeometry, InstanceDescriptor, RT_BUDGET_BYTES, SUBALLOC_ALIGN,
        StripConversion, TlasBuild, blas_sizes, create_strip_pipeline, instance_descriptor,
        new_acceleration_structure, new_buffer, tlas_sizes,
    },
};

#[cfg(test)]
mod capture_tests;
#[cfg(test)]
mod tests;

/// The fixed-function position attribute (`ff_attr_index_for_semantic`).
const POSITION_ATTR: u32 = 0;
/// Instance mask bits: opaque, alpha-tested, figure, leaf.
pub const MASK_OPAQUE: u8 = 1;
pub const MASK_ALPHA: u8 = 2;
/// A skinned figure or a fixed-function draw with vertex blending.
pub const MASK_FIGURE: u8 = 4;
/// A `SpeedTree` leaf.
pub const MASK_LEAF: u8 = 8;
/// New BLAS builds one frame encodes; the rest wait for later frames.
pub const MAX_BUILDS_PER_FRAME: usize = 256;
/// Build scratch one frame's new BLASes may take.
pub const MAX_FRAME_SCRATCH: u64 = 64 << 20;
/// Frames whose TLAS and buffers can be in flight at once.
const MAX_FRAME_SLOTS: usize = 4;
/// Compacted-size queries in flight at once, one `u32` each.
const SIZE_QUERIES: u32 = 1024;
/// `BlasKey::index_type` of a non-indexed draw.
const NO_INDEX: u8 = 0xFF;
/// Bytes of one captured eye position, a `float4` whose `w` is 1.
pub const CAPTURED_POSITION_BYTES: u32 = 16;
/// Bytes of one captured stage-0 coordinate, a `float2`.
pub const CAPTURED_UV_BYTES: u32 = 8;
/// The most vertex ids one captured draw may reach.
const MAX_CAPTURED_VERTICES: u32 = 1 << 20;
/// The object-to-eye rows of a captured draw, whose positions already are in eye space.
const IDENTITY_ROWS: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

/// The geometry one BLAS holds: the contents and range a draw read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BlasKey {
    /// The vertex buffer's `BufferContent`.
    pub vb: u64,
    pub vb_version: u64,
    /// The stream offset the draw bound the vertex buffer at.
    pub vb_offset: u32,
    pub stride: u16,
    /// The position's offset within a vertex.
    pub pos_offset: u16,
    /// `VertexFormat` of the position.
    pub format: u8,
    /// A non-indexed draw's vertex count, 0 for an indexed one.
    pub vertex_count: u32,
    /// The index buffer's `BufferContent`, 0 for a non-indexed draw.
    pub ib: u64,
    pub ib_version: u64,
    pub ib_offset: u32,
    pub index_count: u32,
    /// `IndexType`, [`NO_INDEX`] for a non-indexed draw.
    pub index_type: u8,
    /// The base vertex, or a non-indexed draw's first vertex.
    pub base_vertex: i32,
    /// 0 for a triangle list, 1 for a strip.
    pub prim: u8,
}

impl BlasKey {
    fn vertex_content(&self) -> BufferContent {
        BufferContent {
            id: self.vb,
            version: self.vb_version,
        }
    }

    fn index_content(&self) -> Option<BufferContent> {
        (self.index_type != NO_INDEX).then_some(BufferContent {
            id: self.ib,
            version: self.ib_version,
        })
    }
}

/// Why a captured draw is left out of the TLAS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// A programmable vertex shader that is neither a figure's nor a leaf's.
    Class,
    /// Instanced, or through a crossing fetch.
    Instanced,
    /// No position read from a bound vertex buffer.
    Position,
    /// Vertices or indices inline in the submission's arena.
    Inline,
    /// A position format other than `float3`/`float4`, or a stride Metal rejects.
    Format,
    /// Not triangles, or none.
    Primitive,
    /// A vertex offset Metal cannot build from.
    Offset,
    /// No world-view among its constants.
    Transform,
    /// A capture with no camera projection to take its positions back to eye space.
    Projection,
    /// A capture whose vertex range is unknown or beyond [`MAX_CAPTURED_VERTICES`].
    Extent,
    /// A capture whose vertex shader or pipeline failed to build, or whose buffers could not be made.
    Capture,
}

impl DrawClass {
    /// Whether the ray-traced scene captures this class's geometry every
    /// frame ([`capture_draw`]) rather than reading it from its vertex buffer.
    #[must_use]
    pub const fn captured(self) -> bool {
        matches!(self, Self::FfBlend | Self::Figure | Self::SpeedTree)
    }
}

/// Where a BLAS's indices come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceIndex {
    None,
    Buffer {
        handle: u64,
        offset: u32,
        index_type: IndexType,
    },
}

/// A captured draw's geometry: its key and the live objects a build reads.
#[derive(Clone, Copy, Debug)]
pub struct BlasSource {
    pub key: BlasKey,
    vertex_buffer: u64,
    /// Byte offset of the first position: the bind's offset, the first or
    /// base vertex and the position's own offset folded together.
    vertex_offset: u64,
    stride: u32,
    format: VertexFormat,
    index: SourceIndex,
    strip: bool,
    triangles: u32,
}

/// The geometry of `d` when a BLAS can hold it, else why not.
///
/// # Errors
///
/// The [`Skip`] that leaves the draw out.
pub fn blas_source(d: &CapturedDraw) -> Result<BlasSource, Skip> {
    if d.class != DrawClass::Ff {
        return Err(Skip::Class);
    }
    if d.instances != 1 || d.crossing_fetch() {
        return Err(Skip::Instanced);
    }
    let position = d
        .vertex_attrs()
        .iter()
        .find(|a| a.attr_index == POSITION_ATTR)
        .ok_or(Skip::Position)?;
    if !matches!(position.format, VertexFormat::Float3 | VertexFormat::Float4) {
        return Err(Skip::Format);
    }
    let layout = d
        .snapshot
        .stream_layouts
        .get(position.buffer_index as usize)
        .filter(|l| l.is_used() && l.step == VertexStepFunction::PerVertex)
        .ok_or(Skip::Position)?;
    let (vertex_buffer, bind_offset, content) = match d
        .vertex_binds()
        .iter()
        .find(|b| bind_slot(b) == position.buffer_index)
    {
        Some(&VertexBind::Buffer {
            handle,
            offset,
            content,
            ..
        }) => (handle, offset, content),
        Some(VertexBind::Inline { .. }) => return Err(Skip::Inline),
        _ => return Err(Skip::Position),
    };
    let stride = u16::try_from(layout.stride).map_err(|_| Skip::Format)?;
    let pos_offset = u16::try_from(position.offset).map_err(|_| Skip::Format)?;
    if !stride.is_multiple_of(4)
        || u32::from(stride) < u32::from(pos_offset) + position.format.byte_size()
    {
        return Err(Skip::Format);
    }
    let strip = match d.primitive {
        PrimitiveType::Triangle => false,
        PrimitiveType::TriangleStrip => true,
        _ => return Err(Skip::Primitive),
    };
    let triangles = |count: u32| {
        if strip {
            count.saturating_sub(2)
        } else {
            count / 3
        }
    };
    let mut key = BlasKey {
        vb: content.id,
        vb_version: content.version,
        vb_offset: bind_offset,
        stride,
        pos_offset,
        format: position.format as u8,
        prim: u8::from(strip),
        index_type: NO_INDEX,
        ..BlasKey::default()
    };
    let (first, index, count) = match d.index {
        CapturedIndex::None {
            start_vertex,
            vertex_count,
        } => {
            key.vertex_count = vertex_count;
            key.base_vertex = i32::try_from(start_vertex).map_err(|_| Skip::Offset)?;
            (i64::from(start_vertex), SourceIndex::None, vertex_count)
        }
        CapturedIndex::Buffer {
            handle,
            offset,
            count,
            index_type,
            base_vertex,
            content,
        } => {
            key.ib = content.id;
            key.ib_version = content.version;
            key.ib_offset = offset;
            key.index_count = count;
            key.index_type = index_type as u8;
            key.base_vertex = base_vertex;
            (
                i64::from(base_vertex),
                SourceIndex::Buffer {
                    handle,
                    offset,
                    index_type,
                },
                count,
            )
        }
        CapturedIndex::Inline { .. } => return Err(Skip::Inline),
    };
    let triangles = triangles(count);
    if triangles == 0 {
        return Err(Skip::Primitive);
    }
    let vertex_offset = i64::from(bind_offset) + first * i64::from(stride) + i64::from(pos_offset);
    let vertex_offset = u64::try_from(vertex_offset).map_err(|_| Skip::Offset)?;
    if !vertex_offset.is_multiple_of(4) {
        return Err(Skip::Offset);
    }
    Ok(BlasSource {
        key,
        vertex_buffer,
        vertex_offset,
        stride: u32::from(stride),
        format: position.format,
        index,
        strip,
        triangles,
    })
}

const fn bind_slot(bind: &VertexBind) -> u32 {
    match *bind {
        VertexBind::Buffer { slot, .. }
        | VertexBind::Inline { slot, .. }
        | VertexBind::Zeros { slot, .. } => slot,
    }
}

/// The object-to-eye rows of `d`'s world-view: VS constants 0..3, `p_eye[r] = dot(c[r], (p, 1))`.
fn world_view_rows(d: &CapturedDraw) -> Option<[[f32; 4]; 3]> {
    const BYTES: usize = 3 * 16;
    if (d.vs_constants.len as usize) < BYTES {
        return None;
    }
    // SAFETY: the draw's constants copy in the submission's scratch arena,
    // live through this finalize, at least three rows long.
    let bytes = unsafe { core::slice::from_raw_parts(d.vs_constants.ptr as *const u8, BYTES) };
    let mut rows = [[0.0f32; 4]; 3];
    for (value, chunk) in rows
        .as_flattened_mut()
        .iter_mut()
        .zip(bytes.as_chunks::<4>().0)
    {
        *value = f32::from_le_bytes(*chunk);
    }
    rows.as_flattened()
        .iter()
        .all(|v| v.is_finite())
        .then_some(rows)
}

/// Where one capture stores its vertices, and the inverse projection it binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureTarget {
    /// The `float4` eye-position buffer and the byte offset of vertex id 0.
    pub pos_buf: u64,
    pub pos_offset: u64,
    /// The `float2` stage-0 coordinate buffer and the byte offset of vertex id 0.
    pub uv_buf: u64,
    pub uv_offset: u64,
    /// The vertex ids the draw can reach, `0..vertex_count` ([`capture_extent`]),
    /// which the buffers hold: the shader stores no id at or past it.
    pub vertex_count: u32,
    /// The inverse camera projection ([`inverse_projection_bytes`]).
    pub p_inv: BytesRef,
}

/// A captured draw's geometry: written by its capture, read by its BLAS build and its hit shading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapturedGeometry {
    /// The buffer of one `float4` eye position (`w` 1) per vertex id, and the byte offset of id 0.
    pub pos_buf: u64,
    pub pos_offset: u64,
    /// The buffer of one `float2` stage-0 coordinate per vertex id, and the byte offset of id 0.
    pub uv_buf: u64,
    pub uv_offset: u64,
    /// The vertex ids the draw can reach: `0..vertex_count`.
    pub vertex_count: u32,
    /// The draw's own index source; its base or first vertex counts in the vertex ids.
    pub ib: CapturedIndex,
}

/// The inverse of the camera projection `proj` (`PostBoundary::proj`) as the
/// capture binds it, in this submission's scratch arena; `None` for a
/// projection with no near and far plane.
///
/// Row-vector convention, so MSL's column-major read multiplies a clip
/// position the way `replay::reproject_bytes` documents.
pub fn inverse_projection_bytes(enc: &mut FrameEncoder, proj: [f32; 4]) -> Option<BytesRef> {
    camera_near_far(proj)?;
    let inverse = reproject_for(&IDENTITY, &projection(proj));
    Some(reproject_bytes(enc, &inverse))
}

/// The vertex ids `d` can reach, `0..n`, which its capture's buffers hold:
/// a non-indexed draw's first vertex plus its count, an indexed draw's
/// longest per-vertex stream (its base vertex is in the ids, and an index
/// past every stream reads nothing). The capture stores no id at or past
/// `n`, so an index past the streams (D3D9 forbids it; the encoder cannot
/// read a GPU-only index buffer to tell) writes nothing out of bounds.
///
/// # Errors
///
/// The [`Skip`] that leaves the draw out: instanced, not triangles, inline
/// indices, a non-indexed range past its longest stream, or no range within
/// [`MAX_CAPTURED_VERTICES`].
pub fn capture_extent(d: &CapturedDraw) -> Result<u32, Skip> {
    if d.instances != 1 {
        return Err(Skip::Instanced);
    }
    if !matches!(
        d.primitive,
        PrimitiveType::Triangle | PrimitiveType::TriangleStrip
    ) {
        return Err(Skip::Primitive);
    }
    let count = match d.index {
        CapturedIndex::None {
            start_vertex,
            vertex_count,
        } => start_vertex
            .checked_add(vertex_count)
            .filter(|&end| stream_extent(d).is_none_or(|streams| end <= streams)),
        CapturedIndex::Buffer { .. } => stream_extent(d),
        CapturedIndex::Inline { .. } => return Err(Skip::Inline),
    };
    count
        .filter(|&n| n > 0 && n <= MAX_CAPTURED_VERTICES)
        .ok_or(Skip::Extent)
}

/// The most vertices any per-vertex stream of `d` holds past its bind offset.
fn stream_extent(d: &CapturedDraw) -> Option<u32> {
    d.vertex_binds()
        .iter()
        .filter_map(|bind| {
            let (slot, len, offset) = match *bind {
                VertexBind::Buffer {
                    handle,
                    offset,
                    slot,
                    ..
                } => (slot, buffer_length(handle)?, u64::from(offset)),
                VertexBind::Inline { bytes, slot } => (slot, u64::from(bytes.len), 0),
                VertexBind::Zeros { .. } => return None,
            };
            let stride = d
                .snapshot
                .stream_layouts
                .get(slot as usize)
                .filter(|l| l.is_used() && l.step == VertexStepFunction::PerVertex && l.stride > 0)?
                .stride;
            u32::try_from(len.checked_sub(offset)?.div_ceil(u64::from(stride))).ok()
        })
        .max()
}

/// The length of the live buffer `raw`.
fn buffer_length(raw: u64) -> Option<u64> {
    handle::<MTLBufferKind>(raw)
        .into_retained()
        .map(|buffer| buffer.length() as u64)
}

/// Encode the capture of `d` into `out`: its own vertex shader keyed for
/// the capture ([`VsScene::Capture`]) runs over its vertices with
/// rasterisation off and stores, per vertex id, the eye position and the
/// stage-0 coordinate into `target`.
///
/// Binds every buffer and constant the draw read, a neutral `pos_fixup` (no
/// half-pixel offset, no depth bias; the shader keys no clip planes), so
/// the shader's final position is the camera clip position, `p_inv` at
/// `VS_REPROJECT_SLOT` and `target.vertex_count` at `VS_CAPTURE_COUNT_SLOT`,
/// past which nothing is stored. `out` must go into a pass with no attachment
/// (`Pass::scene_vertex`). Nothing is encoded unless the answer is
/// `Ready`: a shader or pipeline still building answers `Pending`, one that
/// failed `Failed`.
pub fn capture_draw(
    enc: &mut FrameEncoder,
    out: &mut Vec<Command>,
    d: &CapturedDraw,
    target: &CaptureTarget,
) -> Resolution<CapturedGeometry> {
    let vs = match enc.scene_vs(d.vs.as_ref(), VsScene::Capture) {
        Resolution::Ready(vs) => vs,
        Resolution::Pending(ticket) => return Resolution::Pending(ticket),
        Resolution::Failed => return Resolution::Failed,
    };
    let pipeline = match enc.capture_pipeline(&d.snapshot, d.vertex_attrs(), vs) {
        Resolution::Ready(pipeline) => pipeline,
        Resolution::Pending(ticket) => return Resolution::Pending(ticket),
        Resolution::Failed => return Resolution::Failed,
    };
    let (Ok(pos_offset), Ok(uv_offset)) = (
        u32::try_from(target.pos_offset),
        u32::try_from(target.uv_offset),
    ) else {
        return Resolution::Failed;
    };
    let count = BytesRef {
        ptr: enc.alloc_scratch(&target.vertex_count.to_le_bytes()),
        len: 4,
    };
    out.push(Command::set_vertex_buffer(
        target.pos_buf,
        pos_offset,
        VS_CAPTURE_POS_SLOT,
    ));
    out.push(Command::set_vertex_buffer(
        target.uv_buf,
        uv_offset,
        VS_CAPTURE_UV_SLOT,
    ));
    encode_captured_draw(
        out,
        d,
        &ReplayBinding {
            pipeline: pipeline.raw(),
            depth_stencil: 0,
            cull: CullMode::None,
            depth_bias: (0.0, 0.0),
            pos_fixup: neutral_pos_fixup(),
            extra_vs: &[
                (VS_REPROJECT_SLOT, target.p_inv),
                (VS_CAPTURE_COUNT_SLOT, count),
            ],
        },
    );
    Resolution::Ready(CapturedGeometry {
        pos_buf: target.pos_buf,
        pos_offset: target.pos_offset,
        uv_buf: target.uv_buf,
        uv_offset: target.uv_offset,
        vertex_count: target.vertex_count,
        ib: d.index,
    })
}

/// The triangles `count` vertices or indices make: a list's thirds, a strip's count less two.
const fn triangle_count(strip: bool, count: u32) -> u32 {
    if strip {
        count.saturating_sub(2)
    } else {
        count / 3
    }
}

/// The BLAS geometry of `d` from its captured eye positions `g`: the draw's
/// own index source, its base (or first) vertex folded into the vertex
/// offset, since the vertex ids count from it.
fn captured_source(d: &CapturedDraw, g: &CapturedGeometry) -> Result<BlasSource, Skip> {
    let strip = match d.primitive {
        PrimitiveType::Triangle => false,
        PrimitiveType::TriangleStrip => true,
        _ => return Err(Skip::Primitive),
    };
    let (first, index, count) = match g.ib {
        CapturedIndex::None {
            start_vertex,
            vertex_count,
        } => (i64::from(start_vertex), SourceIndex::None, vertex_count),
        CapturedIndex::Buffer {
            handle,
            offset,
            count,
            index_type,
            base_vertex,
            ..
        } => (
            i64::from(base_vertex),
            SourceIndex::Buffer {
                handle,
                offset,
                index_type,
            },
            count,
        ),
        CapturedIndex::Inline { .. } => return Err(Skip::Inline),
    };
    let triangles = triangle_count(strip, count);
    if triangles == 0 {
        return Err(Skip::Primitive);
    }
    let vertex_offset = i64::try_from(g.pos_offset)
        .ok()
        .and_then(|offset| offset.checked_add(first * i64::from(CAPTURED_POSITION_BYTES)))
        .and_then(|offset| u64::try_from(offset).ok())
        .ok_or(Skip::Offset)?;
    Ok(BlasSource {
        key: BlasKey::default(),
        vertex_buffer: g.pos_buf,
        vertex_offset,
        stride: CAPTURED_POSITION_BYTES,
        format: VertexFormat::Float3,
        index,
        strip,
        triangles,
    })
}

/// How far a cached BLAS is in its compaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compaction {
    /// Built in submission `built`; its compacted size lands in query slot `query`.
    Pending {
        built: u64,
        query: Option<u32>,
    },
    /// Its compacted copy is made in submission `seq`.
    Compacting {
        seq: u64,
    },
    Done,
}

#[derive(Clone, Copy, Debug)]
struct BlasEntry {
    blas: u64,
    size: u64,
    /// The submission that last instanced it.
    last_used: u64,
    compaction: Compaction,
}

impl BlasEntry {
    /// The size-query slot it holds, if any.
    const fn query(&self) -> Option<u32> {
        match self.compaction {
            Compaction::Pending { query, .. } => query,
            _ => None,
        }
    }
}

/// The cached BLASes: keyed on their geometry, least recently used first out, within a budget.
pub struct BlasCache {
    entries: FxHashMap<BlasKey, BlasEntry>,
    bytes: u64,
    budget: u64,
    /// Size-query slots of entries that left, for the scene to reuse.
    freed_queries: Vec<u32>,
}

impl Default for BlasCache {
    fn default() -> Self {
        Self::new(RT_BUDGET_BYTES)
    }
}

impl BlasCache {
    #[must_use]
    pub fn new(budget: u64) -> Self {
        Self {
            entries: FxHashMap::default(),
            bytes: 0,
            budget,
            freed_queries: Vec::new(),
        }
    }

    /// The bytes of every cached BLAS.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The BLAS of `key`, marked used in submission `frame`.
    pub fn get(&mut self, key: &BlasKey, frame: u64) -> Option<u64> {
        let entry = self.entries.get_mut(key)?;
        entry.last_used = frame;
        Some(entry.blas)
    }

    /// Cache `blas` of `size` bytes under `key`, used in submission `frame`, fully compacted.
    pub fn insert(&mut self, key: BlasKey, blas: u64, size: u64, frame: u64) {
        let entry = BlasEntry {
            blas,
            size,
            last_used: frame,
            compaction: Compaction::Done,
        };
        self.bytes += size;
        if let Some(old) = self.entries.insert(key, entry) {
            self.bytes -= old.size;
            self.freed_queries.extend(old.query());
        }
    }

    /// Mark `key`'s BLAS as built in submission `built`, its compacted size bound for `query`.
    fn await_compaction(&mut self, key: &BlasKey, built: u64, query: Option<u32>) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.compaction = Compaction::Pending { built, query };
        }
    }

    /// Room for `need` more bytes: the least recently used BLASes not used
    /// in submission `frame` go until it fits, and their handles come back
    /// for retirement. `None`, evicting nothing, when even all of them would
    /// not make room.
    pub fn make_room(&mut self, need: u64, frame: u64) -> Option<Vec<u64>> {
        let total = self.bytes.saturating_add(need);
        if total <= self.budget {
            return Some(Vec::new());
        }
        let deficit = total - self.budget;
        let mut candidates: Vec<(u64, BlasKey, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.last_used < frame)
            .map(|(key, e)| (e.last_used, *key, e.size))
            .collect();
        if candidates.iter().map(|c| c.2).sum::<u64>() < deficit {
            return None;
        }
        candidates.sort_unstable_by_key(|c| c.0);
        let mut freed = 0;
        let mut evicted = Vec::new();
        for (_, key, size) in candidates {
            if freed >= deficit {
                break;
            }
            evicted.extend(self.remove(&key));
            freed += size;
        }
        Some(evicted)
    }

    /// Take `key` out, its handle back for retirement.
    fn remove(&mut self, key: &BlasKey) -> Option<u64> {
        let entry = self.entries.remove(key)?;
        self.bytes -= entry.size;
        self.freed_queries.extend(entry.query());
        Some(entry.blas)
    }

    /// Take out every BLAS whose key `stale` names, their handles back for retirement.
    fn retire_where(&mut self, mut stale: impl FnMut(&BlasKey) -> bool) -> Vec<u64> {
        let keys: Vec<BlasKey> = self.entries.keys().filter(|k| stale(k)).copied().collect();
        keys.iter().filter_map(|key| self.remove(key)).collect()
    }
}

/// One instance of the frame's TLAS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtInstance {
    /// The BLAS it instances.
    pub blas: u64,
    /// Object-to-eye rows: the draw's world-view, row-major; the identity
    /// for a captured draw, whose positions are in eye space.
    pub transform: [[f32; 4]; 3],
    /// [`MASK_OPAQUE`] or [`MASK_ALPHA`], with [`MASK_FIGURE`] for a figure
    /// or vertex blending and [`MASK_LEAF`] for a `SpeedTree` leaf.
    pub mask: u8,
    /// Its index in [`TlasHandle::instances`], which a ray query reports.
    pub user_id: u32,
    /// A captured draw's eye positions and coordinates, `None` for a static one.
    pub geometry: Option<CapturedGeometry>,
}

/// A frame's TLAS: what a ray-traced kernel binds, and what its hits name.
#[derive(Clone, Debug, PartialEq)]
pub struct TlasHandle {
    /// The instance acceleration structure, a raw canonical handle.
    pub tlas: u64,
    pub instances: Vec<RtInstance>,
    /// The captured draw of each instance, by user id.
    pub draws: Vec<u32>,
    /// Every BLAS the TLAS instances, which a kernel tracing it makes resident.
    pub resident: Vec<u64>,
}

/// One buffer or acceleration structure of a frame slot, grown on demand.
#[derive(Clone, Copy, Default)]
struct Grown {
    handle: u64,
    len: u64,
}

/// What one frame's builds use; reused once the GPU is past the frame that last did.
#[derive(Default)]
struct FrameSlot {
    /// The submission that last used it, 0 never.
    last_seq: u64,
    tlas: Grown,
    instances: Grown,
    scratch: Grown,
    lists: Grown,
    /// The frame's captured eye positions and coordinates, Shared.
    captured_pos: Grown,
    captured_uv: Grown,
    /// The frame's BLASes of captured geometry, the frame's `n`th build in the `n`th.
    captured_blas: Vec<Grown>,
}

impl FrameSlot {
    fn bytes(&self) -> u64 {
        self.tlas.len
            + self.instances.len
            + self.scratch.len
            + self.lists.len
            + self.captured_pos.len
            + self.captured_uv.len
            + self.captured_blas.iter().map(|g| g.len).sum::<u64>()
    }
}

/// A BLAS this frame builds, its scratch and list placed in the frame slot's buffers.
struct PendingBuild {
    key: BlasKey,
    blas: u64,
    build: BlasBuild,
    /// Built from captured geometry into the frame slot, not cached.
    captured: bool,
}

/// Where a new BLAS lives.
#[derive(Clone, Copy)]
enum BlasHome {
    /// The cache, under the source's key, compacted once built.
    Cache,
    /// The frame slot's `index`th captured BLAS, built anew every frame.
    Frame { slot: usize, index: usize },
}

/// How far this frame's builds have filled its slot: the submission, the
/// scratch and list bytes placed so far, the captured BLASes made.
struct FrameBuilds {
    seq: u64,
    scratch: u64,
    lists: u64,
    captured: usize,
}

/// What one frame of the scene did, for the tests and the logs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// BLASes built for the cache.
    pub built: u32,
    pub reused: u32,
    pub compacted: u32,
    pub skipped: u32,
    /// Left to a later frame: the frame's build or scratch cap, or the budget.
    pub deferred: u32,
    /// BLASes built from captured geometry, every frame anew.
    pub captured: u32,
    /// Captures whose vertex shader or pipeline is still building; a later frame captures them.
    pub waiting: u32,
}

/// The encoder's ray-traced scene: the BLAS cache, the frames' buffers, the compaction queries.
#[derive(Default)]
pub struct RtScene {
    cache: BlasCache,
    slots: Vec<FrameSlot>,
    /// The Shared buffer compacted sizes land in, one `u32` per slot.
    queries: u64,
    free_queries: Vec<u32>,
    strip_pipeline: MetalHandle<MTLComputePipelineStateKind>,
    stats: FrameStats,
    /// This submission's TLAS, which the ray-traced kernels read.
    frame: Option<TlasHandle>,
}

impl RtScene {
    /// Hand every object the scene owns to the caller's destroys, and release
    /// its pipeline: the encoder's teardown, past its GPU-idle wait.
    pub fn take_all(&mut self, buffers: &mut Vec<u64>, accels: &mut Vec<u64>) {
        accels.extend(self.cache.entries.drain().map(|(_, e)| e.blas));
        self.cache.bytes = 0;
        for slot in self.slots.drain(..) {
            accels.extend(
                core::iter::once(slot.tlas)
                    .chain(slot.captured_blas)
                    .map(|g| g.handle)
                    .filter(|&h| h != 0),
            );
            buffers.extend(
                [
                    slot.instances,
                    slot.scratch,
                    slot.lists,
                    slot.captured_pos,
                    slot.captured_uv,
                ]
                .into_iter()
                .map(|g| g.handle)
                .filter(|&h| h != 0),
            );
        }
        if self.queries != 0 {
            buffers.push(core::mem::take(&mut self.queries));
        }
        self.free_queries.clear();
        if !self.strip_pipeline.is_null() {
            crate::metal::post::destroy_pipeline(self.strip_pipeline.raw());
            self.strip_pipeline = MetalHandle::NULL;
        }
        self.frame = None;
    }

    fn slot_bytes(&self) -> u64 {
        self.slots.iter().map(FrameSlot::bytes).sum()
    }

    /// Count a draw left out, said once per cause.
    fn skip(&mut self, skip: Skip) {
        self.stats.skipped += 1;
        say_skip(skip);
    }
}

impl FrameEncoder {
    /// This submission's TLAS, `None` when it built none.
    #[must_use]
    pub fn rt_frame_tlas(&self) -> Option<&TlasHandle> {
        self.rt.as_ref()?.frame.as_ref()
    }

    /// What this submission's scene did.
    #[cfg(test)]
    #[must_use]
    pub fn rt_stats(&self) -> FrameStats {
        self.rt
            .as_ref()
            .map_or_else(FrameStats::default, |rt| rt.stats)
    }

    /// Record this submission's TLAS for the ray-traced kernels.
    pub(super) fn set_rt_frame_tlas(&mut self, tlas: Option<TlasHandle>) {
        if let Some(rt) = self.rt.as_mut() {
            rt.frame = tlas;
        }
    }

    /// Destroy acceleration structure `raw` once the GPU is past this submission.
    fn retire_acceleration_structure(&mut self, raw: u64) {
        self.retire_rt_object(DestroyKind::AccelerationStructure, raw);
    }

    fn retire_rt_object(&mut self, kind: DestroyKind, raw: u64) {
        if raw == 0 {
            return;
        }
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind,
                handle: raw,
                page_box: None,
                staging_arc: None,
                seq: self.current_submit_seq,
                from_texture: false,
            });
    }
}

/// Build this submission's TLAS over `draws`, queuing its commands on
/// `out` and the captures its builds read on `capture`.
///
/// The commands, in order: the compactions this frame starts, the new
/// BLAS builds (each strip's rewrite first), the TLAS build. `capture`
/// holds the commands of a vertex-only pass (`Pass::scene_vertex`) that has
/// to run before them: one [`capture_draw`] per captured draw instanced.
/// `None`, with nothing queued but compactions and nothing captured, when
/// ray tracing is off, no draw can be instanced or the frame's objects
/// cannot be made. Either way the cache retires every BLAS whose contents
/// its buffer no longer holds.
pub fn build_frame_tlas(
    enc: &mut FrameEncoder,
    draws: &[CapturedDraw],
    out: &mut Vec<BlitCommand>,
    capture: &mut Vec<Command>,
) -> Option<TlasHandle> {
    let mut rt = enc.rt.take()?;
    let tlas = rt.build(enc, draws, (out, capture));
    for raw in rt.cache.retire_where(|key| {
        !enc.buffer_content_current(key.vertex_content())
            || key
                .index_content()
                .is_some_and(|content| !enc.buffer_content_current(content))
    }) {
        enc.retire_acceleration_structure(raw);
    }
    rt.free_queries.append(&mut rt.cache.freed_queries);
    enc.perf
        .add_rt_blas(rt.stats.built, rt.stats.captured, rt.stats.deferred);
    enc.rt = Some(rt);
    tlas
}

/// A draw's way into the frame's TLAS: its BLAS, transform and captured geometry.
struct Instanced {
    blas: u64,
    rows: [[f32; 4]; 3],
    geometry: Option<CapturedGeometry>,
}

impl RtScene {
    fn build(
        &mut self,
        enc: &mut FrameEncoder,
        draws: &[CapturedDraw],
        (out, capture): (&mut Vec<BlitCommand>, &mut Vec<Command>),
    ) -> Option<TlasHandle> {
        self.stats = FrameStats::default();
        let device = enc.device.clone();
        let seq = enc.current_submit_seq;
        let (settled, failed) = enc.upload_gate().unwrap_or((0, 0));
        self.settle_compactions(enc, &device, (settled, failed), out);
        let Some(slot) = self.free_slot(settled) else {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "rt: every frame's acceleration structures are still in flight; a frame goes \
                 without");
            return None;
        };
        let targets = self.plan_captures(enc, &device, slot, draws);

        let mut instances = Vec::with_capacity(draws.len());
        let mut instance_draws = Vec::with_capacity(draws.len());
        let mut builds: Vec<PendingBuild> = Vec::new();
        let mut frame = FrameBuilds {
            seq,
            scratch: 0,
            lists: 0,
            captured: 0,
        };
        let mut captures = Vec::new();
        for ((index, d), target) in (0u32..).zip(draws).zip(&targets) {
            let instanced = match (d.class, target) {
                (DrawClass::Ff, _) => {
                    self.instance_static(enc, &device, d, &mut frame, &mut builds)
                }
                (_, Some(target)) => match target {
                    Ok(target) => self.instance_captured(
                        enc,
                        &device,
                        (d, target, slot),
                        (&mut frame, &mut builds, &mut captures),
                    ),
                    Err(skip) => {
                        self.skip(*skip);
                        None
                    }
                },
                (_, None) => {
                    self.skip(Skip::Class);
                    None
                }
            };
            let Some(instanced) = instanced else {
                continue;
            };
            let base = if d.alpha.is_some() {
                MASK_ALPHA
            } else {
                MASK_OPAQUE
            };
            let class = match d.class {
                DrawClass::Figure | DrawClass::FfBlend => MASK_FIGURE,
                DrawClass::SpeedTree => MASK_LEAF,
                DrawClass::Ff | DrawClass::OtherProgrammable => 0,
            };
            instances.push(RtInstance {
                blas: instanced.blas,
                transform: instanced.rows,
                mask: base | class,
                user_id: u32::try_from(instances.len()).expect("instances fit u32"),
                geometry: instanced.geometry,
            });
            instance_draws.push(index);
        }
        if self.stats.deferred > 0 {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "rt: {} draws wait for a later frame's BLAS builds (per-frame cap or budget)",
                self.stats.deferred);
        }
        if self.stats.waiting > 0 {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "rt: {} captures wait while their vertex shaders or pipelines build; later \
                 frames trace them",
                self.stats.waiting);
        }
        if instances.is_empty() {
            return None;
        }
        let tlas = self.encode_frame(
            enc,
            &device,
            slot,
            (instances, instance_draws),
            (&builds, frame.scratch, frame.lists),
            out,
        );
        if tlas.is_none() {
            // The builds were never queued: their BLASes hold nothing.
            for build in builds.iter().filter(|b| !b.captured) {
                if let Some(raw) = self.cache.remove(&build.key) {
                    enc.retire_acceleration_structure(raw);
                }
            }
            self.stats.built = 0;
            self.stats.captured = 0;
        } else {
            capture.append(&mut captures);
        }
        tlas
    }

    /// Where this frame's captures store their vertices: one entry per draw,
    /// `None` for a draw that is not captured, the buffers of frame slot
    /// `slot` grown to hold them all.
    fn plan_captures(
        &mut self,
        enc: &mut FrameEncoder,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        draws: &[CapturedDraw],
    ) -> Vec<Option<Result<CaptureTarget, Skip>>> {
        let mut plan: Vec<Option<Result<CaptureTarget, Skip>>> = vec![None; draws.len()];
        if !draws.iter().any(|d| d.class.captured()) {
            return plan;
        }
        let p_inv = enc
            .scene
            .boundary()
            .map(|boundary| boundary.proj)
            .and_then(|proj| inverse_projection_bytes(enc, proj));
        let (mut pos_bytes, mut uv_bytes) = (0u64, 0u64);
        for (entry, d) in plan.iter_mut().zip(draws) {
            if !d.class.captured() {
                continue;
            }
            let Some(p_inv) = p_inv else {
                *entry = Some(Err(Skip::Projection));
                continue;
            };
            *entry = Some(capture_extent(d).map(|vertex_count| {
                let target = CaptureTarget {
                    pos_buf: 0,
                    pos_offset: pos_bytes,
                    uv_buf: 0,
                    uv_offset: uv_bytes,
                    vertex_count,
                    p_inv,
                };
                pos_bytes += (u64::from(vertex_count) * u64::from(CAPTURED_POSITION_BYTES))
                    .next_multiple_of(SUBALLOC_ALIGN);
                uv_bytes += (u64::from(vertex_count) * u64::from(CAPTURED_UV_BYTES))
                    .next_multiple_of(SUBALLOC_ALIGN);
                target
            }));
        }
        if pos_bytes == 0 {
            return plan;
        }
        let frame = &mut self.slots[slot];
        let mut retired = Vec::new();
        let grown = grow(
            device,
            &mut frame.captured_pos,
            pos_bytes,
            true,
            "mtld3d-rt-captured-positions",
            &mut retired,
        ) && grow(
            device,
            &mut frame.captured_uv,
            uv_bytes,
            true,
            "mtld3d-rt-captured-uvs",
            &mut retired,
        );
        let (pos_buf, uv_buf) = (frame.captured_pos.handle, frame.captured_uv.handle);
        for (kind, raw) in retired {
            enc.retire_rt_object(kind, raw);
        }
        for target in plan.iter_mut().flatten() {
            if let Ok(t) = target {
                if grown {
                    t.pos_buf = pos_buf;
                    t.uv_buf = uv_buf;
                } else {
                    *target = Err(Skip::Capture);
                }
            }
        }
        plan
    }

    /// A static draw's cached BLAS, built now when the cache has none.
    fn instance_static(
        &mut self,
        enc: &mut FrameEncoder,
        device: &ProtocolObject<dyn MTLDevice>,
        d: &CapturedDraw,
        frame: &mut FrameBuilds,
        builds: &mut Vec<PendingBuild>,
    ) -> Option<Instanced> {
        let source = match blas_source(d) {
            Ok(source) => source,
            Err(skip) => {
                self.skip(skip);
                return None;
            }
        };
        let Some(rows) = world_view_rows(d) else {
            self.skip(Skip::Transform);
            return None;
        };
        let blas = if let Some(blas) = self.cache.get(&source.key, frame.seq) {
            self.stats.reused += 1;
            blas
        } else {
            let build = self.new_blas(enc, device, &source, BlasHome::Cache, frame, builds)?;
            let blas = build.blas;
            builds.push(build);
            blas
        };
        Some(Instanced {
            blas,
            rows,
            geometry: None,
        })
    }

    /// A captured draw's BLAS of this frame, its capture queued on `captures`.
    fn instance_captured(
        &mut self,
        enc: &mut FrameEncoder,
        device: &ProtocolObject<dyn MTLDevice>,
        (d, target, slot): (&CapturedDraw, &CaptureTarget, usize),
        (frame, builds, captures): (&mut FrameBuilds, &mut Vec<PendingBuild>, &mut Vec<Command>),
    ) -> Option<Instanced> {
        let mut commands = Vec::new();
        let geometry = match capture_draw(enc, &mut commands, d, target) {
            Resolution::Ready(geometry) => geometry,
            Resolution::Pending(_) => {
                self.stats.waiting += 1;
                return None;
            }
            Resolution::Failed => {
                self.skip(Skip::Capture);
                return None;
            }
        };
        let source = match captured_source(d, &geometry) {
            Ok(source) => source,
            Err(skip) => {
                self.skip(skip);
                return None;
            }
        };
        let home = BlasHome::Frame {
            slot,
            index: frame.captured,
        };
        let build = self.new_blas(enc, device, &source, home, frame, builds)?;
        frame.captured += 1;
        let blas = build.blas;
        builds.push(build);
        captures.append(&mut commands);
        Some(Instanced {
            blas,
            rows: IDENTITY_ROWS,
            geometry: Some(geometry),
        })
    }

    /// The first slot the GPU is done with, made when fewer than [`MAX_FRAME_SLOTS`] exist.
    fn free_slot(&mut self, settled: u64) -> Option<usize> {
        if let Some(i) = self.slots.iter().position(|s| s.last_seq <= settled) {
            return Some(i);
        }
        if self.slots.len() < MAX_FRAME_SLOTS {
            self.slots.push(FrameSlot::default());
            return Some(self.slots.len() - 1);
        }
        None
    }

    /// Allocate a BLAS for `source` in `home`, placing its scratch and list
    /// after this frame's earlier builds; `None` (counted) when the frame's
    /// caps or the budget leave no room, or an object cannot be made.
    ///
    /// A cached BLAS is new, cached under the source's key and compacted
    /// once built; a captured one reuses the frame slot's acceleration
    /// structure of its index, grown to fit, and is neither cached nor
    /// compacted.
    fn new_blas(
        &mut self,
        enc: &mut FrameEncoder,
        device: &ProtocolObject<dyn MTLDevice>,
        source: &BlasSource,
        home: BlasHome,
        frame: &mut FrameBuilds,
        builds: &[PendingBuild],
    ) -> Option<PendingBuild> {
        let seq = frame.seq;
        let list_bytes = if source.strip {
            u64::from(source.triangles) * 12
        } else {
            0
        };
        let strip_pipeline = if source.strip {
            if self.strip_pipeline.is_null() {
                self.strip_pipeline = create_strip_pipeline(device).unwrap_or(MetalHandle::NULL);
            }
            if self.strip_pipeline.is_null() {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                    "rt: the strip rewrite could not be built; strips are left out");
                self.stats.skipped += 1;
                return None;
            }
            self.strip_pipeline.raw()
        } else {
            0
        };
        // A strip's sizes come from its list; any live buffer stands in for it.
        let mut geometry = BlasGeometry {
            vertex_buffer: source.vertex_buffer,
            vertex_offset: source.vertex_offset,
            vertex_stride: source.stride,
            vertex_format: source.format as u32,
            index_buffer: 0,
            index_offset: 0,
            index_type: IndexType::UInt32 as u32,
            triangle_count: source.triangles,
        };
        match source.index {
            _ if source.strip => {
                geometry.index_buffer = source.vertex_buffer;
            }
            SourceIndex::Buffer {
                handle,
                offset,
                index_type,
            } => {
                geometry.index_buffer = handle;
                geometry.index_offset = u64::from(offset);
                geometry.index_type = index_type as u32;
            }
            SourceIndex::None => {}
        }
        let sizes = blas_sizes(device, &geometry)?;
        let size = sizes.accelerationStructureSize as u64;
        let build_scratch = (sizes.buildScratchBufferSize as u64).next_multiple_of(SUBALLOC_ALIGN);
        if builds.len() >= MAX_BUILDS_PER_FRAME || frame.scratch + build_scratch > MAX_FRAME_SCRATCH
        {
            self.stats.deferred += 1;
            return None;
        }
        // A captured BLAS grows its slot's acceleration structure, already
        // counted among the slots' bytes; a cached one is new.
        let own = match home {
            BlasHome::Cache => size,
            BlasHome::Frame { slot, index } => {
                let pooled = self.slots[slot]
                    .captured_blas
                    .get(index)
                    .map_or(0, |g| g.len);
                size.saturating_sub(pooled)
            }
        };
        let others = self.slot_bytes() + frame.scratch + build_scratch + frame.lists + list_bytes;
        let Some(evicted) = self.cache.make_room(own + others, seq) else {
            self.stats.deferred += 1;
            return None;
        };
        for raw in evicted {
            enc.retire_acceleration_structure(raw);
        }
        let blas = match home {
            BlasHome::Cache => new_acceleration_structure(device, size, "mtld3d-rt-blas")?.raw(),
            BlasHome::Frame { slot, index } => {
                let pool = &mut self.slots[slot].captured_blas;
                if pool.len() <= index {
                    pool.resize(index + 1, Grown::default());
                }
                let mut retired = Vec::new();
                let grown = grow_accel(
                    device,
                    &mut pool[index],
                    size,
                    "mtld3d-rt-blas-captured",
                    &mut retired,
                );
                let blas = pool[index].handle;
                for (kind, raw) in retired {
                    enc.retire_rt_object(kind, raw);
                }
                if !grown {
                    return None;
                }
                blas
            }
        };
        let strip = if source.strip {
            let (source_buffer, source_offset, source_type) = match source.index {
                SourceIndex::Buffer {
                    handle,
                    offset,
                    index_type,
                } => (handle, u64::from(offset), index_type as u32),
                SourceIndex::None => (0, 0, 0),
            };
            let dest_offset = frame.lists;
            frame.lists += list_bytes.next_multiple_of(SUBALLOC_ALIGN);
            geometry.index_buffer = 0;
            geometry.index_offset = dest_offset;
            geometry.index_type = IndexType::UInt32 as u32;
            StripConversion {
                pipeline: strip_pipeline,
                source: source_buffer,
                source_offset,
                source_type,
                triangle_count: source.triangles,
                dest: 0,
                dest_offset,
            }
        } else {
            StripConversion::default()
        };
        let scratch_offset = frame.scratch;
        frame.scratch += build_scratch;
        let query = match home {
            BlasHome::Cache => {
                let query = self
                    .free_queries
                    .pop()
                    .or_else(|| self.first_queries(device));
                self.cache.insert(source.key, blas, size, seq);
                self.cache.await_compaction(&source.key, seq, query);
                self.stats.built += 1;
                query
            }
            BlasHome::Frame { .. } => {
                self.stats.captured += 1;
                None
            }
        };
        Some(PendingBuild {
            key: source.key,
            blas,
            build: BlasBuild {
                geometry,
                strip,
                scratch: 0,
                scratch_offset,
                size_query: 0,
                size_query_offset: query.map_or(0, |q| u64::from(q) * 4),
            },
            captured: matches!(home, BlasHome::Frame { .. }),
        })
    }

    /// Make the size-query buffer on first use and hand out its first slot.
    fn first_queries(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Option<u32> {
        if self.queries != 0 {
            return None;
        }
        self.queries = new_buffer(
            device,
            u64::from(SIZE_QUERIES) * 4,
            true,
            "mtld3d-rt-compacted-sizes",
        )?
        .raw();
        self.free_queries = (1..SIZE_QUERIES).rev().collect();
        Some(0)
    }

    /// Start or finish the compactions whose builds the GPU is done with.
    ///
    /// A BLAS whose build or compaction ran in a failed submission holds
    /// nothing: it leaves the cache, so a later frame builds it again.
    fn settle_compactions(
        &mut self,
        enc: &mut FrameEncoder,
        device: &ProtocolObject<dyn MTLDevice>,
        (settled, failed): (u64, u64),
        out: &mut Vec<BlitCommand>,
    ) {
        let seq = enc.current_submit_seq;
        let sizes = self
            .queries
            .ne(&0)
            .then(|| handle::<MTLBufferKind>(self.queries).into_retained())
            .flatten();
        let mut lost = Vec::new();
        for (key, entry) in &mut self.cache.entries {
            match entry.compaction {
                Compaction::Pending { built, query } if built <= settled => {
                    if failed >= built {
                        lost.push(*key);
                        continue;
                    }
                    entry.compaction = Compaction::Done;
                    let Some(query) = query else {
                        continue;
                    };
                    self.free_queries.push(query);
                    let compacted = sizes.as_ref().map_or(0, |b| read_query(b, query));
                    if compacted == 0 || compacted >= entry.size {
                        continue;
                    }
                    let Some(copy) =
                        new_acceleration_structure(device, compacted, "mtld3d-rt-blas-compact")
                    else {
                        continue;
                    };
                    out.push(BlitCommand::compact_blas(entry.blas, copy.raw()));
                    enc.retire_acceleration_structure(entry.blas);
                    self.cache.bytes -= entry.size - compacted;
                    entry.blas = copy.raw();
                    entry.size = compacted;
                    entry.compaction = Compaction::Compacting { seq };
                    self.stats.compacted += 1;
                }
                Compaction::Compacting { seq: copied } if copied <= settled => {
                    if failed >= copied {
                        lost.push(*key);
                    } else {
                        entry.compaction = Compaction::Done;
                    }
                }
                _ => {}
            }
        }
        for key in lost {
            if let Some(raw) = self.cache.remove(&key) {
                enc.retire_acceleration_structure(raw);
            }
        }
    }

    /// Size the frame slot's objects, write its instances and queue every build.
    fn encode_frame(
        &mut self,
        enc: &mut FrameEncoder,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        (instances, draws): (Vec<RtInstance>, Vec<u32>),
        (builds, scratch, lists): (&[PendingBuild], u64, u64),
        out: &mut Vec<BlitCommand>,
    ) -> Option<TlasHandle> {
        let mut resident: Vec<u64> = Vec::new();
        let mut index_of: FxHashMap<u64, u32> = FxHashMap::default();
        let descriptors: Vec<InstanceDescriptor> = instances
            .iter()
            .map(|instance| {
                let index = *index_of.entry(instance.blas).or_insert_with(|| {
                    resident.push(instance.blas);
                    u32::try_from(resident.len() - 1).expect("BLASes fit u32")
                });
                instance_descriptor(
                    &instance.transform,
                    index,
                    instance.mask,
                    instance.user_id,
                    instance.mask & MASK_ALPHA == 0,
                )
            })
            .collect();
        let count = u32::try_from(descriptors.len()).expect("instances fit u32");
        let sizes = tlas_sizes(device, count, &resident)?;
        let tlas_scratch = sizes.buildScratchBufferSize as u64;
        let instance_bytes = size_of_val(descriptors.as_slice()) as u64;
        let seq = enc.current_submit_seq;
        let frame = &mut self.slots[slot];
        let mut retired = Vec::new();
        let grown = grow(
            device,
            &mut frame.scratch,
            scratch + tlas_scratch,
            false,
            "mtld3d-rt-scratch",
            &mut retired,
        ) && grow(
            device,
            &mut frame.instances,
            instance_bytes,
            true,
            "mtld3d-rt-instances",
            &mut retired,
        ) && (lists == 0
            || grow(
                device,
                &mut frame.lists,
                lists,
                true,
                "mtld3d-rt-lists",
                &mut retired,
            ))
            && grow_accel(
                device,
                &mut frame.tlas,
                sizes.accelerationStructureSize as u64,
                "mtld3d-rt-tlas",
                &mut retired,
            );
        for (kind, raw) in retired {
            enc.retire_rt_object(kind, raw);
        }
        if !grown {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "rt: a frame's acceleration-structure buffers could not be made; it goes without");
            return None;
        }
        frame.last_seq = seq;
        let (frame_scratch, frame_lists, frame_instances, tlas) = (
            frame.scratch.handle,
            frame.lists.handle,
            frame.instances.handle,
            frame.tlas.handle,
        );
        let instance_buffer = handle::<MTLBufferKind>(frame_instances).into_retained()?;
        // SAFETY: a Shared buffer of at least `instance_bytes`, which no
        // command in flight reads (its slot is past the GPU).
        unsafe {
            core::ptr::copy_nonoverlapping(
                descriptors.as_ptr().cast::<u8>(),
                instance_buffer.contents().as_ptr().cast::<u8>(),
                instance_bytes as usize,
            );
        }
        for pending in builds {
            let mut build = pending.build;
            build.scratch = frame_scratch;
            if build.strip.pipeline != 0 {
                build.strip.dest = frame_lists;
                build.geometry.index_buffer = frame_lists;
            }
            build.size_query = if pending.captured {
                0
            } else {
                self.query_buffer_for(&pending.key)
            };
            let (info, _) = enc.scratch.alloc_slice(core::slice::from_ref(&build));
            out.push(BlitCommand::build_blas(info as u64, pending.blas));
        }
        let (blas, _) = enc.scratch.alloc_slice(&resident);
        let info = TlasBuild {
            instances: frame_instances,
            instance_count: count,
            blas_count: u32::try_from(resident.len()).expect("BLASes fit u32"),
            blas: blas as u64,
            scratch: frame_scratch,
            scratch_offset: scratch,
        };
        let (info, _) = enc.scratch.alloc_slice(core::slice::from_ref(&info));
        out.push(BlitCommand::build_tlas(info as u64, tlas));
        Some(TlasHandle {
            tlas,
            instances,
            draws,
            resident,
        })
    }

    /// The size-query buffer when `key`'s BLAS took a query slot, else 0.
    fn query_buffer_for(&self, key: &BlasKey) -> u64 {
        self.cache
            .entries
            .get(key)
            .and_then(BlasEntry::query)
            .map_or(0, |_| self.queries)
    }
}

/// A `MetalHandle` of `raw`, a canonical retain the scene keeps alive.
const fn handle<K>(raw: u64) -> MetalHandle<K> {
    // SAFETY: every caller passes an object the scene owns.
    unsafe { MetalHandle::new(raw) }
}

/// The compacted size the GPU wrote into query slot `query`.
fn read_query(sizes: &Retained<ProtocolObject<dyn MTLBuffer>>, query: u32) -> u64 {
    // SAFETY: a Shared buffer of `SIZE_QUERIES` u32s, slot `query` of which
    // a completed build wrote; nothing in flight writes it.
    let value = unsafe {
        sizes
            .contents()
            .as_ptr()
            .cast::<u32>()
            .add(query as usize)
            .read()
    };
    u64::from(value)
}

/// Make `grown` at least `len` bytes, its old buffer retired; false when it cannot be made.
fn grow(
    device: &ProtocolObject<dyn MTLDevice>,
    grown: &mut Grown,
    len: u64,
    shared: bool,
    label: &str,
    retired: &mut Vec<(DestroyKind, u64)>,
) -> bool {
    if grown.len >= len.max(1) {
        return true;
    }
    let len = len.max(SUBALLOC_ALIGN).next_power_of_two();
    let Some(buffer) = new_buffer(device, len, shared, label) else {
        return false;
    };
    if grown.handle != 0 {
        retired.push((DestroyKind::Buffer, grown.handle));
    }
    *grown = Grown {
        handle: buffer.raw(),
        len,
    };
    true
}

/// [`grow`] for an acceleration structure of a frame slot: its TLAS, a captured BLAS.
fn grow_accel(
    device: &ProtocolObject<dyn MTLDevice>,
    grown: &mut Grown,
    len: u64,
    label: &str,
    retired: &mut Vec<(DestroyKind, u64)>,
) -> bool {
    if grown.len >= len.max(1) {
        return true;
    }
    let len = len.max(SUBALLOC_ALIGN).next_power_of_two();
    let Some(tlas): Option<MetalHandle<MTLAccelerationStructureKind>> =
        new_acceleration_structure(device, len, label)
    else {
        return false;
    };
    if grown.handle != 0 {
        retired.push((DestroyKind::AccelerationStructure, grown.handle));
    }
    *grown = Grown {
        handle: tlas.raw(),
        len,
    };
    true
}

/// Say once why draws are left out of the TLAS.
fn say_skip(skip: Skip) {
    match skip {
        Skip::Class => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: a world draw's programmable vertex shader is neither a figure's nor a \
             SpeedTree leaf's; not traced"),
        Skip::Instanced => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: an instanced or crossing-fetch world draw is not traced"),
        Skip::Position => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: a world draw reads no position from a bound vertex buffer; not traced"),
        Skip::Inline => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: a world draw's vertices or indices are inline (DrawPrimitiveUP, a generated \
             fan); not traced"),
        Skip::Format => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: a world draw's position is not float3/float4 at a 4-byte stride; not traced"),
        Skip::Primitive => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: a world draw draws no triangles; not traced"),
        Skip::Offset => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: a world draw's vertex offset cannot start an acceleration structure; not traced"),
        Skip::Transform => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: a world draw bound no world-view; not traced"),
        Skip::Projection => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: no camera projection reached this submission's boundary; its figures, \
             leaves and vertex-blended draws are not traced"),
        Skip::Extent => mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "rt: a captured world draw's vertex range is unknown or too large; not traced"),
        Skip::Capture => mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "rt: a world draw's capture could not be built; not traced"),
    }
}
