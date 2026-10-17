//! Acceleration structures for the captured world: the BLAS cache and the per-frame instance list.
//!
//! While an `rt.*` key traces rays on a device that can
//! ([`crate::metal::rt::effective_config`]), each submission whose world
//! the scene capture holds whole builds an instance acceleration structure
//! (TLAS) of it in eye space, ahead of the world ([`build_frame_tlas`],
//! called by `scene::assemble`). Its instances are the captured static
//! fixed-function draws: no vertex blending, object-space positions read
//! straight from the bound vertex buffer, each under its own world-view
//! (rows 0..3 of its VS constants). Figures, `SpeedTree` and alpha-tested
//! geometry arrive with later stages.
//!
//! The geometry of a draw is one primitive acceleration structure (BLAS),
//! cached across frames under a [`BlasKey`]: the contents (buffer and
//! content version, `scene_capture::BufferContent`) and the range the draw
//! read, never a Metal handle, so a rename or an in-place upload builds anew
//! and a reused handle never matches. A frame retires every cached BLAS
//! whose contents its buffer no longer holds (an upload, a rename, a
//! release) behind the GPU. Strips become lists through an index rewrite
//! into a list buffer of the frame; fans already are (the shared pattern,
//! its first vertex folded into the vertex offset). A BLAS is compacted once
//! the GPU has finished its build: a size query rides the build, a later
//! frame reads it and copies the BLAS into one that size. The cache and the
//! frames' buffers stay within [`RT_BUDGET_BYTES`], the least recently used
//! BLAS going first; a frame builds at most [`MAX_BUILDS_PER_FRAME`] BLASes
//! with at most [`MAX_FRAME_SCRATCH`] of scratch, leaving the rest to later
//! frames.
//!
//! Draws a BLAS cannot describe are left out and counted, each cause said
//! once: the inline (`DrawPrimitiveUP`, generated fan) vertices or indices
//! of a submission's arena are not copied, a position that is not `float3`
//! or `float4`, an offset Metal cannot build from, an instanced or
//! crossing-fetch draw, a primitive other than triangles.

use mtld3d_shared::{
    BlitCommand, MetalHandle,
    mtl::{DestroyKind, IndexType, PrimitiveType, VertexFormat, VertexStepFunction},
    mtl_handle::{MTLAccelerationStructureKind, MTLBufferKind, MTLComputePipelineStateKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice};
use rustc_hash::FxHashMap;

use super::{
    FrameEncoder, LOG_TARGET, PendingResourceRetention,
    scene_capture::{BufferContent, CapturedDraw, CapturedIndex, DrawClass, VertexBind},
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
mod tests;

/// The fixed-function position attribute (`ff_attr_index_for_semantic`).
const POSITION_ATTR: u32 = 0;
/// Instance mask bits: opaque, alpha-tested, figure, leaf.
pub const MASK_OPAQUE: u8 = 1;
pub const MASK_ALPHA: u8 = 2;
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
    /// Not a static fixed-function draw (vertex blending, a programmable VS).
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
    /// Object-to-eye rows: the draw's world-view, row-major.
    pub transform: [[f32; 4]; 3],
    /// [`MASK_OPAQUE`] or [`MASK_ALPHA`]; figure 4 and leaf 8 arrive with their stages.
    pub mask: u8,
    /// Its index in [`TlasHandle::instances`], which a ray query reports.
    pub user_id: u32,
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
}

impl FrameSlot {
    const fn bytes(&self) -> u64 {
        self.tlas.len + self.instances.len + self.scratch.len + self.lists.len
    }
}

/// A BLAS this frame builds, its scratch and list placed in the frame slot's buffers.
struct PendingBuild {
    key: BlasKey,
    blas: u64,
    build: BlasBuild,
}

/// What one frame of the scene did, for the tests and the logs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub built: u32,
    pub reused: u32,
    pub compacted: u32,
    pub skipped: u32,
    /// Left to a later frame: the frame's build or scratch cap, or the budget.
    pub deferred: u32,
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
            accels.extend(Some(slot.tlas.handle).filter(|&h| h != 0));
            buffers.extend(
                [slot.instances, slot.scratch, slot.lists]
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
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the ray-traced kernels read it")
    )]
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

/// Build this submission's TLAS over `draws`, queuing its commands on `out`.
///
/// The commands, in order: the compactions this frame starts, the new
/// BLAS builds (each strip's rewrite first), the TLAS build. `None`, with
/// nothing queued but compactions, when ray tracing is off, no draw can be
/// instanced or the frame's objects cannot be made. Either way the cache
/// retires every BLAS whose contents its buffer no longer holds.
pub fn build_frame_tlas(
    enc: &mut FrameEncoder,
    draws: &[CapturedDraw],
    out: &mut Vec<BlitCommand>,
) -> Option<TlasHandle> {
    let mut rt = enc.rt.take()?;
    let tlas = rt.build(enc, draws, out);
    for raw in rt.cache.retire_where(|key| {
        !enc.buffer_content_current(key.vertex_content())
            || key
                .index_content()
                .is_some_and(|content| !enc.buffer_content_current(content))
    }) {
        enc.retire_acceleration_structure(raw);
    }
    rt.free_queries.append(&mut rt.cache.freed_queries);
    enc.rt = Some(rt);
    tlas
}

impl RtScene {
    fn build(
        &mut self,
        enc: &mut FrameEncoder,
        draws: &[CapturedDraw],
        out: &mut Vec<BlitCommand>,
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

        let mut instances = Vec::with_capacity(draws.len());
        let mut instance_draws = Vec::with_capacity(draws.len());
        let mut builds: Vec<PendingBuild> = Vec::new();
        let mut scratch = 0u64;
        let mut lists = 0u64;
        for (index, d) in (0u32..).zip(draws) {
            let source = match blas_source(d) {
                Ok(source) => source,
                Err(skip) => {
                    self.skip(skip);
                    continue;
                }
            };
            let Some(rows) = world_view_rows(d) else {
                self.skip(Skip::Transform);
                continue;
            };
            let blas = if let Some(blas) = self.cache.get(&source.key, seq) {
                self.stats.reused += 1;
                blas
            } else {
                let Some(build) = self.new_blas(
                    enc,
                    &device,
                    &source,
                    seq,
                    (&mut scratch, &mut lists),
                    &builds,
                ) else {
                    continue;
                };
                let blas = build.blas;
                builds.push(build);
                blas
            };
            instances.push(RtInstance {
                blas,
                transform: rows,
                mask: if d.alpha.is_some() {
                    MASK_ALPHA
                } else {
                    MASK_OPAQUE
                },
                user_id: u32::try_from(instances.len()).expect("instances fit u32"),
            });
            instance_draws.push(index);
        }
        if self.stats.deferred > 0 {
            mtld3d_shared::log_once_info!(target: LOG_TARGET,
                "rt: {} draws wait for a later frame's BLAS builds (per-frame cap or budget)",
                self.stats.deferred);
        }
        if instances.is_empty() {
            return None;
        }
        let tlas = self.encode_frame(
            enc,
            &device,
            slot,
            (instances, instance_draws),
            (&builds, scratch, lists),
            out,
        );
        if tlas.is_none() {
            // The builds were never queued: their BLASes hold nothing.
            for build in &builds {
                if let Some(raw) = self.cache.remove(&build.key) {
                    enc.retire_acceleration_structure(raw);
                }
            }
            self.stats.built = 0;
        }
        tlas
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

    /// Allocate and cache a BLAS for `source`, placing its scratch and list
    /// after this frame's earlier builds; `None` (counted) when the frame's
    /// caps or the budget leave no room, or an object cannot be made.
    fn new_blas(
        &mut self,
        enc: &mut FrameEncoder,
        device: &ProtocolObject<dyn MTLDevice>,
        source: &BlasSource,
        seq: u64,
        (scratch, lists): (&mut u64, &mut u64),
        builds: &[PendingBuild],
    ) -> Option<PendingBuild> {
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
        if builds.len() >= MAX_BUILDS_PER_FRAME || *scratch + build_scratch > MAX_FRAME_SCRATCH {
            self.stats.deferred += 1;
            return None;
        }
        let others = self.slot_bytes() + *scratch + build_scratch + *lists + list_bytes;
        let Some(evicted) = self.cache.make_room(size + others, seq) else {
            self.stats.deferred += 1;
            return None;
        };
        for raw in evicted {
            enc.retire_acceleration_structure(raw);
        }
        let blas = new_acceleration_structure(device, size, "mtld3d-rt-blas")?;
        let strip = if source.strip {
            let (source_buffer, source_offset, source_type) = match source.index {
                SourceIndex::Buffer {
                    handle,
                    offset,
                    index_type,
                } => (handle, u64::from(offset), index_type as u32),
                SourceIndex::None => (0, 0, 0),
            };
            let dest_offset = *lists;
            *lists += list_bytes.next_multiple_of(SUBALLOC_ALIGN);
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
        let scratch_offset = *scratch;
        *scratch += build_scratch;
        let query = self
            .free_queries
            .pop()
            .or_else(|| self.first_queries(device));
        self.cache.insert(source.key, blas.raw(), size, seq);
        self.cache.await_compaction(&source.key, seq, query);
        self.stats.built += 1;
        Some(PendingBuild {
            key: source.key,
            blas: blas.raw(),
            build: BlasBuild {
                geometry,
                strip,
                scratch: 0,
                scratch_offset,
                size_query: 0,
                size_query_offset: query.map_or(0, |q| u64::from(q) * 4),
            },
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
                    instance.mask == MASK_OPAQUE,
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
            && grow_tlas(
                device,
                &mut frame.tlas,
                sizes.accelerationStructureSize as u64,
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
            build.size_query = self.query_buffer_for(&pending.key);
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

/// [`grow`] for the frame's TLAS.
fn grow_tlas(
    device: &ProtocolObject<dyn MTLDevice>,
    grown: &mut Grown,
    len: u64,
    retired: &mut Vec<(DestroyKind, u64)>,
) -> bool {
    if grown.len >= len.max(1) {
        return true;
    }
    let len = len.max(SUBALLOC_ALIGN).next_power_of_two();
    let Some(tlas): Option<MetalHandle<MTLAccelerationStructureKind>> =
        new_acceleration_structure(device, len, "mtld3d-rt-tlas")
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
            "rt: world draws with vertex blending or a programmable vertex shader are not traced yet"),
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
    }
}
