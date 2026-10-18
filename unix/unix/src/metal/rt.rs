//! Metal ray tracing: the device check, acceleration-structure builds and compaction, the memory budget.
//!
//! The `rt.*` keys trace rays against the world the scene capture keeps
//! (`encoder/rt_scene.rs`): one primitive acceleration structure (BLAS) per
//! distinct static vertex range, cached across frames, instanced each frame
//! into an instance acceleration structure (TLAS) in eye space. This module
//! is the Metal side of that: whether the device can trace at all
//! ([`rt_supported`], [`effective_config`]), the build descriptions the
//! encoder thread writes into a submission's scratch arena ([`BlasBuild`],
//! [`TlasBuild`]), their encode on the submit thread ([`encode_run`]) as
//! [`BlitCommandType::BuildBlas`] / [`BlitCommandType::BuildTlas`] /
//! [`BlitCommandType::CompactBlas`] leading blits, and the strip-to-list
//! index rewrite a BLAS of a triangle strip builds from.
//!
//! Acceleration structures, their scratch and the rewritten lists stay
//! within [`RT_BUDGET_BYTES`] of GPU memory.
//!
//! [`BlitCommandType::BuildBlas`]: mtld3d_shared::BlitCommandType::BuildBlas
//! [`BlitCommandType::BuildTlas`]: mtld3d_shared::BlitCommandType::BuildTlas
//! [`BlitCommandType::CompactBlas`]: mtld3d_shared::BlitCommandType::CompactBlas

use std::sync::Arc;

use mtld3d_core::{config::Mtld3dConfig, rt::RtConfig};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, MetalHandle,
    mtl_handle::{MTLAccelerationStructureKind, MTLBufferKind, MTLComputePipelineStateKind},
    perf::GpuStage,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSString};
use objc2_metal::{
    MTLAccelerationStructure, MTLAccelerationStructureCommandEncoder,
    MTLAccelerationStructureDescriptor, MTLAccelerationStructureGeometryDescriptor,
    MTLAccelerationStructureInstanceDescriptorType, MTLAccelerationStructureInstanceOptions,
    MTLAccelerationStructureSizes, MTLAccelerationStructureTriangleGeometryDescriptor,
    MTLAccelerationStructureUsage, MTLAccelerationStructureUserIDInstanceDescriptor,
    MTLAttributeFormat, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLDevice, MTLDispatchType, MTLIndexType,
    MTLInstanceAccelerationStructureDescriptor, MTLLanguageVersion, MTLLibrary, MTLPackedFloat3,
    MTLPackedFloat4x3, MTLPrimitiveAccelerationStructureDescriptor, MTLResource,
    MTLResourceOptions, MTLSize,
};

use super::{
    handle::{IntoRetained, ReleaseRetain},
    stage_time::{StageTimer, accel_encoder, compute_encoder},
};
use crate::LOG_TARGET;

#[cfg(test)]
mod tests;

/// GPU memory the ray-traced scene may hold: every BLAS, the frames' scratch, lists and TLAS.
pub const RT_BUDGET_BYTES: u64 = 512 << 20;

/// The alignment of every sub-allocation in a scratch or list buffer.
pub const SUBALLOC_ALIGN: u64 = 256;

/// Whether `device` can build and trace acceleration structures.
#[must_use]
pub fn rt_supported(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    device.supportsRaytracing()
}

/// The configuration the native side reads: `config` itself where the
/// device traces rays (`supported`), otherwise `config` with every `rt.*`
/// key off, said once when the file set one.
///
/// Applied before anything reads the configuration (the scene capture and
/// textures, the receivers, the kernels), so a device without ray tracing
/// runs exactly as if the keys were absent: the sun's cascades come back
/// under `shadows.sun`, nothing is captured for `rt.*` alone.
#[must_use]
pub fn effective_config(config: Arc<Mtld3dConfig>, supported: bool) -> Arc<Mtld3dConfig> {
    if supported || config.rt == RtConfig::default() {
        return config;
    }
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "rt: the Metal device does not support ray tracing; every rt.* key is off"
    );
    Arc::new(Mtld3dConfig {
        rt: RtConfig::default(),
        ..(*config).clone()
    })
}

/// The triangles one BLAS is built over.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlasGeometry {
    /// The vertex buffer, a raw canonical `MTLBuffer`.
    pub vertex_buffer: u64,
    /// Byte offset of the first vertex's position.
    pub vertex_offset: u64,
    pub vertex_stride: u32,
    /// `MTLAttributeFormat` of the position: `Float3` or `Float4`.
    pub vertex_format: u32,
    /// The index buffer, 0 for a non-indexed list.
    pub index_buffer: u64,
    pub index_offset: u64,
    /// `mtl::IndexType` of the indices.
    pub index_type: u32,
    pub triangle_count: u32,
    /// Nonzero for opaque triangles; zero for alpha-tested ones, whose
    /// candidate hits go through the kernel's intersection function.
    pub opaque: u32,
    /// `MTLAccelerationStructureUsage` bits: 0 for a cached BLAS,
    /// [`REFIT_USAGE`] for one refit frame after frame.
    pub usage: u32,
}

/// The usage of a BLAS refit frame after frame: refittable (`Refit`, bit 0),
/// built fast (`PreferFastBuild`, bit 1).
pub const REFIT_USAGE: u32 = 0b11;

/// A strip's index rewrite into a list, run before the BLAS build reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StripConversion {
    /// The strip-to-list compute pipeline, 0 for no rewrite.
    pub pipeline: u64,
    /// The strip's index buffer, 0 for a non-indexed strip.
    pub source: u64,
    pub source_offset: u64,
    /// `mtl::IndexType` of the source indices.
    pub source_type: u32,
    pub triangle_count: u32,
    /// The `uint` list the rewrite writes, three indices per triangle.
    pub dest: u64,
    pub dest_offset: u64,
}

/// One BLAS build, as a [`BlitCommandType::BuildBlas`] names it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlasBuild {
    pub geometry: BlasGeometry,
    pub strip: StripConversion,
    pub scratch: u64,
    pub scratch_offset: u64,
    /// A Shared buffer the compacted size is written into (a `u32`), 0 for none.
    pub size_query: u64,
    pub size_query_offset: u64,
    /// The acceleration structure this one is refit from (itself: in place),
    /// built with [`REFIT_USAGE`] over the same triangles; 0 for a full build.
    pub refit: u64,
}

/// One TLAS build, as a [`BlitCommandType::BuildTlas`] names it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TlasBuild {
    /// The [`InstanceDescriptor`]s, a Shared buffer.
    pub instances: u64,
    pub instance_count: u32,
    pub blas_count: u32,
    /// The address of `blas_count` raw BLAS handles in the submission's scratch arena.
    pub blas: u64,
    pub scratch: u64,
    pub scratch_offset: u64,
}

/// One instance of the TLAS: a BLAS (by its index in the TLAS's list), its transform and mask.
pub type InstanceDescriptor = MTLAccelerationStructureUserIDInstanceDescriptor;

/// The instance of BLAS `blas_index` under `rows`, a row-major 3x4
/// object-to-eye transform (`p_eye[r] = dot(rows[r], (p, 1))`).
///
/// Metal's packed 4x3 matrix is column-major: column `c` holds element `c`
/// of each row. Triangles are never culled; `opaque` forces the instance
/// opaque, otherwise each candidate hit goes through the intersection
/// function (its geometry is non-opaque too).
#[must_use]
pub fn instance_descriptor(
    rows: &[[f32; 4]; 3],
    blas_index: u32,
    mask: u8,
    user_id: u32,
    opaque: bool,
) -> InstanceDescriptor {
    let column = |c: usize| MTLPackedFloat3 {
        x: rows[0][c],
        y: rows[1][c],
        z: rows[2][c],
    };
    let opacity = if opaque {
        MTLAccelerationStructureInstanceOptions::Opaque
    } else {
        MTLAccelerationStructureInstanceOptions::NonOpaque
    };
    InstanceDescriptor {
        transformationMatrix: MTLPackedFloat4x3 {
            columns: [column(0), column(1), column(2), column(3)],
        },
        options: MTLAccelerationStructureInstanceOptions::DisableTriangleCulling | opacity,
        mask: u32::from(mask),
        intersectionFunctionTableOffset: 0,
        accelerationStructureIndex: blas_index,
        userID: user_id,
    }
}

/// A `MetalHandle` of `raw`, a canonical retain an owner keeps alive.
const fn handle<K>(raw: u64) -> MetalHandle<K> {
    // SAFETY: every caller passes a handle its owner (the scene, the frame's
    // command buffer, a draw's resolved binding) keeps alive past the use.
    unsafe { MetalHandle::new(raw) }
}

fn buffer(raw: u64) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    handle::<MTLBufferKind>(raw).into_retained()
}

fn acceleration_structure(
    raw: u64,
) -> Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> {
    handle::<MTLAccelerationStructureKind>(raw).into_retained()
}

/// The primitive descriptor of `geometry`, `None` when one of its buffers is gone.
fn primitive_descriptor(
    geometry: &BlasGeometry,
) -> Option<Retained<MTLPrimitiveAccelerationStructureDescriptor>> {
    let triangles = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
    let vertices = buffer(geometry.vertex_buffer)?;
    triangles.setVertexBuffer(Some(&vertices));
    // SAFETY: the build's own range of a live buffer, checked by the caller.
    unsafe {
        triangles.setVertexBufferOffset(geometry.vertex_offset as usize);
    }
    triangles.setVertexFormat(MTLAttributeFormat(geometry.vertex_format as usize));
    triangles.setVertexStride(geometry.vertex_stride as usize);
    if geometry.index_buffer != 0 {
        let indices = buffer(geometry.index_buffer)?;
        triangles.setIndexBuffer(Some(&indices));
        // SAFETY: as the vertex offset.
        unsafe {
            triangles.setIndexBufferOffset(geometry.index_offset as usize);
        }
        triangles.setIndexType(if geometry.index_type == 0 {
            MTLIndexType::UInt16
        } else {
            MTLIndexType::UInt32
        });
    }
    triangles.setTriangleCount(geometry.triangle_count as usize);
    // An alpha-tested draw's candidate hits go through the intersection
    // function; an opaque draw's never do.
    triangles.setOpaque(geometry.opaque != 0);
    let descriptor = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    descriptor.setUsage(MTLAccelerationStructureUsage(geometry.usage as usize));
    let geometry: Retained<MTLAccelerationStructureGeometryDescriptor> =
        Retained::into_super(triangles);
    descriptor.setGeometryDescriptors(Some(&NSArray::from_retained_slice(&[geometry])));
    Some(descriptor)
}

/// The instance descriptor of `build`, `None` when one of its objects is gone.
fn instance_descriptor_set(
    instances: u64,
    instance_count: u32,
    blas: &[u64],
) -> Option<Retained<MTLInstanceAccelerationStructureDescriptor>> {
    let descriptor = MTLInstanceAccelerationStructureDescriptor::descriptor();
    if instances != 0 {
        let instances = buffer(instances)?;
        descriptor.setInstanceDescriptorBuffer(Some(&instances));
    }
    descriptor.setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::UserID);
    // SAFETY: the stride of the descriptor type set above.
    unsafe {
        descriptor.setInstanceDescriptorStride(size_of::<InstanceDescriptor>());
    }
    descriptor.setInstanceCount(instance_count as usize);
    let blas = blas
        .iter()
        .map(|&raw| acceleration_structure(raw))
        .collect::<Option<Vec<_>>>()?;
    descriptor.setInstancedAccelerationStructures(Some(&NSArray::from_retained_slice(&blas)));
    // Rebuilt every frame: build time over trace time.
    descriptor.setUsage(MTLAccelerationStructureUsage::PreferFastBuild);
    Some(descriptor)
}

/// What a BLAS of `geometry` needs: its size and its build's scratch.
#[must_use]
pub fn blas_sizes(
    device: &ProtocolObject<dyn MTLDevice>,
    geometry: &BlasGeometry,
) -> Option<MTLAccelerationStructureSizes> {
    let descriptor = primitive_descriptor(geometry)?;
    Some(device.accelerationStructureSizesWithDescriptor(&descriptor))
}

/// What a TLAS of `instance_count` instances over `blas` needs.
#[must_use]
pub fn tlas_sizes(
    device: &ProtocolObject<dyn MTLDevice>,
    instance_count: u32,
    blas: &[u64],
) -> Option<MTLAccelerationStructureSizes> {
    let descriptor = instance_descriptor_set(0, instance_count, blas)?;
    Some(device.accelerationStructureSizesWithDescriptor(&descriptor))
}

/// A new acceleration structure of `size` bytes, owned by the caller.
#[must_use]
pub fn new_acceleration_structure(
    device: &ProtocolObject<dyn MTLDevice>,
    size: u64,
    label: &str,
) -> Option<MetalHandle<MTLAccelerationStructureKind>> {
    let Some(accel) = device.newAccelerationStructureWithSize(usize::try_from(size).ok()?) else {
        log::error!(target: LOG_TARGET, "rt: a {size}-byte acceleration structure could not be made");
        return None;
    };
    accel.setLabel(Some(&NSString::from_str(label)));
    // SAFETY: the canonical retain is transferred to the caller's owner.
    Some(unsafe { MetalHandle::new(Retained::into_raw(accel) as u64) })
}

/// A new buffer of `len` bytes, CPU-visible when `shared`, owned by the caller.
#[must_use]
pub fn new_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    len: u64,
    shared: bool,
    label: &str,
) -> Option<MetalHandle<MTLBufferKind>> {
    let options = if shared {
        MTLResourceOptions::StorageModeShared
    } else {
        MTLResourceOptions::StorageModePrivate
    };
    let Some(buffer) = device.newBufferWithLength_options(usize::try_from(len).ok()?, options)
    else {
        log::error!(target: LOG_TARGET, "rt: a {len}-byte buffer could not be made");
        return None;
    };
    buffer.setLabel(Some(&NSString::from_str(label)));
    // SAFETY: the canonical retain is transferred to the caller's owner.
    Some(unsafe { MetalHandle::new(Retained::into_raw(buffer) as u64) })
}

/// Release an acceleration structure's canonical retain after GPU retirement.
pub fn destroy_acceleration_structure(raw: u64) {
    // SAFETY: the caller hands over a live canonical retain no command still names.
    unsafe { handle::<MTLAccelerationStructureKind>(raw).release_retain() };
}

/// The strip-to-list rewrite: triangle `t` of a strip is its indices `t`,
/// `t + 1`, `t + 2`, the first two swapped on odd `t` so every triangle
/// keeps the strip's winding.
const STRIP_TO_LIST: &str = r"
#include <metal_stdlib>
using namespace metal;

struct StripParams { uint triangles; uint source_type; };

static uint strip_index(device const uchar *source, constant StripParams &p, uint i) {
    switch (p.source_type) {
        case 0u: return uint(((device const ushort *)source)[i]);
        case 1u: return ((device const uint *)source)[i];
        default: return i;
    }
}

kernel void strip_to_list(device const uchar *source [[buffer(0)]],
                          device uint *list [[buffer(1)]],
                          constant StripParams &p [[buffer(2)]],
                          uint t [[thread_position_in_grid]]) {
    if (t >= p.triangles) {
        return;
    }
    uint first = (t & 1u) != 0u ? t + 1u : t;
    uint second = (t & 1u) != 0u ? t : t + 1u;
    list[3u * t] = strip_index(source, p, first);
    list[3u * t + 1u] = strip_index(source, p, second);
    list[3u * t + 2u] = strip_index(source, p, t + 2u);
}
";

/// `StripParams::source_type` of a non-indexed strip: the index is the position.
const SOURCE_NONE: u32 = 2;

/// Compile the strip-to-list rewrite into a compute pipeline owned by the caller.
#[must_use]
pub fn create_strip_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<MetalHandle<MTLComputePipelineStateKind>> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(STRIP_TO_LIST), Some(&options))
        .map_err(|e| log::error!(target: LOG_TARGET, "rt: strip rewrite library failed: {e}"))
        .ok()?;
    library.setLabel(Some(&NSString::from_str("mtld3d-rt-strip-to-list")));
    let function = library.newFunctionWithName(&NSString::from_str("strip_to_list"))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| log::error!(target: LOG_TARGET, "rt: strip rewrite pipeline failed: {e}"))
        .ok()?;
    // SAFETY: the canonical retain is transferred to the caller's owner.
    Some(unsafe { MetalHandle::new(Retained::into_raw(pipeline) as u64) })
}

/// The encoders one run of acceleration-structure commands took ([`encode_run`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunEncoders {
    /// Compute encoders: every strip rewrite of the run shares one.
    pub compute: u32,
    /// Acceleration-structure encoders: one for the run's BLAS builds and
    /// compactions, one more per TLAS build.
    pub accel: u32,
}

/// The open acceleration-structure encoder of a run, and how many the run opened.
struct Accel<'a> {
    cb: &'a ProtocolObject<dyn MTLCommandBuffer>,
    timer: Option<&'a mut StageTimer>,
    encoder: Option<Retained<ProtocolObject<dyn MTLAccelerationStructureCommandEncoder>>>,
    opened: u32,
}

impl Accel<'_> {
    /// The open encoder, opened on the command buffer (timed under `stage`) if none is.
    fn open(
        &mut self,
        stage: GpuStage,
    ) -> Option<&ProtocolObject<dyn MTLAccelerationStructureCommandEncoder>> {
        if self.encoder.is_none() {
            let Some(encoder) = accel_encoder(self.cb, self.timer.as_deref_mut(), stage) else {
                log::error!(target: LOG_TARGET, "rt: acceleration structure encoder failed");
                return None;
            };
            encoder.setLabel(Some(&NSString::from_str("mtld3d-rt-builds")));
            self.encoder = Some(encoder);
            self.opened += 1;
        }
        self.encoder.as_deref()
    }

    /// End the open encoder, if any.
    fn close(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
    }
}

/// Encode `run`, a run of acceleration-structure commands, on `cb`.
///
/// Every strip rewrite the run's builds read goes first, all on one compute
/// encoder (they write disjoint ranges, so they dispatch concurrently).
/// Then, in the run's order, its BLAS builds and compactions share one
/// acceleration-structure encoder, which lets the GPU overlap them: a build
/// on an encoder of its own waits for the encoder before it, about 0.3 ms
/// a build on Apple GPUs against a few microseconds shared. A TLAS build
/// gets an encoder of its own, after every BLAS command before it. With a
/// `timer` the rewrites and BLAS commands are timed as
/// [`GpuStage::RtBuild`], a TLAS build as [`GpuStage::RtTlas`]. The command
/// buffer retains every object the run names until it completes. `None`,
/// after logging, when a description or one of its objects is missing or
/// an encoder cannot be made.
pub fn encode_run(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    run: &[BlitCommand],
    mut timer: Option<&mut StageTimer>,
) -> Option<RunEncoders> {
    if !cb.retainedReferences() {
        log::error!(target: LOG_TARGET, "rt: command buffer does not retain resources");
        return None;
    }
    let mut strips = Vec::new();
    for command in run {
        if command.cmd == BlitCommandType::BuildBlas as u32 {
            let build = description::<BlasBuild>(command.byte_size)?;
            if build.strip.pipeline != 0 {
                strips.push(build.strip);
            }
        }
    }
    let compute = if strips.is_empty() {
        0
    } else {
        encode_strips(cb, &strips, timer.as_deref_mut())?;
        1
    };
    let mut accel = Accel {
        cb,
        timer,
        encoder: None,
        opened: 0,
    };
    let encoded = run
        .iter()
        .all(|command| encode_command(command, &mut accel));
    accel.close();
    encoded.then_some(RunEncoders {
        compute,
        accel: accel.opened,
    })
}

/// Encode one command of a run on `accel`; its strip rewrite is already encoded.
fn encode_command(command: &BlitCommand, accel: &mut Accel<'_>) -> bool {
    match BlitCommandType::from_repr(command.cmd) {
        Some(BlitCommandType::BuildBlas) => encode_blas(command, accel),
        Some(BlitCommandType::BuildTlas) => {
            accel.close();
            let built = encode_tlas(command, accel);
            accel.close();
            built
        }
        Some(BlitCommandType::CompactBlas) => {
            let (Some(source), Some(compacted)) = (
                acceleration_structure(command.src_handle),
                acceleration_structure(command.dst_handle),
            ) else {
                log::error!(target: LOG_TARGET, "rt: a compaction names a missing acceleration structure");
                return false;
            };
            let Some(encoder) = accel.open(GpuStage::RtBuild) else {
                return false;
            };
            encoder
                .copyAndCompactAccelerationStructure_toAccelerationStructure(&source, &compacted);
            true
        }
        _ => {
            log::error!(target: LOG_TARGET, "rt: command {} is no acceleration-structure command", command.cmd);
            false
        }
    }
}

/// Read the `T` a command names in the submission's scratch arena.
fn description<T: Copy>(address: u64) -> Option<T> {
    if address == 0 {
        log::error!(target: LOG_TARGET, "rt: a build names no description");
        return None;
    }
    // SAFETY: the encoder thread wrote a `T` at `address` in the submission's
    // scratch arena, which outlives this encode; the read copies it out
    // unaligned.
    Some(unsafe { core::ptr::read_unaligned(address as *const T) })
}

fn encode_blas(command: &BlitCommand, accel: &mut Accel<'_>) -> bool {
    let Some(build) = description::<BlasBuild>(command.byte_size) else {
        return false;
    };
    let (Some(target), Some(descriptor), Some(scratch)) = (
        acceleration_structure(command.dst_handle),
        primitive_descriptor(&build.geometry),
        buffer(build.scratch),
    ) else {
        log::error!(target: LOG_TARGET, "rt: a BLAS build names a missing object");
        return false;
    };
    let Some(encoder) = accel.open(GpuStage::RtBuild) else {
        return false;
    };
    if build.refit != 0 {
        let Some(source) = acceleration_structure(build.refit) else {
            log::error!(target: LOG_TARGET, "rt: a BLAS refit names a missing source");
            return false;
        };
        // In place when the source is the target itself.
        let destination = (build.refit != command.dst_handle).then_some(&*target);
        // SAFETY: the scene's objects, alive until the command buffer
        // completes, which retains them; the scratch range is the refit's own.
        unsafe {
            encoder.refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset(
                &source,
                &descriptor,
                destination,
                Some(&scratch),
                build.scratch_offset as usize,
            );
        }
        return true;
    }
    encoder.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
        &target,
        &descriptor,
        &scratch,
        build.scratch_offset as usize,
    );
    if build.size_query != 0 {
        let Some(sizes) = buffer(build.size_query) else {
            log::error!(target: LOG_TARGET, "rt: a BLAS size query names a missing buffer");
            return false;
        };
        encoder.writeCompactedAccelerationStructureSize_toBuffer_offset(
            &target,
            &sizes,
            build.size_query_offset as usize,
        );
    }
    true
}

/// Every rewrite of `strips` on one compute encoder, dispatched concurrently: each writes its own range.
fn encode_strips(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    strips: &[StripConversion],
    timer: Option<&mut StageTimer>,
) -> Option<()> {
    let Some(compute) = compute_encoder(cb, timer, GpuStage::RtBuild, MTLDispatchType::Concurrent)
    else {
        log::error!(target: LOG_TARGET, "rt: strip rewrite encoder failed");
        return None;
    };
    compute.setLabel(Some(&NSString::from_str("mtld3d-rt-strip-to-list")));
    let encoded = strips.iter().all(|strip| encode_strip(&compute, strip));
    compute.endEncoding();
    encoded.then_some(())
}

fn encode_strip(
    compute: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    strip: &StripConversion,
) -> bool {
    // SAFETY: the scene owns the strip pipeline until teardown.
    let pipeline = handle::<MTLComputePipelineStateKind>(strip.pipeline).into_retained();
    let Some(list) = buffer(strip.dest) else {
        log::error!(target: LOG_TARGET, "rt: a strip rewrite names a missing list");
        return false;
    };
    // A non-indexed strip reads no source; the list stands in at the slot.
    let source = if strip.source == 0 {
        Some(list.clone())
    } else {
        buffer(strip.source)
    };
    let (Some(pipeline), Some(source)) = (pipeline, source) else {
        log::error!(target: LOG_TARGET, "rt: a strip rewrite names a missing object");
        return false;
    };
    compute.setComputePipelineState(&pipeline);
    let params = [
        strip.triangle_count,
        if strip.source == 0 {
            SOURCE_NONE
        } else {
            strip.source_type
        },
    ];
    let source_offset = if strip.source == 0 {
        0
    } else {
        strip.source_offset as usize
    };
    // SAFETY: slot 0 is the strip's indices (or the unread list), 1 the list
    // it writes, 2 the parameters, which the encoder copies.
    unsafe {
        compute.setBuffer_offset_atIndex(Some(&source), source_offset, 0);
        compute.setBuffer_offset_atIndex(Some(&list), strip.dest_offset as usize, 1);
        compute.setBytes_length_atIndex(
            core::ptr::NonNull::from(&params).cast(),
            size_of_val(&params),
            2,
        );
    }
    compute.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: strip.triangle_count as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64.min(strip.triangle_count as usize).max(1),
            height: 1,
            depth: 1,
        },
    );
    true
}

fn encode_tlas(command: &BlitCommand, accel: &mut Accel<'_>) -> bool {
    let Some(build) = description::<TlasBuild>(command.byte_size) else {
        return false;
    };
    if build.blas == 0 || build.blas_count == 0 {
        log::error!(target: LOG_TARGET, "rt: a TLAS build instances no BLAS");
        return false;
    }
    // SAFETY: `blas_count` raw handles the encoder thread copied into the
    // submission's scratch arena, which outlives this encode.
    let blas =
        unsafe { core::slice::from_raw_parts(build.blas as *const u64, build.blas_count as usize) };
    let (Some(target), Some(descriptor), Some(scratch)) = (
        acceleration_structure(command.dst_handle),
        instance_descriptor_set(build.instances, build.instance_count, blas),
        buffer(build.scratch),
    ) else {
        log::error!(target: LOG_TARGET, "rt: a TLAS build names a missing object");
        return false;
    };
    let Some(encoder) = accel.open(GpuStage::RtTlas) else {
        return false;
    };
    let descriptor: &MTLAccelerationStructureDescriptor = &descriptor;
    encoder.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
        &target,
        descriptor,
        &scratch,
        build.scratch_offset as usize,
    );
    true
}
