//! The shared MSL of the ray-traced kernels: the instance table, the alpha test and the hit shading.
//!
//! Every ray-traced kernel compiles [`RT_COMMON_MSL`] ahead of its own
//! source ([`create_rt_pipeline`]). It declares the instance table the
//! scene writes each frame (`RtInstanceInfo`, one entry per TLAS instance,
//! indexed by the instance's user id; [`RtInstanceInfo`] here), the triangle
//! intersection function [`ALPHA_INTERSECTION`] the pipeline links and its
//! intersection function table holds at index 0, and the helpers a kernel
//! shades a hit with:
//!
//! - `rt_alpha_passes(func, alpha, alpha_ref)`: `D3DRS_ALPHAFUNC` as the
//!   raster's alpha test compares (`dxso::emit::emit_alpha_test`): `NEVER`
//!   fails, `0`, `ALWAYS` and unknown values pass.
//! - `rt_hit_uv(info, primitive, barycentrics)`: the stage-0 coordinate at a
//!   hit, interpolated from the instance's three vertices.
//! - `rt_alpha_hit(info, primitive, barycentrics)`: whether the hit survives
//!   the instance's alpha test; true for an instance with no texture.
//! - `rt_hit_albedo(info, primitive, barycentrics)`: `texture(uv).rgb *
//!   mat_diffuse`, `mat_diffuse` alone for an instance with no texture.
//!
//! Alpha-tested instances are non-opaque, so an `intersector` tracing with
//! the pipeline's table and a `RtPayload` calls the intersection function on
//! each of their candidate hits, which accepts it iff the alpha test passes.
//! Opaque instances (and their geometry) are opaque: it is never called for
//! them. A kernel walking an `intersection_query` instead calls
//! `rt_alpha_hit` on each non-opaque candidate itself.
//!
//! Textures are sampled at mip 0 (a ray has no footprint). The table names
//! textures, samplers and buffers by their GPU resource ids and addresses,
//! so a kernel tracing it makes each resident ([`bind_scene`]).

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the ray-traced kernels (sun visibility, AO, GI) compile and bind through these"
    )
)]

use mtld3d_shared::{
    MetalHandle,
    mtl_handle::{MTLAccelerationStructureKind, MTLBufferKind, MTLTextureKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSString};
use objc2_metal::{
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLIntersectionFunctionTable,
    MTLIntersectionFunctionTableDescriptor, MTLLanguageVersion, MTLLibrary, MTLLinkedFunctions,
    MTLPipelineOption, MTLResource, MTLResourceUsage,
};

use super::handle::IntoRetained;
use crate::LOG_TARGET;

#[cfg(test)]
mod tests;

/// `RtInstanceInfo::uv_source`: no coordinate, the instance is untextured.
pub const UV_NONE: u32 = 0;
/// `RtInstanceInfo::uv_source`: a vertex-buffer attribute of a static draw.
pub const UV_VB: u32 = 1;
/// `RtInstanceInfo::uv_source`: the coordinates a capture stored (`captured_uv`).
pub const UV_CAPTURED: u32 = 2;

/// `RtInstanceInfo::index_type`: 16-bit indices at `ib`.
pub const INDEX_U16: u32 = 0;
/// `RtInstanceInfo::index_type`: 32-bit indices at `ib`.
pub const INDEX_U32: u32 = 1;
/// `RtInstanceInfo::index_type`: non-indexed, the index is the position.
pub const INDEX_NONE: u32 = 2;
/// `RtInstanceInfo::index_type` flag: a strip, whose triangle `t` is its
/// indices `t`, `t + 1`, `t + 2`, the first two swapped on odd `t` (the
/// order the BLAS's list rewrite keeps).
pub const INDEX_STRIP: u32 = 4;

/// The entry name of the alpha-test intersection function.
pub const ALPHA_INTERSECTION: &str = "rt_alpha_intersection";

/// The kernel slot of the TLAS ([`bind_scene`]).
pub const ACCEL_SLOT: usize = 0;
/// The kernel slot of the intersection function table.
pub const TABLE_SLOT: usize = 1;
/// The kernel slot of the instance table.
pub const INSTANCES_SLOT: usize = 2;

/// One instance's material, as the kernels read it, by user id.
///
/// Field for field the MSL `RtInstanceInfo` of [`RT_COMMON_MSL`], whose
/// `texture2d<float>`, `sampler` and `device const uchar *` members are
/// these 64-bit resource ids and GPU addresses.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RtInstanceInfo {
    /// The stage-0 texture's `gpuResourceID`, 0 for none.
    pub tex_id: u64,
    /// Its sampler's `gpuResourceID` (made with argument-buffer support).
    pub sampler_id: u64,
    /// `D3DRS_ALPHAREF` in `[0, 1]`.
    pub alpha_ref: f32,
    /// `D3DRS_ALPHAFUNC` (`D3DCMP_*`); 0 for an instance with no alpha test.
    pub alpha_func: u32,
    /// [`UV_NONE`], [`UV_VB`] or [`UV_CAPTURED`].
    pub uv_source: u32,
    pub pad0: u32,
    /// The GPU address of the buffer the coordinates are in.
    pub uv_buf: u64,
    /// The byte offset of vertex index 0's coordinate (a `float2`), the
    /// draw's first or base vertex folded in.
    pub uv_offset: u32,
    pub uv_stride: u32,
    /// The GPU address of the draw's first index, 0 for a non-indexed draw.
    pub ib: u64,
    /// [`INDEX_U16`], [`INDEX_U32`] or [`INDEX_NONE`], with [`INDEX_STRIP`] for a strip.
    pub index_type: u32,
    /// The material's diffuse colour.
    pub mat_diffuse: [f32; 3],
}

/// The shared MSL every ray-traced kernel is compiled after, see the module docs.
pub const RT_COMMON_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

constant uint RT_UV_NONE = 0u;
constant uint RT_INDEX_U16 = 0u;
constant uint RT_INDEX_U32 = 1u;
constant uint RT_INDEX_STRIP = 4u;

struct RtInstanceInfo {
    texture2d<float> tex;
    sampler samp;
    float alpha_ref;
    uint alpha_func;
    uint uv_source;
    uint pad0;
    device const uchar *uv_buf;
    uint uv_offset;
    uint uv_stride;
    device const uchar *ib;
    uint index_type;
    packed_float3 mat_diffuse;
};

struct RtPayload {
    device const RtInstanceInfo *instances;
    uint alpha_calls;
};

static bool rt_alpha_passes(uint func, float alpha, float alpha_ref) {
    switch (func) {
        case 1u: return false;
        case 2u: return alpha < alpha_ref;
        case 3u: return alpha == alpha_ref;
        case 4u: return alpha <= alpha_ref;
        case 5u: return alpha > alpha_ref;
        case 6u: return alpha != alpha_ref;
        case 7u: return alpha >= alpha_ref;
        default: return true;
    }
}

static uint3 rt_triangle_vertices(device const RtInstanceInfo &info, uint prim) {
    uint3 slot;
    if ((info.index_type & RT_INDEX_STRIP) != 0u) {
        bool odd = (prim & 1u) != 0u;
        slot = uint3(odd ? prim + 1u : prim, odd ? prim : prim + 1u, prim + 2u);
    } else {
        slot = uint3(3u * prim, 3u * prim + 1u, 3u * prim + 2u);
    }
    switch (info.index_type & 3u) {
        case RT_INDEX_U16: {
            device const ushort *ib = (device const ushort *)info.ib;
            return uint3(ib[slot.x], ib[slot.y], ib[slot.z]);
        }
        case RT_INDEX_U32: {
            device const uint *ib = (device const uint *)info.ib;
            return uint3(ib[slot.x], ib[slot.y], ib[slot.z]);
        }
        default:
            return slot;
    }
}

static float2 rt_vertex_uv(device const RtInstanceInfo &info, uint v) {
    return float2(*(device const packed_float2 *)(info.uv_buf + info.uv_offset + v * info.uv_stride));
}

static bool rt_textured(device const RtInstanceInfo &info) {
    return info.uv_source != RT_UV_NONE && !is_null_texture(info.tex);
}

static float2 rt_hit_uv(device const RtInstanceInfo &info, uint prim, float2 bary) {
    uint3 v = rt_triangle_vertices(info, prim);
    return rt_vertex_uv(info, v.x) * (1.0 - bary.x - bary.y)
         + rt_vertex_uv(info, v.y) * bary.x
         + rt_vertex_uv(info, v.z) * bary.y;
}

static bool rt_alpha_hit(device const RtInstanceInfo &info, uint prim, float2 bary) {
    if (!rt_textured(info)) {
        return true;
    }
    float alpha = info.tex.sample(info.samp, rt_hit_uv(info, prim, bary), level(0.0)).a;
    return rt_alpha_passes(info.alpha_func, alpha, info.alpha_ref);
}

static float3 rt_hit_albedo(device const RtInstanceInfo &info, uint prim, float2 bary) {
    float3 albedo = float3(info.mat_diffuse);
    if (rt_textured(info)) {
        albedo *= info.tex.sample(info.samp, rt_hit_uv(info, prim, bary), level(0.0)).rgb;
    }
    return albedo;
}

[[intersection(triangle, triangle_data, instancing)]]
bool rt_alpha_intersection(uint instance [[user_instance_id]],
                           uint prim [[primitive_id]],
                           float2 bary [[barycentric_coord]],
                           ray_data RtPayload &payload [[payload]]) {
    payload.alpha_calls += 1u;
    return rt_alpha_hit(payload.instances[instance], prim, bary);
}
";

/// A ray-traced compute pipeline and its intersection function table.
///
/// The table holds [`ALPHA_INTERSECTION`] at index 0, the offset every
/// instance and geometry of the scene names; it is never written again, so
/// frames in flight can share it.
pub struct RtPipeline {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub table: Retained<ProtocolObject<dyn MTLIntersectionFunctionTable>>,
}

/// Compile `kernels` after [`RT_COMMON_MSL`] (MSL 3.0, for the resource ids
/// in the instance table) into the pipeline of kernel `entry`, the alpha
/// test linked and in its table. `None`, after logging, when it fails.
#[must_use]
pub fn create_rt_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    kernels: &str,
    entry: &str,
) -> Option<RtPipeline> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version3_0);
    let source = format!("{RT_COMMON_MSL}{kernels}");
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
        .map_err(|e| log::error!(target: LOG_TARGET, "rt: kernel library {entry} failed: {e}"))
        .ok()?;
    library.setLabel(Some(&NSString::from_str(&format!("mtld3d-rt-{entry}"))));
    let function = |name: &str| {
        let found = library.newFunctionWithName(&NSString::from_str(name));
        if found.is_none() {
            log::error!(target: LOG_TARGET, "rt: kernel library has no function {name}");
        }
        found
    };
    let kernel = function(entry)?;
    let alpha: Retained<ProtocolObject<dyn MTLFunction>> = function(ALPHA_INTERSECTION)?;
    let linked = MTLLinkedFunctions::new();
    linked.setFunctions(Some(&NSArray::from_retained_slice(&[alpha.clone()])));
    let descriptor = MTLComputePipelineDescriptor::new();
    descriptor.setComputeFunction(Some(&kernel));
    descriptor.setLinkedFunctions(Some(&linked));
    descriptor.setLabel(Some(&NSString::from_str(&format!("mtld3d-rt-{entry}"))));
    let pipeline = device
        .newComputePipelineStateWithDescriptor_options_reflection_error(
            &descriptor,
            MTLPipelineOption::None,
            None,
        )
        .map_err(|e| log::error!(target: LOG_TARGET, "rt: kernel pipeline {entry} failed: {e}"))
        .ok()?;
    let table_descriptor = MTLIntersectionFunctionTableDescriptor::new();
    table_descriptor.setFunctionCount(1);
    let Some(table) = pipeline.newIntersectionFunctionTableWithDescriptor(&table_descriptor) else {
        log::error!(target: LOG_TARGET, "rt: kernel {entry} has no intersection function table");
        return None;
    };
    let Some(handle) = pipeline.functionHandleWithFunction(&alpha) else {
        log::error!(target: LOG_TARGET, "rt: kernel {entry} did not link the alpha test");
        return None;
    };
    table.setFunction_atIndex(Some(&handle), 0);
    Some(RtPipeline { pipeline, table })
}

/// What a kernel tracing one frame's scene binds and makes resident, every handle raw and canonical.
#[derive(Clone, Copy, Debug)]
pub struct SceneBinding<'a> {
    /// The TLAS.
    pub tlas: u64,
    /// The Shared buffer of the frame's [`RtInstanceInfo`]s.
    pub instances: u64,
    /// Every BLAS the TLAS instances.
    pub blas: &'a [u64],
    /// Every texture the instance table names.
    pub textures: &'a [u64],
    /// Every buffer whose GPU address the instance table holds.
    pub buffers: &'a [u64],
}

/// Bind `pipeline` and `scene` on `encoder`: the TLAS at [`ACCEL_SLOT`], the
/// intersection function table at [`TABLE_SLOT`], the instance table at
/// [`INSTANCES_SLOT`], and every object the TLAS and the table reach made
/// resident for reading. False, after logging, when one is gone.
///
/// The caller's owners (the scene, the frame's draws) keep every handle
/// alive past the dispatch; the command buffer retains what it names.
#[must_use]
pub fn bind_scene(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &RtPipeline,
    scene: &SceneBinding<'_>,
) -> bool {
    // SAFETY: every handle is a canonical retain its owner keeps alive.
    let accel = |raw| unsafe { MetalHandle::<MTLAccelerationStructureKind>::new(raw) };
    // SAFETY: as above.
    let buffer = |raw| unsafe { MetalHandle::<MTLBufferKind>::new(raw) };
    let (Some(tlas), Some(instances)) = (
        accel(scene.tlas).into_retained(),
        buffer(scene.instances).into_retained(),
    ) else {
        log::error!(target: LOG_TARGET, "rt: a trace names a missing TLAS or instance table");
        return false;
    };
    encoder.setComputePipelineState(&pipeline.pipeline);
    // SAFETY: the slots of `RT_COMMON_MSL`'s convention; the objects are live.
    unsafe {
        encoder.setAccelerationStructure_atBufferIndex(Some(&tlas), ACCEL_SLOT);
        encoder.setIntersectionFunctionTable_atBufferIndex(Some(&pipeline.table), TABLE_SLOT);
        encoder.setBuffer_offset_atIndex(Some(&instances), 0, INSTANCES_SLOT);
    }
    let resident = |resource: Option<&ProtocolObject<dyn MTLResource>>| {
        let Some(resource) = resource else {
            log::error!(target: LOG_TARGET, "rt: a trace names a missing resource");
            return false;
        };
        encoder.useResource_usage(resource, MTLResourceUsage::Read);
        true
    };
    scene.blas.iter().all(|&raw| {
        let blas = accel(raw).into_retained();
        resident(blas.as_deref().map(ProtocolObject::from_ref))
    }) && scene.textures.iter().all(|&raw| {
        // SAFETY: as above.
        let texture = unsafe { MetalHandle::<MTLTextureKind>::new(raw) }.into_retained();
        resident(texture.as_deref().map(ProtocolObject::from_ref))
    }) && scene.buffers.iter().all(|&raw| {
        let buffer = buffer(raw).into_retained();
        resident(buffer.as_deref().map(ProtocolObject::from_ref))
    })
}
