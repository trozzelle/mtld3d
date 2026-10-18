//! DXSO (D3D9 shader bytecode) parsing and → MSL translation.
//!
//! Platform-independent pure Rust. The `parser` submodule turns a `&[u32]` of
//! DXSO bytecode into a structured `DxsoProgram`; the MSL emitters
//! (`emit_vs_programmable` / `emit_ps_programmable` for DXSO and `emit_vs_ff`
//! / `emit_ps_ff` for fixed-function) each produce a standalone translation
//! unit containing a single `mtld3d_vs` or `mtld3d_ps` entry point. A render
//! pipeline can mix any VS source with any PS source.

mod emit;
mod ff;
mod ir;
mod link;
mod opcode;
mod parser;

/// `log` target used by every call inside this module.
///
/// Consumed via the `log` crate macros (`warn!(target: LOG_TARGET, …)`) and
/// by `log_once_warn!`. No logger is registered here — mtld3d-core is an rlib
/// statically linked into `d3d9.dll`, which owns the `env_logger` init; tests
/// without a registered logger simply no-op.
pub const LOG_TARGET: &str = "mtld3d::dxso";

pub use emit::{
    COLOR_STYLE_SCENE_FLOAT, COLOR_STYLE_STORE_CLAMP, DEFAULT_PS_ENTRY, DEFAULT_VS_ENTRY,
    EmitError, SCENE_VIS_AO, SCENE_VIS_GI, SCENE_VIS_POINTS, SCENE_VIS_SUN, VariantFlags,
    VariantKey, VsPlans, VsSamplerKinds, VsScene, bound_sampler_type, declared_ps_samplers,
    emit_ps_programmable, emit_ps_programmable_named, emit_vs_programmable,
    emit_vs_programmable_named, explicit_lod_samplers, ff_vs_color_style, fog_blend_active,
    scene_vis_hash_tag,
};
pub use ff::{
    FF_PASSTHROUGH_ATTR_BASE, FF_VS_PALETTE_BASE_ROW, FfPsKey, FfStage, FfStageFlags,
    FfStageResult, FfVsFlags, FfVsKey, MAX_VERTEX_BLEND_MATRIX_INDEX, emit_ps_ff, emit_ps_ff_named,
    emit_vs_ff, emit_vs_ff_named, ff_attr_index_for_semantic, is_bump_env_op, tci_entry, tt_count,
    tt_projected,
};
pub use ir::{
    CmpFunc, DeclUsage, Declaration, DxsoError, DxsoProgram, Instruction, RegKind, ShaderType,
    TextureType,
};
pub use link::{LinkInputs, MAX_LINKED_INPUTS, SemanticSet, decl_passthrough_code};
pub use opcode::Opcode;
pub use parser::{operand_token_count, parse};

#[cfg(test)]
mod tests;
