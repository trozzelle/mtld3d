//! Scene replay: a captured world draw drawn again, depth only, from another view.
//!
//! The sun's cascades and the camera depth pre-pass draw the frame's world
//! again from a [`CapturedDraw`] (`scene_capture.rs`). A replay keeps the
//! draw's own vertex shader, constants and buffers, keyed for the scene
//! re-projection: the shader multiplies its final clip-space position by a
//! `reproject` matrix bound at `VS_REPROJECT_SLOT`, after the half-pixel
//! fixup, the user clip distances and the depth bias it computes from the
//! `pos_fixup` the replay binds. The caller picks both: a cascade binds the
//! camera-to-light matrix and a neutral `pos_fixup` ([`neutral_pos_fixup`]),
//! the camera pre-pass the identity and the draw's own captured one. The
//! replayed key has no user clip planes.
//!
//! The pipeline is the draw's depth-only sibling ([`FrameEncoder::caster_pipeline`]):
//! a `Depth32Float` target, one sample, no colour. Its pixel function is a
//! native caster shader: one that writes nothing for an opaque draw, and for
//! an alpha-tested one a sample of stage 0 at the draw's coordinate varying,
//! tested against `alpha_ref` with the draw's function, so the holes of a
//! leaf or a fence stay holes in the depth.

use mtld3d_core::{
    async_compile::{JobTicket, Resolution},
    depth_stencil_state::DepthStencilSnapshot,
    draw_data::{
        FixedVsSource, ProgrammableVsSource, PsSourceView, ShaderSourceFlags, VsSourceView,
    },
    dxso::FfVsKey,
    perf::PairShaderId,
};
use mtld3d_shared::{
    Command, MetalHandle, mtl::VS_REPROJECT_SLOT, mtl_handle::MTLFunctionKind, perf::ShaderTimings,
};
use mtld3d_types::{
    D3DCMP_EQUAL, D3DCMP_GREATER, D3DCMP_GREATEREQUAL, D3DCMP_LESS, D3DCMP_LESSEQUAL,
    D3DCMP_NOTEQUAL,
};
use rustc_hash::FxHashMap;

use super::{
    FrameEncoder, LOG_TARGET, StageLibHandles, compile_stage_library,
    scene_capture::{BytesRef, CapturedDraw, ReplayBinding, encode_captured_draw},
};

#[cfg(test)]
mod tests;

/// The identity re-projection: the camera's own clip space (the depth pre-pass).
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the depth pre-pass replays with it")
)]
pub const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// A `pos_fixup` for a view that is not the camera's.
///
/// `[x, y, z, w, depth_bias]` as `emit_draw` packs it: no half-pixel offset,
/// no pre-transformed depth clamp, one render pixel per logical pixel and
/// no constant depth bias.
static NEUTRAL_POS_FIXUP: [f32; 5] = [0.0, 0.0, 0.0, 1.0, 0.0];

/// [`NEUTRAL_POS_FIXUP`] as the bytes a replay binds; static, so valid for every submission.
#[cfg_attr(not(test), expect(dead_code, reason = "the cascades bind it"))]
#[must_use]
pub fn neutral_pos_fixup() -> BytesRef {
    BytesRef {
        ptr: NEUTRAL_POS_FIXUP.as_ptr() as u64,
        len: u32::try_from(size_of_val(&NEUTRAL_POS_FIXUP)).expect("20 bytes"),
    }
}

/// How one replay ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the cascades and the depth pre-pass replay")
)]
pub enum Replayed {
    /// The draw is in the pass.
    Drawn,
    /// Its re-projected vertex shader or caster pipeline is still building
    /// under this ticket; nothing was encoded.
    Pending(JobTicket),
    /// It cannot be replayed; nothing was encoded.
    Failed,
}

/// The depth state of every replay: test `LESSEQUAL`, write, no stencil.
const CASTER_DEPTH: DepthStencilSnapshot = DepthStencilSnapshot {
    depth_enable: 1,
    depth_write: 1,
    depth_func: 4,
    ..DepthStencilSnapshot::inert()
};
const _: () = assert!(CASTER_DEPTH.depth_func as u32 == D3DCMP_LESSEQUAL);

/// The alpha test a caster pixel function makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AlphaCaster {
    /// The `texcoord<n>` varying stage 0 samples at.
    uv_attr: u8,
    /// `D3DCMP_*`, never `ALWAYS` or `NEVER`.
    func: u8,
    /// Stage 0 divides its coordinate by `.w` (`D3DTTFF_PROJECTED`).
    projected: bool,
}

/// The native caster pixel functions, built on first use and kept until teardown.
///
/// A build that failed is kept as `None`, so it costs one compile.
#[derive(Default)]
pub struct CasterShaders {
    opaque: Option<Option<StageLibHandles>>,
    alpha: FxHashMap<AlphaCaster, Option<StageLibHandles>>,
}

impl CasterShaders {
    /// Every built function and its library, emptying the set (teardown).
    pub fn take_all(&mut self) -> impl Iterator<Item = StageLibHandles> {
        self.opaque
            .take()
            .flatten()
            .into_iter()
            .chain(self.alpha.drain().filter_map(|(_, handles)| handles))
    }
}

/// The comparison `D3DCMP_*` `func` makes, `None` for `ALWAYS`, `NEVER` and unknown values.
const fn alpha_compare(func: u8) -> Option<&'static str> {
    match func as u32 {
        D3DCMP_LESS => Some("<"),
        D3DCMP_EQUAL => Some("=="),
        D3DCMP_LESSEQUAL => Some("<="),
        D3DCMP_GREATER => Some(">"),
        D3DCMP_NOTEQUAL => Some("!="),
        D3DCMP_GREATEREQUAL => Some(">="),
        _ => None,
    }
}

/// The entry name of a caster pixel function.
fn caster_entry(alpha: Option<AlphaCaster>) -> String {
    alpha.map_or_else(
        || "mtld3d_ps_caster_opaque".to_owned(),
        |a| {
            format!(
                "mtld3d_ps_caster_alpha_t{}_f{}{}",
                a.uv_attr,
                a.func,
                if a.projected { "_p" } else { "" }
            )
        },
    )
}

/// The MSL of a caster pixel function, `None` for an alpha function it cannot test.
///
/// An opaque caster writes nothing: the rasterized depth is the whole
/// result. An alpha caster samples stage 0 (`s0`, `samp0`, a 2D texture)
/// at its `texcoord<n>` varying, which Metal links to the vertex output of
/// that name, and discards what the draw's alpha test rejects against
/// `alpha_ref` (fragment slot 14, as the world's pixel stages read it).
fn caster_ps_msl(alpha: Option<AlphaCaster>) -> Option<String> {
    let entry = caster_entry(alpha);
    let Some(alpha) = alpha else {
        return Some(format!(
            "#include <metal_stdlib>\nusing namespace metal;\n\nfragment void {entry}() {{}}\n"
        ));
    };
    let cmp = alpha_compare(alpha.func)?;
    let uv = alpha.uv_attr;
    let coord = if alpha.projected {
        format!("in.texcoord{uv}.xy / in.texcoord{uv}.w")
    } else {
        format!("in.texcoord{uv}.xy")
    };
    Some(format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         \n\
         struct CasterIn {{\n    float4 texcoord{uv};\n}};\n\
         \n\
         fragment void {entry}(\n    \
             CasterIn in [[stage_in]],\n    \
             texture2d<float> s0 [[texture(0)]],\n    \
             sampler samp0 [[sampler(0)]],\n    \
             constant float &alpha_ref [[buffer(14)]]) {{\n    \
             if (!(s0.sample(samp0, {coord}).a {cmp} alpha_ref)) discard_fragment();\n\
         }}\n"
    ))
}

/// The diagnostics identity of a caster pixel function.
fn caster_ps_id(alpha: Option<AlphaCaster>) -> PairShaderId {
    let hash = alpha.map_or(0xCA57_0000, |a| {
        0xCA57_0000
            | 1 << 24
            | u64::from(a.uv_attr) << 16
            | u64::from(a.func) << 8
            | u64::from(a.projected)
    });
    PairShaderId {
        is_programmable: false,
        hash,
    }
}

/// Encode `d` into `out` as a depth-only caster, re-projected by `reproject`.
///
/// `reproject` is column-major, as MSL reads a `float4x4` (`reproject[c]` is
/// column `c`); it multiplies the final clip-space position the draw's
/// vertex shader computed against `pos_fixup` (see the module docs).
/// `depth_bias` is `setDepthBias(constant, slope)`. Binds a `LESSEQUAL`
/// depth-writing state, the draw's cull mode and every buffer and constant
/// the draw read; the pass, its viewport and its `Depth32Float` attachment
/// are the caller's. Nothing is encoded unless the answer is
/// [`Replayed::Drawn`]: a vertex shader or pipeline still building answers
/// `Pending` (a later frame finds it built), a draw no caster can draw
/// `Failed`.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the cascades and the depth pre-pass replay")
)]
pub fn replay_draw(
    enc: &mut FrameEncoder,
    out: &mut Vec<Command>,
    d: &CapturedDraw,
    reproject: &[[f32; 4]; 4],
    pos_fixup: BytesRef,
    depth_bias: (f32, f32),
) -> Replayed {
    let alpha = match d.alpha {
        None => None,
        Some(alpha) => {
            let Some(alpha) = alpha_caster(d, alpha.func, alpha.uv_attr) else {
                return Replayed::Failed;
            };
            Some(alpha)
        }
    };
    let (vs_fn, vs_id) = match enc.caster_vs(d.vs.as_ref()) {
        Resolution::Ready(vs) => vs,
        Resolution::Pending(ticket) => return Replayed::Pending(ticket),
        Resolution::Failed => return Replayed::Failed,
    };
    let Some(ps_fn) = enc.caster_ps(alpha) else {
        return Replayed::Failed;
    };
    let pipeline = match enc.caster_pipeline(
        &d.snapshot,
        d.vertex_attrs(),
        (vs_fn, vs_id),
        (ps_fn, caster_ps_id(alpha)),
    ) {
        Resolution::Ready(pipeline) => pipeline,
        Resolution::Pending(ticket) => return Replayed::Pending(ticket),
        Resolution::Failed => return Replayed::Failed,
    };
    let depth_stencil = enc.get_or_create_depth_stencil(&CASTER_DEPTH, false);
    if depth_stencil == 0 {
        return Replayed::Failed;
    }
    let mut matrix = [0u8; 64];
    for (bytes, value) in matrix
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(reproject.as_flattened())
    {
        *bytes = value.to_le_bytes();
    }
    let reproject = BytesRef {
        ptr: enc.alloc_scratch(&matrix),
        len: 64,
    };
    encode_captured_draw(
        out,
        d,
        &ReplayBinding {
            pipeline: pipeline.raw(),
            depth_stencil,
            cull: d.cull,
            depth_bias,
            pos_fixup,
            extra_vs: &[(VS_REPROJECT_SLOT, reproject)],
        },
    );
    Replayed::Drawn
}

/// The caster alpha test of an alpha-tested `d`, `None` (said once) when no caster can make it.
///
/// The caster samples stage 0 as a plain 2D texture; a cube, volume, depth
/// or gather binding there is not replayed.
fn alpha_caster(d: &CapturedDraw, func: u8, uv_attr: u8) -> Option<AlphaCaster> {
    let v = &d.variant;
    if (v.volume_sampler_mask | v.cube_sampler_mask | v.depth_sampler_mask | v.fetch4_mask) & 1 != 0
        || alpha_compare(func).is_none()
    {
        mtld3d_shared::log_once_info!(target: LOG_TARGET,
            "scene: an alpha-tested world draw's stage 0 is not a plain 2D sample; not replayed");
        return None;
    }
    let projected = match d.ps.as_ref() {
        PsSourceView::FixedFunction(source) => source.key.tt_projected_mask & 1 != 0,
        PsSourceView::Programmable(_) => v.tt_projected_mask & 1 != 0,
    };
    Some(AlphaCaster {
        uv_attr,
        func,
        projected,
    })
}

impl FrameEncoder {
    /// The draw's vertex shader keyed for the scene re-projection, with no user clip planes.
    ///
    /// Resolved through the same library index and cache as every vertex
    /// shader, on a copy of the draw's source record with the re-projection
    /// set: `FfVsKey::SCENE_REPROJECT` or `ShaderSourceFlags::REPROJECT`.
    fn caster_vs(
        &mut self,
        vs: VsSourceView<'_>,
    ) -> Resolution<(MetalHandle<MTLFunctionKind>, PairShaderId)> {
        let ff;
        let programmable;
        let caster = match vs {
            VsSourceView::FixedFunction(source) => {
                ff = FixedVsSource {
                    key: FfVsKey {
                        scene: source.key.scene | FfVsKey::SCENE_REPROJECT,
                        clip_plane_count: 0,
                        ..source.key.clone()
                    },
                    max_row_count: source.max_row_count,
                    reserved: [0; 2],
                };
                VsSourceView::FixedFunction(&ff)
            }
            VsSourceView::Programmable(source) => {
                programmable = ProgrammableVsSource {
                    vs_id: source.vs_id,
                    max_const_used: source.max_const_used,
                    provided_input_mask: source.provided_input_mask,
                    flags: ShaderSourceFlags::from_bits_retain(
                        source.flags.bits() | ShaderSourceFlags::REPROJECT.bits(),
                    ),
                    clip_plane_count: 0,
                    sampler_kinds: source.sampler_kinds,
                    reserved: [0; 7],
                };
                VsSourceView::Programmable(&programmable)
            }
        };
        match self.resolve_vs_library(caster) {
            Resolution::Ready(handles) => Resolution::Ready((
                handles.func,
                PairShaderId {
                    is_programmable: matches!(caster, VsSourceView::Programmable(_)),
                    hash: caster.disk_key(),
                },
            )),
            Resolution::Pending(ticket) => Resolution::Pending(ticket),
            Resolution::Failed => Resolution::Failed,
        }
    }

    /// The caster pixel function for `alpha` (`None`: opaque), built now on first use.
    fn caster_ps(&mut self, alpha: Option<AlphaCaster>) -> Option<MetalHandle<MTLFunctionKind>> {
        let built = match alpha {
            None => self.casters.opaque,
            Some(key) => self.casters.alpha.get(&key).copied(),
        };
        if let Some(built) = built {
            return built.map(|handles| handles.func);
        }
        let handles = caster_ps_msl(alpha).and_then(|msl| {
            compile_stage_library(
                self.device_handle,
                &msl,
                &caster_entry(alpha),
                &mut ShaderTimings::new(),
            )
        });
        if handles.is_none() {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "scene: a caster pixel function failed to build; its draws are not replayed");
        }
        match alpha {
            None => self.casters.opaque = Some(handles),
            Some(key) => {
                self.casters.alpha.insert(key, handles);
            }
        }
        handles.map(|handles| handles.func)
    }
}
