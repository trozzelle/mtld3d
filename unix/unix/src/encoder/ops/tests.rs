//! `color.sceneFloat` routing: every copy into or out of a half-float back
//! buffer goes through a render quad keyed at `Rgba16Float`, and a BGRA8 back
//! buffer keeps the raw blit-encoder copy it always took.
//!
//! The handles are opaque values the recorded passes and blits name; nothing
//! here submits, so nothing dereferences them. The pipelines and the scratch
//! textures are real, built on the default device.

use std::sync::atomic::AtomicU64;

use mtld3d_core::{
    encoder_data::{ColorFillTarget, ColorRegionUpdate, StretchSurfaceFlags},
    encoder_records as records,
    passes::{BackbufferContents, FrameReset},
    render_scale::RenderScale,
    stretch_rect::StretchRegion,
};
use mtld3d_shared::{BlitCommandType, MetalHandle, mtl::PixelFormat};
use objc2_metal::MTLDevice as _;

use super::{StretchBlitParams, emit_stretch_rect_blit};
use crate::encoder::FrameEncoder;

const BB: u64 = 0xB000;
const RT: u64 = 0xA000;
const SIZE: (u32, u32) = (16, 8);

fn frame(enc: &mut FrameEncoder, format: PixelFormat) {
    enc.pass_state.reset_frame(&FrameReset {
        // SAFETY: an opaque value the recorded passes name but nothing dereferences.
        backbuffer: unsafe { MetalHandle::new(BB) },
        backbuffer_srgb: MetalHandle::NULL,
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: SIZE,
        backbuffer_format: format,
        backbuffer_contents: BackbufferContents::Preserved,
        depth_texture: MetalHandle::NULL,
        depth_size: (0, 0),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
}

/// Run `body` on a fresh encoder whose frame has a back buffer of `format`.
fn with_encoder(format: PixelFormat, body: impl FnOnce(&mut FrameEncoder)) {
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = crate::encoder::tests::post_encoder(
        &coherent,
        &retained,
        mtld3d_core::post::PostConfig::default(),
    ) else {
        return;
    };
    // The scratch textures a conversion stages through need a device record.
    let queue = enc
        .device
        .newCommandQueue()
        .expect("a queue on the default device");
    // SAFETY: the raw address carries this retain, which the record adopts.
    let queue = unsafe { MetalHandle::new(objc2::rc::Retained::into_raw(queue) as u64) };
    enc.record = Some(crate::metal::DeviceRecord::new(
        queue,
        None,
        mtld3d_shared::mtl::PresentDebugFlags::empty(),
    ));
    frame(&mut enc, format);
    body(&mut enc);
    enc.shutdown_cleanup();
}

/// The colour format of the last recorded pass on `handle`, if any.
fn pass_format(enc: &FrameEncoder, handle: u64) -> Option<PixelFormat> {
    enc.pass_state
        .passes()
        .iter()
        .rev()
        .find(|pass| pass.color_texture().raw() == handle)
        .map(mtld3d_core::passes::Pass::color_format)
}

fn pending_copies(enc: &mut FrameEncoder, cmd: BlitCommandType, dst: u64) -> usize {
    enc.pass_state
        .take_pending_leading_blits()
        .iter()
        .filter(|blit| blit.cmd == cmd as u32 && blit.dst_handle == dst)
        .count()
}

/// An X8R8G8B8 render-target surface record over `handle`.
fn surface(handle: u64) -> records::SurfaceRecord {
    records::SurfaceRecord {
        identity: records::SurfaceIdentityRecord {
            texture: records::TextureRecord::ZERO,
            handle,
            kind: 1,
            reserved: 0,
        },
        autogen_id: 0,
        msaa: 0,
        msaa_srgb: 0,
        width: SIZE.0,
        height: SIZE.1,
        texture_width: SIZE.0,
        texture_height: SIZE.1,
        scale: 100,
        format: mtld3d_types::D3DFMT_X8R8G8B8,
        mip_level: 0,
        slice: 0,
        pool: 0,
        flags: u32::from(StretchSurfaceFlags::IS_RENDER_TARGET.bits()),
        sample_count: 1,
        present: 0,
    }
}

fn stretch(enc: &mut FrameEncoder, src: u64, dst: u64) {
    let whole = StretchRegion {
        x: 0,
        y: 0,
        w: SIZE.0,
        h: SIZE.1,
    };
    emit_stretch_rect_blit(
        enc,
        &surface(src),
        &surface(dst),
        &StretchBlitParams {
            source: src,
            destination: dst,
            src_region: whole,
            dst_region: whole,
            mip_level: 0,
            render_quad: false,
            filter: mtld3d_types::D3DTEXF_POINT,
        },
    );
}

#[test]
fn stretch_rect_into_a_float_backbuffer_converts() {
    with_encoder(PixelFormat::Rgba16Float, |enc| {
        stretch(enc, RT, BB);
        assert_eq!(pass_format(enc, BB), Some(PixelFormat::Rgba16Float));
        assert_eq!(
            pending_copies(enc, BlitCommandType::CopyTextureToTexture, BB),
            0
        );
    });
    with_encoder(PixelFormat::Bgra8Unorm, |enc| {
        stretch(enc, RT, BB);
        assert_eq!(pass_format(enc, BB), None);
        assert_eq!(
            pending_copies(enc, BlitCommandType::CopyTextureToTexture, BB),
            1
        );
    });
}

#[test]
fn stretch_rect_out_of_a_float_backbuffer_converts() {
    with_encoder(PixelFormat::Rgba16Float, |enc| {
        stretch(enc, BB, RT);
        assert_eq!(pass_format(enc, RT), Some(PixelFormat::Bgra8Unorm));
        assert_eq!(
            pending_copies(enc, BlitCommandType::CopyTextureToTexture, RT),
            0
        );
    });
    with_encoder(PixelFormat::Bgra8Unorm, |enc| {
        stretch(enc, BB, RT);
        assert_eq!(pass_format(enc, RT), None);
        assert_eq!(
            pending_copies(enc, BlitCommandType::CopyTextureToTexture, RT),
            1
        );
    });
}

fn update_surface(enc: &mut FrameEncoder) {
    let rows = vec![0x80u8; (SIZE.0 * SIZE.1 * 4) as usize];
    enc.update_color_region(
        BB,
        &ColorRegionUpdate {
            level: 0,
            format: PixelFormat::Bgra8Unorm,
            origin: (0, 0),
            extent: SIZE,
            logical: SIZE,
            texture: SIZE,
            scale: RenderScale::IDENTITY,
            bytes_per_row: SIZE.0 * 4,
        },
        &rows,
    );
}

#[test]
fn update_surface_into_a_float_backbuffer_converts() {
    with_encoder(PixelFormat::Rgba16Float, |enc| {
        update_surface(enc);
        assert_eq!(pass_format(enc, BB), Some(PixelFormat::Rgba16Float));
        assert_eq!(
            pending_copies(enc, BlitCommandType::CopyBufferToTexture, BB),
            0
        );
    });
    with_encoder(PixelFormat::Bgra8Unorm, |enc| {
        update_surface(enc);
        assert_eq!(pass_format(enc, BB), None);
        assert_eq!(
            pending_copies(enc, BlitCommandType::CopyBufferToTexture, BB),
            1
        );
    });
}

/// The `UploadColor` write-back of a back-buffer `UnlockRect` / `ReleaseDC`.
fn write_back(enc: &mut FrameEncoder) -> usize {
    let rows = vec![0x80u8; (SIZE.0 * SIZE.1 * 4) as usize];
    enc.upload_color_write_back(BB, &rows, SIZE, SIZE.0 * 4);
    enc.frame_blit_commands
        .iter()
        .filter(|blit| {
            blit.cmd == BlitCommandType::CopyBufferToTexture as u32 && blit.dst_handle == BB
        })
        .count()
}

#[test]
fn write_back_into_a_float_backbuffer_converts() {
    with_encoder(PixelFormat::Rgba16Float, |enc| {
        assert_eq!(
            write_back(enc),
            0,
            "no raw copy into the half-float texture"
        );
        assert_eq!(pass_format(enc, BB), Some(PixelFormat::Rgba16Float));
    });
    with_encoder(PixelFormat::Bgra8Unorm, |enc| {
        assert_eq!(write_back(enc), 1);
        assert_eq!(pass_format(enc, BB), None);
    });
}

fn color_fill(enc: &mut FrameEncoder) {
    enc.color_fill_target(&ColorFillTarget {
        // SAFETY: as `frame`'s back buffer.
        texture: unsafe { MetalHandle::new(BB) },
        logical_size: SIZE,
        texture_size: SIZE,
        format: PixelFormat::Bgra8Unorm,
        scale: RenderScale::IDENTITY,
        subresource: (0, 0),
        rect: (2, 2, 4, 4),
        rgba: (0, 0, 0, 0),
        msaa: MetalHandle::NULL,
        msaa_srgb: MetalHandle::NULL,
        sample_count: 1,
        regenerate_mipmaps: false,
    });
}

#[test]
fn color_fill_on_a_float_backbuffer_keys_its_real_format() {
    with_encoder(PixelFormat::Rgba16Float, |enc| {
        color_fill(enc);
        assert_eq!(pass_format(enc, BB), Some(PixelFormat::Rgba16Float));
    });
    with_encoder(PixelFormat::Bgra8Unorm, |enc| {
        color_fill(enc);
        assert_eq!(pass_format(enc, BB), Some(PixelFormat::Bgra8Unorm));
    });
}
