//! Unit tests for present routing, the geometry settle filter and the blit copy guard.
//!
//! `present_route` is checked across the geometries a present can produce: an enlargement
//! in both axes reaches `MetalFX` only if it exists, and anything else, equal extents
//! included, falls to the present shader. The settle tests pin the other half of the
//! decision, that a scaler is spent on a geometry that held still rather than on a ratio,
//! which is what keeps a window drag from building one scaler per frame.
//!
//! The present passes run on the GPU into Private, render-target-only destinations shaped
//! like a framebuffer-only drawable. The SDR pass without a ramp, and the exact copy the
//! `MetalFX` route ends with, at matching odd extents must give back the source byte for
//! byte. A ramp pass that cannot be encoded must leave the plain copy on the destination,
//! and the HDR pass-through must write the linear values of the source into an
//! `RGBA16Float` destination.
//!
//! `copy_texture_reject` is checked against each condition Metal validates on
//! `copyFromTexture:`, plus the pairs it accepts: an identical pair, a sub-rect inside a
//! mip level, and a linear format against its sRGB twin.
//!
//! The buffer/texture pair, `copy_buffer_to_texture_reject` and its readback mirror, is
//! checked on both of its bounds: the region against the addressed mip level, rounded up
//! to the block grid so a compressed level below one block still takes a whole block, and
//! the buffer against the rows and slices the strides walk.
//!
//! `first_pending` is the lookup behind a GPU-retire wait: it is checked to answer with the
//! smallest registered seq at or past the target on the waiting device, and never with
//! another device's entry, however the seqs of the two interleave.

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use mtld3d_shared::{
    ExtraColorDesc, MetalHandle, PassDescriptor,
    mtl::{BlockLayout, LoadAction, PixelFormat, PresentDebugFlags, StoreAction},
};
use objc2::{
    Message as _,
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
};
use objc2_foundation::{NSDictionary, NSError, NSLocalizedDescriptionKey, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLOrigin, MTLPixelFormat,
    MTLResource, MTLResourceOptions, MTLSharedEvent, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};

use super::{
    CopyBufferEndpoint, CopyEndpoint, CopyRegion, CopyRejectReason, DeviceRecord, EncodeContext,
    PendingCmdBuf, PresentGeometry, PresentRoute, SETTLED_PRESENTS, command_buffer_error,
    commit_registered, copy_buffer_to_texture_reject, copy_texture_reject,
    copy_texture_to_buffer_reject, encode_upload_cmd_buf, first_pending, geometry_settled,
    install_frame_handler, present_route, publish_idle_upload, readback_completed, retire_finished,
    submit_frame, submit_frame_with, wait_for_gpu_retire,
};
use crate::metal::{
    depth_transfer::PlanePool,
    submission::{FrameSubmission, RetirementCounter, SubmissionOutcome, SubmitDescription},
    transient::{SubmitStamp, UploadRing},
};

/// Two device identities that sort either side of each other's seqs.
const DEVICE_A: u64 = 0x1000;
const DEVICE_B: u64 = 0x2000;

/// A wait answers with its own device's smallest seq at or past the target.
///
/// Device B holds the seq A's wait would find first in a registry keyed by
/// seq alone.
#[test]
fn a_wait_answers_with_its_own_devices_next_seq() {
    let map: BTreeMap<(u64, u64), u32> = [
        ((DEVICE_A, 5), 15),
        ((DEVICE_B, 5), 25),
        ((DEVICE_B, 6), 26),
        ((DEVICE_A, 7), 17),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        first_pending(&map, DEVICE_A, 5).map(|(_, value)| value),
        Some(&15)
    );
    assert_eq!(
        first_pending(&map, DEVICE_A, 6).map(|(_, value)| value),
        Some(&17)
    );
    assert_eq!(
        first_pending(&map, DEVICE_A, 7).map(|(_, value)| value),
        Some(&17)
    );
    assert_eq!(
        first_pending(&map, DEVICE_A, 8).map(|(_, value)| value),
        Some(&17)
    );
    assert_eq!(
        first_pending(&map, DEVICE_B, 6).map(|(_, value)| value),
        Some(&26)
    );
}

/// Another device's entries never answer a wait, whatever seqs it holds.
#[test]
fn another_devices_entries_never_answer_a_wait() {
    let map: BTreeMap<(u64, u64), u32> = (1..=10).map(|seq| ((DEVICE_B, seq), 20)).collect();
    assert_eq!(first_pending(&map, DEVICE_A, 1), None);
    assert_eq!(first_pending(&map, DEVICE_A, 0), None);
    assert_eq!(
        first_pending(&map, DEVICE_B, 3).map(|(_, value)| value),
        Some(&20)
    );
    assert_eq!(
        first_pending(&map, DEVICE_B, 11).map(|(_, value)| value),
        Some(&20)
    );
}

/// Matching extents take the shader's exact copy whether or not `MetalFX` exists.
///
/// The drawable is framebuffer-only, so no blit can write it.
#[test]
fn equal_extents_route_to_the_shader() {
    assert_eq!(
        present_route((1920, 1080), (1920, 1080), true),
        PresentRoute::Shader
    );
    assert_eq!(
        present_route((1920, 1080), (1920, 1080), false),
        PresentRoute::Shader
    );
}

/// A larger drawable is `MetalFX`'s job, and the shader's without it.
#[test]
fn enlargement_routes_to_metalfx_when_present() {
    assert_eq!(
        present_route((1280, 720), (1920, 1080), true),
        PresentRoute::Upscale
    );
    assert_eq!(
        present_route((1280, 720), (1920, 1080), false),
        PresentRoute::Shader
    );
}

/// The scaler only enlarges, so a smaller drawable is always the shader's.
#[test]
fn minification_routes_to_the_shader() {
    assert_eq!(
        present_route((1920, 1080), (1280, 720), true),
        PresentRoute::Shader
    );
}

/// A drawable larger in one axis and smaller in the other is a stretch.
///
/// `MTLFXSpatialScaler` rejects the pair; the shader covers every pixel of
/// the drawable whatever the ratio.
#[test]
fn mixed_axis_change_routes_to_the_shader() {
    assert_eq!(
        present_route((1920, 720), (1280, 1080), true),
        PresentRoute::Shader
    );
    assert_eq!(
        present_route((1280, 1080), (1920, 720), true),
        PresentRoute::Shader
    );
}

/// One axis equal and the other larger still enlarges.
#[test]
fn single_axis_enlargement_routes_to_metalfx() {
    assert_eq!(
        present_route((1920, 1080), (1920, 1200), true),
        PresentRoute::Upscale
    );
}

/// Every `render.scale` the config accepts reaches the quality path.
///
/// The knob's range is `(0, 1.0]`, and the smallest enlargement a user can
/// ask for deliberately (`0.99`, a ratio of 1.0098) sits *inside* the band
/// a live window resize produces, which is why routing does not judge an
/// enlargement by its ratio. The back-buffer dimension mirrors
/// `RenderScale::dimension` in `mtld3d-core`, which lives in the other
/// workspace and is not a dependency here.
#[test]
fn every_render_scale_setting_routes_to_metalfx() {
    let dimension = |logical: usize, percent: usize| ((logical * percent + 50) / 100).max(1);
    for percent in 1..100 {
        let src = (dimension(2560, percent), dimension(1600, percent));
        assert_eq!(
            present_route(src, (2560, 1600), true),
            PresentRoute::Upscale,
            "render.scale = {percent}% must reach MetalFX"
        );
    }
}

/// A resize drag is filtered by never settling, not by its ratio.
///
/// These are geometries measured off a live drag. Each is a legitimate
/// `Upscale` on geometry alone; what keeps them off the scaler is that
/// the next present carries a different pair.
#[test]
fn a_resize_drag_is_filtered_by_settling_not_by_geometry() {
    let drag = [
        ((2452, 1532), (2454, 1534)),
        ((2474, 1546), (2484, 1552)),
        ((2400, 1498), (2408, 1504)),
    ];
    let mut seen = None;
    for (src, dst) in drag {
        assert_eq!(present_route(src, dst, true), PresentRoute::Upscale);
        assert!(
            !geometry_settled(&mut seen, PresentGeometry { src, dst }),
            "{src:?} → {dst:?} lasted one present and must not build a scaler"
        );
    }
}

/// A geometry settles only after holding still for consecutive presents.
#[test]
fn geometry_settles_after_holding_still() {
    let steady = PresentGeometry {
        src: (960, 540),
        dst: (1920, 1080),
    };
    let mut seen = None;
    let settled: Vec<bool> = (0..=SETTLED_PRESENTS)
        .map(|_| geometry_settled(&mut seen, steady))
        .collect();
    let expected: Vec<bool> = (1..=SETTLED_PRESENTS + 1)
        .map(|n| n >= SETTLED_PRESENTS)
        .collect();
    assert_eq!(settled, expected);
}

/// A window being dragged larger never settles, so it never builds a scaler.
///
/// Each frame of the drag is a different enlargement, which is exactly the
/// case that would otherwise leak one `MTLFXSpatialScaler` per frame.
#[test]
fn a_geometry_that_changes_every_present_never_settles() {
    let mut seen = None;
    for height in 0..64 {
        let dragging = PresentGeometry {
            src: (960, 540),
            dst: (1920, 1080 + height),
        };
        assert!(
            !geometry_settled(&mut seen, dragging),
            "a geometry seen once must not count as settled"
        );
    }
}

/// Settling restarts from scratch after the geometry changes.
#[test]
fn a_changed_geometry_restarts_the_count() {
    let before = PresentGeometry {
        src: (960, 540),
        dst: (1920, 1080),
    };
    let after = PresentGeometry {
        src: (960, 540),
        dst: (1920, 1200),
    };
    let mut seen = None;
    for _ in 0..SETTLED_PRESENTS * 2 {
        geometry_settled(&mut seen, before);
    }
    assert!(!geometry_settled(&mut seen, after));
    assert_eq!(
        geometry_settled(&mut seen, before),
        SETTLED_PRESENTS <= 2,
        "returning to a geometry starts its count over, it does not resume"
    );
}

/// A square `BGRA8` texture with one mip level, copied from its origin.
fn endpoint(size: usize) -> CopyEndpoint {
    CopyEndpoint {
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        sample_count: 1,
        width: size,
        height: size,
        depth: 1,
        level: 0,
        levels: 1,
        origin_x: 0,
        origin_y: 0,
    }
}

/// A pair that agrees on everything copies.
#[test]
fn a_matching_pair_is_accepted() {
    assert_eq!(
        copy_texture_reject(&endpoint(256), &endpoint(256), 256, 256, 1),
        None
    );
}

/// Differing sample counts are the case the RESZ resolve exists for.
#[test]
fn a_sample_count_change_is_rejected() {
    let mut src = endpoint(256);
    src.sample_count = 4;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 256, 256, 1),
        Some(CopyRejectReason::SampleCountMismatch)
    );
}

/// Two unrelated formats of the same size are still a reject.
#[test]
fn a_format_change_is_rejected() {
    let mut dst = endpoint(256);
    dst.pixel_format = MTLPixelFormat::RGBA8Unorm;
    assert_eq!(
        copy_texture_reject(&endpoint(256), &dst, 256, 256, 1),
        Some(CopyRejectReason::FormatMismatch)
    );
}

/// A linear format and its sRGB twin are two views of one base format.
#[test]
fn an_srgb_twin_is_accepted_in_either_direction() {
    let mut srgb = endpoint(256);
    srgb.pixel_format = MTLPixelFormat::BGRA8Unorm_sRGB;
    assert_eq!(
        copy_texture_reject(&endpoint(256), &srgb, 256, 256, 1),
        None
    );
    assert_eq!(
        copy_texture_reject(&srgb, &endpoint(256), 256, 256, 1),
        None
    );
}

/// The format check runs before the sample-count check, so it reports first.
#[test]
fn a_pair_that_differs_in_both_reports_the_format() {
    let mut src = endpoint(256);
    src.pixel_format = MTLPixelFormat::RGBA8Unorm;
    src.sample_count = 4;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 256, 256, 1),
        Some(CopyRejectReason::FormatMismatch)
    );
}

/// The region is bounds-checked against the source as well as the destination.
#[test]
fn a_region_leaving_either_end_is_rejected() {
    assert_eq!(
        copy_texture_reject(&endpoint(128), &endpoint(256), 256, 256, 1),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
    assert_eq!(
        copy_texture_reject(&endpoint(256), &endpoint(128), 256, 256, 1),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// The origin counts towards the bound, so a sub-rect can still overrun.
#[test]
fn an_offset_sub_rect_is_bounded_by_the_origin() {
    let mut src = endpoint(256);
    src.origin_x = 128;
    src.origin_y = 128;
    assert_eq!(copy_texture_reject(&src, &endpoint(256), 128, 128, 1), None);
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 129, 128, 1),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

/// Bounds are the addressed mip level's extent, not the base level's.
#[test]
fn the_bound_is_the_addressed_mip_level() {
    let mut src = endpoint(256);
    src.levels = 9;
    src.level = 2;
    let mut dst = endpoint(64);
    dst.levels = 7;
    assert_eq!(copy_texture_reject(&src, &dst, 64, 64, 1), None);
    assert_eq!(
        copy_texture_reject(&src, &dst, 65, 64, 1),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

/// A level past the end of either chain has no extent to copy through.
#[test]
fn a_missing_mip_level_is_rejected() {
    let mut src = endpoint(256);
    src.level = 1;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 1, 1, 1),
        Some(CopyRejectReason::SourceLevelMissing)
    );
    let mut dst = endpoint(256);
    dst.level = 1;
    assert_eq!(
        copy_texture_reject(&endpoint(256), &dst, 1, 1, 1),
        Some(CopyRejectReason::DestinationLevelMissing)
    );
}

/// Each reason keys its own one-shot warn.
#[test]
fn every_reject_reason_has_a_distinct_key() {
    let reasons = [
        CopyRejectReason::FormatMismatch,
        CopyRejectReason::SampleCountMismatch,
        CopyRejectReason::SourceLevelMissing,
        CopyRejectReason::DestinationLevelMissing,
        CopyRejectReason::SourceRegionOutOfBounds,
        CopyRejectReason::DestinationRegionOutOfBounds,
        CopyRejectReason::SourceBufferTooShort,
        CopyRejectReason::DestinationBufferTooShort,
    ];
    let mut keys: Vec<u64> = reasons.iter().map(|r| r.key()).collect();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), reasons.len());
    assert!(reasons.iter().all(|r| !r.as_str().is_empty()));
}

/// A `BGRA8` staging buffer holding `rows` tightly packed rows of `size` pixels.
fn upload_buffer(size: usize, rows: usize) -> CopyBufferEndpoint {
    CopyBufferEndpoint {
        length: size * 4 * rows,
        offset: 0,
        bytes_per_row: size * 4,
        bytes_per_image: size * 4 * rows,
    }
}

/// A single-slice region of `w` by `h` pixels.
const fn region(w: usize, h: usize) -> CopyRegion {
    CopyRegion {
        width: w,
        height: h,
        depth: 1,
    }
}

/// The block layout every `BGRA8` upload in these tests is measured in.
fn bgra8_block() -> BlockLayout {
    PixelFormat::Bgra8Unorm.block_layout()
}

/// A whole-level upload out of a buffer that holds exactly the level.
#[test]
fn a_matching_upload_is_accepted() {
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(256, 256),
            &endpoint(256),
            &region(256, 256),
            bgra8_block(),
        ),
        None
    );
}

/// An overhanging destination is rejected before Metal sees it.
///
/// This is the shape Metal reports as a destination origin plus a source width
/// exceeding the level's width.
#[test]
fn an_upload_overhanging_the_level_is_rejected() {
    let mut dst = endpoint(256);
    dst.origin_x = 128;
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(256, 256),
            &dst,
            &region(256, 256),
            bgra8_block(),
        ),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
    // Half the width still fits at that origin.
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(256, 256),
            &dst,
            &region(128, 256),
            bgra8_block(),
        ),
        None
    );
}

/// Bounds are the addressed level's extent, not the base level's.
#[test]
fn an_upload_is_bounded_by_the_addressed_level() {
    let mut dst = endpoint(256);
    dst.levels = 9;
    dst.level = 2;
    assert_eq!(
        copy_buffer_to_texture_reject(&upload_buffer(64, 64), &dst, &region(64, 64), bgra8_block(),),
        None
    );
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(128, 128),
            &dst,
            &region(65, 64),
            bgra8_block(),
        ),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// A level past the end of the chain has no extent to upload into.
#[test]
fn an_upload_to_a_missing_level_is_rejected() {
    let mut dst = endpoint(256);
    dst.level = 1;
    assert_eq!(
        copy_buffer_to_texture_reject(&upload_buffer(1, 1), &dst, &region(1, 1), bgra8_block(),),
        Some(CopyRejectReason::DestinationLevelMissing)
    );
}

/// One byte short of the rows the strides walk is a reject.
#[test]
fn a_short_source_buffer_is_rejected() {
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(256, 256),
            &endpoint(256),
            &region(256, 256),
            bgra8_block(),
        ),
        None
    );
    let mut short = upload_buffer(256, 256);
    short.length -= 1;
    assert_eq!(
        copy_buffer_to_texture_reject(&short, &endpoint(256), &region(256, 256), bgra8_block()),
        Some(CopyRejectReason::SourceBufferTooShort)
    );
}

/// The copy stops at the last row's own pixels, not at the end of its stride.
///
/// A sub-rect at the right edge of the last row of a staging buffer starts that
/// row part-way in, so charging it a whole stride would reject an upload whose
/// bytes the buffer holds.
#[test]
fn the_last_row_is_bounded_by_its_pixels_not_its_stride() {
    let mut src = upload_buffer(256, 256);
    src.offset = 255 * 256 * 4 + 128 * 4;
    let mut dst = endpoint(512);
    dst.origin_x = 128;
    dst.origin_y = 255;
    assert_eq!(
        copy_buffer_to_texture_reject(&src, &dst, &region(128, 1), bgra8_block()),
        None
    );
    // One pixel more than the 512 bytes left in the buffer.
    assert_eq!(
        copy_buffer_to_texture_reject(&src, &dst, &region(129, 1), bgra8_block()),
        Some(CopyRejectReason::SourceBufferTooShort)
    );
}

/// Compressed rows are counted in blocks, so a `BC1` level needs an eighth of the bytes.
#[test]
fn a_compressed_upload_counts_block_rows() {
    let block = PixelFormat::Bc1Rgba.block_layout();
    let mut dst = endpoint(128);
    dst.pixel_format = MTLPixelFormat::BC1_RGBA;
    // 32 block rows of 32 blocks, 8 bytes each.
    let exact = CopyBufferEndpoint {
        length: 256 * 32,
        offset: 0,
        bytes_per_row: 256,
        bytes_per_image: 256 * 32,
    };
    assert_eq!(
        copy_buffer_to_texture_reject(&exact, &dst, &region(128, 128), block),
        None
    );
    let short = CopyBufferEndpoint {
        length: 256 * 32 - 1,
        offset: 0,
        bytes_per_row: 256,
        bytes_per_image: 256 * 32,
    };
    assert_eq!(
        copy_buffer_to_texture_reject(&short, &dst, &region(128, 128), block),
        Some(CopyRejectReason::SourceBufferTooShort)
    );
}

/// A compressed level below one block still addresses a whole block.
#[test]
fn a_compressed_level_under_one_block_takes_a_whole_block() {
    let block = PixelFormat::Bc1Rgba.block_layout();
    let mut dst = endpoint(4);
    dst.pixel_format = MTLPixelFormat::BC1_RGBA;
    dst.levels = 3;
    dst.level = 2;
    let one_block = CopyBufferEndpoint {
        length: 8,
        offset: 0,
        bytes_per_row: 8,
        bytes_per_image: 8,
    };
    // The level is one pixel; the copy names the 4x4 block that holds it.
    assert_eq!(
        copy_buffer_to_texture_reject(&one_block, &dst, &region(4, 4), block),
        None
    );
    // Two blocks wide is past the level however the extent is rounded.
    assert_eq!(
        copy_buffer_to_texture_reject(&one_block, &dst, &region(8, 4), block),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// A volume upload reads one slice stride per slice past the first.
#[test]
fn a_volume_upload_counts_every_slice() {
    let mut dst = endpoint(32);
    dst.depth = 4;
    let box_region = CopyRegion {
        width: 32,
        height: 32,
        depth: 4,
    };
    let exact = CopyBufferEndpoint {
        length: 32 * 4 * 32 * 4,
        offset: 0,
        bytes_per_row: 32 * 4,
        bytes_per_image: 32 * 4 * 32,
    };
    assert_eq!(
        copy_buffer_to_texture_reject(&exact, &dst, &box_region, bgra8_block()),
        None
    );
    let three_slices = CopyBufferEndpoint {
        length: 32 * 4 * 32 * 3,
        offset: 0,
        bytes_per_row: 32 * 4,
        bytes_per_image: 32 * 4 * 32,
    };
    assert_eq!(
        copy_buffer_to_texture_reject(&three_slices, &dst, &box_region, bgra8_block()),
        Some(CopyRejectReason::SourceBufferTooShort)
    );
    // The same box against a texture that has one slice.
    assert_eq!(
        copy_buffer_to_texture_reject(&exact, &endpoint(32), &box_region, bgra8_block()),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// The readback mirror reports the texture as the source and the buffer as the destination.
#[test]
fn a_readback_reports_the_ends_the_other_way_round() {
    assert_eq!(
        copy_texture_to_buffer_reject(
            &endpoint(256),
            &upload_buffer(256, 256),
            &region(256, 256),
            bgra8_block(),
        ),
        None
    );
    // The caller asks for more pixels than the source holds, which is what a
    // declined resolve of a render-resolution frame leaves behind.
    assert_eq!(
        copy_texture_to_buffer_reject(
            &endpoint(128),
            &upload_buffer(256, 256),
            &region(256, 256),
            bgra8_block(),
        ),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
    assert_eq!(
        copy_texture_to_buffer_reject(
            &endpoint(256),
            &upload_buffer(256, 255),
            &region(256, 256),
            bgra8_block(),
        ),
        Some(CopyRejectReason::DestinationBufferTooShort)
    );
}

/// A missing final sequence still waits for earlier committed GPU work.
#[test]
fn missing_final_submit_waits_for_earlier_work() {
    let queue = test_queue();
    let event = queue.device().newSharedEvent().expect("shared event");
    let cb = queue.commandBuffer().expect("command buffer");
    cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
    let coherent = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let counter = atomic_address(&coherent);
    let record = test_record(&queue);
    record
        .pending()
        .lock()
        .insert((counter, 1), PendingCmdBuf(cb.clone()));
    cb.commit();
    let (done, watchdog) = release_event_after_wait(event);
    wait_for_gpu_retire(record.pending(), 2, counter, 0, atomic_address(&failed));
    let status_at_return = cb.status();
    let retired_at_return = coherent.load(Ordering::Acquire);
    let _ = done.send(());
    watchdog.join().unwrap();
    cb.waitUntilCompleted();
    record.pending().lock().remove(&(counter, 1));
    assert_eq!(status_at_return, MTLCommandBufferStatus::Completed);
    assert_eq!(
        retired_at_return, 1,
        "the missing sequence was not submitted"
    );
}

/// Both failure positions protect committed work and publish failure before retirement.
#[test]
fn cpu_submit_failure_drains_draw_and_upload_handlers() {
    cpu_submit_failure_drain(false);
}

#[test]
fn cpu_submit_failure_after_upload_commit_drains_handlers() {
    cpu_submit_failure_drain(true);
}

fn cpu_submit_failure_drain(upload_committed: bool) {
    let queue = test_queue();
    let event = queue.device().newSharedEvent().expect("shared event");
    let cb = queue.commandBuffer().expect("command buffer");
    if !upload_committed {
        cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
    }
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let draw_counter = atomic_address(&coherent);
    let upload_counter = atomic_address(&upload);
    let handler_done = Arc::new(AtomicBool::new(false));
    let handler_flag = Arc::clone(&handler_done);
    let handler = block2::RcBlock::new(
        move |_cb: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            handler_flag.store(true, Ordering::Release);
        },
    );
    // SAFETY: Metal retains the block, whose only capture owns its atomic.
    unsafe { cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler)) };
    let record = test_record(&queue);
    record
        .pending()
        .lock()
        .insert((draw_counter, 1), PendingCmdBuf(cb.clone()));
    cb.commit();
    if upload_committed {
        cb.waitUntilCompleted();
    }
    // In the upload case only the upload is parked. Waiting for the older
    // draw alone cannot satisfy the lifetime assertion.
    let upload_gate = queue.commandBuffer().expect("upload gate");
    if upload_committed {
        upload_gate.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
    }
    upload_gate.commit();
    let params = test_submit_params(&coherent, &upload, &failed);
    let texture = upload_test_texture(&queue);
    let upload_pass = upload_test_pass(&texture);
    let (done, watchdog) = release_event_after_wait(event);
    let mut committed_upload = None;
    let frame = FrameSubmission {
        description: &params,
        blits: &[],
        post_constants: &[],
        passes: &[],
    };
    let success = submit_frame_with(&record, &frame, |_| {
        if upload_committed {
            let upload_cb = encode_test_upload(
                &record,
                &queue,
                core::slice::from_ref(&upload_pass),
                &params,
            )
            .expect("an upload buffer");
            commit_registered(
                record.pending(),
                &upload_cb,
                params.upload_retirement.address(),
                params.submit_seq,
            );
            // The upload at seq 2 is parked behind its own gate. Register
            // a draw at the same seq to prove the counter identities do not collide.
            record
                .pending()
                .lock()
                .insert((draw_counter, 2), PendingCmdBuf(cb.clone()));
            let pending = record.pending().lock();
            assert!(pending.contains_key(&(draw_counter, 2)));
            committed_upload = pending.get(&(upload_counter, 2)).map(|cb| cb.0.clone());
            drop(pending);
            assert!(committed_upload.is_some());
        }
        false
    });
    let upload_status_at_return = committed_upload.as_ref().map(|cb| cb.status());
    let status_at_return = cb.status();
    let handler_at_return = handler_done.load(Ordering::Acquire);
    let counters_at_return = (
        coherent.load(Ordering::Acquire),
        upload.load(Ordering::Acquire),
        failed.load(Ordering::Acquire),
    );
    let _ = done.send(());
    watchdog.join().unwrap();
    cb.waitUntilCompleted();
    // Cleanup precedes assertions so the pre-fix reproduction frees no live sink.
    let pending_upload = record
        .pending()
        .lock()
        .get(&(upload_counter, 2))
        .map(|cb| cb.0.clone());
    if let Some(upload_cb) = pending_upload {
        upload_cb.waitUntilCompleted();
    }
    upload_gate.waitUntilCompleted();
    record
        .pending()
        .lock()
        .retain(|&(counter, _), _| counter != draw_counter && counter != upload_counter);
    assert!(!success.success);
    assert_eq!(status_at_return, MTLCommandBufferStatus::Completed);
    assert!(handler_at_return, "callback sinks must be unused on return");
    if upload_committed {
        assert_eq!(
            upload_status_at_return,
            Some(MTLCommandBufferStatus::Completed)
        );
    }
    assert_eq!(counters_at_return, (2, 2, 2));
}

/// Encode `passes` into an upload command buffer with a ring of the test's own.
fn encode_test_upload(
    record: &Arc<DeviceRecord>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    passes: &[PassDescriptor],
    params: &SubmitDescription,
) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
    let device = queue.device();
    let mut ring = UploadRing::default();
    let mut planes = PlanePool::default();
    let mut ctx = EncodeContext {
        device: &device,
        stamp: SubmitStamp::new(params).upload(),
        ring: &mut ring,
        planes: &mut planes,
        post: &[],
    };
    encode_upload_cmd_buf(
        record,
        queue,
        &[],
        passes,
        params,
        &mut SubmissionOutcome::new(),
        &mut ctx,
    )
}

fn test_queue() -> Retained<ProtocolObject<dyn MTLCommandQueue>> {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    device.newCommandQueue().expect("command queue")
}

fn atomic_address(value: &AtomicU64) -> u64 {
    core::ptr::from_ref(value) as u64
}

/// A record around the test's queue, as `create_command_queue` would build one.
///
/// Takes a retain of its own, which the record releases when it drops, so the
/// caller's `Retained` stays valid. No presenter thread: the submissions here
/// carry no present.
fn test_record(queue: &ProtocolObject<dyn MTLCommandQueue>) -> Arc<DeviceRecord> {
    let retained = queue.retain();
    // SAFETY: `Retained::into_raw` transfers this test's extra retain into the
    // handle, which the record's `Drop` releases.
    let handle = unsafe { MetalHandle::new(Retained::into_raw(retained) as u64) };
    DeviceRecord::new(handle, None, PresentDebugFlags::empty())
}

fn test_submit_params(
    coherent: &AtomicU64,
    upload: &AtomicU64,
    failed: &AtomicU64,
) -> SubmitDescription {
    SubmitDescription {
        blit_commands_need_encoder: false,
        upload_pass_count: 0,
        present_layer: MetalHandle::NULL,
        present_texture: MetalHandle::NULL,
        present_view: MetalHandle::NULL,
        present_scene_linear: false,
        submit_seq: 2,
        // SAFETY: each test drains its callbacks before these counters leave scope.
        draw_retirement: unsafe { RetirementCounter::from_address(atomic_address(coherent)) },
        // SAFETY: each test drains its callbacks before these counters leave scope.
        upload_retirement: unsafe { RetirementCounter::from_address(atomic_address(upload)) },
        // SAFETY: each test drains its callbacks before these counters leave scope.
        failed_submission: unsafe { RetirementCounter::from_address(atomic_address(failed)) },
    }
}

/// The watchdog also releases immediately when a broken wait returns early.
fn release_event_after_wait(
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
) -> (mpsc::Sender<()>, thread::JoinHandle<()>) {
    let (done, waiting) = mpsc::channel();
    let watchdog = thread::spawn(move || {
        let _ = waiting.recv_timeout(Duration::from_millis(100));
        event.setSignaledValue(1);
    });
    (done, watchdog)
}

/// Volume depth shrinks per mip and is checked at both ends independently.
#[test]
fn texture_copy_depth_is_bounded_by_each_mip() {
    let mut src = endpoint(8);
    src.depth = 8;
    src.levels = 4;
    src.level = 1;
    let mut dst = endpoint(4);
    dst.depth = 4;
    dst.levels = 3;
    assert_eq!(copy_texture_reject(&src, &dst, 4, 4, 4), None);
    assert_eq!(
        copy_texture_reject(&src, &dst, 4, 4, 5),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
    dst.depth = 2;
    assert_eq!(
        copy_texture_reject(&src, &dst, 4, 4, 3),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
    assert_eq!(copy_texture_reject(&src, &dst, 4, 4, 2), None);
    src.level = 3;
    dst.level = 2;
    assert_eq!(copy_texture_reject(&src, &dst, 1, 1, 1), None);
    assert_eq!(
        copy_texture_reject(&src, &dst, 1, 1, 2),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

/// Ordinary 2D and cube mip copies address one depth plane.
#[test]
fn flat_texture_copies_reject_multiple_depth_planes() {
    assert_eq!(
        copy_texture_reject(&endpoint(4), &endpoint(4), 4, 4, 1),
        None
    );
    assert_eq!(
        copy_texture_reject(&endpoint(4), &endpoint(4), 4, 4, 2),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

#[test]
fn upload_prefix_finishes_before_its_retirement_signal() {
    let queue = test_queue();
    let texture = upload_test_texture(&queue);
    let pass = upload_test_pass(&texture);
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let params = test_submit_params(&coherent, &upload, &failed);
    let record = test_record(&queue);
    let upload_cb =
        encode_test_upload(&record, &queue, &[pass], &params).expect("an upload buffer");
    commit_registered(
        record.pending(),
        &upload_cb,
        params.upload_retirement.address(),
        params.submit_seq,
    );
    wait_for_gpu_retire(
        record.pending(),
        params.submit_seq,
        atomic_address(&upload),
        0,
        atomic_address(&failed),
    );
    assert_eq!(upload.load(Ordering::Acquire), params.submit_seq);
    assert_eq!(coherent.load(Ordering::Acquire), 0);
    assert_eq!(failed.load(Ordering::Acquire), 0);
    assert_eq!(upload_test_pixel(&queue, &texture), [0, 0, 255, 255]);
}

#[test]
fn frame_submission_accepts_empty_no_upload_and_all_upload_prefixes() {
    for (pass_count, upload_count, separate_upload) in
        [(0, 0, true), (1, 0, true), (1, 1, true), (1, 1, false)]
    {
        let queue = test_queue();
        let texture = upload_test_texture(&queue);
        let pass = upload_test_pass(&texture);
        let coherent = AtomicU64::new(0);
        let upload = AtomicU64::new(0);
        let failed = AtomicU64::new(0);
        let mut params = test_submit_params(&coherent, &upload, &failed);
        params.upload_pass_count = upload_count;
        if !separate_upload {
            params.upload_retirement = RetirementCounter::NONE;
        }
        let passes = if pass_count == 0 {
            &[][..]
        } else {
            core::slice::from_ref(&pass)
        };
        let frame = FrameSubmission {
            description: &params,
            blits: &[],
            post_constants: &[],
            passes,
        };
        let record = test_record(&queue);
        assert!(submit_frame(&record, &frame).success);
        wait_for_gpu_retire(
            record.pending(),
            params.submit_seq,
            atomic_address(&coherent),
            params.upload_retirement.address(),
            atomic_address(&failed),
        );
        assert_eq!(coherent.load(Ordering::Acquire), params.submit_seq);
        assert_eq!(failed.load(Ordering::Acquire), 0);
        // A frame without an upload buffer is published on the upload counter
        // too, since none is in flight.
        assert_eq!(
            upload.load(Ordering::Acquire),
            if separate_upload {
                params.submit_seq
            } else {
                0
            }
        );
        if pass_count != 0 {
            assert_eq!(upload_test_pixel(&queue, &texture), [0, 0, 255, 255]);
        }
    }
}

#[test]
fn frame_submission_rejects_invalid_upload_prefix() {
    for (pass_count, upload_count) in [(0, 1), (1, 2)] {
        let queue = test_queue();
        let texture = upload_test_texture(&queue);
        let pass = upload_test_pass(&texture);
        let coherent = AtomicU64::new(0);
        let upload = AtomicU64::new(0);
        let failed = AtomicU64::new(0);
        let mut params = test_submit_params(&coherent, &upload, &failed);
        params.upload_pass_count = upload_count;
        let frame = FrameSubmission {
            description: &params,
            blits: &[],
            post_constants: &[],
            passes: if pass_count == 0 {
                &[]
            } else {
                core::slice::from_ref(&pass)
            },
        };
        let record = test_record(&queue);
        let outcome = submit_frame(&record, &frame);
        assert!(!outcome.success);
        assert_eq!(outcome.drawable_wait_ns, 0);
        assert_eq!(outcome.present_wait_ns, 0);
        assert!(outcome.snapshot_flags.is_empty());
        assert_eq!(failed.load(Ordering::Acquire), params.submit_seq);
        assert_eq!(coherent.load(Ordering::Acquire), params.submit_seq);
        assert_eq!(upload.load(Ordering::Acquire), params.submit_seq);
    }
}

fn upload_test_texture(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    // SAFETY: a 1x1 non-mipmapped BGRA8 texture is a valid 2D descriptor.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::BGRA8Unorm,
            1,
            1,
            false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(MTLTextureUsage::RenderTarget);
    let texture = queue
        .device()
        .newTextureWithDescriptor(&desc)
        .expect("upload texture");
    texture.setLabel(Some(&NSString::from_str("mtld3d-test-upload-texture")));
    texture
}

fn upload_test_pass(texture: &ProtocolObject<dyn MTLTexture>) -> PassDescriptor {
    PassDescriptor {
        // SAFETY: the test owns this texture until the command buffer retires.
        color_texture: unsafe { MetalHandle::new(core::ptr::from_ref(texture) as u64) },
        color_resolve_texture: MetalHandle::NULL,
        depth_texture: MetalHandle::NULL,
        commands_ptr: 0,
        visibility_result_buffer: MetalHandle::NULL,
        leading_blits_ptr: 0,
        color_load_action: LoadAction::Clear,
        color_store_action: StoreAction::Store,
        clear_r: 1.0f32.to_bits(),
        clear_g: 0,
        clear_b: 0,
        clear_a: 1.0f32.to_bits(),
        depth_load_action: LoadAction::DontCare,
        depth_store_action: StoreAction::DontCare,
        depth_clear_value: 0,
        stencil_load_action: LoadAction::DontCare,
        stencil_store_action: StoreAction::DontCare,
        stencil_clear_value: 0,
        command_count: 0,
        leading_blits_count: 0,
        pass_flags: PassDescriptor::pack_flags(false, 0, 0, 0, 0),
        reserved: 0,
        extra_color: [ExtraColorDesc::NONE; 3],
    }
}

fn upload_test_pixel(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
) -> [u8; 4] {
    let buffer = queue
        .device()
        .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
        .expect("readback buffer");
    buffer.setLabel(Some(&NSString::from_str("mtld3d-test-upload-pixel")));
    let cmd = queue.commandBuffer().expect("readback command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-upload-readback")));
    let blit = cmd.blitCommandEncoder().expect("readback blit");
    blit.setLabel(Some(&NSString::from_str("mtld3d-test-upload-copy")));
    // SAFETY: the source is 1x1, the destination holds a 256-byte aligned
    // row, and both resources stay alive until the copy completes.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: 1, height: 1, depth: 1 }, &buffer, 0, 256, 256,
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the shared buffer holds the completed four-byte pixel copy.
    unsafe { buffer.contents().cast::<[u8; 4]>().read() }
}

/// Completion is success only for the successful terminal state.
#[test]
fn readback_completion_rejects_failed_and_unfinished_states() {
    for status in [
        MTLCommandBufferStatus::NotEnqueued,
        MTLCommandBufferStatus::Enqueued,
        MTLCommandBufferStatus::Committed,
        MTLCommandBufferStatus::Scheduled,
        MTLCommandBufferStatus::Error,
    ] {
        assert!(!readback_completed(status, || None), "{status:?}");
    }
    assert!(readback_completed(
        MTLCommandBufferStatus::Completed,
        || { panic!("a successful readback must not fetch error diagnostics") }
    ));
}

/// A real `NSError` crosses the same completion decision without any GPU submission.
#[test]
fn readback_completion_preserves_driver_error_details() {
    let description = "Caused GPU Hang Error (00000003:kIOAccelCommandBufferCallbackErrorHang)";
    let value = NSString::from_str(description);
    // SAFETY: Foundation exports this immutable NSString key.
    let key = unsafe { NSLocalizedDescriptionKey };
    let info = NSDictionary::from_slices(&[key], &[AsRef::<AnyObject>::as_ref(&*value)]);
    // SAFETY: the user-info dictionary contains an NSString under the description key.
    let error = unsafe {
        NSError::errorWithDomain_code_userInfo(
            &NSString::from_str("MTLCommandBufferErrorDomain"),
            2,
            Some(&info),
        )
    };
    assert_eq!(
        command_buffer_error(Some(&error)),
        (2, description.to_owned())
    );
    assert_eq!(command_buffer_error(None), (0, String::new()));
    let mut inspected = false;
    assert!(!readback_completed(MTLCommandBufferStatus::Error, || {
        inspected = true;
        Some(error)
    }));
    assert!(inspected, "a failed readback must inspect its driver error");
}

/// A failed upscale after HDR preflight must still convert the original SDR frame.
#[test]
fn hdr_upscale_failure_reencodes_the_original_source() {
    use objc2_metal::{MTLClearColor, MTLLoadAction, MTLRenderPassDescriptor, MTLStoreAction};
    let queue = test_queue();
    let device = queue.device();
    if !crate::metal::upscale::is_available(&device) {
        eprintln!("MetalFX unsupported, skipping HDR fallback regression");
        return;
    }
    let cache = crate::metal::upscale::UpscaleCache::new();
    let src =
        crate::metal::upscale::scratch_target(&device, &cache, 32, 32, PixelFormat::Bgra8Unorm)
            .expect("source");
    let dst =
        crate::metal::upscale::scratch_target(&device, &cache, 64, 64, PixelFormat::Rgba16Float)
            .expect("destination");
    let cmd = queue.commandBuffer().expect("reference command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-reference")));
    let pass = MTLRenderPassDescriptor::new();
    // SAFETY: attachment zero exists on every render pass descriptor.
    let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
    color.setTexture(Some(&src));
    color.setLoadAction(MTLLoadAction::Clear);
    color.setStoreAction(MTLStoreAction::Store);
    color.setClearColor(MTLClearColor {
        red: 0.75,
        green: 0.5,
        blue: 0.25,
        alpha: 1.0,
    });
    let encoder = cmd
        .renderCommandEncoderWithDescriptor(&pass)
        .expect("clear encoder");
    encoder.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-source")));
    encoder.endEncoding();
    assert!(super::encode_hdr_present(
        &cmd,
        &src,
        &dst,
        super::HdrTarget {
            peak: 2.0,
            scene_linear: false
        },
        0
    ));
    cmd.commit();
    cmd.waitUntilCompleted();
    let expected = upload_test_pixel(&queue, &dst);
    assert_ne!(expected, [0; 4]);
    let cmd = queue.commandBuffer().expect("fallback command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-fallback")));
    super::clear_drawable(&cmd, &dst);
    let invoked = std::cell::Cell::new(false);
    assert!(super::encode_hdr_present_upscaled_with(
        &cmd,
        &cache,
        &src,
        &dst,
        super::HdrTarget {
            peak: 2.0,
            scene_linear: false
        },
        0,
        |_| {
            invoked.set(true);
            false
        }
    ));
    assert!(
        invoked.get(),
        "the failure must occur after successful preflight"
    );
    crate::metal::upscale::retire_evicted(&cmd, &cache);
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(upload_test_pixel(&queue, &dst), expected);
    crate::metal::upscale::retire(&cache);
}

/// Invalid second-plane uploads must not publish the already encoded depth plane.
#[test]
fn depth_plane_failure_aborts_the_pair_and_retry_retains_sources() {
    use mtld3d_shared::{BlitCommand, BlitCommandType, CopyBufferToTextureInfo};
    use objc2_metal::{MTLBlitOption, MTLEvent};

    let queue = test_queue();
    let device = queue.device();
    // SAFETY: a 2x1 private combined depth texture has one valid mip.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::Depth32Float_Stencil8,
            2,
            1,
            false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(MTLTextureUsage::RenderTarget);
    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("depth destination");
    texture.setLabel(Some(&NSString::from_str("mtld3d-test-depth-pair")));
    let color = upload_test_texture(&queue);
    let readback = device
        .newBufferWithLength_options(512, MTLResourceOptions::StorageModeShared)
        .expect("plane readback");
    readback.setLabel(Some(&NSString::from_str("mtld3d-test-depth-readback")));
    let read = || {
        let cb = queue.commandBuffer().expect("read command buffer");
        cb.setLabel(Some(&NSString::from_str("mtld3d-test-depth-read")));
        let blit = cb.blitCommandEncoder().expect("read encoder");
        blit.setLabel(Some(&NSString::from_str("mtld3d-test-depth-read-planes")));
        for (offset, plane) in [
            (0, MTLBlitOption::DepthFromDepthStencil),
            (256, MTLBlitOption::StencilFromDepthStencil),
        ] {
            // SAFETY: both 2x1 planes fit their disjoint aligned rows, retained until completion.
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                    &texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: 2, height: 1, depth: 1 },
                    &readback, offset, 256, 256, plane,
                );
            }
        }
        blit.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: completion made the shared depth row CPU-visible.
        let depths = unsafe { readback.contents().cast::<[f32; 2]>().read() };
        let masks = readback
            .contents()
            .cast::<u8>()
            .as_ptr()
            .wrapping_add(256)
            .cast::<[u8; 2]>();
        // SAFETY: the completed stencil copy occupies the second aligned row.
        let stencil = unsafe { masks.read() };
        (depths, stencil)
    };
    for (generation, depths, stencil) in
        [(0, [0.25f32, 0.75], [17u8, 239]), (1, [0.5, 1.0], [91, 75])]
    {
        let depth = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .expect("depth upload");
        depth.setLabel(Some(&NSString::from_str("mtld3d-test-depth-upload")));
        let masks = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .expect("stencil upload");
        masks.setLabel(Some(&NSString::from_str("mtld3d-test-stencil-upload")));
        // SAFETY: the fresh shared depth buffer holds two float values.
        unsafe { depth.contents().cast::<[f32; 2]>().write(depths) };
        // SAFETY: the fresh shared stencil buffer holds two stencil bytes.
        unsafe { masks.contents().cast::<[u8; 2]>().write(stencil) };
        let make = |buffer: &ProtocolObject<dyn MTLBuffer>, kind| {
            let mut cmd = BlitCommand::copy_buffer_to_texture(&CopyBufferToTextureInfo {
                buffer_handle: core::ptr::from_ref(buffer) as u64,
                buffer_offset: 0,
                bytes_per_row: 256,
                texture_handle: core::ptr::from_ref(&*texture) as u64,
                destination_slice: 0,
                mip_level: 0,
                origin_x: 0,
                origin_y: 0,
                region_w: 2,
                region_h: 1,
                depth: 1,
                bytes_per_image: 256,
            });
            cmd.cmd = kind as u32;
            cmd
        };
        let commands = [
            make(&depth, BlitCommandType::CopyBufferToDepth),
            make(&masks, BlitCommandType::CopyBufferToStencil),
        ];
        if generation != 0 {
            for bad in 0..6 {
                let mut rejected = commands;
                match bad {
                    0 => rejected[1].src_handle = 0,
                    1 => rejected[1].dst_handle = 0,
                    2 => rejected[1].mip_level = 1,
                    3 => rejected[1].src_offset = 256,
                    4 => rejected[1].region_w = 3,
                    _ => rejected[1].dst_handle = core::ptr::from_ref(&*color) as u64,
                }
                let coherent = AtomicU64::new(0);
                let upload = AtomicU64::new(0);
                let failed = AtomicU64::new(0);
                let mut params = test_submit_params(&coherent, &upload, &failed);
                params.blit_commands_need_encoder = true;
                let frame = FrameSubmission {
                    description: &params,
                    blits: &rejected,
                    post_constants: &[],
                    passes: &[],
                };
                let record = test_record(&queue);
                assert!(
                    !submit_frame(&record, &frame).success,
                    "invalid plane case {bad}"
                );
                assert_eq!(failed.load(Ordering::Acquire), params.submit_seq);
                assert_eq!(coherent.load(Ordering::Acquire), params.submit_seq);
                assert_eq!(read(), ([0.25, 0.75], [17, 239]));
            }
        }
        let event = device.newSharedEvent().expect("completion gate");
        event.setLabel(Some(&NSString::from_str("mtld3d-test-depth-gate")));
        let cb = queue.commandBuffer().expect("upload command buffer");
        cb.setLabel(Some(&NSString::from_str("mtld3d-test-depth-pair-upload")));
        cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
        let coherent = AtomicU64::new(0);
        let upload = AtomicU64::new(0);
        let failed = AtomicU64::new(0);
        let params = test_submit_params(&coherent, &upload, &failed);
        let mut ring = UploadRing::default();
        let mut planes = PlanePool::default();
        let mut ctx = EncodeContext {
            device: &device,
            stamp: SubmitStamp::new(&params),
            ring: &mut ring,
            planes: &mut planes,
            post: &[],
        };
        assert!(super::encode_leading_blits(
            &cb,
            &commands,
            true,
            super::BlitSite::FrameLeading,
            &mut ctx,
        ));
        drop(depth);
        drop(masks);
        cb.commit();
        event.setSignaledValue(1);
        cb.waitUntilCompleted();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(read(), (depths, stencil));
    }
}

/// A frame without uploads moves the upload counter only while no upload buffer is in flight.
#[test]
fn an_idle_upload_counter_follows_frames_that_upload_nothing() {
    let queue = test_queue();
    let record = test_record(&queue);
    let upload = AtomicU64::new(1);
    let counter = atomic_address(&upload);
    publish_idle_upload(record.pending(), counter, 4);
    assert_eq!(upload.load(Ordering::Acquire), 4);
    let cb = queue.commandBuffer().expect("buffer");
    record
        .pending()
        .lock()
        .insert((counter, 5), PendingCmdBuf(cb));
    publish_idle_upload(record.pending(), counter, 6);
    assert_eq!(upload.load(Ordering::Acquire), 4);
    record.pending().lock().remove(&(counter, 5));
    publish_idle_upload(record.pending(), counter, 7);
    assert_eq!(upload.load(Ordering::Acquire), 7);
}

/// A buffer released uncommitted runs its handlers and still moves neither counter.
///
/// An ended buffer sits registered at seq 1 on both counters, so a handler
/// that went on to retire would publish it: only the early return keeps the
/// counters at 0.
#[test]
fn a_buffer_released_uncommitted_never_advances_a_counter() {
    let queue = test_queue();
    let record = test_record(&queue);
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let params = test_submit_params(&coherent, &upload, &failed);
    for counter in [
        params.draw_retirement.address(),
        params.upload_retirement.address(),
    ] {
        let ended = queue.commandBuffer().expect("ended");
        ended.commit();
        ended.waitUntilCompleted();
        record
            .pending()
            .lock()
            .insert((counter, 1), PendingCmdBuf(ended));
    }
    let texture = upload_test_texture(&queue);
    let upload_pass = upload_test_pass(&texture);
    objc2::rc::autoreleasepool(|_| {
        let frame = queue.commandBuffer().expect("frame");
        install_frame_handler(&frame, &record, &params);
        let upload_cb = encode_test_upload(
            &record,
            &queue,
            core::slice::from_ref(&upload_pass),
            &params,
        )
        .expect("an upload buffer");
        drop((frame, upload_cb));
    });
    assert_eq!(coherent.load(Ordering::Acquire), 0);
    assert_eq!(upload.load(Ordering::Acquire), 0);
    assert_eq!(failed.load(Ordering::Acquire), 0);
    for counter in [
        params.draw_retirement.address(),
        params.upload_retirement.address(),
    ] {
        retire_finished(record.pending(), counter, 0, "test");
    }
    assert_eq!(coherent.load(Ordering::Acquire), 1);
    assert_eq!(upload.load(Ordering::Acquire), 1);
}

/// A counter moves over ended buffers from the oldest and stops at the first still running.
///
/// The later buffer ends first: the counter must not name it while the
/// earlier one still runs, and names both once that one ends.
#[test]
fn a_counter_never_passes_a_buffer_still_running() {
    let queue = test_queue();
    let record = test_record(&queue);
    let coherent = AtomicU64::new(0);
    let counter = atomic_address(&coherent);
    let event = queue.device().newSharedEvent().expect("gate");
    let first = queue.commandBuffer().expect("first");
    first.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
    first.commit();
    // A second queue, so the later buffer does not queue behind the parked one.
    let other = test_queue();
    let second = other.commandBuffer().expect("second");
    second.commit();
    second.waitUntilCompleted();
    {
        let mut map = record.pending().lock();
        map.insert((counter, 1), PendingCmdBuf(first.clone()));
        map.insert((counter, 2), PendingCmdBuf(second));
    }
    retire_finished(record.pending(), counter, 0, "test");
    assert_eq!(coherent.load(Ordering::Acquire), 0);
    event.setSignaledValue(1);
    first.waitUntilCompleted();
    retire_finished(record.pending(), counter, 0, "test");
    assert_eq!(coherent.load(Ordering::Acquire), 2);
    assert!(record.pending().lock().is_empty());
}

/// A wait publishes a frame without uploads that the submit-time publication had to skip.
///
/// The previous frame's upload buffer is still registered when a mid-frame
/// submission with no upload buffer commits, so that submission leaves the
/// upload counter behind. The retirement wait that follows proves every
/// upload buffer up to the draw it waited for, so the upload counter reaches
/// the same sequence and a gate on both counters does not stop at the older
/// upload.
#[test]
fn a_retirement_wait_publishes_uploads_through_the_draw_it_waited_for() {
    let queue = test_queue();
    let record = test_record(&queue);
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let (draw_counter, upload_counter) = (atomic_address(&coherent), atomic_address(&upload));
    for (counter, seq) in [(upload_counter, 1), (draw_counter, 1), (draw_counter, 2)] {
        let cb = queue.commandBuffer().expect("buffer");
        cb.commit();
        record
            .pending()
            .lock()
            .insert((counter, seq), PendingCmdBuf(cb));
    }
    publish_idle_upload(record.pending(), upload_counter, 2);
    assert_eq!(upload.load(Ordering::Acquire), 0);
    wait_for_gpu_retire(
        record.pending(),
        2,
        draw_counter,
        upload_counter,
        atomic_address(&failed),
    );
    assert_eq!(coherent.load(Ordering::Acquire), 2);
    assert_eq!(upload.load(Ordering::Acquire), 2);
    assert!(record.pending().lock().is_empty());
}

/// A submitted frame finishes for the latency bound when its draw buffer's handler retires it.
///
/// Drives the real path: the submission notes itself on the device's signal
/// before its draw buffer commits, and the completion handler wakes the
/// `Present` waiting for it.
#[test]
fn a_committed_frame_wakes_the_frame_latency_wait() {
    use std::{sync::mpsc, time::Duration};

    use crate::metal::FrameWait;

    let queue = test_queue();
    let record = test_record(&queue);
    let floor = AtomicU64::new(0);
    let coherent = Arc::new(AtomicU64::new(0));
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    // SAFETY: `floor` outlives every use of the signal in this test.
    record
        .present()
        .frame_signal()
        .attach(unsafe { RetirementCounter::from_address(atomic_address(&floor)) });
    let (sender, receiver) = mpsc::channel();
    let waiter = {
        let signal = Arc::clone(record.present().frame_signal());
        let coherent = Arc::clone(&coherent);
        std::thread::spawn(move || {
            // SAFETY: the waiter owns a reference to the atomic it reads.
            let draw = unsafe { RetirementCounter::from_address(atomic_address(&coherent)) };
            let _ = sender.send(signal.wait(draw, 2, crate::metal::PACED_PRESENT_CAP));
        })
    };
    assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
    let params = test_submit_params(&coherent, &upload, &failed);
    let frame = FrameSubmission {
        description: &params,
        blits: &[],
        post_constants: &[],
        passes: &[],
    };
    assert!(
        submit_frame(&record, &frame).success,
        "an empty frame submits"
    );
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(5)),
        Ok(FrameWait::Finished)
    );
    waiter.join().unwrap();
    assert_eq!(
        floor.load(Ordering::Acquire),
        2,
        "the floor follows the noted submission"
    );
    wait_for_gpu_retire(
        record.pending(),
        2,
        atomic_address(&coherent),
        atomic_address(&upload),
        atomic_address(&failed),
    );
}

/// Odd extents a present copies at, off any power-of-two texel grid.
const EXACT_EXTENTS: [(usize, usize); 2] = [(1366, 768), (1311, 847)];

/// The present shader's SDR pass without a ramp copies matching extents byte for byte.
///
/// At matching extents that pass is the whole present, into a drawable no
/// blit can write. A fragment that missed its texel centre would filter in a
/// neighbour, and random contents make every such blend visible.
#[test]
fn the_sdr_present_pass_copies_matching_extents_exactly() {
    let queue = test_queue();
    for (width, height) in EXACT_EXTENTS {
        let (src, expected) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, width, height);
        let dst = drawable_shaped(&queue, MTLPixelFormat::BGRA8Unorm, width, height);
        let cmd = queue.commandBuffer().expect("command buffer");
        cmd.setLabel(Some(&NSString::from_str("mtld3d-test-present-copy")));
        assert!(super::encode_present_copy(&cmd, &src, &dst, 0));
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(
            cmd.status(),
            MTLCommandBufferStatus::Completed,
            "{:?}",
            cmd.error()
        );
        assert_texels_equal(&texture_bytes(&queue, &dst), &expected, 4, width);
    }
}

/// The copy that carries a `MetalFX` output onto the drawable reproduces it in both formats.
///
/// The SDR drawable is `BGRA8Unorm` and the HDR one `RGBA16Float`, whose
/// random contents are finite normal halves, positive and negative, beyond
/// `1.0` included.
#[test]
fn the_exact_copy_reproduces_its_source_in_both_drawable_formats() {
    let queue = test_queue();
    for format in [MTLPixelFormat::BGRA8Unorm, MTLPixelFormat::RGBA16Float] {
        for (width, height) in EXACT_EXTENTS {
            let (src, expected) = random_source(&queue, format, width, height);
            let dst = drawable_shaped(&queue, format, width, height);
            let cmd = queue.commandBuffer().expect("command buffer");
            cmd.setLabel(Some(&NSString::from_str("mtld3d-test-exact-copy")));
            assert!(super::encode_exact_copy(&cmd, &src, &dst), "{format:?}");
            cmd.commit();
            cmd.waitUntilCompleted();
            assert_eq!(
                cmd.status(),
                MTLCommandBufferStatus::Completed,
                "{:?}",
                cmd.error()
            );
            assert_texels_equal(
                &texture_bytes(&queue, &dst),
                &expected,
                bytes_per_texel(format),
                width,
            );
        }
    }
}

/// A ramp pass that cannot be encoded leaves the plain copy on the destination.
///
/// The ramp pipeline here is a null handle, so its pass is skipped the way a
/// pass whose gamma table went away is. The plain pass must then write the
/// render-target-only destination, which no blit could.
#[test]
fn a_ramp_pass_that_fails_falls_back_to_the_plain_copy() {
    let queue = test_queue();
    let (width, height) = EXACT_EXTENTS[1];
    let (src, expected) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, width, height);
    let dst = drawable_shaped(&queue, MTLPixelFormat::BGRA8Unorm, width, height);
    let resources =
        crate::metal::present::ensure_resources(&queue.device()).expect("present pipelines");
    let cmd = queue.commandBuffer().expect("command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-ramp-fallback")));
    assert!(super::encode_present_pass(
        &cmd,
        &src,
        &dst,
        &super::PresentStage {
            plain: resources.copy,
            ramp: Some(0),
            uniforms: None,
            gamma_layer: 1,
        },
    ));
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    assert_texels_equal(&texture_bytes(&queue, &dst), &expected, 4, width);
}

/// The HDR pass-through writes the source's linear values into an `RGBA16Float` destination.
///
/// The destination is Private and render-target only, like the float
/// drawable of an EDR layer. One texel value covers the three branches of the
/// decode: zero, a mid-tone on the power curve, and full scale.
#[test]
fn the_hdr_passthrough_writes_linear_values_onto_a_float_destination() {
    let queue = test_queue();
    let (width, height) = EXACT_EXTENTS[1];
    // Blue, green, red, alpha of a `BGRA8Unorm` texel.
    let texel = [0_u8, 128, 255, 255];
    let src = uploaded(
        &queue,
        MTLPixelFormat::BGRA8Unorm,
        (width, height),
        &texel.repeat(width * height),
    );
    let dst = drawable_shaped(&queue, MTLPixelFormat::RGBA16Float, width, height);
    let cmd = queue.commandBuffer().expect("command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-passthrough")));
    assert!(super::encode_hdr_present(
        &cmd,
        &src,
        &dst,
        super::HdrTarget {
            peak: 1.0,
            scene_linear: false
        },
        0
    ));
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    let mid_tone = ((128.0_f32 / 255.0 + 0.055) / 1.055).powf(2.4);
    let expected = [1.0, mid_tone, 0.0, 1.0];
    let bytes = texture_bytes(&queue, &dst);
    for (index, texel) in bytes.as_chunks::<8>().0.iter().enumerate() {
        let channels = texel
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| half_to_f32(u16::from_le_bytes(*half)));
        for (channel, (read, wanted)) in channels.zip(expected).enumerate() {
            assert!(
                (read - wanted).abs() <= 1.0e-3,
                "texel {index} channel {channel} reads {read}, the linear value is {wanted}"
            );
        }
    }
}

/// An EDR present of a tone-mapped frame passes it through: the sRGB
/// decode alone, 1.0 on paper white and the headroom above it, whatever the
/// panel's peak. Any other frame keeps BT.2446 when the panel has headroom.
#[test]
fn edr_present_passthrough_selected_only_with_scene_float() {
    use crate::metal::present::GammaStage;
    assert!(super::hdr_stage(2.0, false) == GammaStage::Bt2446);
    assert!(super::hdr_stage(2.0, true) == GammaStage::Passthrough);
    assert!(super::hdr_stage(1.0, false) == GammaStage::Passthrough);
    assert!(super::hdr_stage(1.0, true) == GammaStage::Passthrough);

    let queue = test_queue();
    let (width, height) = EXACT_EXTENTS[1];
    // A tone-mapped texel above SDR white: 1.5 encoded, red; then green 0.5, blue 0, alpha 1.
    let texel: [u16; 4] = [0x3e00, 0x3800, 0x0000, 0x3c00];
    let bytes: Vec<u8> = texel
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect::<Vec<u8>>()
        .repeat(width * height);
    let src = uploaded(&queue, MTLPixelFormat::RGBA16Float, (width, height), &bytes);
    let present = |scene_linear: bool| {
        let dst = drawable_shaped(&queue, MTLPixelFormat::RGBA16Float, width, height);
        let cmd = queue.commandBuffer().expect("command buffer");
        cmd.setLabel(Some(&NSString::from_str("mtld3d-test-edr-scene-linear")));
        assert!(super::encode_hdr_present(
            &cmd,
            &src,
            &dst,
            super::HdrTarget {
                peak: 4.0,
                scene_linear,
            },
            0
        ));
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
        texture_bytes(&queue, &dst)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| half_to_f32(u16::from_le_bytes(*half)))
            .collect::<Vec<f32>>()
    };
    let eotf = |e: f32| ((e + 0.055) / 1.055).powf(2.4);
    let expected = [eotf(1.5), eotf(0.5), 0.0, 1.0];
    let linear = present(true);
    for (index, texel) in linear.as_chunks::<4>().0.iter().enumerate() {
        for (channel, (read, wanted)) in texel.iter().zip(expected).enumerate() {
            assert!(
                (read - wanted).abs() <= 2.0e-3 * wanted.max(1.0),
                "texel {index} channel {channel} reads {read}, the linear value is {wanted}"
            );
        }
    }
    assert_ne!(
        present(false),
        linear,
        "without the tone map a panel with headroom gets BT.2446"
    );
}

/// A finite IEEE half's value.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = bits & 0x3ff;
    if exponent == 0 {
        return sign * f32::from(mantissa) / 16_777_216.0;
    }
    let normal = f32::from_bits(((exponent + 112) << 23) | (u32::from(mantissa) << 13));
    sign * normal
}

/// Bytes per texel of the two drawable formats, `R16Float` and `Depth32Float`.
fn bytes_per_texel(format: MTLPixelFormat) -> usize {
    if format == MTLPixelFormat::RGBA16Float {
        8
    } else if format == MTLPixelFormat::R16Float {
        2
    } else {
        4
    }
}

/// A Private, shader-readable texture filled with random texels, and those texels' bytes.
///
/// `RGBA16Float` texels are finite normal halves: a NaN payload or a denormal
/// is not something a drawable ever carries.
fn random_source(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    format: MTLPixelFormat,
    width: usize,
    height: usize,
) -> (Retained<ProtocolObject<dyn MTLTexture>>, Vec<u8>) {
    let bpp = bytes_per_texel(format);
    let mut state =
        0x9e37_79b9_7f4a_7c15_u64 ^ u64::try_from(width * 4096 + height).expect("a small extent");
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let bytes: Vec<u8> = if bpp == 8 {
        (0..width * height * 4)
            .flat_map(|_| {
                let random = next();
                let sign = u16::from(random & 1 == 1) << 15;
                let exponent = u16::try_from(1 + (random >> 1) % 30).expect("below 31") << 10;
                let mantissa = u16::try_from((random >> 8) & 0x3ff).expect("ten bits");
                (sign | exponent | mantissa).to_le_bytes()
            })
            .collect()
    } else {
        (0..width * height)
            .flat_map(|_| {
                u32::try_from(next() >> 32)
                    .expect("the high half")
                    .to_le_bytes()
            })
            .collect()
    };
    (uploaded(queue, format, (width, height), &bytes), bytes)
}

/// A Private, shader-readable texture holding `bytes`, rows packed.
///
/// Uploaded through a blit from a shared buffer, which works whether or not
/// the GPU has unified memory.
fn uploaded(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    format: MTLPixelFormat,
    (width, height): (usize, usize),
    bytes: &[u8],
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    let device = queue.device();
    let row = width * bytes_per_texel(format);
    assert_eq!(bytes.len(), row * height, "one packed image");
    let buffer = device
        .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModeShared)
        .expect("upload buffer");
    buffer.setLabel(Some(&NSString::from_str("mtld3d-test-exact-upload")));
    // SAFETY: the shared buffer was created with exactly `bytes.len()` bytes
    // and nothing else references it yet.
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes.len(),
        );
    }
    let texture = texture_2d(
        &device,
        format,
        (width, height),
        MTLTextureUsage::ShaderRead,
    );
    let cmd = queue.commandBuffer().expect("upload command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-exact-upload")));
    let blit = cmd.blitCommandEncoder().expect("upload blit");
    blit.setLabel(Some(&NSString::from_str("mtld3d-test-exact-upload-copy")));
    // SAFETY: the buffer holds `height` rows of `row` bytes, exactly the
    // texture's extent, and both stay alive until the copy completes.
    unsafe {
        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            &buffer, 0, row, row * height, MTLSize { width, height, depth: 1 },
            &texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    texture
}

/// A texture shaped like a framebuffer-only drawable: Private and render-target only.
fn drawable_shaped(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    format: MTLPixelFormat,
    width: usize,
    height: usize,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    texture_2d(
        &queue.device(),
        format,
        (width, height),
        MTLTextureUsage::RenderTarget,
    )
}

/// One Private, single-level 2D texture.
fn texture_2d(
    device: &ProtocolObject<dyn MTLDevice>,
    format: MTLPixelFormat,
    (width, height): (usize, usize),
    usage: MTLTextureUsage,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    // SAFETY: objc2 typed binding; a single-level 2D descriptor of a colour
    // format at an extent within Metal's limits.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format, width, height, false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(usage);
    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("test texture");
    texture.setLabel(Some(&NSString::from_str("mtld3d-test-exact")));
    texture
}

/// Every texel of `texture`, rows packed, read back through a staging buffer.
fn texture_bytes(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let row = width * bytes_per_texel(texture.pixelFormat());
    let stride = row.next_multiple_of(256);
    let buffer = queue
        .device()
        .newBufferWithLength_options(stride * height, MTLResourceOptions::StorageModeShared)
        .expect("readback buffer");
    buffer.setLabel(Some(&NSString::from_str("mtld3d-test-exact-readback")));
    let cmd = queue.commandBuffer().expect("readback command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-exact-readback")));
    let blit = cmd.blitCommandEncoder().expect("readback blit");
    blit.setLabel(Some(&NSString::from_str("mtld3d-test-exact-readback-copy")));
    // SAFETY: the buffer holds `height` rows of `stride` bytes, enough for
    // the whole texture, and both stay alive until the copy completes.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width, height, depth: 1 },
            &buffer, 0, stride, stride * height,
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the completed copy initialized every row of the shared buffer,
    // which is `stride * height` bytes long.
    let all = unsafe {
        core::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), stride * height)
    };
    all.chunks_exact(stride)
        .flat_map(|line| &line[..row])
        .copied()
        .collect()
}

/// Assert two packed images are equal, naming the first texel that differs.
fn assert_texels_equal(actual: &[u8], expected: &[u8], bpp: usize, width: usize) {
    assert_eq!(actual.len(), expected.len());
    let pairs = || actual.chunks_exact(bpp).zip(expected.chunks_exact(bpp));
    if let Some(first) = pairs().position(|(read, wrote)| read != wrote) {
        let differing = pairs().filter(|(read, wrote)| read != wrote).count();
        panic!(
            "texel ({}, {}) reads {:?} where the source holds {:?}; {differing} texels differ",
            first % width,
            first / width,
            &actual[first * bpp..(first + 1) * bpp],
            &expected[first * bpp..(first + 1) * bpp],
        );
    }
}

/// The debug effect, run as `FrameEncoder::post_boundary` queues it: a compute
/// dispatch reading the back buffer into a scratch texture, then the copy back.
/// Every colour channel is inverted and alpha kept.
#[test]
fn post_invert_runs_in_the_leading_blits_and_copies_back() {
    use mtld3d_shared::{BlitCommand, mtl::PixelFormat};

    use crate::metal::post::{Kernel, PostConstants};

    let queue = test_queue();
    let device = queue.device();
    let (width, height) = (5_u32, 3_u32);
    let (back_buffer, bytes) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, 5, 3);
    let scratch =
        crate::metal::texture::create_post_scratch(&device, width, height, PixelFormat::Bgra8Unorm)
            .expect("a ShaderRead | ShaderWrite BGRA8 scratch");
    let pipeline = crate::metal::post::create_pipeline(&device, Kernel::Invert)
        .expect("the invert kernel compiles");
    let back_buffer_raw = core::ptr::from_ref(&*back_buffer) as u64;
    let mut effect = BlitCommand::post_effect(&mtld3d_shared::PostEffectInfo {
        kernel: Kernel::Invert as u32,
        pipeline: pipeline.raw(),
        input: back_buffer_raw,
        output: scratch.raw(),
        depth: 0,
        aux: 0,
        constants_row: 0,
        width,
        height,
    });
    let copy_back = BlitCommand::copy_texture_to_texture_full_mip(
        scratch.raw(),
        back_buffer_raw,
        0,
        width,
        height,
    );
    let constants = [PostConstants::new(
        &mtld3d_core::encoder_data::PostBoundary {
            proj: [1.0; 4],
            sun_dir_eye: None,
            sun_color: [0.0; 3],
            sun_above_horizon: false,
            ..mtld3d_core::encoder_data::PostBoundary::default()
        },
        mtld3d_core::post::PostConfig::default(),
        width,
        height,
        false,
        1.0,
    )];
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let params = test_submit_params(&coherent, &upload, &failed);
    let mut ring = UploadRing::default();
    let mut planes = PlanePool::default();
    let mut ctx = EncodeContext {
        device: &device,
        stamp: SubmitStamp::new(&params),
        ring: &mut ring,
        planes: &mut planes,
        post: &constants,
    };
    let cb = queue.commandBuffer().expect("post command buffer");
    cb.setLabel(Some(&NSString::from_str("mtld3d-test-post-invert")));
    assert!(super::encode_leading_blits(
        &cb,
        &[effect, copy_back],
        true,
        super::BlitSite::FrameLeading,
        &mut ctx,
    ));
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    let expected: Vec<u8> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[b, g, r, a]| [255 - b, 255 - g, 255 - r, a])
        .collect();
    assert_eq!(texture_bytes(&queue, &back_buffer), expected);

    // A constants index past the table fails the encode instead of reading past it.
    effect.byte_size = 1;
    let cb = queue.commandBuffer().expect("post command buffer");
    assert!(!super::encode_leading_blits(
        &cb,
        &[effect],
        true,
        super::BlitSite::FrameLeading,
        &mut ctx,
    ));
    drop(cb);
    crate::metal::destroy_texture(scratch.raw());
    crate::metal::post::destroy_pipeline(pipeline.raw());
}

/// The game's world projection terms (`_11`, `_22`, `_33`, `_43`): a
/// left-handed D3D perspective with near plane 10 and far plane 16000.
fn daoc_projection() -> [f32; 4] {
    let (near, far) = (10.0_f32, 16_000.0_f32);
    let p33 = far / (far - near);
    [1.3, 1.73, p33, -near * p33]
}

/// [`daoc_projection`] through a long lens (`_11 = _22 = 80`), so that near
/// the far plane a pixel of a 32-wide raster spans 12.5 world units and the
/// 64-unit search reaches its neighbours.
fn long_lens_projection() -> [f32; 4] {
    let [_, _, p33, p43] = daoc_projection();
    [80.0, 80.0, p33, p43]
}

/// The D3D depth of eye depth `z` under [`daoc_projection`], `_33 + _43 / z`;
/// 1 (the far plane) for an infinite `z`.
fn daoc_depth(z: f32) -> f32 {
    let [_, _, p33, p43] = daoc_projection();
    if z.is_finite() { p33 + p43 / z } else { 1.0 }
}

/// What AO's passes left: the horizon pass's occlusion and the blurred
/// occlusion per half-extent texel, and the colour before and after the effect.
struct AoRun {
    occlusion: Vec<f32>,
    blurred: Vec<f32>,
    before: Vec<u8>,
    after: Vec<u8>,
}

/// Run AO's passes as `FrameEncoder::post_boundary` queues them, on a random
/// BGRA8 colour over the eye depths `eye_z` (rows packed; infinite for the
/// far plane) under `proj`, at strength `ao_pct`. `None` without a Metal device.
fn run_ao(
    (width, height): (u32, u32),
    proj: [f32; 4],
    eye_z: &[f32],
    ao_pct: u16,
) -> Option<AoRun> {
    use mtld3d_shared::{BlitCommand, PostEffectInfo, mtl::PixelFormat};

    use crate::metal::{
        handle::IntoRetained,
        post::{Effect, Input, Output, PostConstants, Slot},
    };

    if MTLCreateSystemDefaultDevice().is_none() {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return None;
    }
    let queue = test_queue();
    let device = queue.device();
    let (w, h) = (width as usize, height as usize);
    let (colour, before) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, w, h);
    let depth_bytes: Vec<u8> = eye_z
        .iter()
        .flat_map(|&z| daoc_depth(z).to_le_bytes())
        .collect();
    let depth = uploaded(&queue, MTLPixelFormat::Depth32Float, (w, h), &depth_bytes);
    let raster = (width, height);
    let slots = [Slot::ColourA, Slot::AoRaw, Slot::AoBlurred].map(|slot| {
        let (sw, sh, format) = slot.key(raster, PixelFormat::Bgra8Unorm);
        (
            slot,
            crate::metal::texture::create_post_scratch(&device, sw, sh, format)
                .expect("a post scratch texture"),
        )
    });
    let handle = |slot: Slot| {
        slots
            .iter()
            .find(|(s, _)| *s == slot)
            .expect("a slot AO uses")
            .1
    };
    let colour_raw = core::ptr::from_ref(&*colour) as u64;
    let texture = |input: Input| match input {
        Input::Colour => colour_raw,
        Input::Slot(slot) => handle(slot).raw(),
    };
    let constants = PostConstants::new(
        &mtld3d_core::encoder_data::PostBoundary {
            proj,
            sun_dir_eye: None,
            sun_color: [0.0; 3],
            sun_above_horizon: false,
            ..mtld3d_core::encoder_data::PostBoundary::default()
        },
        mtld3d_core::post::PostConfig {
            ao_pct,
            ..mtld3d_core::post::PostConfig::default()
        },
        width,
        height,
        true,
        1.0,
    );
    let cb = queue.commandBuffer().expect("AO command buffer");
    cb.setLabel(Some(&NSString::from_str("mtld3d-test-post-ao")));
    let mut pipelines = Vec::new();
    for pass in Effect::Ao.passes() {
        let pipeline = crate::metal::post::create_pipeline(&device, pass.kernel)
            .expect("the AO kernels compile");
        pipelines.push(pipeline);
        let (pw, ph) = pass.extent().of(raster);
        let command = BlitCommand::post_effect(&PostEffectInfo {
            kernel: pass.kernel as u32,
            pipeline: pipeline.raw(),
            input: texture(pass.input),
            output: match pass.output {
                Output::Colour => handle(Slot::ColourA),
                Output::Slot(slot) => handle(slot),
            }
            .raw(),
            depth: core::ptr::from_ref(&*depth) as u64,
            aux: pass.aux.map_or(0, texture),
            constants_row: 0,
            width: pw,
            height: ph,
        });
        assert!(crate::metal::post::encode(&cb, &command, &constants));
    }
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    let halves = |slot: Slot| -> Vec<f32> {
        let texture = handle(slot).into_retained().expect("an AO texture");
        texture_bytes(&queue, &texture)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&bytes| half_to_f32(u16::from_le_bytes(bytes)))
            .collect()
    };
    let occlusion = halves(Slot::AoRaw);
    let blurred = halves(Slot::AoBlurred);
    let result = handle(Slot::ColourA)
        .into_retained()
        .expect("the colour scratch");
    let after = texture_bytes(&queue, &result);
    drop(result);
    for (_, scratch) in slots {
        crate::metal::destroy_texture(scratch.raw());
    }
    for pipeline in pipelines {
        crate::metal::post::destroy_pipeline(pipeline.raw());
    }
    Some(AoRun {
        occlusion,
        blurred,
        before,
        after,
    })
}

/// The far-plane scene of [`ao_ignores_far_depth`], 32×32 under
/// [`long_lens_projection`], sky (infinite) wherever nothing else is:
/// - terrain at eye depth 15900 over x 4..20, y 8..20;
/// - in it, a 4×4 pit 25 units deeper at x 16..20, y 8..12, which the sky
///   borders on the right (x 20) and above (y 7);
/// - a slope receding to the far plane over x 4..=10, y 22..30, 40 units per
///   pixel up to 15990 at x 10, with the sky beyond it from x 11.
fn far_plane_scene() -> Vec<f32> {
    (0_u16..32 * 32)
        .map(|i| {
            let (x, y) = (i % 32, i / 32);
            if (16..20).contains(&x) && (8..12).contains(&y) {
                15_925.0
            } else if (4..20).contains(&x) && (8..20).contains(&y) {
                15_900.0
            } else if (4..=10).contains(&x) && (22..30).contains(&y) {
                15_590.0 + f32::from(40 * x)
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

/// The sky and the far plane (depth ≥ 1) occlude nothing and are never
/// darkened, right next to geometry that is occluded.
#[test]
fn ao_ignores_far_depth() {
    // 4×4: the left half a wall 100 units away, the right half the far plane.
    let eye_z: Vec<f32> = (0..16)
        .map(|i| if i % 4 < 2 { 100.0 } else { f32::INFINITY })
        .collect();
    let Some(run) = run_ao((4, 4), daoc_projection(), &eye_z, 200) else {
        return;
    };
    // At half extent, column 1 is full-extent column 2: the far plane.
    for y in 0..2 {
        assert_eq!(run.occlusion[y * 2 + 1], 1.0, "row {y}");
        assert!((0.0..=1.0).contains(&run.occlusion[y * 2]), "row {y}");
    }
    // A frame of nothing but sky is unchanged.
    let Some(run) = run_ao((4, 4), daoc_projection(), &[f32::INFINITY; 16], 200) else {
        return;
    };
    assert_eq!(run.occlusion, [1.0; 4]);
    assert_eq!(run.after, run.before);

    // A scene that occludes, at the far plane where the sky is in reach.
    let eye_z = far_plane_scene();
    let Some(run) = run_ao((32, 32), long_lens_projection(), &eye_z, 200) else {
        return;
    };
    let sky = |x: usize, y: usize| eye_z[y * 32 + x].is_infinite();
    let texel = |q: (usize, usize)| q.1 * 16 + q.0;
    // The pit's floor is occluded by its rim: the scene has something to bleed.
    let pit = [(8, 4), (9, 4), (8, 5), (9, 5)].map(|q| run.occlusion[texel(q)]);
    assert!(pit.iter().all(|&o| o < 1.0), "{pit:?}");
    assert!(pit.iter().any(|&o| o < 0.75), "{pit:?}");
    // The sky's own occlusion is 1 before and after the blur, though the slope
    // and the terrain sit in its search and in its blur footprint.
    for qy in 0..16 {
        for qx in 0..16 {
            if sky(qx * 2, qy * 2) {
                let q = (qx, qy);
                assert_eq!(run.occlusion[texel(q)], 1.0, "sky texel {q:?}");
                assert_eq!(run.blurred[texel(q)], 1.0, "blurred sky texel {q:?}");
            }
        }
    }
    // The pit texel beside the sky blurs only with the pit below it: the sky
    // taps of its footprint carry no weight.
    let mean = f32::midpoint(run.occlusion[texel((9, 4))], run.occlusion[texel((9, 5))]);
    let edge = run.blurred[texel((9, 4))];
    assert!((edge - mean).abs() < 2.0e-3, "{edge} vs {mean}");
    // The slope's last texel before the sky is unoccluded: the sky beyond it
    // neither occludes it nor bends its normal.
    for qy in 11..15 {
        let o = run.occlusion[texel((5, qy))];
        assert!(o > 0.97, "slope texel (5, {qy}): {o}");
    }
    // At the strongest setting no sky pixel changes, while the occluded pit
    // beside it darkens.
    let pixel = |x: usize, y: usize| (y * 32 + x) * 4;
    for y in 0..32 {
        for x in 0..32 {
            if sky(x, y) {
                let i = pixel(x, y);
                assert_eq!(run.after[i..i + 4], run.before[i..i + 4], "sky ({x}, {y})");
            }
        }
    }
    let i = pixel(17, 9);
    assert!(
        run.after[i..i + 3]
            .iter()
            .zip(&run.before[i..i + 3])
            .all(|(a, b)| a <= b)
            && run.after[i..i + 3] != run.before[i..i + 3],
        "the pit darkens"
    );
}

/// `post.ao = 0` (or `off`) runs nothing; and a zero strength reaching the
/// kernel leaves the colour exactly as it was, however occluded the scene.
#[test]
fn ao_strength_zero_is_identity() {
    let off = mtld3d_core::post::PostConfig::default();
    assert_eq!(off.ao_pct, 0);
    assert_eq!(
        crate::metal::post::enabled_effects(off, false).as_slice(),
        []
    );
    assert!(!off.boundary_wanted());
    // 32×32: a plane 100 units away with an 8×8 pit 30 units deeper in its
    // middle, whose rim occludes the pit's floor.
    let eye_z: Vec<f32> = (0..32 * 32)
        .map(|i| {
            let (x, y) = (i % 32, i / 32);
            if (12..20).contains(&x) && (12..20).contains(&y) {
                130.0
            } else {
                100.0
            }
        })
        .collect();
    let Some(run) = run_ao((32, 32), daoc_projection(), &eye_z, 0) else {
        return;
    };
    let darkest = run.occlusion.iter().copied().fold(1.0_f32, f32::min);
    assert!(darkest < 0.9, "the pit is occluded: {darkest}");
    // The flat plane around it is not.
    assert_eq!(run.occlusion[0], 1.0);
    assert_eq!(run.after, run.before, "a zero strength changes nothing");
    // The same scene at strength 1 darkens it.
    let Some(run) = run_ao((32, 32), daoc_projection(), &eye_z, 100) else {
        return;
    };
    assert_ne!(run.after, run.before);
}

/// A kernel that reads depth or a second input is never dispatched with that
/// texture unbound: the encode fails instead.
#[test]
fn post_pass_without_its_depth_or_second_input_is_refused() {
    use mtld3d_shared::{BlitCommand, PostEffectInfo, mtl::PixelFormat};

    use crate::metal::post::{Kernel, PostConstants};

    if MTLCreateSystemDefaultDevice().is_none() {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    }
    let queue = test_queue();
    let device = queue.device();
    let (colour, _) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, 4, 4);
    let depth = uploaded(
        &queue,
        MTLPixelFormat::Depth32Float,
        (4, 4),
        &[0_u8; 4 * 4 * 4],
    );
    let scratch =
        crate::metal::texture::create_post_scratch(&device, 4, 4, PixelFormat::Bgra8Unorm)
            .expect("a post scratch texture");
    let constants = PostConstants::new(
        &mtld3d_core::encoder_data::PostBoundary {
            proj: daoc_projection(),
            sun_dir_eye: None,
            sun_color: [0.0; 3],
            sun_above_horizon: false,
            ..mtld3d_core::encoder_data::PostBoundary::default()
        },
        mtld3d_core::post::PostConfig::default(),
        4,
        4,
        false,
        1.0,
    );
    let pass = |kernel: Kernel, depth: u64, aux: u64| {
        let pipeline = crate::metal::post::create_pipeline(&device, kernel).expect("compiles");
        let command = BlitCommand::post_effect(&PostEffectInfo {
            kernel: kernel as u32,
            pipeline: pipeline.raw(),
            input: core::ptr::from_ref(&*colour) as u64,
            output: scratch.raw(),
            depth,
            aux,
            constants_row: 0,
            width: 4,
            height: 4,
        });
        let cb = queue.commandBuffer().expect("post command buffer");
        let encoded = crate::metal::post::encode(&cb, &command, &constants);
        drop(cb);
        crate::metal::post::destroy_pipeline(pipeline.raw());
        encoded
    };
    let depth_raw = core::ptr::from_ref(&*depth) as u64;
    assert!(!pass(Kernel::AoHorizon, 0, 0), "no depth snapshot");
    assert!(!pass(Kernel::AoApply, depth_raw, 0), "no second input");
    assert!(
        !pass(Kernel::AoApply, 0, scratch.raw()),
        "no depth snapshot"
    );
    crate::metal::destroy_texture(scratch.raw());
}

/// What the god-ray passes left: the far-depth mask and the blurred rays per
/// half-extent texel (rgba), and the colour before and after the effect.
struct GodRaysRun {
    mask: Vec<[f32; 4]>,
    rays: Vec<[f32; 4]>,
    before: Vec<u8>,
    after: Vec<u8>,
}

/// Run the god-ray passes as `FrameEncoder::post_boundary` queues them, on a
/// random 32×32 BGRA8 colour over the eye depths `eye_z` (infinite for the
/// far plane), with the sun along −`sun_dir_eye` in `sun_color`, above the
/// horizon, at strength 1. `None` without a Metal device.
fn run_god_rays(eye_z: &[f32], sun_dir_eye: [f32; 3], sun_color: [f32; 3]) -> Option<GodRaysRun> {
    use mtld3d_shared::{BlitCommand, PostEffectInfo, mtl::PixelFormat};

    use crate::metal::{
        handle::IntoRetained,
        post::{Effect, Input, Output, PostConstants, Slot},
    };

    if MTLCreateSystemDefaultDevice().is_none() {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return None;
    }
    let queue = test_queue();
    let device = queue.device();
    let (width, height) = (32_u32, 32_u32);
    let (w, h) = (width as usize, height as usize);
    let (colour, before) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, w, h);
    let depth_bytes: Vec<u8> = eye_z
        .iter()
        .flat_map(|&z| daoc_depth(z).to_le_bytes())
        .collect();
    let depth = uploaded(&queue, MTLPixelFormat::Depth32Float, (w, h), &depth_bytes);
    let raster = (width, height);
    let slots = [Slot::ColourA, Slot::GodRaysMask, Slot::GodRays].map(|slot| {
        let (sw, sh, format) = slot.key(raster, PixelFormat::Bgra8Unorm);
        (
            slot,
            crate::metal::texture::create_post_scratch(&device, sw, sh, format)
                .expect("a post scratch texture"),
        )
    });
    let handle = |slot: Slot| {
        slots
            .iter()
            .find(|(s, _)| *s == slot)
            .expect("a slot god rays use")
            .1
    };
    let colour_raw = core::ptr::from_ref(&*colour) as u64;
    let texture = |input: Input| match input {
        Input::Colour => colour_raw,
        Input::Slot(slot) => handle(slot).raw(),
    };
    let boundary = mtld3d_core::encoder_data::PostBoundary {
        proj: daoc_projection(),
        sun_dir_eye: Some(sun_dir_eye),
        sun_color,
        sun_above_horizon: true,
        ..mtld3d_core::encoder_data::PostBoundary::default()
    };
    let constants = PostConstants::new(
        &boundary,
        mtld3d_core::post::PostConfig {
            god_rays_pct: 100,
            ..mtld3d_core::post::PostConfig::default()
        },
        width,
        height,
        true,
        1.0,
    );
    let cb = queue.commandBuffer().expect("god rays command buffer");
    cb.setLabel(Some(&NSString::from_str("mtld3d-test-post-god-rays")));
    let mut pipelines = Vec::new();
    for pass in Effect::GodRays.passes() {
        let pipeline = crate::metal::post::create_pipeline(&device, pass.kernel)
            .expect("the god-ray kernels compile");
        pipelines.push(pipeline);
        let (pw, ph) = pass.extent().of(raster);
        let command = BlitCommand::post_effect(&PostEffectInfo {
            kernel: pass.kernel as u32,
            pipeline: pipeline.raw(),
            input: texture(pass.input),
            output: match pass.output {
                Output::Colour => handle(Slot::ColourA),
                Output::Slot(slot) => handle(slot),
            }
            .raw(),
            depth: if pass.kernel.reads_depth() {
                core::ptr::from_ref(&*depth) as u64
            } else {
                0
            },
            aux: pass.aux.map_or(0, texture),
            constants_row: 0,
            width: pw,
            height: ph,
        });
        assert!(crate::metal::post::encode(&cb, &command, &constants));
    }
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    let texels = |slot: Slot| -> Vec<[f32; 4]> {
        let texture = handle(slot).into_retained().expect("a god-ray texture");
        texture_bytes(&queue, &texture)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|t| {
                core::array::from_fn(|c| half_to_f32(u16::from_le_bytes([t[2 * c], t[2 * c + 1]])))
            })
            .collect()
    };
    let mask = texels(Slot::GodRaysMask);
    let rays = texels(Slot::GodRays);
    let result = handle(Slot::ColourA)
        .into_retained()
        .expect("the colour scratch");
    let after = texture_bytes(&queue, &result);
    drop(result);
    for (_, scratch) in slots {
        crate::metal::destroy_texture(scratch.raw());
    }
    for pipeline in pipelines {
        crate::metal::post::destroy_pipeline(pipeline.raw());
    }
    Some(GodRaysRun {
        mask,
        rays,
        before,
        after,
    })
}

/// A sun straight ahead: the screen's centre.
const CENTRED_SUN: [f32; 3] = [0.0, 0.0, -1.0];

/// Only the sky lights the rays: geometry is black in the mask, a frame of
/// geometry alone gets no rays at all, and the rays add the sun's colour.
#[test]
fn godrays_mask_is_far_depth_only() {
    // The left half is a wall 100 units away, the right half sky.
    let half_sky: Vec<f32> = (0..32 * 32)
        .map(|i| if i % 32 < 16 { 100.0 } else { f32::INFINITY })
        .collect();
    let Some(run) = run_god_rays(&half_sky, CENTRED_SUN, [1.0, 0.9, 0.8]) else {
        return;
    };
    for qy in 0..16 {
        for qx in 0..16 {
            let m = run.mask[qy * 16 + qx];
            if qx < 8 {
                assert_eq!(m[..3], [0.0; 3], "wall texel ({qx}, {qy})");
            } else {
                // The sky's colour, averaged over the 2×2 block (BGRA bytes).
                for (c, byte) in [2, 1, 0].into_iter().enumerate() {
                    let mean = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .map(|&(dx, dy)| {
                            let p = (2 * qy + dy) * 32 + 2 * qx + dx;
                            f32::from(run.before[p * 4 + byte]) / 255.0
                        })
                        .sum::<f32>()
                        / 4.0;
                    assert!(
                        (m[c] - mean).abs() < 2e-3,
                        "sky texel ({qx}, {qy}) channel {c}: {} vs {mean}",
                        m[c]
                    );
                }
            }
        }
    }
    // The rays add light and keep alpha; the sky beside the sun brightens.
    let mut brighter = 0;
    for (p, (after, before)) in run
        .after
        .as_chunks::<4>()
        .0
        .iter()
        .zip(run.before.as_chunks::<4>().0)
        .enumerate()
    {
        assert_eq!(after[3], before[3], "alpha at {p}");
        for c in 0..3 {
            assert!(
                after[c] >= before[c],
                "pixel {p} darkened: {after:?} {before:?}"
            );
        }
        brighter += usize::from(after != before);
    }
    assert!(brighter > 64, "the rays light the sky: {brighter} pixels");
    assert!(
        run.rays.iter().any(|r| r[0] > 0.05),
        "the rays carry the sky"
    );

    // Geometry everywhere: no mask, no rays, the colour unchanged.
    let Some(run) = run_god_rays(&[100.0; 32 * 32], CENTRED_SUN, [1.0, 0.9, 0.8]) else {
        return;
    };
    assert!(
        run.mask.iter().all(|m| m[..3] == [0.0; 3]),
        "geometry masks out"
    );
    assert!(
        run.rays.iter().all(|r| r[..3] == [0.0; 3]),
        "no rays from geometry"
    );
    assert_eq!(run.after, run.before, "geometry contributes nothing");

    // A black sun tints the rays to nothing.
    let Some(run) = run_god_rays(&[f32::INFINITY; 32 * 32], CENTRED_SUN, [0.0; 3]) else {
        return;
    };
    assert!(
        run.rays.iter().any(|r| r[0] > 0.05),
        "an all-sky frame has rays"
    );
    assert_eq!(run.after, run.before, "the rays take the sun's colour");
}

/// The rays run from the sky away from the sun, with v down the raster: a
/// small patch of sky in the middle under a sun up and left (uv 0.25, 0.25)
/// lights the texels below and right of it, beyond it from the sun, and
/// nothing between it and the sun or across either axis.
#[test]
fn godrays_stream_away_from_an_off_centre_sun() {
    use crate::metal::post::sun_screen;

    // Sky only over pixels 14..18 on both axes: half-extent texels 7 and 8.
    let patch: Vec<f32> = (0..32 * 32)
        .map(|i| {
            let (x, y) = (i % 32, i / 32);
            if (14..18).contains(&x) && (14..18).contains(&y) {
                f32::INFINITY
            } else {
                100.0
            }
        })
        .collect();
    // NDC (−0.5, 0.5) under `daoc_projection`: the sun at eye (x, y, 1).
    let [p11, p22, _, _] = daoc_projection();
    let sun_dir_eye = [0.5 / p11, -0.5 / p22, -1.0];
    let [u, v] = sun_screen(&mtld3d_core::encoder_data::PostBoundary {
        proj: daoc_projection(),
        sun_dir_eye: Some(sun_dir_eye),
        sun_color: [1.0; 3],
        sun_above_horizon: true,
        ..mtld3d_core::encoder_data::PostBoundary::default()
    })
    .expect("a sun on screen");
    assert!(
        (u - 0.25).abs() < 1e-6 && (v - 0.25).abs() < 1e-6,
        "{u} {v}"
    );
    let Some(run) = run_god_rays(&patch, sun_dir_eye, [1.0; 3]) else {
        return;
    };
    let light = |qx: usize, qy: usize| run.rays[qy * 16 + qx][..3].iter().sum::<f32>();
    // Beyond the patch from the sun, down and right: lit.
    assert!(light(11, 11) > 0.01, "beyond the patch: {}", light(11, 11));
    // Between the sun and the patch: dark (lit if the blur ran backwards).
    assert_eq!(light(5, 5), 0.0, "between the sun and the patch");
    // Mirrored across either axis: dark (lit if v or u were flipped).
    assert_eq!(light(11, 4), 0.0, "up and right");
    assert_eq!(light(4, 11), 0.0, "down and left");
    let brighter = run
        .after
        .as_chunks::<4>()
        .0
        .iter()
        .zip(run.before.as_chunks::<4>().0)
        .filter(|(a, b)| a != b)
        .count();
    assert!(brighter > 0, "the rays reach the colour");
}

/// Run bloom's passes as `FrameEncoder::post_boundary` queues them on a
/// `width` × `height` BGRA8 colour holding `bgra`, at strength 1 and the
/// threshold `threshold_pct`; the colour bytes after. No depth is bound: bloom
/// reads none. `None` without a Metal device.
fn run_bloom(width: u32, height: u32, bgra: &[u8], threshold_pct: u16) -> Option<Vec<u8>> {
    use mtld3d_shared::{BlitCommand, PostEffectInfo, mtl::PixelFormat};

    use crate::metal::{
        handle::IntoRetained,
        post::{Effect, Input, Output, PostConstants, Slot},
    };

    if MTLCreateSystemDefaultDevice().is_none() {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return None;
    }
    let queue = test_queue();
    let device = queue.device();
    let colour = uploaded(
        &queue,
        MTLPixelFormat::BGRA8Unorm,
        (width as usize, height as usize),
        bgra,
    );
    let raster = (width, height);
    let mut used = vec![Slot::ColourA];
    for pass in Effect::Bloom.passes() {
        for input in [Some(pass.input), pass.aux].into_iter().flatten() {
            if let Input::Slot(slot) = input
                && !used.contains(&slot)
            {
                used.push(slot);
            }
        }
        if let Output::Slot(slot) = pass.output
            && !used.contains(&slot)
        {
            used.push(slot);
        }
    }
    let slots: Vec<_> = used
        .into_iter()
        .map(|slot| {
            let (sw, sh, format) = slot.key(raster, PixelFormat::Bgra8Unorm);
            assert!(sw >= 1 && sh >= 1, "{slot:?} has no zero-size level");
            (
                slot,
                crate::metal::texture::create_post_scratch(&device, sw, sh, format)
                    .expect("a post scratch texture"),
            )
        })
        .collect();
    let handle = |slot: Slot| {
        slots
            .iter()
            .find(|(s, _)| *s == slot)
            .expect("a slot bloom uses")
            .1
    };
    let colour_raw = core::ptr::from_ref(&*colour) as u64;
    let texture = |input: Input| match input {
        Input::Colour => colour_raw,
        Input::Slot(slot) => handle(slot).raw(),
    };
    let constants = PostConstants::new(
        &mtld3d_core::encoder_data::PostBoundary {
            proj: daoc_projection(),
            sun_dir_eye: None,
            sun_color: [0.0; 3],
            sun_above_horizon: false,
            ..mtld3d_core::encoder_data::PostBoundary::default()
        },
        mtld3d_core::post::PostConfig {
            bloom_pct: 100,
            bloom_threshold_pct: threshold_pct,
            ..mtld3d_core::post::PostConfig::default()
        },
        width,
        height,
        false,
        1.0,
    );
    let cb = queue.commandBuffer().expect("bloom command buffer");
    cb.setLabel(Some(&NSString::from_str("mtld3d-test-post-bloom")));
    let mut pipelines = Vec::new();
    for pass in Effect::Bloom.passes() {
        let pipeline = crate::metal::post::create_pipeline(&device, pass.kernel)
            .expect("the bloom kernels compile");
        pipelines.push(pipeline);
        let (pw, ph) = pass.extent().of(raster);
        let command = BlitCommand::post_effect(&PostEffectInfo {
            kernel: pass.kernel as u32,
            pipeline: pipeline.raw(),
            input: texture(pass.input),
            output: match pass.output {
                Output::Colour => handle(Slot::ColourA),
                Output::Slot(slot) => handle(slot),
            }
            .raw(),
            depth: 0,
            aux: pass.aux.map_or(0, texture),
            constants_row: 0,
            width: pw,
            height: ph,
        });
        assert!(
            crate::metal::post::encode(&cb, &command, &constants),
            "{:?}",
            pass.kernel
        );
    }
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    let result = handle(Slot::ColourA)
        .into_retained()
        .expect("the colour scratch");
    let after = texture_bytes(&queue, &result);
    drop(result);
    for (_, scratch) in slots {
        crate::metal::destroy_texture(scratch.raw());
    }
    for pipeline in pipelines {
        crate::metal::post::destroy_pipeline(pipeline.raw());
    }
    Some(after)
}

/// A BGRA8 image of `width` × `height` in one grey, alpha opaque.
fn grey_bgra(width: u32, height: u32, value: u8) -> Vec<u8> {
    (0..width * height)
        .flat_map(|_| [value, value, value, 255])
        .collect()
}

/// The knee starts at the threshold: an image at 0.79 under a threshold of
/// 0.8 is left exactly as it was, while the same image at 0.95 brightens.
#[test]
fn bloom_below_threshold_adds_nothing() {
    // 201 / 255 ≈ 0.788.
    let dim = grey_bgra(32, 32, 201);
    let Some(after) = run_bloom(32, 32, &dim, 80) else {
        return;
    };
    assert_eq!(after, dim, "nothing below the threshold glows");
    // 242 / 255 ≈ 0.949: above the threshold and the knee, so it glows.
    let bright = grey_bgra(32, 32, 242);
    let after = run_bloom(32, 32, &bright, 80).expect("a device");
    for (i, (a, b)) in after.iter().zip(&bright).enumerate() {
        if i % 4 == 3 {
            assert_eq!(a, b, "alpha is kept");
        } else {
            assert!(a > b, "byte {i}: {a} not above {b}");
        }
    }
    // A bright 4×4 patch at (16..20, 16..20) glows into its dim surround.
    let mut spot = grey_bgra(32, 32, 64);
    for y in 16..20 {
        for x in 16..20 {
            let i = (y * 32 + x) * 4;
            spot[i..i + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    let after = run_bloom(32, 32, &spot, 80).expect("a device");
    let near = (17 * 32 + 22) * 4;
    assert!(
        after[near] > spot[near],
        "the glow reaches 3 px past the patch"
    );
}

/// Five levels on a raster smaller than 2^5 clamp each level to 1 px: every
/// pass encodes and runs, and the image still glows.
#[test]
fn bloom_runs_on_a_tiny_raster() {
    for (w, h) in [(1, 1), (3, 2), (17, 5)] {
        let bright = grey_bgra(w, h, 242);
        let Some(after) = run_bloom(w, h, &bright, 80) else {
            return;
        };
        assert!(after[0] > bright[0], "{w}x{h}");
    }
}

/// The glow of a small highlight reaches far: the coarse levels hold a
/// white 4×4 patch's faint share (0.1 at the default threshold, averaged down
/// to under 1/255) in half float, where 8-bit levels would round it to 0.
#[test]
fn bloom_small_highlight_glows_far() {
    let mut spot = grey_bgra(32, 32, 64);
    for y in 16..20 {
        for x in 16..20 {
            let i = (y * 32 + x) * 4;
            spot[i..i + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    let Some(after) = run_bloom(32, 32, &spot, 80) else {
        return;
    };
    // 12 px left of the patch: only the coarsest levels reach it.
    let far = (17 * 32 + 4) * 4;
    assert!(
        after[far] > spot[far],
        "{} not above {}",
        after[far],
        spot[far]
    );
}

/// Read a 4x1 half-float back buffer through `blit_texture_to_buffer` as `planes`.
///
/// The texels are `rgba` halves; the returned bytes are what the PE page
/// received, or `None` when the readback declined.
fn read_half_backbuffer(
    planes: mtld3d_shared::mtl::ReadbackPlanes,
    rgba: &[[u16; 4]; 4],
    bytes_per_row: u32,
) -> Option<Vec<u8>> {
    let queue = test_queue();
    let device = queue.device();
    // SAFETY: the handle borrows `device`, which outlives every use below;
    // the creation path only bumps the retain count.
    let device_handle = unsafe {
        MetalHandle::<mtld3d_shared::mtl_handle::MTLDeviceKind>::new(
            Retained::as_ptr(&device) as usize as u64
        )
    };
    let (texture_handle, twin) =
        super::super::create_backbuffer(device_handle, 4, 1, PixelFormat::Rgba16Float)
            .expect("half-float back buffer");
    assert_eq!(twin, 0);
    let texture = {
        use crate::metal::handle::IntoRetained;
        texture_handle.into_retained().expect("back buffer texture")
    };
    // Fill the Private back buffer the way a frame would reach it: through
    // the GPU, from a staging buffer of the halves.
    let halves: Vec<u8> = rgba
        .iter()
        .flatten()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let staging = device
        .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
        .expect("staging buffer");
    // SAFETY: the shared buffer holds 256 bytes, more than the 32 copied.
    unsafe {
        core::ptr::copy_nonoverlapping(
            halves.as_ptr(),
            staging.contents().as_ptr().cast::<u8>(),
            halves.len(),
        );
    }
    let cmd = queue.commandBuffer().expect("fill command buffer");
    let blit = cmd.blitCommandEncoder().expect("fill blit");
    // SAFETY: the 4x1 region fits the texture, the 32-byte row fits the
    // buffer, and both resources stay alive until the copy completes.
    unsafe {
        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            &staging, 0, 32, 32, MTLSize { width: 4, height: 1, depth: 1 }, &texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    drop(texture);

    let record = test_record(&queue);
    let mut page = mtld3d_core::page_box::PageBox::new_zeroed(64);
    let ok = super::blit_texture_to_buffer(&super::BlitArgs {
        planes,
        stencil_bytes_per_row: 0,
        stencil_offset: 0,
        record: &record,
        device_handle,
        tex_handle: texture_handle,
        dst_ptr: page.as_mut_ptr() as u64,
        dst_len: page.len() as u64,
        mip_level: 0,
        slice: 0,
        origin_x: 0,
        origin_y: 0,
        width: 4,
        height: 1,
        bytes_per_row,
        source_width: 4,
        source_height: 1,
        block_height: 1,
    });
    super::super::destroy_texture(texture_handle.raw());
    ok.then(|| page.as_slice()[..bytes_per_row as usize].to_vec())
}

/// Under `color.sceneFloat` the back buffer is half float, but every read of
/// it keeps the BGRA8 layout of its 8-bit D3D9 surface: the GPU converts,
/// clamping to [0, 1] the way the SDR screen does.
#[test]
fn float_backbuffer_readback_is_bgra8() {
    const ZERO: u16 = 0x0000;
    const QUARTER: u16 = 0x3400;
    const THREE_QUARTERS: u16 = 0x3A00;
    const ONE: u16 = 0x3C00;
    const ONE_AND_A_HALF: u16 = 0x3E00;
    const TWO: u16 = 0x4000;
    const MINUS_QUARTER: u16 = 0xB400;
    if MTLCreateSystemDefaultDevice().is_none() {
        return;
    }
    let texels = [
        [QUARTER, THREE_QUARTERS, ONE, THREE_QUARTERS],
        [ONE_AND_A_HALF, QUARTER, ZERO, THREE_QUARTERS],
        [MINUS_QUARTER, TWO, THREE_QUARTERS, ONE],
        [ZERO, ZERO, ZERO, QUARTER],
    ];
    let bgra = read_half_backbuffer(mtld3d_shared::mtl::ReadbackPlanes::ColorBgra8, &texels, 16)
        .expect("the converting readback succeeds");
    assert_eq!(
        bgra,
        [
            255, 191, 64, 191, // in range: exact
            0, 64, 255, 191, // red above 1 clamps to 255
            191, 255, 0, 255, // red below 0 clamps to 0, green above 1 to 255
            0, 0, 0, 64, // alpha carried
        ]
    );
    // The raw colour read still hands back the texture's own layout, which is
    // why the PE has to ask for the conversion.
    let raw = read_half_backbuffer(mtld3d_shared::mtl::ReadbackPlanes::Color, &texels, 32)
        .expect("the raw readback succeeds");
    let expected_raw: Vec<u8> = texels
        .iter()
        .flatten()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    assert_eq!(raw, expected_raw);
}

/// A finite f32's nearest-below IEEE half bits (normal halves and zero only).
fn f32_to_half(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).expect("one bit");
    let exponent = i32::try_from((bits >> 23) & 0xff).expect("eight bits") - 127 + 15;
    assert!((1..31).contains(&exponent), "{value} is not a normal half");
    let mantissa = u16::try_from((bits >> 13) & 0x3ff).expect("ten bits");
    sign | (u16::try_from(exponent).expect("five bits") << 10) | mantissa
}

/// The extended sRGB OETF the tone map writes with.
fn srgb_oetf(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055_f32.mul_add(linear.powf(1.0 / 2.4), -0.055)
    }
}

/// Run the tone map at `headroom` over one row of grey texels, each `inputs`
/// value (sRGB-encoded, as the back buffer holds it) in rgb and alpha 0.25,
/// followed by an inf, a NaN and a −inf texel; the output texels.
fn run_tone_map(inputs: &[f32], headroom: f32) -> Vec<[f32; 4]> {
    use mtld3d_shared::{BlitCommand, mtl::PixelFormat};

    use crate::metal::post::{Kernel, PostConstants};

    let queue = test_queue();
    let device = queue.device();
    let mut halves: Vec<u16> = inputs.iter().map(|&v| f32_to_half(v)).collect();
    halves.extend([0x7c00, 0x7e00, 0xfc00]);
    let width = halves.len();
    let alpha = f32_to_half(0.25);
    let bytes: Vec<u8> = halves
        .iter()
        .flat_map(|&h| [h, h, h, alpha])
        .flat_map(u16::to_le_bytes)
        .collect();
    let src = uploaded(&queue, MTLPixelFormat::RGBA16Float, (width, 1), &bytes);
    let w = u32::try_from(width).expect("a short row");
    let dst = crate::metal::texture::create_post_scratch(&device, w, 1, PixelFormat::Rgba16Float)
        .expect("a ShaderRead | ShaderWrite float scratch");
    let pipeline = crate::metal::post::create_pipeline(&device, Kernel::ToneMap)
        .expect("the tone map compiles");
    let command = BlitCommand::post_effect(&mtld3d_shared::PostEffectInfo {
        kernel: Kernel::ToneMap as u32,
        pipeline: pipeline.raw(),
        input: core::ptr::from_ref(&*src) as u64,
        output: dst.raw(),
        depth: 0,
        aux: 0,
        constants_row: 0,
        width: w,
        height: 1,
    });
    let constants = PostConstants::new(
        &mtld3d_core::encoder_data::PostBoundary {
            proj: [1.0; 4],
            sun_dir_eye: None,
            sun_color: [0.0; 3],
            sun_above_horizon: false,
            ..mtld3d_core::encoder_data::PostBoundary::default()
        },
        mtld3d_core::post::PostConfig::default(),
        w,
        1,
        false,
        headroom,
    );
    let cb = queue.commandBuffer().expect("tone map command buffer");
    cb.setLabel(Some(&NSString::from_str("mtld3d-test-tone-map")));
    assert!(crate::metal::post::encode(&cb, &command, &constants));
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    let written = crate::metal::handle::IntoRetained::into_retained(dst).expect("the scratch");
    let out: Vec<[f32; 4]> = texture_bytes(&queue, &written)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|texel| {
            core::array::from_fn(|i| {
                let half = texel.as_chunks::<2>().0[i];
                let bits = u16::from_le_bytes(half);
                assert_ne!(bits & 0x7c00, 0x7c00, "an inf or NaN half: {bits:#06x}");
                half_to_f32(bits)
            })
        })
        .collect();
    crate::metal::destroy_texture(dst.raw());
    crate::metal::post::destroy_pipeline(pipeline.raw());
    out
}

/// The inputs both tone-map tests run: zero, two display values, white, and
/// three far above it, as the unclamped lighting writes them.
const TONE_MAP_INPUTS: [f32; 7] = [0.0, 0.5, 0.8, 1.0, 2.0, 10.0, 1000.0];

/// At headroom 1 (an SDR display) every output is in `[0, 1]`, rises with
/// its input, leaves the dark and mid values as they were, keeps alpha, and
/// an inf or NaN input gives a finite pixel.
#[test]
fn tonemap_sdr_maps_to_unit_range() {
    let out = run_tone_map(&TONE_MAP_INPUTS, 1.0);
    for (index, texel) in out.iter().enumerate() {
        for &c in &texel[..3] {
            assert!(
                (0.0..=1.0 + 1.0e-3).contains(&c),
                "texel {index}: {texel:?}"
            );
        }
        assert_eq!(texel[3], 0.25, "texel {index} keeps its alpha");
    }
    let grey: Vec<f32> = out.iter().map(|t| t[0]).collect();
    assert!(
        grey[..TONE_MAP_INPUTS.len()].is_sorted(),
        "monotonic: {grey:?}"
    );
    assert_eq!(grey[0], 0.0);
    assert!(
        (grey[1] - 0.5).abs() < 1.0e-3,
        "0.5 is untouched: {}",
        grey[1]
    );
    // Above the knee the curve compresses: white comes out below white, and
    // the brightest input lands near, never past, white.
    assert!(grey[3] < 0.99, "1.0 is compressed: {}", grey[3]);
    assert!(grey[6] > 0.99, "1000 nears white: {}", grey[6]);
    assert!(grey[4] > grey[3], "{grey:?}");
}

/// At headroom 4 (an EDR display) the outputs stay under `oetf(4)`, every
/// input up to white passes untouched, the brightest approach `oetf(4)`,
/// +inf maps as the headroom does, and NaN and −inf come out black.
#[test]
fn tonemap_respects_headroom() {
    let out = run_tone_map(&TONE_MAP_INPUTS, 4.0);
    let ceiling = srgb_oetf(4.0);
    for (index, texel) in out.iter().enumerate() {
        for &c in &texel[..3] {
            assert!(
                (0.0..=ceiling + 2.0e-3).contains(&c),
                "texel {index}: {texel:?} over {ceiling}"
            );
        }
    }
    for (index, &input) in TONE_MAP_INPUTS[..4].iter().enumerate() {
        let read = out[index][0];
        assert!(
            (read - input).abs() <= 2.0e-3,
            "{input} is untouched under headroom: {read}"
        );
    }
    for index in 5..7 {
        let read = out[index][0];
        assert!(
            (ceiling - read).abs() < 5.0e-3,
            "input {} approaches {ceiling}: {read}",
            TONE_MAP_INPUTS[index]
        );
    }
    // +inf maps as the headroom H does; NaN and −inf floor to black first.
    let [inf, nan, neg_inf] = out[TONE_MAP_INPUTS.len()..] else {
        panic!("the inf, NaN and −inf texels: {out:?}");
    };
    assert!(inf.iter().all(|c| c.is_finite()), "{inf:?}");
    assert!(
        inf[0] > out[3][0] && inf[0] <= ceiling + 2.0e-3,
        "+inf is the bright end: {inf:?}"
    );
    assert_eq!(nan[..3], [0.0; 3], "NaN is black: {nan:?}");
    assert_eq!(neg_inf[..3], [0.0; 3], "−inf is black: {neg_inf:?}");
}
