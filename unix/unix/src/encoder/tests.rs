use std::sync::mpsc;

use mtld3d_core::{
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineSnapshot, StreamLayout,
    },
    shader_cache::ShaderRecordRef,
    shader_key::CachedKind,
};
use mtld3d_shared::{MetalHandle, mtl::PixelFormat};
use mtld3d_types::MAX_STREAMS;
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState,
};

use super::{
    BufferGpuState, DepthSnapshot, DestroyKind, MipStagingBuffer, StageLibHandles, StretchScratch,
    TextureGpuState, WarmCache, cached_buffer_handles, destroy_resources_bulk,
    drain_source_scratch, staging_wrapped_bytes, take_released_buffer,
    take_released_staging_wrapper, take_source_scratch, take_staging_wrapper,
};

struct NativeObjects {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    function: Retained<ProtocolObject<dyn MTLFunction>>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

impl NativeObjects {
    /// Compiles a vertex function named `entry` and a pipeline that uses it.
    ///
    /// Each test passes its own `entry`. Metal returns the same `MTLFunction`
    /// object for two libraries compiled from identical source, so tests that
    /// shared one source would change each other's retain counts when they run
    /// in parallel.
    fn new(entry: &str) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device for ownership test");
        let source = NSString::from_str(&format!(
            "#include <metal_stdlib>\nusing namespace metal;\nvertex void {entry}(uint id [[vertex_id]]) {{}}",
        ));
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("library");
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .expect("function");
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&function));
        descriptor.setRasterizationEnabled(false);
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .expect("pipeline");
        Self {
            library,
            function,
            pipeline,
        }
    }

    fn counts(&self) -> [usize; 3] {
        [
            self.pipeline.retainCount(),
            self.function.retainCount(),
            self.library.retainCount(),
        ]
    }

    fn warm(&self) -> WarmCache {
        // SAFETY: each handle owns the real object's extra retain transferred by into_raw.
        let library = unsafe { MetalHandle::new(Retained::into_raw(self.library.clone()) as u64) };
        // SAFETY: this handle owns the real function's transferred extra retain.
        let func = unsafe { MetalHandle::new(Retained::into_raw(self.function.clone()) as u64) };
        // SAFETY: this handle owns the real pipeline's transferred extra retain.
        let pipeline =
            unsafe { MetalHandle::new(Retained::into_raw(self.pipeline.clone()) as u64) };
        let snapshot = PipelineSnapshot {
            vdecl_hash: 0,
            vs_fn: func,
            ps_fn: MetalHandle::NULL,
            stream_layouts: [StreamLayout::UNUSED; MAX_STREAMS as usize],
            color_format: PixelFormat::Bgra8Unorm,
            attach: PipelineAttachFlags::empty(),
            rs: PipelineRsBits::default(),
            extra: ExtraColorAttachments::NONE,
            ps_color_out_mask: 0,
            sample_count: 1,
        };
        WarmCache {
            libraries: vec![(
                ShaderRecordRef::new(CachedKind::Sm2Vs, 1),
                StageLibHandles { library, func },
            )],
            pipelines: vec![(
                mtld3d_core::pipeline_state::key_from_snapshot(&snapshot, &[]),
                pipeline,
            )],
            no_color_siblings: vec![(pipeline.raw(), pipeline)],
        }
    }
}

#[test]
fn failed_prewarm_send_releases_native_payload() {
    objc2::rc::autoreleasepool(|_| {
        let objects = NativeObjects::new("failed_send_probe");
        let baseline = objects.counts();
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        let result = sender.send(objects.warm());
        assert!(result.is_err());
        drop(result);
        assert_eq!(
            objects.counts(),
            baseline,
            "failed delivery releases all three retains"
        );
    });
}

#[test]
fn unread_prewarm_payload_releases_native_objects() {
    objc2::rc::autoreleasepool(|_| {
        let objects = NativeObjects::new("unread_payload_probe");
        let baseline = objects.counts();
        let (sender, receiver) = mpsc::sync_channel(1);
        assert!(sender.send(objects.warm()).is_ok());
        drop(receiver);
        assert_eq!(
            objects.counts(),
            baseline,
            "queued payload is still an owner"
        );
    });
}

#[test]
fn adopted_prewarm_retains_transfer_once_and_release_in_dependency_order() {
    objc2::rc::autoreleasepool(|_| {
        let objects = NativeObjects::new("adopted_prewarm_probe");
        let baseline = objects.counts();
        let mut source = objects.warm();
        let retained = objects.counts();
        let mut destination = WarmCache {
            libraries: std::mem::take(&mut source.libraries),
            pipelines: std::mem::take(&mut source.pipelines),
            no_color_siblings: std::mem::take(&mut source.no_color_siblings),
        };
        drop(source);
        assert_eq!(
            objects.counts(),
            retained,
            "adoption leaves the old owner empty"
        );
        let mut releases = Vec::new();
        destination.release_with(|kind, handles| {
            releases.push((kind, handles.to_vec()));
            destroy_resources_bulk(kind, handles);
        });
        assert_eq!(
            releases
                .iter()
                .map(|(kind, handles)| (*kind, handles.len()))
                .collect::<Vec<_>>(),
            vec![
                (DestroyKind::RenderPipeline, 1),
                (DestroyKind::ShaderFunction, 1),
                (DestroyKind::ShaderLibrary, 1),
            ]
        );
        drop(destination);
        assert_eq!(
            objects.counts(),
            baseline,
            "aliases and empty owners release nothing extra"
        );
    });
}

#[test]
fn native_shader_build_resets_empty_inputs_and_retains_outputs_after_pool_drain() {
    use mtld3d_shared::perf::ShaderTimings;

    use crate::metal::handle::IntoRetained;

    objc2::rc::autoreleasepool(|_| {
        for (source, entry) in [("", "probe"), ("invalid MSL", "")] {
            let mut timings = ShaderTimings {
                preparation_ns: u64::MAX,
                library_ns: u64::MAX,
                function_ns: u64::MAX,
            };
            assert!(
                super::compile_stage_library(MetalHandle::NULL, source, entry, &mut timings)
                    .is_none()
            );
            assert_eq!(
                (
                    timings.preparation_ns,
                    timings.library_ns,
                    timings.function_ns
                ),
                (0, 0, 0),
                "empty input resets all phases before native compilation"
            );
        }

        let device = MTLCreateSystemDefaultDevice().expect("Metal device for shader lifetime test");
        // SAFETY: device owns this live Metal device until the build completes.
        let device_handle = unsafe { MetalHandle::new(Retained::as_ptr(&device) as u64) };
        let source = "#include <metal_stdlib>\nusing namespace metal;\nvertex float4 probe(uint id [[vertex_id]]) { return float4(float(id), 0, 0, 1); }";
        let handles =
            super::compile_stage_library(device_handle, source, "probe", &mut ShaderTimings::new())
                .expect("native shader build succeeds");
        // The build's inner pool has drained; only canonical output retains remain.
        let library = handles
            .library
            .into_retained()
            .expect("live library output");
        let function = handles.func.into_retained().expect("live function output");
        assert_eq!(library.label().expect("library label").to_string(), "probe");
        assert_eq!(function.name().to_string(), "probe");
        drop(function);
        drop(library);
        destroy_resources_bulk(DestroyKind::ShaderFunction, &[handles.func.raw()]);
        destroy_resources_bulk(DestroyKind::ShaderLibrary, &[handles.library.raw()]);
    });
}

#[test]
fn payload_rotation_reuses_warm_scratch_chunks_across_the_submit_thread() {
    use std::collections::BTreeSet;

    use mtld3d_core::{passes::PassState, scratch::ScratchArena};

    use super::{FramePayload, SUBMIT_PAYLOAD_CAP};

    const FRAMES: usize = 48;
    const WARM_FRAMES: usize = 8;
    let (work_tx, work_rx) = mpsc::sync_channel::<FramePayload>(1);
    let (return_tx, return_rx) = mpsc::channel::<FramePayload>();
    // Stands in for the submit thread: it reads a payload and hands the whole of it back.
    let submit = std::thread::spawn(move || {
        while let Ok(payload) = work_rx.recv() {
            assert!(
                payload.scratch.bytes_used() > 0,
                "the frame's snapshots rode along"
            );
            if return_tx.send(payload).is_err() {
                break;
            }
        }
    });
    let mut live = ScratchArena::new();
    let mut blits = Vec::new();
    let mut post_constants = Vec::new();
    let mut pass_state = PassState::new();
    let mut pool: Vec<FramePayload> = Vec::new();
    let mut created = 0_u32;
    let mut seen = BTreeSet::new();
    let snapshot = vec![0x5a_u8; 30 * 1024];
    for frame in 0..FRAMES {
        // `begin_frame`, then two snapshot decodes' worth of storage and a constant block,
        // which together need two chunks.
        live.clear();
        for _ in 0..3 {
            live.alloc(&snapshot);
        }
        let chunks = live
            .allocation_ranges()
            .map(|(address, _)| address)
            .collect::<Vec<_>>();
        assert_eq!(chunks.len(), 2);
        if frame >= WARM_FRAMES {
            assert!(
                chunks.iter().all(|address| seen.contains(address)),
                "frame {frame} recorded into a chunk no earlier frame had"
            );
        }
        seen.extend(chunks);
        // `acquire_clean_payload`: reclaim what came back, reuse, create up to the cap, wait.
        while let Ok(mut returned) = return_rx.try_recv() {
            returned.clear(&mut pass_state);
            pool.push(returned);
        }
        let mut payload = pool.pop().unwrap_or_else(|| {
            if created < SUBMIT_PAYLOAD_CAP {
                created += 1;
                FramePayload::default()
            } else {
                let mut returned = return_rx.recv().unwrap();
                returned.clear(&mut pass_state);
                returned
            }
        });
        payload.adopt_frame_buffers(&mut live, &mut blits, &mut post_constants);
        work_tx.send(payload).unwrap();
    }
    drop(work_tx);
    submit.join().unwrap();
    while let Ok(mut returned) = return_rx.try_recv() {
        returned.clear(&mut pass_state);
        pool.push(returned);
    }
    // The encoder's arena and at most two payloads' arenas exist, each warmed to two chunks.
    let arenas = pool.iter().map(|payload| &payload.scratch).chain([&live]);
    let chunk_total = arenas.map(ScratchArena::chunk_count).sum::<u32>();
    assert_eq!(pool.len(), usize::try_from(created).unwrap());
    assert_eq!(chunk_total, 2 * (created + 1));
    assert_eq!(seen.len(), usize::try_from(chunk_total).unwrap());
}

/// An opaque texture handle, never dereferenced.
fn texture(raw: u64) -> MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind> {
    // SAFETY: tests; opaque values never dereferenced.
    unsafe { MetalHandle::new(raw) }
}

/// A destroyed source takes its depth snapshot and `StretchRect` scratch with it.
///
/// Both caches are keyed by the source texture's handle. A source destroyed
/// without its entries leaked a full-size Private copy per recreated depth
/// target or back buffer, and a texture Metal later created at the same
/// address was handed the old copy. Shutdown takes whatever is left.
#[test]
fn a_destroyed_source_takes_its_scratch_copies_and_leaves_the_others() {
    let mut snapshots = rustc_hash::FxHashMap::default();
    let mut scratch = rustc_hash::FxHashMap::default();
    for (source, copy) in [(0x100, 0x1100), (0x200, 0x1200)] {
        snapshots.insert(
            source,
            DepthSnapshot {
                handle: texture(copy),
                width: 64,
                height: 64,
                format: PixelFormat::Depth32Float,
                epoch: 0,
            },
        );
    }
    for (source, copy) in [(0x100, 0x2100), (0x300, 0x2300)] {
        scratch.insert(
            source,
            StretchScratch {
                handle: texture(copy),
                width: 64,
                height: 64,
                format: PixelFormat::Bgra8Unorm,
            },
        );
    }

    let taken = take_source_scratch(&mut snapshots, &mut scratch, 0x100);
    assert_eq!(
        taken.map(|copy| copy.map(MetalHandle::raw)),
        [Some(0x1100), Some(0x2100)],
        "both copies of the destroyed source are handed back for retirement"
    );
    assert_eq!(
        take_source_scratch(&mut snapshots, &mut scratch, 0x100).map(|copy| copy.is_some()),
        [false, false],
        "a texture created later at the same address finds no copy of the old one"
    );
    assert!(
        snapshots.contains_key(&0x200),
        "another source keeps its snapshot"
    );
    assert!(
        scratch.contains_key(&0x300),
        "another source keeps its scratch"
    );

    let mut rest: Vec<u64> = drain_source_scratch(&mut snapshots, &mut scratch)
        .into_iter()
        .map(MetalHandle::raw)
        .collect();
    rest.sort_unstable();
    assert_eq!(rest, [0x1200, 0x2300], "shutdown takes every copy left");
    assert!(snapshots.is_empty() && scratch.is_empty());
}

/// An opaque buffer handle, never dereferenced.
fn buffer(raw: u64) -> MetalHandle<mtld3d_shared::mtl_handle::MTLBufferKind> {
    // SAFETY: tests; opaque values never dereferenced.
    unsafe { MetalHandle::new(raw) }
}

/// A `Staged` entry whose CPU backing was released, the shape a `WRITEONLY` DEFAULT buffer reaches.
fn staged(device_buffer: u64, last_submit_seq: u64) -> BufferGpuState {
    BufferGpuState {
        mtl_buffer: MetalHandle::NULL,
        device_buffer: buffer(device_buffer),
        is_staged: true,
        backing_ptr: 0,
        length: 4096,
        backing_generation: 0,
        last_submit_seq,
    }
}

/// A `Direct` entry wrapping a CPU backing.
fn direct(wrapper: u64) -> BufferGpuState {
    BufferGpuState {
        mtl_buffer: buffer(wrapper),
        device_buffer: MetalHandle::NULL,
        is_staged: false,
        backing_ptr: 0x10_0000,
        length: 4096,
        backing_generation: 1,
        last_submit_seq: 3,
    }
}

/// A released `Staged` buffer with no backing hands its device buffer to the retention queue.
///
/// A `D3DPOOL_DEFAULT` `D3DUSAGE_WRITEONLY` buffer drops its CPU copy after
/// its upload, so its release sends no backing through the retention intake,
/// the one path that took the entry out before; its Private device buffer
/// leaked on every release, and every DEFAULT buffer is recreated around a
/// `Reset`. The destroy waits for the later of the frame that released the
/// buffer and the last frame that drew with it.
#[test]
fn a_released_staged_buffer_retires_its_device_buffer() {
    use mtld3d_core::ids::BufferId;

    let mut cache = rustc_hash::FxHashMap::default();
    let (released, kept, wrapped) = (
        BufferId::from_raw(1),
        BufferId::from_raw(2),
        BufferId::from_raw(3),
    );
    cache.insert(released, staged(0xD100, 7));
    cache.insert(kept, staged(0xD200, 7));
    cache.insert(wrapped, direct(0xC300));

    let entry = take_released_buffer(&mut cache, released, 5).expect("the device buffer retires");
    assert!(matches!(entry.kind, DestroyKind::Buffer));
    assert_eq!(entry.handle, 0xD100);
    assert_eq!(entry.seq, 7, "gated on the last frame that drew with it");
    assert!(entry.page_box.is_none(), "no CPU backing rides along");
    assert!(!cache.contains_key(&released));
    assert!(
        take_released_buffer(&mut cache, released, 9).is_none(),
        "a second destroy finds nothing"
    );
    assert_eq!(
        take_released_buffer(&mut cache, kept, 9).map(|entry| entry.seq),
        Some(9),
        "a release after the last draw waits for its own frame"
    );
    assert!(
        take_released_buffer(&mut cache, wrapped, 9).is_none(),
        "a Direct wrapper retires with its backing, not here"
    );
    assert!(cache.contains_key(&wrapped));
}

/// Shutdown collects the `Staged` device buffers as well as the `Direct` wrappers.
#[test]
fn shutdown_collects_staged_device_buffers_and_direct_wrappers() {
    use mtld3d_core::ids::BufferId;

    let mut cache = rustc_hash::FxHashMap::default();
    cache.insert(BufferId::from_raw(1), staged(0xD100, 1));
    cache.insert(BufferId::from_raw(2), direct(0xC200));
    // A staged entry whose warmup create failed holds no buffer yet.
    cache.insert(BufferId::from_raw(3), staged(0, 1));

    let mut handles = cached_buffer_handles(&cache);
    handles.sort_unstable();
    assert_eq!(handles, [0xC200, 0xD100]);
}

/// A cached level wrapper over `length` staging bytes, or an empty slot for a null `handle`.
fn wrapper(handle: u64, length: u64) -> MipStagingBuffer {
    MipStagingBuffer {
        handle: buffer(handle),
        backing_ptr: handle << 16,
        length,
        keepalive: None,
        kept_after_release: false,
    }
}

/// A cached texture whose level slots are `levels`.
fn wrapped_texture(levels: Vec<MipStagingBuffer>) -> TextureGpuState {
    TextureGpuState {
        views: mtld3d_shared::texture_views::TextureViews::EMPTY,
        mip_staging_buffers: levels,
        render_target: false,
    }
}

/// The wrapper gauge sums every populated level slot of every cached texture.
///
/// An empty slot (a level never uploaded, or one whose wrapper was parked)
/// adds nothing, whatever length it last recorded.
#[test]
fn the_wrapper_gauge_counts_populated_level_slots_only() {
    let mut cache = rustc_hash::FxHashMap::default();
    assert_eq!(staging_wrapped_bytes(&cache), 0);
    cache.insert(
        mtld3d_core::ids::TextureId::from_raw(1),
        wrapped_texture(vec![
            wrapper(0x10, 64 << 10),
            wrapper(0x11, 16 << 10),
            wrapper(0, 32 << 10),
        ]),
    );
    cache.insert(
        mtld3d_core::ids::TextureId::from_raw(2),
        wrapped_texture(vec![wrapper(0, 16 << 10), wrapper(0x20, 1 << 20)]),
    );
    assert_eq!(
        staging_wrapped_bytes(&cache),
        (64 << 10) + (16 << 10) + (1 << 20)
    );
}

/// Retiring a level's wrapper empties exactly that slot and hands back its keepalive.
///
/// The keepalive is the native owner of the guest pages, so it has to travel
/// with the wrapper into the retention queue rather than stay in the cache;
/// an empty slot, another level, an uncached texture and an index past the
/// texture's levels hand back nothing.
#[test]
fn taking_a_staging_wrapper_empties_its_slot_and_keeps_the_others() {
    let pages = std::sync::Arc::new(mtld3d_core::page_box::PageBox::new_zeroed(2));
    let id = mtld3d_core::ids::TextureId::from_raw(7);
    let mut cache = rustc_hash::FxHashMap::default();
    cache.insert(
        id,
        wrapped_texture(vec![
            MipStagingBuffer {
                keepalive: Some(std::sync::Arc::clone(&pages)),
                ..wrapper(0x70, 32 << 10)
            },
            wrapper(0x71, 16 << 10),
            wrapper(0, 16 << 10),
        ]),
    );

    let taken = take_staging_wrapper(&mut cache, id, 0).expect("level 0 is wrapped");
    assert_eq!(taken.handle.raw(), 0x70);
    assert!(
        taken
            .keepalive
            .as_ref()
            .is_some_and(|owner| std::sync::Arc::ptr_eq(owner, &pages)),
        "the pages' owner leaves the cache with the wrapper"
    );
    assert!(
        take_staging_wrapper(&mut cache, id, 0).is_none(),
        "the slot is empty afterwards, so a later upload creates a fresh wrapper"
    );
    assert!(
        take_staging_wrapper(&mut cache, id, 2).is_none(),
        "an empty slot"
    );
    assert!(
        take_staging_wrapper(&mut cache, id, 3).is_none(),
        "past the levels"
    );
    assert!(
        take_staging_wrapper(&mut cache, mtld3d_core::ids::TextureId::from_raw(8), 0).is_none(),
        "an uncached texture"
    );
    assert_eq!(
        staging_wrapped_bytes(&cache),
        16 << 10,
        "the other level keeps its wrapper"
    );
    drop(taken);
    assert_eq!(std::sync::Arc::strong_count(&pages), 1);
}

/// A release answer retires a level's wrapper once per backing.
///
/// The first answer takes the wrapper and leaves the backing's address in
/// the slot. An upload that wraps the same backing again shows the PE side
/// kept it (a level rewritten every frame, whose answer a newer upload
/// overtakes), so that wrapper stays cached through later answers instead
/// of being created and destroyed every frame. A different backing starts
/// over, and a backing change still retires a kept wrapper.
#[test]
fn a_release_retires_a_levels_wrapper_once_per_backing() {
    let id = mtld3d_core::ids::TextureId::from_raw(9);
    let pages = std::sync::Arc::new(mtld3d_core::page_box::PageBox::new_zeroed(2));
    let mut cache = rustc_hash::FxHashMap::default();
    cache.insert(id, wrapped_texture(vec![wrapper(0x90, 32 << 10)]));
    let slot = |cache: &rustc_hash::FxHashMap<_, TextureGpuState>| {
        let state: &TextureGpuState = &cache[&id];
        let slot = &state.mip_staging_buffers[0];
        (slot.handle.raw(), slot.backing_ptr, slot.kept_after_release)
    };

    let first = take_released_staging_wrapper(&mut cache, id, 0).expect("first release");
    assert_eq!(first.handle.raw(), 0x90);
    assert_eq!(
        slot(&cache),
        (0, 0x90 << 16, false),
        "the emptied slot remembers the backing"
    );
    assert!(
        take_released_staging_wrapper(&mut cache, id, 0).is_none(),
        "an empty slot has nothing to retire"
    );

    let state = cache.get_mut(&id).expect("cached");
    let prior = &state.mip_staging_buffers[0];
    let again = MipStagingBuffer::created(
        buffer(0x91),
        0x90 << 16,
        32 << 10,
        std::sync::Arc::clone(&pages),
        prior,
    );
    state.mip_staging_buffers[0] = again;
    assert_eq!(slot(&cache), (0x91, 0x90 << 16, true));
    assert!(
        take_released_staging_wrapper(&mut cache, id, 0).is_none(),
        "the second wrapper over the same backing stays cached"
    );
    assert_eq!(staging_wrapped_bytes(&cache), 32 << 10);
    assert!(
        take_staging_wrapper(&mut cache, id, 0).is_some(),
        "a backing change still retires a kept wrapper"
    );

    for (prior, backing, length) in [
        (wrapper(0, 32 << 10), 0x90 << 16, 32 << 10),
        (wrapper(0x92, 32 << 10), 0x92 << 16, 32 << 10),
        (
            MipStagingBuffer {
                handle: buffer(0),
                backing_ptr: 0x90 << 16,
                length: 32 << 10,
                ..MipStagingBuffer::default()
            },
            0x90 << 16,
            16 << 10,
        ),
    ] {
        let fresh = MipStagingBuffer::created(
            buffer(0x93),
            backing,
            length,
            std::sync::Arc::clone(&pages),
            &prior,
        );
        assert!(
            !fresh.kept_after_release,
            "a never-released slot, a live wrapper and another length start over"
        );
    }
    drop(first);
}

/// A submission that fails on the CPU ends the frame-latency wait of a `Present` behind it.
///
/// The failed frame publishes its sequence as retired, but it was never noted
/// as submitted and never presents; the failure is what has to end the wait,
/// and it reaches the device's mailbox too.
#[test]
fn a_failed_submission_ends_the_frame_latency_wait() {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicU32, AtomicU64, Ordering},
        },
        time::Duration,
    };

    use mtld3d_shared::mtl::PresentDebugFlags;
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

    use crate::metal::{
        DeviceRecord, FrameWait,
        submission::{RetirementCounter, SubmitDescription},
    };

    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> = device
        .newCommandQueue()
        .expect("a queue on the default device");
    // SAFETY: the raw address carries this retain, which the record adopts and
    // releases when it drops at the end of this test.
    let handle = unsafe { MetalHandle::new(Retained::into_raw(queue) as u64) };
    let record = DeviceRecord::new(handle, None, PresentDebugFlags::empty());
    let failure = AtomicU32::new(0);
    let floor = AtomicU64::new(0);
    let draw = Arc::new(AtomicU64::new(0));
    // SAFETY: `floor` outlives every use of the signal in this test.
    let floor_counter =
        unsafe { RetirementCounter::from_address(core::ptr::from_ref(&floor) as u64) };
    record.present().frame_signal().attach(floor_counter);
    let (sender, receiver) = mpsc::channel();
    let waiter = {
        let signal = Arc::clone(record.present().frame_signal());
        let draw = Arc::clone(&draw);
        std::thread::spawn(move || {
            // SAFETY: the waiter owns a reference to the atomic it reads.
            let draw = unsafe { RetirementCounter::from_address(Arc::as_ptr(&draw) as u64) };
            let _ = sender.send(signal.wait(draw, 3, crate::metal::PACED_PRESENT_CAP));
        })
    };
    assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
    // An upload prefix longer than the pass list fails the encode on the CPU.
    let description = SubmitDescription {
        blit_commands_need_encoder: false,
        upload_pass_count: 1,
        present_layer: MetalHandle::NULL,
        present_texture: MetalHandle::NULL,
        present_view: MetalHandle::NULL,
        present_scene_linear: false,
        submit_seq: 3,
        // SAFETY: the waiter's own reference keeps this atomic alive past the submission.
        draw_retirement: unsafe { RetirementCounter::from_address(Arc::as_ptr(&draw) as u64) },
        upload_retirement: RetirementCounter::NONE,
        failed_submission: RetirementCounter::NONE,
    };
    let (_payload, outcome) = super::execute_submit(
        Some(&record),
        &description,
        super::FramePayload::default(),
        core::ptr::from_ref(&failure) as u64,
    );
    assert_ne!(outcome.status, 0, "the submission failed");
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(5)),
        Ok(FrameWait::Ended)
    );
    waiter.join().unwrap();
    assert_eq!(
        failure.load(Ordering::Acquire),
        1,
        "the mailbox holds the failure"
    );
}

/// A grown-out fan index pattern counts in the retained bytes while it waits, and leaves them.
///
/// The retained-bytes counter is the API thread's retention-cap input and the
/// HUD's in-flight figure; a box the drain subtracts without having added it
/// wraps the counter to near `u64::MAX`. Shutdown, like a reset, drains the
/// queue without waiting for the gate, and has to subtract what it takes too.
#[test]
fn a_grown_fan_pattern_is_counted_in_the_retained_bytes_until_it_drains() {
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    use mtld3d_core::{
        config::Mtld3dConfig, convert::fan_pattern_bytes, gpu_caps::GpuCaps, page_box::PageBox,
    };
    use mtld3d_shared::{clock_calibration::ClockCalibration, record_handle::DeviceRecordHandle};

    use crate::encoder_service::EncoderContext;

    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    // Nothing is submitted, so every queued entry's sequence (0) has retired.
    let coherent = AtomicU64::new(0);
    let retained = AtomicU64::new(0);
    let context = EncoderContext {
        // SAFETY: `device` stays retained for the whole test; the encoder takes a retain of its own.
        device_handle: unsafe { MetalHandle::new(Retained::as_ptr(&device) as u64) },
        record_handle: DeviceRecordHandle::NULL,
        coherent_seq_ptr: core::ptr::from_ref(&coherent) as u64,
        upload_coherent_seq_ptr: 0,
        failed_submit_seq_ptr: 0,
        retained_bytes_ptr: core::ptr::from_ref(&retained) as u64,
        paced_present_seq_ptr: 0,
        largest_free_word_ptr: 0,
        guest_page_bytes_ptr: 0,
        guest_pool_bytes_ptr: 0,
    };
    let mut enc = super::FrameEncoder::new(
        GpuCaps::apple_silicon_default(),
        Arc::new(Mtld3dConfig::default()),
        None,
        Arc::new(ClockCalibration::new()),
        &context,
        super::NativeSubmitSpawner,
    )
    .expect("an encoder on the default device");

    assert_ne!(enc.fan_index_buffer(300), 0);
    assert_eq!(
        retained.load(Ordering::Relaxed),
        0,
        "the live pattern is not retention"
    );
    assert_ne!(enc.fan_index_buffer(1000), 0);
    assert_eq!(
        retained.load(Ordering::Relaxed),
        PageBox::padded_len(fan_pattern_bytes(300)) as u64,
        "the grown-out pattern waits in retention"
    );
    enc.drain_retired_resource_retention();
    assert_eq!(
        retained.load(Ordering::Relaxed),
        0,
        "the drain takes back exactly what was added"
    );

    assert_ne!(enc.fan_index_buffer(5000), 0);
    assert_ne!(retained.load(Ordering::Relaxed), 0);
    enc.shutdown_cleanup();
    assert_eq!(
        retained.load(Ordering::Relaxed),
        0,
        "shutdown's drain takes back what it empties"
    );
    drop(enc);
}

/// The slot-6 per-pixel specular uniform the encoder binds follows its
/// config: scale, numeric power or `auto` (0), and the fallback colour or
/// none.
#[test]
fn ppl_params_follow_the_encoder_config() {
    use std::sync::{Arc, atomic::AtomicU64};

    use mtld3d_core::{
        config::Mtld3dConfig,
        gpu_caps::GpuCaps,
        ppl::{PPL_PARAMS_BYTES, PplConfig},
    };
    use mtld3d_shared::{clock_calibration::ClockCalibration, record_handle::DeviceRecordHandle};

    use crate::encoder_service::EncoderContext;

    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let coherent = AtomicU64::new(0);
    let retained = AtomicU64::new(0);
    let context = EncoderContext {
        // SAFETY: `device` stays retained for the whole test; the encoder takes a retain of its own.
        device_handle: unsafe { MetalHandle::new(Retained::as_ptr(&device) as u64) },
        record_handle: DeviceRecordHandle::NULL,
        coherent_seq_ptr: core::ptr::from_ref(&coherent) as u64,
        upload_coherent_seq_ptr: 0,
        failed_submit_seq_ptr: 0,
        retained_bytes_ptr: core::ptr::from_ref(&retained) as u64,
        paced_present_seq_ptr: 0,
        largest_free_word_ptr: 0,
        guest_page_bytes_ptr: 0,
        guest_pool_bytes_ptr: 0,
    };
    let lanes = |bytes: &[u8; PPL_PARAMS_BYTES]| -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let cases = [
        (
            PplConfig::default(),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            PplConfig {
                specular: true,
                specular_scale_pct: 40,
                specular_power: Some(16),
                specular_default_pct: Some([50, 100, 0]),
                ..PplConfig::default()
            },
            [0.4, 16.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0],
        ),
    ];
    for (ppl, expected) in cases {
        let mut enc = super::FrameEncoder::new(
            GpuCaps::apple_silicon_default(),
            Arc::new(Mtld3dConfig {
                ppl,
                ..Mtld3dConfig::default()
            }),
            None,
            Arc::new(ClockCalibration::new()),
            &context,
            super::NativeSubmitSpawner,
        )
        .expect("an encoder on the default device");
        assert_eq!(lanes(enc.ppl_params()), expected, "{ppl:?}");
        enc.shutdown_cleanup();
        drop(enc);
    }
}

/// `post.debug` alone.
const POST_DEBUG: mtld3d_core::post::PostConfig = mtld3d_core::post::PostConfig {
    debug: true,
    ao_pct: 0,
    god_rays_pct: 0,
    bloom_pct: 0,
    bloom_threshold_pct: 80,
};

/// A `FrameEncoder` on the default device with the `post` keys, or `None` without a device.
pub(super) fn post_encoder(
    coherent: &std::sync::atomic::AtomicU64,
    retained: &std::sync::atomic::AtomicU64,
    post: mtld3d_core::post::PostConfig,
) -> Option<(Retained<ProtocolObject<dyn MTLDevice>>, super::FrameEncoder)> {
    post_encoder_with(coherent, retained, post, false)
}

/// [`post_encoder`] with `color.sceneFloat` set to `scene_float`.
fn post_encoder_with(
    coherent: &std::sync::atomic::AtomicU64,
    retained: &std::sync::atomic::AtomicU64,
    post: mtld3d_core::post::PostConfig,
    scene_float: bool,
) -> Option<(Retained<ProtocolObject<dyn MTLDevice>>, super::FrameEncoder)> {
    encoder_with_config(
        coherent,
        retained,
        mtld3d_core::config::Mtld3dConfig {
            post,
            scene_float,
            ..mtld3d_core::config::Mtld3dConfig::default()
        },
    )
}

/// A `FrameEncoder` on the default device with `config`, or `None` without a device.
pub(super) fn encoder_with_config(
    coherent: &std::sync::atomic::AtomicU64,
    retained: &std::sync::atomic::AtomicU64,
    config: mtld3d_core::config::Mtld3dConfig,
) -> Option<(Retained<ProtocolObject<dyn MTLDevice>>, super::FrameEncoder)> {
    use std::sync::Arc;

    use mtld3d_core::gpu_caps::GpuCaps;
    use mtld3d_shared::{clock_calibration::ClockCalibration, record_handle::DeviceRecordHandle};

    use crate::encoder_service::EncoderContext;

    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return None;
    };
    let context = EncoderContext {
        // SAFETY: the caller keeps `device` retained for the whole test; the encoder takes a retain of its own.
        device_handle: unsafe { MetalHandle::new(Retained::as_ptr(&device) as u64) },
        record_handle: DeviceRecordHandle::NULL,
        coherent_seq_ptr: core::ptr::from_ref(coherent) as u64,
        upload_coherent_seq_ptr: 0,
        failed_submit_seq_ptr: 0,
        retained_bytes_ptr: core::ptr::from_ref(retained) as u64,
        paced_present_seq_ptr: 0,
        largest_free_word_ptr: 0,
        guest_page_bytes_ptr: 0,
        guest_pool_bytes_ptr: 0,
    };
    let enc = super::FrameEncoder::new(
        GpuCaps::apple_silicon_default(),
        Arc::new(config),
        None,
        Arc::new(ClockCalibration::new()),
        &context,
        super::NativeSubmitSpawner,
    )
    .expect("an encoder on the default device");
    Some((device, enc))
}

/// Start a frame on an opaque back-buffer handle the post chain only names,
/// with an opaque depth attachment of the same size when `depth` is set.
pub(super) fn post_reset_frame(
    enc: &mut super::FrameEncoder,
    size: (u32, u32),
    samples: u8,
    depth: bool,
) {
    use mtld3d_core::{
        passes::{BackbufferContents, FrameReset},
        render_scale::RenderScale,
    };
    enc.pass_state.reset_frame(&FrameReset {
        // SAFETY: an opaque value the queued blits carry but nothing dereferences.
        backbuffer: unsafe { MetalHandle::new(0xB000) },
        backbuffer_srgb: MetalHandle::NULL,
        backbuffer_msaa: if samples > 1 {
            // SAFETY: as above.
            unsafe { MetalHandle::new(0xB100) }
        } else {
            MetalHandle::NULL
        },
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: samples,
        backbuffer_size: size,
        backbuffer_format: PixelFormat::Bgra8Unorm,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: if depth {
            // SAFETY: as above; the snapshot copy names it but nothing dereferences it.
            unsafe { MetalHandle::new(0xD000) }
        } else {
            MetalHandle::NULL
        },
        depth_size: if depth { size } else { (0, 0) },
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    if depth {
        enc.set_depth_attachment_desc(size.0, size.1, PixelFormat::Depth32Float);
    }
}

pub(super) fn post_boundary_op() -> mtld3d_core::encoder_data::PostBoundary {
    mtld3d_core::encoder_data::PostBoundary {
        proj: [1.5, 2.0, 1.001, -0.1],
        sun_dir_eye: None,
        sun_color: [0.0; 3],
        sun_above_horizon: false,
        ..mtld3d_core::encoder_data::PostBoundary::default()
    }
}

/// Draws into a multisampled back buffer land in its companion, so a chain run
/// on the resolve target would be lost: the boundary queues nothing.
#[test]
fn post_skips_msaa_backbuffer() {
    use std::sync::atomic::AtomicU64;
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = post_encoder(&coherent, &retained, POST_DEBUG) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 4, false);
    enc.post_boundary(&post_boundary_op());
    assert_eq!(enc.pass_state.take_pending_leading_blits().len(), 0);
    assert_eq!(enc.post_constants.len(), 0);
    assert!(
        enc.post
            .scratch_handle(crate::metal::post::Slot::ColourA)
            .is_none()
    );
    enc.shutdown_cleanup();
}

/// The boundary queues one effect and the copy back; a new back-buffer size
/// replaces the colour scratch and retires the old one behind the GPU.
#[test]
fn post_boundary_queues_the_chain_and_retires_resized_scratch() {
    use std::sync::atomic::AtomicU64;

    use mtld3d_shared::BlitCommandType;
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = post_encoder(&coherent, &retained, POST_DEBUG) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, false);
    enc.post_boundary(&post_boundary_op());
    let blits = enc.pass_state.take_pending_leading_blits();
    assert_eq!(blits.len(), 2);
    let a = enc
        .post
        .scratch_handle(crate::metal::post::Slot::ColourA)
        .expect("a colour scratch");
    assert_eq!(blits[0].cmd, BlitCommandType::PostEffect as u32);
    assert_eq!(
        blits[0].src_handle, 0xB000,
        "effect 0 reads the back buffer"
    );
    assert_eq!(blits[0].dst_handle, a.raw());
    assert_eq!(blits[0].byte_size, 0, "the first constants row");
    assert_eq!((blits[0].region_w, blits[0].region_h), (64, 64));
    assert_ne!(blits[0].src_offset, 0, "the compute pipeline");
    assert_eq!(blits[0].dst_offset, 0, "the debug effect reads no depth");
    assert_eq!(blits[1].cmd, BlitCommandType::CopyTextureToTexture as u32);
    assert_eq!(
        (blits[1].src_handle, blits[1].dst_handle),
        (a.raw(), 0xB000)
    );
    assert_eq!(enc.post_constants.len(), 1);
    assert_eq!(enc.post_constants[0].proj, [1.5, 2.0, 1.001, -0.1]);
    assert_eq!(enc.post_constants[0].extent, [64, 64, 0, 0]);

    let before = enc.pending_resource_retention.len();
    post_reset_frame(&mut enc, (128, 64), 1, false);
    enc.post_boundary(&post_boundary_op());
    let retired: Vec<u64> = enc
        .pending_resource_retention
        .iter()
        .skip(before)
        .map(|entry| entry.handle)
        .collect();
    assert!(
        retired.contains(&a.raw()),
        "the 64x64 colour scratch waits for the GPU"
    );
    assert_eq!(retired.len(), 1, "one effect uses one colour slot");
    let b = enc
        .post
        .scratch_handle(crate::metal::post::Slot::ColourA)
        .expect("a new colour scratch");
    assert_ne!(b.raw(), a.raw());
    let blits = enc.pass_state.take_pending_leading_blits();
    assert_eq!((blits[0].region_w, blits[0].region_h), (128, 64));
    // Rows accumulate until the payload swap at submit takes the table.
    assert_eq!(blits[0].byte_size, 1, "the second constants row");
    enc.shutdown_cleanup();
}

/// Ambient occlusion reads depth: without a depth attachment to snapshot it
/// steps aside alone, and the rest of the chain runs as before.
#[test]
fn ao_steps_aside_without_depth() {
    use std::sync::atomic::AtomicU64;

    use mtld3d_shared::BlitCommandType;

    use crate::metal::post::Kernel;
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let ao_only = mtld3d_core::post::PostConfig {
        ao_pct: 100,
        debug: false,
        god_rays_pct: 0,
        bloom_pct: 0,
        bloom_threshold_pct: 80,
    };
    let Some((_device, mut enc)) = post_encoder(&coherent, &retained, ao_only) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, false);
    enc.post_boundary(&post_boundary_op());
    assert_eq!(enc.pass_state.take_pending_leading_blits().len(), 0);
    assert_eq!(enc.post_constants.len(), 0);
    enc.shutdown_cleanup();

    let both = mtld3d_core::post::PostConfig {
        ao_pct: 100,
        debug: true,
        god_rays_pct: 0,
        bloom_pct: 0,
        bloom_threshold_pct: 80,
    };
    let Some((_device, mut enc)) = post_encoder(&coherent, &retained, both) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, false);
    enc.post_boundary(&post_boundary_op());
    let blits = enc.pass_state.take_pending_leading_blits();
    assert_eq!(blits.len(), 2, "the invert and the copy back");
    assert_eq!(blits[0].cmd, BlitCommandType::PostEffect as u32);
    assert_eq!(blits[0].mip_level, Kernel::Invert as u32);
    assert_eq!(enc.post_constants[0].extent, [64, 64, 0, 0]);
    enc.shutdown_cleanup();
}

/// With a depth attachment, AO runs its three passes after the depth
/// snapshot: the horizon search and the blur at half extent into their own
/// slots, then the full-extent pass that darkens the colour.
#[test]
fn ao_queues_its_passes_after_the_depth_snapshot() {
    use std::sync::atomic::AtomicU64;

    use mtld3d_shared::BlitCommandType;

    use crate::metal::post::{Kernel, Slot};
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let both = mtld3d_core::post::PostConfig {
        ao_pct: 150,
        debug: true,
        god_rays_pct: 0,
        bloom_pct: 0,
        bloom_threshold_pct: 80,
    };
    let Some((_device, mut enc)) = post_encoder(&coherent, &retained, both) else {
        return;
    };
    post_reset_frame(&mut enc, (63, 33), 1, true);
    // Taking a real snapshot needs a device record, which a test encoder has
    // not; seed the cache with a stale copy so the boundary refreshes it.
    enc.depth_snapshots.insert(
        0xD000,
        super::DepthSnapshot {
            // SAFETY: an opaque value the queued blits carry but nothing dereferences.
            handle: unsafe { MetalHandle::new(0xD100) },
            width: 63,
            height: 33,
            format: PixelFormat::Depth32Float,
            epoch: enc.depth_write_epoch.wrapping_sub(1),
        },
    );
    enc.post_boundary(&post_boundary_op());
    let blits = enc.pass_state.take_pending_leading_blits();
    let slot = |slot| {
        enc.post
            .scratch_handle(slot)
            .expect("a scratch texture")
            .raw()
    };
    assert_eq!(
        blits.len(),
        6,
        "snapshot, three AO passes, invert, copy back"
    );
    assert_eq!(blits[0].cmd, BlitCommandType::CopyTextureToTexture as u32);
    assert_eq!(
        (blits[0].src_handle, blits[0].dst_handle),
        (0xD000, 0xD100),
        "the depth snapshot is refreshed first"
    );
    let snapshot = blits[0].dst_handle;
    let kernels: Vec<u32> = blits[1..5].iter().map(|b| b.mip_level).collect();
    assert_eq!(
        kernels,
        [
            Kernel::AoHorizon as u32,
            Kernel::AoBlur as u32,
            Kernel::AoApply as u32,
            Kernel::Invert as u32
        ]
    );
    let [horizon, blur, apply, invert] = [blits[1], blits[2], blits[3], blits[4]];
    for pass in [horizon, blur, apply] {
        assert_eq!(pass.cmd, BlitCommandType::PostEffect as u32);
        assert_eq!(pass.dst_offset, snapshot, "every AO pass reads depth");
    }
    assert_eq!((horizon.region_w, horizon.region_h), (32, 17));
    assert_eq!(horizon.dst_handle, slot(Slot::AoRaw));
    assert_eq!((blur.region_w, blur.region_h), (32, 17));
    assert_eq!(blur.src_handle, slot(Slot::AoRaw));
    assert_eq!(blur.dst_handle, slot(Slot::AoBlurred));
    assert_eq!((apply.region_w, apply.region_h), (63, 33));
    assert_eq!(apply.src_handle, 0xB000, "AO reads the world's colour");
    assert_eq!(apply.bytes_per_row, slot(Slot::AoBlurred));
    assert_eq!(apply.dst_handle, slot(Slot::ColourA));
    assert_eq!(invert.src_handle, slot(Slot::ColourA));
    assert_eq!(invert.dst_handle, slot(Slot::ColourB));
    assert_eq!(invert.dst_offset, 0, "the invert reads no depth");
    assert_eq!(
        (blits[5].src_handle, blits[5].dst_handle),
        (slot(Slot::ColourB), 0xB000)
    );
    let k = enc.post_constants[0];
    assert_eq!(k.extent, [63, 33, 1, 0]);
    assert_eq!(k.strength, [1.5, 0.0, 0.0, 0.0]);
    // The seeded snapshot is not a texture the shutdown could destroy.
    enc.depth_snapshots.clear();
    enc.shutdown_cleanup();
}

/// God rays with no visible sun drop out at the boundary: no depth snapshot,
/// no pass and no constants row of theirs; with a sun ahead they run their
/// three passes after the snapshot.
#[test]
fn godrays_queue_nothing_without_a_visible_sun() {
    use std::sync::atomic::AtomicU64;

    use mtld3d_shared::BlitCommandType;

    use crate::metal::post::{Kernel, Slot};
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let rays = mtld3d_core::post::PostConfig {
        god_rays_pct: 100,
        ..mtld3d_core::post::PostConfig::default()
    };
    let Some((_device, mut enc)) = post_encoder(&coherent, &retained, rays) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, true);
    enc.depth_snapshots.insert(
        0xD000,
        super::DepthSnapshot {
            // SAFETY: an opaque value the queued blits carry but nothing dereferences.
            handle: unsafe { MetalHandle::new(0xD100) },
            width: 64,
            height: 64,
            format: PixelFormat::Depth32Float,
            epoch: enc.depth_write_epoch.wrapping_sub(1),
        },
    );
    // No sun, a set sun straight ahead, a sun behind, one far off screen.
    for (sun_dir_eye, sun_above_horizon) in [
        (None, false),
        (Some([0.0, 0.0, -1.0]), false),
        (Some([0.0, 0.0, 1.0]), true),
        (Some([-2.0, 0.0, -1.0]), true),
    ] {
        enc.post_boundary(&mtld3d_core::encoder_data::PostBoundary {
            sun_dir_eye,
            sun_above_horizon,
            ..post_boundary_op()
        });
        assert_eq!(
            enc.pass_state.take_pending_leading_blits().len(),
            0,
            "{sun_dir_eye:?}"
        );
        assert_eq!(enc.post_constants.len(), 0, "{sun_dir_eye:?}");
    }
    enc.post_boundary(&mtld3d_core::encoder_data::PostBoundary {
        sun_dir_eye: Some([0.0, 0.0, -1.0]),
        sun_color: [1.0, 0.9, 0.8],
        sun_above_horizon: true,
        ..post_boundary_op()
    });
    let blits = enc.pass_state.take_pending_leading_blits();
    assert_eq!(blits.len(), 5, "snapshot, three god-ray passes, copy back");
    assert_eq!(blits[0].cmd, BlitCommandType::CopyTextureToTexture as u32);
    let kernels: Vec<u32> = blits[1..4].iter().map(|b| b.mip_level).collect();
    assert_eq!(
        kernels,
        [
            Kernel::GodRaysMask as u32,
            Kernel::GodRaysBlur as u32,
            Kernel::GodRaysApply as u32
        ]
    );
    let slot = |slot| {
        enc.post
            .scratch_handle(slot)
            .expect("a scratch texture")
            .raw()
    };
    let [mask, blur, apply] = [blits[1], blits[2], blits[3]];
    assert_eq!(mask.dst_offset, 0xD100, "the mask reads depth");
    assert_eq!((mask.region_w, mask.region_h), (32, 32));
    assert_eq!(mask.dst_handle, slot(Slot::GodRaysMask));
    assert_eq!(blur.src_handle, slot(Slot::GodRaysMask));
    assert_eq!(blur.dst_handle, slot(Slot::GodRays));
    assert_eq!((blur.region_w, blur.region_h), (32, 32));
    assert_eq!(apply.src_handle, 0xB000);
    assert_eq!(apply.bytes_per_row, slot(Slot::GodRays));
    assert_eq!((apply.region_w, apply.region_h), (64, 64));
    let k = enc.post_constants[0];
    assert_eq!(k.strength, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(k.sun_screen, [0.5, 0.5, 1.0, 0.0]);
    enc.depth_snapshots.clear();
    enc.shutdown_cleanup();
}

/// Bloom reads no depth: without a depth attachment it still runs, ten
/// passes down to a sixteenth of the raster and back, then the copy back.
#[test]
fn bloom_runs_without_depth() {
    use std::sync::atomic::AtomicU64;

    use mtld3d_shared::BlitCommandType;

    use crate::metal::post::{Kernel, Slot};
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let cfg = mtld3d_core::post::PostConfig {
        ao_pct: 100,
        bloom_pct: 50,
        ..mtld3d_core::post::PostConfig::default()
    };
    let Some((_device, mut enc)) = post_encoder(&coherent, &retained, cfg) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, false);
    enc.post_boundary(&post_boundary_op());
    let blits = enc.pass_state.take_pending_leading_blits();
    assert_eq!(blits.len(), 11, "ten bloom passes and the copy back");
    let slot = |slot| {
        enc.post
            .scratch_handle(slot)
            .expect("a scratch texture")
            .raw()
    };
    for pass in &blits[..10] {
        assert_eq!(pass.cmd, BlitCommandType::PostEffect as u32);
        assert_eq!(pass.dst_offset, 0, "bloom reads no depth");
    }
    let kernels: Vec<u32> = blits[..10].iter().map(|b| b.mip_level).collect();
    let d = Kernel::BloomDownsample as u32;
    let u = Kernel::BloomUpsample as u32;
    assert_eq!(
        kernels,
        [
            Kernel::BloomThreshold as u32,
            d,
            d,
            d,
            d,
            u,
            u,
            u,
            u,
            Kernel::BloomApply as u32
        ]
    );
    let sizes: Vec<(u32, u32)> = blits[..10]
        .iter()
        .map(|b| (b.region_w, b.region_h))
        .collect();
    assert_eq!(
        sizes,
        [
            (32, 32),
            (16, 16),
            (8, 8),
            (4, 4),
            (2, 2),
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64)
        ]
    );
    assert_eq!(
        blits[0].src_handle, 0xB000,
        "bloom reads the world's colour"
    );
    assert_eq!(blits[0].dst_handle, slot(Slot::BloomDown0));
    assert_eq!(blits[4].dst_handle, slot(Slot::BloomDown4));
    let first_up = blits[5];
    assert_eq!(first_up.src_handle, slot(Slot::BloomDown3));
    assert_eq!(first_up.bytes_per_row, slot(Slot::BloomDown4));
    assert_eq!(first_up.dst_handle, slot(Slot::BloomUp3));
    let apply = blits[9];
    assert_eq!(apply.src_handle, 0xB000);
    assert_eq!(apply.bytes_per_row, slot(Slot::BloomUp0));
    assert_eq!(apply.dst_handle, slot(Slot::ColourA));
    assert_eq!(
        (blits[10].src_handle, blits[10].dst_handle),
        (slot(Slot::ColourA), 0xB000)
    );
    let k = enc.post_constants[0];
    assert_eq!(k.extent, [64, 64, 0, 0]);
    assert_eq!(k.strength, [1.0, 0.0, 0.5, 0.8]);
    enc.shutdown_cleanup();
}

/// `color.sceneFloat` puts the tone map last in the chain, so the boundary's
/// leading blits end with it and the copy back; the UI draw that follows is
/// encoded after them in that same pass, on the tone-mapped colour, with the
/// stock clamps and a saturated store.
#[test]
fn ui_after_boundary_not_tonemapped() {
    use std::sync::atomic::AtomicU64;

    use mtld3d_shared::{BlitCommandType, Command, mtl::PrimitiveType};

    use crate::metal::post::{Kernel, Slot};
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let Some((_device, mut enc)) = post_encoder_with(&coherent, &retained, POST_DEBUG, true) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, false);
    let world = Command::draw_primitives(PrimitiveType::Triangle, 0, 3);
    let ui = Command::draw_primitives(PrimitiveType::Triangle, 3, 6);
    use mtld3d_core::dxso::{COLOR_STYLE_SCENE_FLOAT, COLOR_STYLE_STORE_CLAMP};
    // The world draw floors its colour; the UI draw after the tone map keeps
    // the stock clamps and saturates its final colour, storing [0, 1].
    assert_eq!(enc.ps_color_style(), COLOR_STYLE_SCENE_FLOAT, "the world");
    enc.pass_state.emit_command(world);
    enc.post_boundary(&post_boundary_op());
    assert_eq!(enc.ps_color_style(), COLOR_STYLE_STORE_CLAMP, "the UI");
    enc.pass_state.emit_command(ui);
    enc.pass_state.end_current_pass("test");
    let passes = enc.pass_state.passes();
    assert_eq!(passes.len(), 2, "the world's pass, then the UI's");
    let draws = |pass: &mtld3d_core::passes::Pass| -> Vec<u64> {
        pass.commands()
            .iter()
            .filter(|c| c.cmd == world.cmd)
            .map(|c| c.param_b)
            .collect()
    };
    assert!(passes[0].leading_blits().is_empty());
    assert_eq!(draws(&passes[0]), [world.param_b]);
    let blits = passes[1].leading_blits();
    let kernels: Vec<u32> = blits
        .iter()
        .filter(|b| b.cmd == BlitCommandType::PostEffect as u32)
        .map(|b| b.mip_level)
        .collect();
    assert_eq!(kernels, [Kernel::Invert as u32, Kernel::ToneMap as u32]);
    let [.., tone_map, copy_back] = blits else {
        panic!("the tone map and the copy back lead the UI pass");
    };
    assert_eq!(tone_map.cmd, BlitCommandType::PostEffect as u32);
    assert_eq!(tone_map.mip_level, Kernel::ToneMap as u32);
    let slot = |slot| {
        enc.post
            .scratch_handle(slot)
            .expect("a scratch texture")
            .raw()
    };
    assert_eq!(tone_map.src_handle, slot(Slot::ColourA), "after the invert");
    assert_eq!(tone_map.dst_handle, slot(Slot::ColourB));
    assert_eq!(copy_back.cmd, BlitCommandType::CopyTextureToTexture as u32);
    assert_eq!(
        (copy_back.src_handle, copy_back.dst_handle),
        (slot(Slot::ColourB), 0xB000)
    );
    // The UI draw is the pass's only draw: it runs after every leading blit,
    // on the tone-mapped back buffer, and is not tone-mapped itself.
    assert_eq!(draws(&passes[1]), [ui.param_b]);
    assert_eq!(enc.post_constants.len(), 1);
    assert_eq!(
        enc.post_constants[0].tonemap,
        [1.0, 0.0, 0.0, 0.0],
        "no attachment: SDR headroom"
    );
    // A mid-frame flush keeps the UI style; the present starts a world again.
    assert!(!enc.take_present_scene_linear(true));
    assert_eq!(enc.ps_color_style(), COLOR_STYLE_STORE_CLAMP);
    assert!(enc.take_present_scene_linear(false));
    assert_eq!(enc.ps_color_style(), COLOR_STYLE_SCENE_FLOAT, "next frame");
    enc.shutdown_cleanup();
}

/// The present is scene-linear only when the tone map was queued: the key
/// on and a boundary that ran the chain. The flag waits through a mid-frame
/// flush for the present-bearing submit, which takes it.
#[test]
fn edr_present_passthrough_selected_only_with_scene_float() {
    use std::sync::atomic::AtomicU64;

    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let off = mtld3d_core::post::PostConfig::default();
    // Key on, a boundary: the present-bearing submit takes the flag once.
    let Some((_device, mut enc)) = post_encoder_with(&coherent, &retained, off, true) else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, false);
    assert!(!enc.take_present_scene_linear(false), "no boundary yet");
    enc.post_boundary(&post_boundary_op());
    assert_eq!(enc.pass_state.take_pending_leading_blits().len(), 2);
    assert!(
        !enc.take_present_scene_linear(true),
        "a flush presents nothing"
    );
    assert!(enc.take_present_scene_linear(false));
    assert!(!enc.take_present_scene_linear(false), "one frame only");
    // Key on, a multisampled back buffer: the chain steps aside.
    post_reset_frame(&mut enc, (64, 64), 4, false);
    enc.post_boundary(&post_boundary_op());
    assert_eq!(enc.pass_state.take_pending_leading_blits().len(), 0);
    assert!(!enc.take_present_scene_linear(false));
    enc.shutdown_cleanup();
    // Key off, a boundary with another effect: no tone map, no flag.
    let Some((_device, mut enc)) = post_encoder_with(&coherent, &retained, POST_DEBUG, false)
    else {
        return;
    };
    post_reset_frame(&mut enc, (64, 64), 1, false);
    enc.post_boundary(&post_boundary_op());
    assert_eq!(enc.pass_state.take_pending_leading_blits().len(), 2);
    assert!(!enc.take_present_scene_linear(false));
    enc.shutdown_cleanup();
}

/// AO, god rays with a visible sun and bloom together, over two frames of
/// one size: the second frame finds every scratch texture of the first and
/// creates or retires none.
#[test]
fn post_scratch_survives_frames_with_every_effect() {
    use std::sync::atomic::AtomicU64;

    use crate::metal::post::Slot;
    let (coherent, retained) = (AtomicU64::new(0), AtomicU64::new(0));
    let cfg = mtld3d_core::post::PostConfig {
        ao_pct: 100,
        god_rays_pct: 100,
        bloom_pct: 100,
        ..mtld3d_core::post::PostConfig::default()
    };
    let Some((_device, mut enc)) = post_encoder(&coherent, &retained, cfg) else {
        return;
    };
    let sun = mtld3d_core::encoder_data::PostBoundary {
        sun_dir_eye: Some([0.0, 0.0, -1.0]),
        sun_color: [1.0, 0.9, 0.8],
        sun_above_horizon: true,
        ..post_boundary_op()
    };
    let frame = |enc: &mut super::FrameEncoder| {
        post_reset_frame(enc, (1920, 1080), 1, true);
        enc.depth_snapshots.insert(
            0xD000,
            super::DepthSnapshot {
                // SAFETY: an opaque value the queued blits carry but nothing dereferences.
                handle: unsafe { MetalHandle::new(0xD100) },
                width: 1920,
                height: 1080,
                format: PixelFormat::Depth32Float,
                epoch: enc.depth_write_epoch.wrapping_sub(1),
            },
        );
        enc.post_boundary(&sun);
        let blits = enc.pass_state.take_pending_leading_blits();
        let handles: Vec<Option<u64>> = Slot::ALL
            .iter()
            .map(|&slot| enc.post.scratch_handle(slot).map(MetalHandle::raw))
            .collect();
        (blits, handles)
    };
    let (first, first_handles) = frame(&mut enc);
    assert_eq!(first.len(), 18, "snapshot, 16 passes, copy back");
    assert!(
        first_handles.iter().all(Option::is_some),
        "every slot is used"
    );
    let before = enc.pending_resource_retention.len();
    let (second, second_handles) = frame(&mut enc);
    assert_eq!(
        enc.pending_resource_retention.len(),
        before,
        "nothing retired on the second frame"
    );
    assert_eq!(second_handles, first_handles, "no scratch re-created");
    let names = |blits: &[mtld3d_shared::BlitCommand]| -> Vec<[u64; 3]> {
        blits
            .iter()
            .map(|b| [b.src_handle, b.dst_handle, b.bytes_per_row])
            .collect()
    };
    assert_eq!(
        names(&second),
        names(&first),
        "the same textures each frame"
    );
    enc.depth_snapshots.clear();
    enc.shutdown_cleanup();
}
