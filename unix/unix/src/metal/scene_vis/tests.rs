//! The scene kernels: their constants' layout on both sides, and that every kernel builds.

use core::mem::offset_of;

use mtld3d_core::{
    encoder_data::PostBoundary,
    shadows::{ShadowConfig, cascades::CascadeSet},
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use super::{
    CASCADE_BLEND, Kernel, PCF_MIN_TEXELS, SCENE_CONSTANTS_MSL, SceneConstants, create_pipeline,
    horizon_fade,
};

/// The members of `SceneConstants`, in declaration order, with their Rust offsets.
const MEMBERS: [(&str, usize); 9] = [
    ("proj", offset_of!(SceneConstants, proj)),
    ("viewport", offset_of!(SceneConstants, viewport)),
    ("extent", offset_of!(SceneConstants, extent)),
    ("sun", offset_of!(SceneConstants, sun)),
    ("split_far", offset_of!(SceneConstants, split_far)),
    ("blend_start", offset_of!(SceneConstants, blend_start)),
    ("texel", offset_of!(SceneConstants, texel)),
    ("pcf_radius", offset_of!(SceneConstants, pcf_radius)),
    ("cascades", offset_of!(SceneConstants, cascades)),
];

/// The Metal compiler lays the MSL mirror out as Rust lays `SceneConstants`
/// out: a kernel compiled against the mirror writes the size and every
/// member's offset, and they match the Rust struct's.
#[test]
fn scene_constants_layout_matches_msl() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let writes: String = MEMBERS
        .iter()
        .enumerate()
        .map(|(i, (name, _))| {
            format!(
                "    out[{}] = uint(__builtin_offsetof(SceneConstants, {name}));\n",
                i + 1
            )
        })
        .collect();
    let msl = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{SCENE_CONSTANTS_MSL}kernel void \
         layout(device uint* out [[buffer(0)]]) {{\n    out[0] = \
         uint(sizeof(SceneConstants));\n{writes}}}\n"
    );
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&msl), None)
        .expect("the layout kernel compiles");
    let function = library
        .newFunctionWithName(&NSString::from_str("layout"))
        .expect("the layout function");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("the layout pipeline");
    let count = MEMBERS.len() + 1;
    let buffer = device
        .newBufferWithLength_options(count * 4, MTLResourceOptions::StorageModeShared)
        .expect("the layout buffer");
    let queue = device.newCommandQueue().expect("command queue");
    let cb = queue.commandBuffer().expect("command buffer");
    let compute = cb.computeCommandEncoder().expect("compute encoder");
    compute.setComputePipelineState(&pipeline);
    // SAFETY: the kernel writes `count` uints into the retained buffer.
    unsafe { compute.setBuffer_offset_atIndex(Some(&buffer), 0, 0) };
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    compute.dispatchThreadgroups_threadsPerThreadgroup(one, one);
    compute.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the completed kernel wrote `count` uints.
    let written =
        unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u32>(), count) };
    let as_u32 = |v: usize| u32::try_from(v).expect("small");
    assert_eq!(written[0], as_u32(size_of::<SceneConstants>()), "size");
    for (i, (name, offset)) in MEMBERS.iter().enumerate() {
        assert_eq!(written[i + 1], as_u32(*offset), "offset of {name}");
    }
    assert!(size_of::<SceneConstants>() <= 512);
}

/// Every scene kernel compiles into a pipeline.
#[test]
fn every_scene_kernel_builds() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    for kernel in Kernel::ALL {
        let pipeline = create_pipeline(&device, kernel).expect("the kernel builds");
        crate::metal::post::destroy_pipeline(pipeline.raw());
        assert_eq!(Kernel::from_id(kernel as u32), Some(kernel));
    }
    assert_eq!(Kernel::from_id(Kernel::ALL.len() as u32), None);
}

/// A boundary under the identity view (world −Z up is eye −Z) whose sun is `elevation` degrees up.
fn boundary_at(elevation: f32) -> PostBoundary {
    let e = elevation.to_radians();
    PostBoundary {
        proj: [1.0, 1.0, 1.001, -10.01],
        // Travelling down (+Z) and along +X.
        sun_dir_eye: Some([e.cos(), 0.0, e.sin()]),
        sun_above_horizon: elevation > 0.0,
        ..PostBoundary::default()
    }
}

/// The fade is linear in the sun's elevation: none at 6° and above, half at 3°, full at the horizon and below.
#[test]
fn horizon_fade_follows_the_elevation() {
    for (elevation, fade) in [
        (90.0, 1.0),
        (6.0, 1.0),
        (4.5, 0.75),
        (3.0, 0.5),
        (0.0, 0.0),
        (-10.0, 0.0),
    ] {
        let got = horizon_fade(&boundary_at(elevation));
        assert!((got - fade).abs() < 1e-4, "{elevation}°: {got} vs {fade}");
    }
    let no_sun = PostBoundary {
        sun_dir_eye: None,
        ..boundary_at(45.0)
    };
    assert_eq!(horizon_fade(&no_sun), 0.0);
}

/// A cascade set of three cascades whose texels are 1, 2 and 8 world units at `res`.
fn three_cascades(res: u16) -> CascadeSet {
    let scaled = |texel: f32| {
        let per_unit = 2.0 / (f32::from(res) * texel);
        [
            [per_unit, 0.0, 0.0, 0.0],
            [0.0, per_unit, 0.0, 0.0],
            [0.0, 0.0, 0.001, 0.0],
            [0.0, 0.0, 0.5, 1.0],
        ]
    };
    CascadeSet {
        matrices: [scaled(1.0), scaled(2.0), scaled(8.0), [[0.0; 4]; 4]],
        splits: [10.0, 100.0, 1000.0, 8000.0, 8000.0],
        count: 3,
    }
}

/// The PCF radius is `softness` cascade-0 texels of world, in each
/// cascade's own texels and never under one; the blend starts 10 % before
/// each cascade's end; the matrices go in unbiased; without cascades the
/// count is 0.
#[test]
fn constants_carry_the_cascades_in_world_units() {
    let shadows = ShadowConfig {
        sun: true,
        resolution: 1024,
        softness_pct: 300,
        ..ShadowConfig::default()
    };
    let set = three_cascades(1024);
    let boundary = boundary_at(30.0);
    let k = SceneConstants::new(
        &boundary,
        Some(&set),
        &shadows,
        (640, 360),
        [0.0, 0.0, 640.0, 360.0],
    );
    assert_eq!(k.extent, [640, 360, 3, 1024]);
    assert_eq!(k.proj, boundary.proj);
    let close = |a: [f32; 4], b: [f32; 4]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-3);
    assert!(close(k.texel, [1.0, 2.0, 8.0, 0.0]), "{:?}", k.texel);
    // 3 world units: 3, 1.5 and 0.375 → 1 texel.
    assert!(
        close(k.pcf_radius, [3.0, 1.5, PCF_MIN_TEXELS, 0.0]),
        "{:?}",
        k.pcf_radius
    );
    assert_eq!(k.split_far, [100.0, 1000.0, 8000.0, 0.0]);
    let starts = [
        100.0 - CASCADE_BLEND * 90.0,
        1000.0 - CASCADE_BLEND * 900.0,
        8000.0 - CASCADE_BLEND * 7000.0,
        0.0,
    ];
    assert!(close(k.blend_start, starts), "{:?}", k.blend_start);
    assert_eq!(k.cascades[..3], set.matrices[..3]);
    assert!((k.sun[3] - 1.0).abs() < 1e-6, "30° is above the fade");

    let none = SceneConstants::new(
        &boundary,
        None,
        &shadows,
        (640, 360),
        [0.0, 0.0, 640.0, 360.0],
    );
    assert_eq!(none.extent[2], 0);
    assert_eq!(none.pcf_radius, [0.0; 4]);
}
