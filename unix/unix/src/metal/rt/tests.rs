//! The instance packing and the build descriptions' shape.

use objc2_metal::MTLAccelerationStructureInstanceOptions;

use super::{BlasBuild, InstanceDescriptor, TlasBuild, instance_descriptor};

/// A row-major 3x4 transform lands in Metal's column-major packed 4x3:
/// column `c` holds element `c` of each row, the translation last.
#[test]
fn instance_descriptor_packs_rows_as_columns() {
    let rows = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ];
    let d = instance_descriptor(&rows, 3, 2, 7, false);
    let columns = d.transformationMatrix.columns.map(|c| [c.x, c.y, c.z]);
    assert_eq!(
        columns,
        [
            [1.0, 5.0, 9.0],
            [2.0, 6.0, 10.0],
            [3.0, 7.0, 11.0],
            [4.0, 8.0, 12.0]
        ]
    );
    assert_eq!((d.accelerationStructureIndex, d.mask, d.userID), (3, 2, 7));
    assert_eq!(
        d.options,
        MTLAccelerationStructureInstanceOptions::DisableTriangleCulling
            | MTLAccelerationStructureInstanceOptions::NonOpaque
    );
    let opaque = instance_descriptor(&rows, 0, 1, 0, true);
    assert!(
        opaque
            .options
            .contains(MTLAccelerationStructureInstanceOptions::Opaque)
    );
}

/// The layouts the encoder thread writes and the submit thread reads.
#[test]
fn build_descriptions_have_fixed_layouts() {
    assert_eq!(
        size_of::<InstanceDescriptor>(),
        68,
        "Metal's user-id instance"
    );
    assert_eq!(size_of::<BlasBuild>(), 128);
    assert_eq!(size_of::<TlasBuild>(), 40);
}
