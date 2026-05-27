use cgns_io::{read, write, Error};
use mefikit::mesh::ElementType as MefiEt;
use ndarray::arr2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tmp(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("cgns_io_{name}.cgns"))
        .to_string_lossy()
        .into_owned()
}

/// Run cgnscheck and return its full stdout+stderr output.
fn cgnscheck(path: &str) -> String {
    let out = std::process::Command::new("cgnscheck")
        .arg(path)
        .output()
        .expect("cgnscheck must be on PATH (/usr/local/bin/cgnscheck)");
    String::from_utf8_lossy(&out.stdout).into_owned()
        + &String::from_utf8_lossy(&out.stderr)
}

/// Assert cgnscheck reports no ERROR lines for the given file.
fn assert_cgnscheck_clean(path: &str) {
    let output = cgnscheck(path);
    let errors: Vec<&str> = output.lines().filter(|l| l.starts_with("ERROR:")).collect();
    assert!(
        errors.is_empty(),
        "cgnscheck found errors in {path}:\n{}\n\nFull output:\n{output}",
        errors.join("\n")
    );
}

fn make_tri3_mesh() -> mefikit::mesh::UMesh {
    // Unit square: 4 nodes, 2 triangles (2-D).
    let coords = arr2(&[[0.0f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    let mut mesh = mefikit::mesh::UMesh::new(coords.into_shared());
    mesh.add_regular_block(
        MefiEt::TRI3,
        arr2(&[[0usize, 1, 2], [0, 2, 3]]).into_shared(),
        None,
    );
    mesh
}

fn make_tet4_mesh() -> mefikit::mesh::UMesh {
    // Regular tetrahedron: 4 nodes, 1 tet (3-D).
    let coords = arr2(&[
        [0.0f64, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let mut mesh = mefikit::mesh::UMesh::new(coords.into_shared());
    mesh.add_regular_block(
        MefiEt::TET4,
        arr2(&[[0usize, 1, 2, 3]]).into_shared(),
        None,
    );
    mesh
}

fn make_hexa8_mesh() -> mefikit::mesh::UMesh {
    // Unit cube: 8 nodes, 1 hex (3-D).
    let coords = arr2(&[
        [0.0f64, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]);
    let mut mesh = mefikit::mesh::UMesh::new(coords.into_shared());
    mesh.add_regular_block(
        MefiEt::HEX8,
        arr2(&[[0usize, 1, 2, 3, 4, 5, 6, 7]]).into_shared(),
        None,
    );
    mesh
}

fn make_mixed_mesh() -> mefikit::mesh::UMesh {
    // 5 nodes: two surface triangles + one volume tet (3-D).
    let coords = arr2(&[
        [0.0f64, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]);
    let mut mesh = mefikit::mesh::UMesh::new(coords.into_shared());
    mesh.add_regular_block(
        MefiEt::TRI3,
        arr2(&[[0usize, 1, 4], [0, 4, 2]]).into_shared(),
        None,
    );
    mesh.add_regular_block(
        MefiEt::TET4,
        arr2(&[[0usize, 1, 2, 3]]).into_shared(),
        None,
    );
    mesh
}

// ---------------------------------------------------------------------------
// Round-trip + cgnscheck: TRI3 2-D
// ---------------------------------------------------------------------------

#[test]
fn tri3_roundtrip() {
    let path = tmp("tri3");
    let _ = std::fs::remove_file(&path);

    let original = make_tri3_mesh();
    write(&path, original.view()).expect("write");

    assert_cgnscheck_clean(&path);

    let loaded = read(&path).expect("read");

    assert_eq!(loaded.coords().shape(), &[4, 2]);
    assert!((loaded.coords()[[0, 0]] - 0.0).abs() < 1e-15);
    assert!((loaded.coords()[[1, 0]] - 1.0).abs() < 1e-15);
    assert!((loaded.coords()[[2, 1]] - 1.0).abs() < 1e-15);

    let conn = loaded.regular_connectivity(MefiEt::TRI3).expect("TRI3");
    assert_eq!(conn.shape(), &[2, 3]);
    assert_eq!(conn[[0, 0]], 0);
    assert_eq!(conn[[0, 1]], 1);
    assert_eq!(conn[[0, 2]], 2);
    assert_eq!(conn[[1, 0]], 0);
    assert_eq!(conn[[1, 1]], 2);
    assert_eq!(conn[[1, 2]], 3);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Round-trip + cgnscheck: TET4 3-D
// ---------------------------------------------------------------------------

#[test]
fn tet4_roundtrip() {
    let path = tmp("tet4");
    let _ = std::fs::remove_file(&path);

    let original = make_tet4_mesh();
    write(&path, original.view()).expect("write");

    assert_cgnscheck_clean(&path);

    let loaded = read(&path).expect("read");

    assert_eq!(loaded.coords().shape(), &[4, 3]);

    let conn = loaded.regular_connectivity(MefiEt::TET4).expect("TET4");
    assert_eq!(conn.shape(), &[1, 4]);
    assert_eq!(conn.row(0).as_slice().unwrap(), &[0, 1, 2, 3]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Round-trip + cgnscheck: HEXA8 3-D
// ---------------------------------------------------------------------------

#[test]
fn hexa8_roundtrip() {
    let path = tmp("hexa8");
    let _ = std::fs::remove_file(&path);

    let original = make_hexa8_mesh();
    write(&path, original.view()).expect("write");

    assert_cgnscheck_clean(&path);

    let loaded = read(&path).expect("read");

    assert_eq!(loaded.coords().shape(), &[8, 3]);

    let conn = loaded.regular_connectivity(MefiEt::HEX8).expect("HEX8");
    assert_eq!(conn.shape(), &[1, 8]);
    assert_eq!(conn.row(0).as_slice().unwrap(), &[0, 1, 2, 3, 4, 5, 6, 7]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Round-trip + cgnscheck: mixed TRI3 + TET4
// ---------------------------------------------------------------------------

#[test]
fn mixed_tri3_tet4_roundtrip() {
    let path = tmp("mixed");
    let _ = std::fs::remove_file(&path);

    let original = make_mixed_mesh();
    write(&path, original.view()).expect("write");

    assert_cgnscheck_clean(&path);

    let loaded = read(&path).expect("read");

    assert_eq!(loaded.coords().shape(), &[5, 3]);

    let tri = loaded.regular_connectivity(MefiEt::TRI3).expect("TRI3");
    assert_eq!(tri.shape(), &[2, 3]);

    let tet = loaded.regular_connectivity(MefiEt::TET4).expect("TET4");
    assert_eq!(tet.shape(), &[1, 4]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Read cgns-io example: yf17_hdf5.cgns (TRI3 + TET4, 97104 nodes)
// ---------------------------------------------------------------------------

const YF17: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/yf17_hdf5.cgns");

#[test]
fn read_yf17_node_and_element_counts() {
    let mesh = read(YF17).expect("read yf17");

    assert_eq!(mesh.coords().shape()[0], 97104, "node count");
    assert_eq!(mesh.coords().shape()[1], 3, "3-D coordinates");

    let tets = mesh.regular_connectivity(MefiEt::TET4).expect("TET4");
    assert_eq!(tets.shape()[0], 528_915, "tet count");

    let tris = mesh.regular_connectivity(MefiEt::TRI3).expect("TRI3");
    assert_eq!(tris.shape()[0], 27_656, "tri count");
}

#[test]
fn write_yf17_passes_cgnscheck() {
    let path = tmp("yf17_roundtrip");
    let _ = std::fs::remove_file(&path);

    let mesh = read(YF17).expect("read yf17");
    write(&path, mesh.view()).expect("write");

    assert_cgnscheck_clean(&path);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Read cgns-io example: particles_example.cgns (no standard CGNS zones)
// ---------------------------------------------------------------------------

const PARTICLES: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/examples/particles_example.cgns");

#[test]
fn read_particles_returns_no_zone() {
    let result = read(PARTICLES);
    assert!(
        matches!(result, Err(Error::NoZone)),
        "expected NoZone, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Read cgns-rs test_grid.cgns: BaseUnstructured (TRI3 + HEX8 + TET4)
// No coordinates in this zone — verifies element blocks only.
// ---------------------------------------------------------------------------

const TEST_GRID: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../cgns-rs/cgns-tools/tests/data/test_grid.cgns"
);

#[test]
fn read_cgns_rs_test_grid_unstructured() {
    let mesh = read(TEST_GRID).expect("read test_grid");

    // The BaseUnstructured zone has 8 declared vertices, no coords written.
    assert_eq!(mesh.coords().shape()[0], 8);

    // 2 TRI3 + 1 HEX8 + 1 TET4 = 4 elements total, three blocks.
    let tris = mesh.regular_connectivity(MefiEt::TRI3).expect("TRI3");
    assert_eq!(tris.shape(), &[2, 3]);

    let hexs = mesh.regular_connectivity(MefiEt::HEX8).expect("HEX8");
    assert_eq!(hexs.shape(), &[1, 8]);

    let tets = mesh.regular_connectivity(MefiEt::TET4).expect("TET4");
    assert_eq!(tets.shape(), &[1, 4]);
}
