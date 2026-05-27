use cgns::data::ElementType as CgnsEt;
use cgns::CgnsFile;
use mefikit::mesh::{UMeshView, ElementType as MefiEt, Regularity};

use crate::Error;

fn mefi_to_cgns(et: MefiEt) -> Option<CgnsEt> {
    match et {
        MefiEt::VERTEX => Some(CgnsEt::Node),
        MefiEt::SEG2 => Some(CgnsEt::Bar2),
        MefiEt::SEG3 => Some(CgnsEt::Bar3),
        MefiEt::TRI3 => Some(CgnsEt::Tri3),
        MefiEt::TRI6 => Some(CgnsEt::Tri6),
        MefiEt::QUAD4 => Some(CgnsEt::Quad4),
        MefiEt::QUAD8 => Some(CgnsEt::Quad8),
        MefiEt::QUAD9 => Some(CgnsEt::Quad9),
        MefiEt::TET4 => Some(CgnsEt::Tetra4),
        MefiEt::TET10 => Some(CgnsEt::Tetra10),
        MefiEt::HEX8 => Some(CgnsEt::Hexa8),
        _ => None,
    }
}

/// Write a [`UMeshView`] to a CGNS file.
///
/// Creates a single base and zone named `"Base"` / `"Zone"`.
/// `phys_dim` is taken from the coordinate space dimension; `cell_dim` from the highest
/// topological dimension present in the mesh.
///
/// One CGNS section is written per element block, named after the element type.
/// Connectivity indices are converted from mefikit 0-based to CGNS 1-based.
///
/// Returns [`Error::UnsupportedElementType`] for poly blocks (PGON, PHED, SPLINE) or for
/// element types without a CGNS equivalent (SEG4, TRI7, HEX21).
pub fn write(path: &str, mesh: UMeshView<'_>) -> Result<(), Error> {
    let _op = crate::HDF5_OP_MUTEX
        .lock()
        .map_err(|e| Error::Shape(format!("HDF5 op mutex poisoned: {e}")))?;
    let coords = mesh.coords();
    let num_nodes = coords.shape()[0];
    let phys_dim = coords.shape()[1] as i32;

    let cell_dim = mesh
        .topological_dimension()
        .ok_or_else(|| Error::Shape("empty mesh".into()))?;
    let cell_dim_i32 = u8::from(cell_dim) as i32;

    // CGNS zone size requires only the count of highest-topological-dimension elements.
    let num_elements: usize = mesh
        .blocks()
        .filter(|(et, _): &(&MefiEt, _)| et.dimension() == cell_dim)
        .map(|(_, b)| b.len())
        .sum();

    let file = CgnsFile::create(path)?;
    let base = file.create_base("Base", cell_dim_i32, phys_dim)?;
    let zone = base.create_zone_unstructured("Zone", num_nodes as i64, num_elements as i64)?;

    // Write coordinates — one array per spatial dimension.
    let coord_names = ["CoordinateX", "CoordinateY", "CoordinateZ"];
    for col in 0..phys_dim as usize {
        let col_data: Vec<f64> = coords.column(col).to_vec();
        zone.write_coord_f64(coord_names[col], &col_data)?;
    }

    // Write one section per element block.
    let mut elem_start: i64 = 1;
    for (et, block) in mesh.blocks() {
        let et: MefiEt = *et;
        if et.regularity() == Regularity::Poly {
            return Err(Error::UnsupportedElementType(format!("{et:?}")));
        }
        let cgns_et = mefi_to_cgns(et)
            .ok_or_else(|| Error::UnsupportedElementType(format!("{et:?}")))?;

        let num_elems = block.len() as i64;
        let elem_end = elem_start + num_elems - 1;

        // 0-based → 1-based flat connectivity.
        let conn_view = mesh
            .regular_connectivity(et)
            .map_err(Error::Shape)?;
        let flat: Vec<i64> = conn_view.iter().map(|&x| x as i64 + 1).collect();

        let section_name = format!("{et:?}");
        zone.write_section(&section_name, cgns_et, elem_start, elem_end, 0, &flat)?;

        elem_start = elem_end + 1;
    }

    Ok(())
}
