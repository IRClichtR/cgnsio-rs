use cgns::data::{ElementType as CgnsEt, ZoneType};
use cgns::CgnsFile;
use mefikit::mesh::{UMesh, ElementType as MefiEt};
use ndarray::Array2;

use crate::Error;

fn cgns_to_mefi(et: CgnsEt) -> Option<MefiEt> {
    match et {
        CgnsEt::Node => Some(MefiEt::VERTEX),
        CgnsEt::Bar2 => Some(MefiEt::SEG2),
        CgnsEt::Bar3 => Some(MefiEt::SEG3),
        CgnsEt::Tri3 => Some(MefiEt::TRI3),
        CgnsEt::Tri6 => Some(MefiEt::TRI6),
        CgnsEt::Quad4 => Some(MefiEt::QUAD4),
        CgnsEt::Quad8 => Some(MefiEt::QUAD8),
        CgnsEt::Quad9 => Some(MefiEt::QUAD9),
        CgnsEt::Tetra4 => Some(MefiEt::TET4),
        CgnsEt::Tetra10 => Some(MefiEt::TET10),
        CgnsEt::Hexa8 => Some(MefiEt::HEX8),
        _ => None,
    }
}

/// Read the first unstructured zone of the first base from a CGNS file into a [`UMesh`].
///
/// Coordinates are stored as `(num_nodes, space_dim)` with columns ordered X, Y, Z.
/// Connectivity indices are converted from CGNS 1-based to mefikit 0-based.
///
/// Sections whose element type has no mefikit equivalent are skipped silently.
/// Returns [`Error::NoZone`] when no unstructured zone has any supported element type.
pub fn read(path: &str) -> Result<UMesh, Error> {
    let _op = crate::HDF5_OP_MUTEX
        .lock()
        .map_err(|e| Error::Shape(format!("HDF5 op mutex poisoned: {e}")))?;
    let file = CgnsFile::open(path)?;

    for base in file.bases()? {
        let zones = match base.zones() {
            Ok(z) => z,
            Err(_) => continue,
        };
        for zone in zones {
            if zone.zone_type().ok() != Some(ZoneType::Unstructured) {
                continue;
            }

            let section_list: Vec<_> = match zone.sections() {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Skip zones where every section has an unsupported element type.
            if !section_list.is_empty()
                && section_list.iter().all(|s| {
                    s.info()
                        .ok()
                        .and_then(|info| cgns_to_mefi(info.element_type))
                        .is_none()
                })
            {
                continue;
            }

            let size = zone.zone_size()?;
            let num_nodes = size[0] as usize;

            // Build (num_nodes × space_dim) coordinate array.
            let coord_names = zone.coord_names()?;
            let space_dim = coord_names.len().min(3);
            let rmin = [1i64];
            let rmax = [num_nodes as i64];

            let mut coords = Array2::<f64>::zeros((num_nodes, space_dim));
            for (col, name) in coord_names.iter().enumerate().take(space_dim) {
                let mut col_data = vec![0.0f64; num_nodes];
                zone.read_coord_f64(name, &rmin, &rmax, &mut col_data)?;
                for (row, &v) in col_data.iter().enumerate() {
                    coords[[row, col]] = v;
                }
            }

            let mut mesh = UMesh::new(coords.into_shared());

            // Read each element section, skipping unsupported types.
            for section in section_list {
                let info = section.info()?;
                let mefi_et = match cgns_to_mefi(info.element_type) {
                    Some(et) => et,
                    None => continue,
                };
                let npe = mefi_et.num_nodes().unwrap();

                let flat_cgns = section.read_connectivity()?;
                let num_elems = flat_cgns.len() / npe;

                // 1-based → 0-based.
                let conn: Vec<usize> = flat_cgns.iter().map(|&x| (x - 1) as usize).collect();
                let conn_arr = Array2::from_shape_vec((num_elems, npe), conn)
                    .map_err(|e| Error::Shape(e.to_string()))?
                    .into_shared();

                mesh.add_regular_block(mefi_et, conn_arr, None);
            }

            return Ok(mesh);
        }
    }

    Err(Error::NoZone)
}
