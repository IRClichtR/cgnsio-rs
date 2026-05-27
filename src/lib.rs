pub mod read;
pub mod write;

pub use read::read;
pub use write::write;

// HDF5 serial (non-thread-safe build) maintains per-file context in global C
// state that persists across API calls.  Per-call locking in cgns-sys is not
// enough: interleaved read/write sessions from different threads corrupt that
// state.  This mutex serialises whole read() / write() operations.
pub(crate) static HDF5_OP_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug)]
pub enum Error {
    Cgns(cgns::error::CgnsError),
    UnsupportedElementType(String),
    NoZone,
    Shape(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cgns(e) => write!(f, "CGNS: {e}"),
            Self::UnsupportedElementType(s) => write!(f, "unsupported element type: {s}"),
            Self::NoZone => write!(f, "no unstructured zone found"),
            Self::Shape(s) => write!(f, "array shape error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<cgns::error::CgnsError> for Error {
    fn from(e: cgns::error::CgnsError) -> Self {
        Self::Cgns(e)
    }
}
