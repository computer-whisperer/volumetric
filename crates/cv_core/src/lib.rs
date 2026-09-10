//! Computer vision on pixels, pure Rust: ArUco markers found with
//! sub-pixel corners ([`detect`]), a picture's pose from the markers of a
//! view set's map ([`pnp`]), the EXIF focal seed ([`exif`]), the
//! dictionaries as data ([`dict`]), and a synthetic board renderer with
//! exact ground truth for the tests ([`board`]).
//!
//! Pixel coordinates follow the view set: the picture spans `0..width`
//! and pixel centres sit at `+0.5`.

pub mod board;
pub mod detect;
pub mod dict;
mod dict_tables;
pub mod exif;
pub mod gray;
pub mod linalg;
pub mod pnp;

pub use detect::{DetectParams, Detection, detect};
pub use dict::Dictionary;
pub use exif::{Exif, read_exif};
pub use gray::Gray;
pub use pnp::{Estimate, MarkerFit, PoseSolve, SolveOptions, solve_view};
