//! Computer vision on pixels, pure Rust: ArUco markers found with
//! sub-pixel corners ([`detect`]), a picture's pose from the markers of a
//! view set's map ([`pnp`]), the EXIF focal seed ([`exif`]), the
//! dictionaries as data ([`dict`]), the whole still-to-view pipeline the
//! command and the operator share ([`still`]), and a synthetic board
//! renderer with exact ground truth for the tests ([`board`]).
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
pub mod still;

pub use detect::{DetectParams, Detection, detect};
pub use dict::Dictionary;
pub use exif::{Exif, read_exif};
pub use gray::Gray;
pub use pnp::{Estimate, MarkerFit, PoseSolve, SolveOptions, solve_view};
pub use still::{StillOptions, StillSolve, append_view, solve_still};
