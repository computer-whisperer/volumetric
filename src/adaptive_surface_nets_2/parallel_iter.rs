//! Conditional parallel iteration helpers.
//!
//! These functions provide parallel iteration when the `native` feature is enabled
//! (using rayon), and fall back to sequential iteration on web (wasm32).

#[cfg(feature = "native")]
use rayon::prelude::*;

/// Process a Vec in parallel (native) or sequentially (web), returning results.
#[cfg(feature = "native")]
pub fn map_vec<T, R, F>(items: Vec<T>, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(T) -> R + Sync + Send,
{
    items.into_par_iter().map(f).collect()
}

#[cfg(not(feature = "native"))]
pub fn map_vec<T, R, F>(items: Vec<T>, f: F) -> Vec<R>
where
    F: Fn(T) -> R,
{
    items.into_iter().map(f).collect()
}

/// Process a range in parallel (native) or sequentially (web), returning results.
#[cfg(feature = "native")]
pub fn map_range<R, F>(range: std::ops::Range<usize>, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    range.into_par_iter().map(f).collect()
}

#[cfg(not(feature = "native"))]
pub fn map_range<R, F>(range: std::ops::Range<usize>, f: F) -> Vec<R>
where
    F: Fn(usize) -> R,
{
    range.into_iter().map(f).collect()
}
