//! Sample Cache for Research Experiments
//!
//! Provides a simple caching layer for sampler calls to:
//! 1. Avoid redundant samples when multiple algorithms probe the same point
//! 2. Track sample counts for comparing algorithm efficiency
//! 3. Enable reproducibility by recording exact sample sequences
//!
//! The cache uses a spatial hash with quantization to handle floating-point keys.

use std::cell::RefCell;
use std::collections::HashMap;

/// Statistics about cache usage
#[derive(Clone, Debug, Default)]
pub struct SampleCacheStats {
    /// Number of cache hits (sample was already cached)
    pub hits: u64,
    /// Number of cache misses (sample was computed fresh)
    pub misses: u64,
    /// Total sample calls (hits + misses)
    pub total_calls: u64,
}

impl SampleCacheStats {
    /// Hit rate as a fraction (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_calls as f64
        }
    }

    /// Number of actual sampler invocations (misses)
    pub fn actual_samples(&self) -> u64 {
        self.misses
    }
}

/// A cache key for 3D positions
///
/// Quantizes floating-point coordinates to avoid floating-point comparison issues.
/// Uses a fixed precision that should be sufficient for most use cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PositionKey {
    x: i64,
    y: i64,
    z: i64,
}

impl PositionKey {
    /// Quantization factor: positions are rounded to this precision
    /// 1e-9 gives sub-nanometer precision, more than enough for any reasonable use case
    const QUANTIZATION: f64 = 1e9;

    fn from_position(x: f64, y: f64, z: f64) -> Self {
        Self {
            x: (x * Self::QUANTIZATION).round() as i64,
            y: (y * Self::QUANTIZATION).round() as i64,
            z: (z * Self::QUANTIZATION).round() as i64,
        }
    }
}

/// A sample cache that stores sampler results for reuse
///
/// Uses interior mutability (RefCell) to allow sampling through shared references.
/// This is appropriate for single-threaded research code.
pub struct SampleCache<F> {
    sampler: F,
    cache: RefCell<HashMap<PositionKey, f32>>,
    stats: RefCell<SampleCacheStats>,
}

impl<F> SampleCache<F>
where
    F: Fn(f64, f64, f64) -> f32,
{
    /// Create a new sample cache wrapping the given sampler
    pub fn new(sampler: F) -> Self {
        Self {
            sampler,
            cache: RefCell::new(HashMap::new()),
            stats: RefCell::new(SampleCacheStats::default()),
        }
    }

    /// Sample at a point, using cache if available
    pub fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        let key = PositionKey::from_position(x, y, z);
        let mut stats = self.stats.borrow_mut();
        stats.total_calls += 1;

        // Check cache
        if let Some(&value) = self.cache.borrow().get(&key) {
            stats.hits += 1;
            return value;
        }

        // Cache miss: compute and store
        stats.misses += 1;
        let value = (self.sampler)(x, y, z);
        self.cache.borrow_mut().insert(key, value);
        value
    }

    /// Check if a point is inside (sample > 0)
    pub fn is_inside(&self, x: f64, y: f64, z: f64) -> bool {
        self.sample(x, y, z) > 0.0
    }

    /// Get cache statistics
    pub fn stats(&self) -> SampleCacheStats {
        self.stats.borrow().clone()
    }

    /// Reset statistics (but keep cached values)
    pub fn reset_stats(&self) {
        *self.stats.borrow_mut() = SampleCacheStats::default();
    }

    /// Clear the cache and reset statistics
    pub fn clear(&self) {
        self.cache.borrow_mut().clear();
        self.reset_stats();
    }

    /// Number of cached entries
    pub fn cache_size(&self) -> usize {
        self.cache.borrow().len()
    }

    /// Get the underlying sampler (for direct access when caching not needed)
    pub fn sampler(&self) -> &F {
        &self.sampler
    }
}

/// A cache-aware sampler wrapper that can be passed to algorithms
///
/// This implements the sampling interface while tracking statistics.
/// Useful for comparing sample efficiency across algorithms.
pub struct CachedSampler<'a, F> {
    cache: &'a SampleCache<F>,
}

impl<'a, F> CachedSampler<'a, F>
where
    F: Fn(f64, f64, f64) -> f32,
{
    pub fn new(cache: &'a SampleCache<F>) -> Self {
        Self { cache }
    }

    pub fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        self.cache.sample(x, y, z)
    }

    pub fn is_inside(&self, x: f64, y: f64, z: f64) -> bool {
        self.cache.is_inside(x, y, z)
    }
}

/// Binary search to find surface crossing along a ray
///
/// Given two points with opposite signs (one inside, one outside),
/// finds the crossing point to the specified precision.
///
/// # Arguments
/// * `cache` - The sample cache to use
/// * `p_inside` - A point known to be inside (sample > 0)
/// * `p_outside` - A point known to be outside (sample <= 0)
/// * `iterations` - Number of binary search iterations
///
/// # Returns
/// The estimated crossing point
pub fn binary_search_crossing<F>(
    cache: &SampleCache<F>,
    p_inside: (f64, f64, f64),
    p_outside: (f64, f64, f64),
    iterations: usize,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut inside = p_inside;
    let mut outside = p_outside;

    for _ in 0..iterations {
        let mid = (
            (inside.0 + outside.0) / 2.0,
            (inside.1 + outside.1) / 2.0,
            (inside.2 + outside.2) / 2.0,
        );

        if cache.is_inside(mid.0, mid.1, mid.2) {
            inside = mid;
        } else {
            outside = mid;
        }
    }

    // Return midpoint of final interval
    (
        (inside.0 + outside.0) / 2.0,
        (inside.1 + outside.1) / 2.0,
        (inside.2 + outside.2) / 2.0,
    )
}

/// Find surface crossing in a given direction from a starting point
///
/// Probes outward from `start` in direction `dir` until a sign change is found,
/// then binary searches to refine the crossing point.
///
/// # Arguments
/// * `cache` - The sample cache to use
/// * `start` - Starting point (should be inside or outside consistently)
/// * `dir` - Direction to search (will be normalized)
/// * `max_distance` - Maximum distance to search
/// * `initial_step` - Initial step size for probing
/// * `binary_iterations` - Number of binary search iterations for refinement
///
/// # Returns
/// Some((crossing_point, distance)) if a crossing was found, None otherwise
pub fn find_crossing_in_direction<F>(
    cache: &SampleCache<F>,
    start: (f64, f64, f64),
    dir: (f64, f64, f64),
    max_distance: f64,
    initial_step: f64,
    binary_iterations: usize,
) -> Option<((f64, f64, f64), f64)>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let len = (dir.0 * dir.0 + dir.1 * dir.1 + dir.2 * dir.2).sqrt();
    if len < 1e-12 {
        return None;
    }
    let dir = (dir.0 / len, dir.1 / len, dir.2 / len);

    let start_inside = cache.is_inside(start.0, start.1, start.2);
    let mut prev_point = start;
    let mut t = initial_step;

    while t <= max_distance {
        let current = (start.0 + dir.0 * t, start.1 + dir.1 * t, start.2 + dir.2 * t);

        let current_inside = cache.is_inside(current.0, current.1, current.2);

        if current_inside != start_inside {
            // Found a crossing between prev_point and current
            let (p_inside, p_outside) = if start_inside {
                (prev_point, current)
            } else {
                (current, prev_point)
            };

            let crossing = binary_search_crossing(cache, p_inside, p_outside, binary_iterations);
            let dist = ((crossing.0 - start.0).powi(2)
                + (crossing.1 - start.1).powi(2)
                + (crossing.2 - start.2).powi(2))
            .sqrt();

            return Some((crossing, dist));
        }

        prev_point = current;
        t += initial_step;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sphere(x: f64, y: f64, z: f64) -> f32 {
        (1.0 - x * x - y * y - z * z) as f32
    }

    #[test]
    fn test_cache_basic() {
        let cache = SampleCache::new(unit_sphere);

        // First call should be a miss
        let v1 = cache.sample(0.5, 0.0, 0.0);
        assert!(v1 > 0.0);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Second call at same point should be a hit
        let v2 = cache.sample(0.5, 0.0, 0.0);
        assert!((v1 - v2).abs() < 1e-6);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 1);

        // Different point should be a miss
        cache.sample(0.6, 0.0, 0.0);
        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn test_binary_search_crossing() {
        let cache = SampleCache::new(unit_sphere);

        // Start inside at origin, end outside at (2, 0, 0)
        let crossing =
            binary_search_crossing(&cache, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), 20);

        // Should be close to (1, 0, 0)
        assert!((crossing.0 - 1.0).abs() < 0.001);
        assert!(crossing.1.abs() < 0.001);
        assert!(crossing.2.abs() < 0.001);
    }

    #[test]
    fn test_find_crossing_in_direction() {
        let cache = SampleCache::new(unit_sphere);

        // Search outward from origin in +X direction
        let result = find_crossing_in_direction(
            &cache,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            2.0,
            0.1,
            20,
        );

        assert!(result.is_some());
        let (crossing, dist) = result.unwrap();

        // Should be close to (1, 0, 0)
        assert!((crossing.0 - 1.0).abs() < 0.001);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quantization_precision() {
        let cache = SampleCache::new(unit_sphere);

        // These should hash to different keys
        cache.sample(0.0, 0.0, 0.0);
        cache.sample(1e-8, 0.0, 0.0); // Different by 1e-8

        assert_eq!(cache.stats().misses, 2, "Close but different points should be separate");

        // These should hash to the same key
        cache.sample(0.5, 0.0, 0.0);
        cache.sample(0.5 + 1e-12, 0.0, 0.0); // Different by only 1e-12

        // The second should be a hit because 1e-12 is below quantization precision
        assert_eq!(cache.stats().hits, 1);
    }
}
