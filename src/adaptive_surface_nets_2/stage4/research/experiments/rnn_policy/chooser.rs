//! Chooser head: maps latent state to sample positions.
//!
//! The chooser outputs 8 weights for the cube corners, and the sample position
//! is computed as a weighted average of the corners.

use super::gru::{Rng, HIDDEN_DIM};
use super::math::{matvec, outer_product, softmax, vec_add};

/// Number of octants (cube corners).
pub const NUM_OCTANTS: usize = 8;

/// Chooser weights: linear layer from hidden state to octant weights.
#[derive(Clone)]
pub struct ChooserWeights {
    /// Weight matrix: [NUM_OCTANTS x HIDDEN_DIM].
    pub w: Vec<f64>,
    /// Bias: [NUM_OCTANTS].
    pub b: Vec<f64>,
}

impl ChooserWeights {
    /// Create new chooser weights with small random initialization.
    pub fn new(rng: &mut Rng) -> Self {
        let scale = 0.1;
        Self {
            w: (0..NUM_OCTANTS * HIDDEN_DIM)
                .map(|_| rng.next_range(scale))
                .collect(),
            b: vec![0.0; NUM_OCTANTS],
        }
    }

    /// Total number of parameters.
    pub fn param_count(&self) -> usize {
        NUM_OCTANTS * HIDDEN_DIM + NUM_OCTANTS
    }
}

/// Intermediate values from chooser forward pass, needed for backprop.
#[derive(Clone)]
pub struct ChooserCache {
    /// Input hidden state.
    pub h: Vec<f64>,
    /// Raw logits (before softmax).
    pub logits: Vec<f64>,
    /// Softmax probabilities.
    pub probs: Vec<f64>,
    /// Sampled action index (for policy gradient).
    pub action: usize,
}

/// Chooser forward pass.
///
/// Returns (logits, softmax_probs, cache).
pub fn chooser_forward(weights: &ChooserWeights, h: &[f64]) -> (Vec<f64>, Vec<f64>, ChooserCache) {
    // logits = W @ h + b
    let logits_raw = matvec(&weights.w, h, NUM_OCTANTS, HIDDEN_DIM);
    let logits = vec_add(&logits_raw, &weights.b);
    let probs = softmax(&logits);

    let cache = ChooserCache {
        h: h.to_vec(),
        logits: logits.clone(),
        probs: probs.clone(),
        action: 0, // Will be set during sampling
    };

    (logits, probs, cache)
}

/// Gradients for chooser weights.
#[derive(Clone)]
pub struct ChooserGradients {
    pub dw: Vec<f64>,
    pub db: Vec<f64>,
}

impl ChooserGradients {
    /// Create zero-initialized gradients.
    pub fn zeros() -> Self {
        Self {
            dw: vec![0.0; NUM_OCTANTS * HIDDEN_DIM],
            db: vec![0.0; NUM_OCTANTS],
        }
    }

    /// Add another gradient.
    pub fn add(&mut self, other: &ChooserGradients) {
        for (a, b) in self.dw.iter_mut().zip(other.dw.iter()) {
            *a += b;
        }
        for (a, b) in self.db.iter_mut().zip(other.db.iter()) {
            *a += b;
        }
    }

    /// Scale gradients.
    pub fn scale(&mut self, s: f64) {
        for v in self.dw.iter_mut() {
            *v *= s;
        }
        for v in self.db.iter_mut() {
            *v *= s;
        }
    }

    /// Clip gradient norms.
    pub fn clip(&mut self, max_norm: f64) {
        clip_vec(&mut self.dw, max_norm);
        clip_vec(&mut self.db, max_norm);
    }
}

fn clip_vec(v: &mut [f64], max_norm: f64) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
}

/// Chooser backward pass for policy gradient.
///
/// For REINFORCE, we need d(log pi(a)) / d(params), where pi(a) = softmax(logits)[a].
/// d(log pi(a)) / d(logits[i]) = 1(i == a) - pi(i)
///
/// Returns (gradients, dL/dh).
pub fn chooser_backward_policy_gradient(
    weights: &ChooserWeights,
    cache: &ChooserCache,
    advantage: f64,
) -> (ChooserGradients, Vec<f64>) {
    // Gradient of log probability w.r.t. logits
    // d(log pi[action]) / d(logits[i]) = 1(i == action) - probs[i]
    let mut d_logits = vec![0.0; NUM_OCTANTS];
    for i in 0..NUM_OCTANTS {
        d_logits[i] = if i == cache.action { 1.0 } else { 0.0 };
        d_logits[i] -= cache.probs[i];
    }

    // Scale by advantage (REINFORCE)
    for v in d_logits.iter_mut() {
        *v *= advantage;
    }

    // Gradients for W and b
    // logits = W @ h + b
    // d/dW = d_logits @ h^T
    // d/db = d_logits
    let dw = outer_product(&d_logits, &cache.h);
    let db = d_logits.clone();

    // Gradient w.r.t. h
    // d/dh = W^T @ d_logits
    let dh = matvec_transpose(&weights.w, &d_logits, NUM_OCTANTS, HIDDEN_DIM);

    (ChooserGradients { dw, db }, dh)
}

/// Transpose matrix-vector multiplication.
fn matvec_transpose(matrix: &[f64], vec: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), rows * cols);
    debug_assert_eq!(vec.len(), rows);

    let mut result = vec![0.0; cols];
    for i in 0..rows {
        let row_start = i * cols;
        for j in 0..cols {
            result[j] += matrix[row_start + j] * vec[i];
        }
    }
    result
}

/// Get the 8 cube corner positions for a given vertex and cell_size.
///
/// Corner index encodes signs: bit0=X, bit1=Y, bit2=Z.
/// Index 0: (-, -, -), Index 7: (+, +, +).
pub fn octant_corners(vertex: (f64, f64, f64), cell_size: f64) -> [(f64, f64, f64); 8] {
    let half = cell_size * 0.5;
    [
        (vertex.0 - half, vertex.1 - half, vertex.2 - half), // 0: ---
        (vertex.0 + half, vertex.1 - half, vertex.2 - half), // 1: +--
        (vertex.0 - half, vertex.1 + half, vertex.2 - half), // 2: -+-
        (vertex.0 + half, vertex.1 + half, vertex.2 - half), // 3: ++-
        (vertex.0 - half, vertex.1 - half, vertex.2 + half), // 4: --+
        (vertex.0 + half, vertex.1 - half, vertex.2 + half), // 5: +-+
        (vertex.0 - half, vertex.1 + half, vertex.2 + half), // 6: -++
        (vertex.0 + half, vertex.1 + half, vertex.2 + half), // 7: +++
    ]
}

/// Compute weighted average position from octant weights.
pub fn weighted_position(
    weights: &[f64],
    corners: &[(f64, f64, f64); 8],
) -> (f64, f64, f64) {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;
    let mut sum = 0.0;

    for (i, &w) in weights.iter().enumerate() {
        // Use exp(w) to ensure positive weights
        let w_pos = w.exp();
        x += w_pos * corners[i].0;
        y += w_pos * corners[i].1;
        z += w_pos * corners[i].2;
        sum += w_pos;
    }

    if sum > 1e-12 {
        (x / sum, y / sum, z / sum)
    } else {
        // Fallback to center
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for corner in corners.iter() {
            cx += corner.0;
            cy += corner.1;
            cz += corner.2;
        }
        (cx / 8.0, cy / 8.0, cz / 8.0)
    }
}

/// Compute position from softmax probabilities over octants.
pub fn position_from_probs(
    probs: &[f64],
    corners: &[(f64, f64, f64); 8],
) -> (f64, f64, f64) {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        x += p * corners[i].0;
        y += p * corners[i].1;
        z += p * corners[i].2;
    }

    (x, y, z)
}

/// Get corner position for a specific octant index.
pub fn octant_corner(vertex: (f64, f64, f64), cell_size: f64, index: usize) -> (f64, f64, f64) {
    let half = cell_size * 0.5;
    let x = if index & 1 == 0 { vertex.0 - half } else { vertex.0 + half };
    let y = if index & 2 == 0 { vertex.1 - half } else { vertex.1 + half };
    let z = if index & 4 == 0 { vertex.2 - half } else { vertex.2 + half };
    (x, y, z)
}

/// Sample an action from categorical distribution.
pub fn sample_categorical(probs: &[f64], rng: &mut Rng) -> usize {
    let mut r = rng.next_f64();
    for (i, &p) in probs.iter().enumerate() {
        if r <= p {
            return i;
        }
        r -= p;
    }
    probs.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octant_corners() {
        let vertex = (0.0, 0.0, 0.0);
        let cell_size = 1.0;
        let corners = octant_corners(vertex, cell_size);

        assert_eq!(corners[0], (-0.5, -0.5, -0.5));
        assert_eq!(corners[7], (0.5, 0.5, 0.5));
    }

    #[test]
    fn test_weighted_position() {
        let vertex = (0.0, 0.0, 0.0);
        let cell_size = 1.0;
        let corners = octant_corners(vertex, cell_size);

        // Equal weights should give center
        let weights = vec![0.0; 8];
        let pos = weighted_position(&weights, &corners);
        assert!((pos.0 - 0.0).abs() < 1e-10);
        assert!((pos.1 - 0.0).abs() < 1e-10);
        assert!((pos.2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_from_probs() {
        let vertex = (0.0, 0.0, 0.0);
        let cell_size = 1.0;
        let corners = octant_corners(vertex, cell_size);

        // All probability on corner 7 (+++)
        let mut probs = vec![0.0; 8];
        probs[7] = 1.0;
        let pos = position_from_probs(&probs, &corners);
        assert!((pos.0 - 0.5).abs() < 1e-10);
        assert!((pos.1 - 0.5).abs() < 1e-10);
        assert!((pos.2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_chooser_forward_shape() {
        let mut rng = Rng::new(42);
        let weights = ChooserWeights::new(&mut rng);
        let h = vec![0.1; HIDDEN_DIM];

        let (logits, probs, _cache) = chooser_forward(&weights, &h);

        assert_eq!(logits.len(), NUM_OCTANTS);
        assert_eq!(probs.len(), NUM_OCTANTS);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
