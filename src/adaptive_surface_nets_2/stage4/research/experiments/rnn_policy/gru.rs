//! Stacked GRU (Gated Recurrent Unit) implementation.
//!
//! Implements a multi-layer GRU with forward and backward passes for gradient computation.
//! Each layer processes the output of the previous layer, building hierarchical representations.
//!
//! ## Architecture
//!
//! ```text
//! Input x ──► GRU Layer 0 ──► h[0] ──► GRU Layer 1 ──► h[1] ──► ... ──► h[N-1]
//!                  ▲                        ▲                              │
//!             h_prev[0]                h_prev[1]                     final output
//! ```
//!
//! - Layer 0: input_dim → hidden_dim
//! - Layer 1+: hidden_dim → hidden_dim
//! - Final layer's hidden state is used by downstream heads (chooser, classifier)
//!
//! ## Configuration
//!
//! - `HIDDEN_DIM`: Size of hidden state per layer (default: 32)
//! - `NUM_LAYERS`: Number of stacked GRU layers (default: 2)
//!
//! ## GRU Equations (per layer)
//!
//! ```text
//! z = sigmoid(W_z @ x + U_z @ h_prev + b_z)       // update gate
//! r = sigmoid(W_r @ x + U_r @ h_prev + b_r)       // reset gate
//! h_tilde = tanh(W_h @ x + U_h @ (r * h_prev) + b_h)  // candidate
//! h = (1 - z) * h_prev + z * h_tilde             // new hidden state
//! ```

use super::math::{
    matvec, outer_product, vec_add, vec_mul, vec_sigmoid, vec_tanh,
    sigmoid_derivative, tanh_derivative,
};

/// Hidden dimension for the GRU.
pub const HIDDEN_DIM: usize = 32;

/// Number of stacked GRU layers.
pub const NUM_LAYERS: usize = 2;

/// GRU weights for a single layer.
///
/// GRU equations:
///   z = sigmoid(W_z @ x + U_z @ h_prev + b_z)  // update gate
///   r = sigmoid(W_r @ x + U_r @ h_prev + b_r)  // reset gate
///   h_tilde = tanh(W_h @ x + U_h @ (r * h_prev) + b_h)  // candidate
///   h = (1 - z) * h_prev + z * h_tilde  // new hidden state
#[derive(Clone)]
pub struct GruLayerWeights {
    /// Input dimension for this layer.
    pub input_dim: usize,
    /// Update gate weights for input: W_z [hidden_dim x input_dim].
    pub w_z: Vec<f64>,
    /// Update gate weights for hidden: U_z [hidden_dim x hidden_dim].
    pub u_z: Vec<f64>,
    /// Update gate bias: b_z [hidden_dim].
    pub b_z: Vec<f64>,
    /// Reset gate weights for input: W_r [hidden_dim x input_dim].
    pub w_r: Vec<f64>,
    /// Reset gate weights for hidden: U_r [hidden_dim x hidden_dim].
    pub u_r: Vec<f64>,
    /// Reset gate bias: b_r [hidden_dim].
    pub b_r: Vec<f64>,
    /// Candidate weights for input: W_h [hidden_dim x input_dim].
    pub w_h: Vec<f64>,
    /// Candidate weights for hidden: U_h [hidden_dim x hidden_dim].
    pub u_h: Vec<f64>,
    /// Candidate bias: b_h [hidden_dim].
    pub b_h: Vec<f64>,
}

impl GruLayerWeights {
    /// Create new GRU layer weights initialized to small random values.
    pub fn new(input_dim: usize, rng: &mut Rng) -> Self {
        let hidden = HIDDEN_DIM;
        let scale = 0.1;

        Self {
            input_dim,
            w_z: random_vec(hidden * input_dim, scale, rng),
            u_z: random_vec(hidden * hidden, scale, rng),
            b_z: vec![0.0; hidden],
            w_r: random_vec(hidden * input_dim, scale, rng),
            u_r: random_vec(hidden * hidden, scale, rng),
            b_r: vec![0.0; hidden],
            w_h: random_vec(hidden * input_dim, scale, rng),
            u_h: random_vec(hidden * hidden, scale, rng),
            b_h: vec![0.0; hidden],
        }
    }

    /// Total number of parameters for this layer.
    pub fn param_count(&self) -> usize {
        let h = HIDDEN_DIM;
        let i = self.input_dim;
        // 3 gates, each with W (h*i), U (h*h), b (h)
        3 * (h * i + h * h + h)
    }
}

/// Stacked GRU weights for multiple layers.
///
/// Layer 0 takes external input (input_dim), subsequent layers take
/// the hidden state from the previous layer (hidden_dim).
#[derive(Clone)]
pub struct GruWeights {
    /// Weights for each layer.
    pub layers: Vec<GruLayerWeights>,
}

impl GruWeights {
    /// Create new stacked GRU weights.
    /// Layer 0: input_dim -> hidden_dim
    /// Layer 1+: hidden_dim -> hidden_dim
    pub fn new(input_dim: usize, rng: &mut Rng) -> Self {
        let mut layers = Vec::with_capacity(NUM_LAYERS);

        for layer_idx in 0..NUM_LAYERS {
            let layer_input_dim = if layer_idx == 0 { input_dim } else { HIDDEN_DIM };
            layers.push(GruLayerWeights::new(layer_input_dim, rng));
        }

        Self { layers }
    }

    /// Total number of parameters across all layers.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }

    /// Get the input dimension (for layer 0).
    pub fn input_dim(&self) -> usize {
        self.layers[0].input_dim
    }
}

/// Intermediate values from a single GRU layer forward pass.
#[derive(Clone)]
pub struct GruLayerCache {
    /// Input vector to this layer.
    pub x: Vec<f64>,
    /// Previous hidden state for this layer.
    pub h_prev: Vec<f64>,
    /// Update gate pre-activation.
    pub z_pre: Vec<f64>,
    /// Update gate output.
    pub z: Vec<f64>,
    /// Reset gate pre-activation.
    pub r_pre: Vec<f64>,
    /// Reset gate output.
    pub r: Vec<f64>,
    /// Candidate pre-activation.
    pub h_tilde_pre: Vec<f64>,
    /// Candidate output.
    pub h_tilde: Vec<f64>,
    /// Reset gate applied to h_prev.
    pub r_h_prev: Vec<f64>,
    /// New hidden state for this layer.
    pub h: Vec<f64>,
}

/// Intermediate values from stacked GRU forward pass.
#[derive(Clone)]
pub struct GruCache {
    /// Cache for each layer.
    pub layers: Vec<GruLayerCache>,
}

/// Single GRU layer forward pass.
fn gru_layer_forward(
    weights: &GruLayerWeights,
    x: &[f64],
    h_prev: &[f64],
) -> (Vec<f64>, GruLayerCache) {
    let h_dim = HIDDEN_DIM;

    // Update gate: z = sigmoid(W_z @ x + U_z @ h_prev + b_z)
    let wx_z = matvec(&weights.w_z, x, h_dim, weights.input_dim);
    let uh_z = matvec(&weights.u_z, h_prev, h_dim, h_dim);
    let z_pre = vec_add(&vec_add(&wx_z, &uh_z), &weights.b_z);
    let z = vec_sigmoid(&z_pre);

    // Reset gate: r = sigmoid(W_r @ x + U_r @ h_prev + b_r)
    let wx_r = matvec(&weights.w_r, x, h_dim, weights.input_dim);
    let uh_r = matvec(&weights.u_r, h_prev, h_dim, h_dim);
    let r_pre = vec_add(&vec_add(&wx_r, &uh_r), &weights.b_r);
    let r = vec_sigmoid(&r_pre);

    // Candidate: h_tilde = tanh(W_h @ x + U_h @ (r * h_prev) + b_h)
    let r_h_prev = vec_mul(&r, h_prev);
    let wx_h = matvec(&weights.w_h, x, h_dim, weights.input_dim);
    let uh_h = matvec(&weights.u_h, &r_h_prev, h_dim, h_dim);
    let h_tilde_pre = vec_add(&vec_add(&wx_h, &uh_h), &weights.b_h);
    let h_tilde = vec_tanh(&h_tilde_pre);

    // New hidden: h = (1 - z) * h_prev + z * h_tilde
    let one_minus_z: Vec<f64> = z.iter().map(|&zi| 1.0 - zi).collect();
    let h: Vec<f64> = one_minus_z
        .iter()
        .zip(h_prev.iter())
        .zip(z.iter().zip(h_tilde.iter()))
        .map(|((&omz, &hp), (&zi, &ht))| omz * hp + zi * ht)
        .collect();

    let cache = GruLayerCache {
        x: x.to_vec(),
        h_prev: h_prev.to_vec(),
        z_pre,
        z,
        r_pre,
        r,
        h_tilde_pre,
        h_tilde,
        r_h_prev,
        h: h.clone(),
    };

    (h, cache)
}

/// Stacked GRU forward pass.
///
/// Takes input x and previous hidden states for all layers.
/// Returns new hidden states for all layers and cache for backward pass.
///
/// h_prev should have NUM_LAYERS elements, one per layer.
/// Returns h_new with NUM_LAYERS elements.
pub fn gru_forward(
    weights: &GruWeights,
    x: &[f64],
    h_prev: &[Vec<f64>],
) -> (Vec<Vec<f64>>, GruCache) {
    debug_assert_eq!(h_prev.len(), NUM_LAYERS);
    debug_assert_eq!(weights.layers.len(), NUM_LAYERS);

    let mut layer_caches: Vec<GruLayerCache> = Vec::with_capacity(NUM_LAYERS);
    let mut h_new: Vec<Vec<f64>> = Vec::with_capacity(NUM_LAYERS);

    for layer_idx in 0..NUM_LAYERS {
        let layer_weights = &weights.layers[layer_idx];

        // Input to this layer: external x for layer 0, previous layer's output otherwise
        let layer_input: &[f64] = if layer_idx == 0 {
            x
        } else {
            &h_new[layer_idx - 1]
        };

        let (h_out, cache) = gru_layer_forward(layer_weights, layer_input, &h_prev[layer_idx]);
        layer_caches.push(cache);
        h_new.push(h_out);
    }

    let cache = GruCache {
        layers: layer_caches,
    };

    (h_new, cache)
}

/// Initialize hidden states for all layers (zeros).
pub fn init_hidden() -> Vec<Vec<f64>> {
    (0..NUM_LAYERS).map(|_| vec![0.0; HIDDEN_DIM]).collect()
}

/// Get the final layer's hidden state (for downstream use like classifier).
pub fn final_hidden(h: &[Vec<f64>]) -> &[f64] {
    &h[NUM_LAYERS - 1]
}

/// Gradients for a single GRU layer.
#[derive(Clone)]
pub struct GruLayerGradients {
    pub dw_z: Vec<f64>,
    pub du_z: Vec<f64>,
    pub db_z: Vec<f64>,
    pub dw_r: Vec<f64>,
    pub du_r: Vec<f64>,
    pub db_r: Vec<f64>,
    pub dw_h: Vec<f64>,
    pub du_h: Vec<f64>,
    pub db_h: Vec<f64>,
}

impl GruLayerGradients {
    /// Create zero-initialized gradients for a layer.
    pub fn zeros(input_dim: usize) -> Self {
        let h = HIDDEN_DIM;
        Self {
            dw_z: vec![0.0; h * input_dim],
            du_z: vec![0.0; h * h],
            db_z: vec![0.0; h],
            dw_r: vec![0.0; h * input_dim],
            du_r: vec![0.0; h * h],
            db_r: vec![0.0; h],
            dw_h: vec![0.0; h * input_dim],
            du_h: vec![0.0; h * h],
            db_h: vec![0.0; h],
        }
    }

    /// Add another gradient (for accumulation).
    pub fn add(&mut self, other: &GruLayerGradients) {
        for (a, b) in self.dw_z.iter_mut().zip(other.dw_z.iter()) {
            *a += b;
        }
        for (a, b) in self.du_z.iter_mut().zip(other.du_z.iter()) {
            *a += b;
        }
        for (a, b) in self.db_z.iter_mut().zip(other.db_z.iter()) {
            *a += b;
        }
        for (a, b) in self.dw_r.iter_mut().zip(other.dw_r.iter()) {
            *a += b;
        }
        for (a, b) in self.du_r.iter_mut().zip(other.du_r.iter()) {
            *a += b;
        }
        for (a, b) in self.db_r.iter_mut().zip(other.db_r.iter()) {
            *a += b;
        }
        for (a, b) in self.dw_h.iter_mut().zip(other.dw_h.iter()) {
            *a += b;
        }
        for (a, b) in self.du_h.iter_mut().zip(other.du_h.iter()) {
            *a += b;
        }
        for (a, b) in self.db_h.iter_mut().zip(other.db_h.iter()) {
            *a += b;
        }
    }

    /// Scale gradients.
    pub fn scale(&mut self, s: f64) {
        for v in self.dw_z.iter_mut() {
            *v *= s;
        }
        for v in self.du_z.iter_mut() {
            *v *= s;
        }
        for v in self.db_z.iter_mut() {
            *v *= s;
        }
        for v in self.dw_r.iter_mut() {
            *v *= s;
        }
        for v in self.du_r.iter_mut() {
            *v *= s;
        }
        for v in self.db_r.iter_mut() {
            *v *= s;
        }
        for v in self.dw_h.iter_mut() {
            *v *= s;
        }
        for v in self.du_h.iter_mut() {
            *v *= s;
        }
        for v in self.db_h.iter_mut() {
            *v *= s;
        }
    }

    /// Clip gradient norms.
    pub fn clip(&mut self, max_norm: f64) {
        clip_vec(&mut self.dw_z, max_norm);
        clip_vec(&mut self.du_z, max_norm);
        clip_vec(&mut self.db_z, max_norm);
        clip_vec(&mut self.dw_r, max_norm);
        clip_vec(&mut self.du_r, max_norm);
        clip_vec(&mut self.db_r, max_norm);
        clip_vec(&mut self.dw_h, max_norm);
        clip_vec(&mut self.du_h, max_norm);
        clip_vec(&mut self.db_h, max_norm);
    }
}

/// Gradients for all GRU layers.
#[derive(Clone)]
pub struct GruGradients {
    pub layers: Vec<GruLayerGradients>,
}

impl GruGradients {
    /// Create zero-initialized gradients for all layers.
    pub fn zeros(weights: &GruWeights) -> Self {
        let layers = weights
            .layers
            .iter()
            .map(|l| GruLayerGradients::zeros(l.input_dim))
            .collect();
        Self { layers }
    }

    /// Add another gradient (for accumulation).
    pub fn add(&mut self, other: &GruGradients) {
        for (a, b) in self.layers.iter_mut().zip(other.layers.iter()) {
            a.add(b);
        }
    }

    /// Scale gradients.
    pub fn scale(&mut self, s: f64) {
        for layer in &mut self.layers {
            layer.scale(s);
        }
    }

    /// Clip gradient norms.
    pub fn clip(&mut self, max_norm: f64) {
        for layer in &mut self.layers {
            layer.clip(max_norm);
        }
    }
}

/// Clip vector norm to max_norm.
fn clip_vec(v: &mut [f64], max_norm: f64) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
}

/// Single GRU layer backward pass.
///
/// Returns (layer_gradients, dL/dh_prev, dL/dx).
fn gru_layer_backward(
    weights: &GruLayerWeights,
    cache: &GruLayerCache,
    dh: &[f64],
) -> (GruLayerGradients, Vec<f64>, Vec<f64>) {
    let h_dim = HIDDEN_DIM;

    // h = (1 - z) * h_prev + z * h_tilde
    // dh/dz = h_tilde - h_prev
    // dh/dh_tilde = z
    // dh/dh_prev = 1 - z

    // Gradient w.r.t. h_tilde
    let dh_tilde: Vec<f64> = dh.iter().zip(cache.z.iter()).map(|(&dhi, &zi)| dhi * zi).collect();

    // Gradient w.r.t. z
    let dz: Vec<f64> = dh
        .iter()
        .zip(cache.h_tilde.iter().zip(cache.h_prev.iter()))
        .map(|(&dhi, (&ht, &hp))| dhi * (ht - hp))
        .collect();

    // Gradient through h_tilde = tanh(h_tilde_pre)
    let dh_tilde_pre: Vec<f64> = dh_tilde
        .iter()
        .zip(cache.h_tilde.iter())
        .map(|(&dht, &ht)| dht * tanh_derivative(ht))
        .collect();

    // Gradients for candidate computation
    // h_tilde_pre = W_h @ x + U_h @ (r * h_prev) + b_h
    let db_h = dh_tilde_pre.clone();
    let dw_h = outer_product(&dh_tilde_pre, &cache.x);
    let du_h = outer_product(&dh_tilde_pre, &cache.r_h_prev);

    // Gradient w.r.t. r_h_prev
    let dr_h_prev = matvec_transpose(&weights.u_h, &dh_tilde_pre, h_dim, h_dim);

    // Gradient w.r.t. r (through r_h_prev = r * h_prev)
    let dr: Vec<f64> = dr_h_prev
        .iter()
        .zip(cache.h_prev.iter())
        .map(|(&drh, &hp)| drh * hp)
        .collect();

    // Gradient through z = sigmoid(z_pre)
    let dz_pre: Vec<f64> = dz
        .iter()
        .zip(cache.z.iter())
        .map(|(&dzi, &zi)| dzi * sigmoid_derivative(zi))
        .collect();

    // Gradients for update gate
    let db_z = dz_pre.clone();
    let dw_z = outer_product(&dz_pre, &cache.x);
    let du_z = outer_product(&dz_pre, &cache.h_prev);

    // Gradient through r = sigmoid(r_pre)
    let dr_pre: Vec<f64> = dr
        .iter()
        .zip(cache.r.iter())
        .map(|(&dri, &ri)| dri * sigmoid_derivative(ri))
        .collect();

    // Gradients for reset gate
    let db_r = dr_pre.clone();
    let dw_r = outer_product(&dr_pre, &cache.x);
    let du_r = outer_product(&dr_pre, &cache.h_prev);

    // Gradient w.r.t. h_prev (for BPTT to previous timestep)
    // From: h = (1 - z) * h_prev + z * h_tilde
    let dh_prev_from_h: Vec<f64> = dh
        .iter()
        .zip(cache.z.iter())
        .map(|(&dhi, &zi)| dhi * (1.0 - zi))
        .collect();

    // From: z_pre = ... + U_z @ h_prev
    let dh_prev_from_z = matvec_transpose(&weights.u_z, &dz_pre, h_dim, h_dim);

    // From: r_pre = ... + U_r @ h_prev
    let dh_prev_from_r = matvec_transpose(&weights.u_r, &dr_pre, h_dim, h_dim);

    // From: r_h_prev = r * h_prev
    let dh_prev_from_rh: Vec<f64> = dr_h_prev
        .iter()
        .zip(cache.r.iter())
        .map(|(&drh, &ri)| drh * ri)
        .collect();

    // Sum all contributions to dh_prev
    let dh_prev: Vec<f64> = (0..h_dim)
        .map(|i| {
            dh_prev_from_h[i] + dh_prev_from_z[i] + dh_prev_from_r[i] + dh_prev_from_rh[i]
        })
        .collect();

    // Gradient w.r.t. input x (for backprop to previous layer)
    // From: z_pre = W_z @ x + ...
    let dx_from_z = matvec_transpose(&weights.w_z, &dz_pre, h_dim, weights.input_dim);
    // From: r_pre = W_r @ x + ...
    let dx_from_r = matvec_transpose(&weights.w_r, &dr_pre, h_dim, weights.input_dim);
    // From: h_tilde_pre = W_h @ x + ...
    let dx_from_h = matvec_transpose(&weights.w_h, &dh_tilde_pre, h_dim, weights.input_dim);

    let dx: Vec<f64> = (0..weights.input_dim)
        .map(|i| dx_from_z[i] + dx_from_r[i] + dx_from_h[i])
        .collect();

    let grads = GruLayerGradients {
        dw_z,
        du_z,
        db_z,
        dw_r,
        du_r,
        db_r,
        dw_h,
        du_h,
        db_h,
    };

    (grads, dh_prev, dx)
}

/// Stacked GRU backward pass.
///
/// Given dL/dh for the final layer output, computes gradients for all layers
/// and returns dL/dh_prev for all layers (needed for BPTT).
///
/// Returns (gradients, dL/dh_prev for all layers).
pub fn gru_backward(
    weights: &GruWeights,
    cache: &GruCache,
    dh_final: &[f64],
) -> (GruGradients, Vec<Vec<f64>>) {
    let mut layer_grads = vec![];
    let mut dh_prev_all = vec![vec![]; NUM_LAYERS];

    // Backprop through layers in reverse order
    let mut dh_current = dh_final.to_vec();

    for layer_idx in (0..NUM_LAYERS).rev() {
        let (layer_grad, dh_prev, dx) = gru_layer_backward(
            &weights.layers[layer_idx],
            &cache.layers[layer_idx],
            &dh_current,
        );
        layer_grads.push(layer_grad);
        dh_prev_all[layer_idx] = dh_prev;

        // dx becomes dh for the previous layer (if any)
        if layer_idx > 0 {
            dh_current = dx;
        }
    }

    // Reverse layer_grads since we collected them in reverse order
    layer_grads.reverse();

    let grads = GruGradients { layers: layer_grads };

    (grads, dh_prev_all)
}

/// Transpose matrix-vector multiplication: result = M^T @ v.
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

/// Simple RNG for weight initialization.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64 / u32::MAX as f64).min(0.999999)
    }

    pub fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max.max(1)
    }

    /// Random f64 in range [-scale, scale].
    pub fn next_range(&mut self, scale: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * scale
    }

    /// Generate a Gaussian random number (mean=0, std=1) using Box-Muller transform.
    pub fn next_gaussian(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-10); // Avoid log(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate a uniform random rotation matrix (uniform on SO(3)).
    ///
    /// Uses the quaternion method: sample 4 Gaussians, normalize to unit quaternion,
    /// then convert to rotation matrix.
    pub fn next_rotation_matrix(&mut self) -> [[f64; 3]; 3] {
        // Sample 4 Gaussian values for quaternion (w, x, y, z)
        let w = self.next_gaussian();
        let x = self.next_gaussian();
        let y = self.next_gaussian();
        let z = self.next_gaussian();

        // Normalize to unit quaternion
        let len = (w * w + x * x + y * y + z * z).sqrt();
        let (w, x, y, z) = if len > 1e-10 {
            (w / len, x / len, y / len, z / len)
        } else {
            (1.0, 0.0, 0.0, 0.0) // Fallback to identity
        };

        // Convert quaternion to rotation matrix
        // https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    }
}

/// Generate random vector with values in [-scale, scale].
fn random_vec(len: usize, scale: f64, rng: &mut Rng) -> Vec<f64> {
    (0..len).map(|_| rng.next_range(scale)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gru_forward_shape() {
        let mut rng = Rng::new(42);
        let input_dim = 10;
        let weights = GruWeights::new(input_dim, &mut rng);

        let x = vec![0.1; input_dim];
        let h_prev = init_hidden();

        let (h, cache) = gru_forward(&weights, &x, &h_prev);

        // h should have NUM_LAYERS entries
        assert_eq!(h.len(), NUM_LAYERS);
        for layer_h in &h {
            assert_eq!(layer_h.len(), HIDDEN_DIM);
        }

        // cache should have NUM_LAYERS layer caches
        assert_eq!(cache.layers.len(), NUM_LAYERS);
        for layer_cache in &cache.layers {
            assert_eq!(layer_cache.z.len(), HIDDEN_DIM);
            assert_eq!(layer_cache.r.len(), HIDDEN_DIM);
            assert_eq!(layer_cache.h_tilde.len(), HIDDEN_DIM);
        }
    }

    #[test]
    fn test_gru_backward_shape() {
        let mut rng = Rng::new(42);
        let input_dim = 10;
        let weights = GruWeights::new(input_dim, &mut rng);

        let x = vec![0.1; input_dim];
        let h_prev = init_hidden();

        let (h, cache) = gru_forward(&weights, &x, &h_prev);
        let dh = vec![1.0; HIDDEN_DIM]; // gradient for final layer output

        let (grads, dh_prev_all) = gru_backward(&weights, &cache, &dh);

        // dh_prev_all should have NUM_LAYERS entries
        assert_eq!(dh_prev_all.len(), NUM_LAYERS);
        for dh_prev in &dh_prev_all {
            assert_eq!(dh_prev.len(), HIDDEN_DIM);
        }

        // grads should have NUM_LAYERS layer gradients
        assert_eq!(grads.layers.len(), NUM_LAYERS);

        // Layer 0 has input_dim inputs
        assert_eq!(grads.layers[0].dw_z.len(), HIDDEN_DIM * input_dim);
        assert_eq!(grads.layers[0].du_z.len(), HIDDEN_DIM * HIDDEN_DIM);
        assert_eq!(grads.layers[0].db_z.len(), HIDDEN_DIM);

        // Layer 1+ has HIDDEN_DIM inputs
        for layer_idx in 1..NUM_LAYERS {
            assert_eq!(grads.layers[layer_idx].dw_z.len(), HIDDEN_DIM * HIDDEN_DIM);
        }
    }

    #[test]
    fn test_param_count() {
        let mut rng = Rng::new(42);
        let input_dim = 10;
        let weights = GruWeights::new(input_dim, &mut rng);
        let count = weights.param_count();

        // Layer 0: 3 * (hidden*input + hidden*hidden + hidden)
        let layer0_params = 3 * (HIDDEN_DIM * input_dim + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM);
        // Layer 1+: 3 * (hidden*hidden + hidden*hidden + hidden)
        let layern_params = 3 * (HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM);

        let expected = layer0_params + (NUM_LAYERS - 1) * layern_params;
        assert_eq!(count, expected);
    }

    #[test]
    fn test_init_hidden() {
        let h = init_hidden();
        assert_eq!(h.len(), NUM_LAYERS);
        for layer_h in &h {
            assert_eq!(layer_h.len(), HIDDEN_DIM);
            for &val in layer_h {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn test_final_hidden() {
        let mut h = init_hidden();
        // Put some values in final layer
        h[NUM_LAYERS - 1][0] = 1.5;
        h[NUM_LAYERS - 1][1] = 2.5;

        let final_h = final_hidden(&h);
        assert_eq!(final_h.len(), HIDDEN_DIM);
        assert_eq!(final_h[0], 1.5);
        assert_eq!(final_h[1], 2.5);
    }

    #[test]
    fn test_rotation_matrix_is_valid() {
        let mut rng = Rng::new(12345);

        // Generate several random rotation matrices and verify properties
        for _ in 0..10 {
            let r = rng.next_rotation_matrix();

            // Check that R^T * R = I (orthogonality)
            let mut rtc = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        rtc[i][j] += r[k][i] * r[k][j]; // R^T * R
                    }
                }
            }

            // Should be identity
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (rtc[i][j] - expected).abs() < 1e-10,
                        "R^T * R not identity at [{},{}]: got {}",
                        i, j, rtc[i][j]
                    );
                }
            }

            // Check determinant is +1 (proper rotation, not reflection)
            let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                    - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                    + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
            assert!(
                (det - 1.0).abs() < 1e-10,
                "Determinant should be 1, got {}",
                det
            );
        }
    }

    #[test]
    fn test_rotation_preserves_length() {
        let mut rng = Rng::new(42);
        let r = rng.next_rotation_matrix();

        // Apply rotation to a unit vector
        let v = (1.0, 0.0, 0.0);
        let rv = (
            r[0][0] * v.0 + r[0][1] * v.1 + r[0][2] * v.2,
            r[1][0] * v.0 + r[1][1] * v.1 + r[1][2] * v.2,
            r[2][0] * v.0 + r[2][1] * v.1 + r[2][2] * v.2,
        );

        let len = (rv.0 * rv.0 + rv.1 * rv.1 + rv.2 * rv.2).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-10,
            "Rotation should preserve length, got {}",
            len
        );
    }
}
