//! GRU (Gated Recurrent Unit) implementation.
//!
//! Implements a GRU cell with forward and backward passes for gradient computation.
//! The GRU is simpler than LSTM and sufficient for short episodes (~50 steps).

use super::math::{
    matvec, outer_product, vec_add, vec_mul, vec_sigmoid, vec_tanh,
    sigmoid_derivative, tanh_derivative,
};

/// Hidden dimension for the GRU.
pub const HIDDEN_DIM: usize = 64;

/// GRU weights for a single cell.
///
/// GRU equations:
///   z = sigmoid(W_z @ x + U_z @ h_prev + b_z)  // update gate
///   r = sigmoid(W_r @ x + U_r @ h_prev + b_r)  // reset gate
///   h_tilde = tanh(W_h @ x + U_h @ (r * h_prev) + b_h)  // candidate
///   h = (1 - z) * h_prev + z * h_tilde  // new hidden state
#[derive(Clone)]
pub struct GruWeights {
    /// Input dimension.
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

impl GruWeights {
    /// Create new GRU weights initialized to small random values.
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

    /// Total number of parameters.
    pub fn param_count(&self) -> usize {
        let h = HIDDEN_DIM;
        let i = self.input_dim;
        // 3 gates, each with W (h*i), U (h*h), b (h)
        3 * (h * i + h * h + h)
    }
}

/// Intermediate values from GRU forward pass, needed for backpropagation.
#[derive(Clone)]
pub struct GruCache {
    /// Input vector.
    pub x: Vec<f64>,
    /// Previous hidden state.
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
    /// New hidden state.
    pub h: Vec<f64>,
}

/// GRU forward pass.
///
/// Returns (new_hidden_state, cache_for_backward).
pub fn gru_forward(weights: &GruWeights, x: &[f64], h_prev: &[f64]) -> (Vec<f64>, GruCache) {
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

    let cache = GruCache {
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

/// Gradients for GRU weights.
#[derive(Clone)]
pub struct GruGradients {
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

impl GruGradients {
    /// Create zero-initialized gradients.
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
    pub fn add(&mut self, other: &GruGradients) {
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

/// GRU backward pass.
///
/// Given dL/dh (gradient of loss w.r.t. output hidden state),
/// computes gradients w.r.t. weights and gradient w.r.t. h_prev.
///
/// Returns (gradients, dL/dh_prev).
pub fn gru_backward(
    weights: &GruWeights,
    cache: &GruCache,
    dh: &[f64],
) -> (GruGradients, Vec<f64>) {
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

    // Gradient w.r.t. h_prev
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

    // Sum all contributions
    let dh_prev: Vec<f64> = (0..h_dim)
        .map(|i| {
            dh_prev_from_h[i] + dh_prev_from_z[i] + dh_prev_from_r[i] + dh_prev_from_rh[i]
        })
        .collect();

    let grads = GruGradients {
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

    (grads, dh_prev)
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
        let h_prev = vec![0.0; HIDDEN_DIM];

        let (h, cache) = gru_forward(&weights, &x, &h_prev);

        assert_eq!(h.len(), HIDDEN_DIM);
        assert_eq!(cache.z.len(), HIDDEN_DIM);
        assert_eq!(cache.r.len(), HIDDEN_DIM);
        assert_eq!(cache.h_tilde.len(), HIDDEN_DIM);
    }

    #[test]
    fn test_gru_backward_shape() {
        let mut rng = Rng::new(42);
        let input_dim = 10;
        let weights = GruWeights::new(input_dim, &mut rng);

        let x = vec![0.1; input_dim];
        let h_prev = vec![0.0; HIDDEN_DIM];

        let (h, cache) = gru_forward(&weights, &x, &h_prev);
        let dh = vec![1.0; HIDDEN_DIM];

        let (grads, dh_prev) = gru_backward(&weights, &cache, &dh);

        assert_eq!(dh_prev.len(), HIDDEN_DIM);
        assert_eq!(grads.dw_z.len(), HIDDEN_DIM * input_dim);
        assert_eq!(grads.du_z.len(), HIDDEN_DIM * HIDDEN_DIM);
        assert_eq!(grads.db_z.len(), HIDDEN_DIM);
    }

    #[test]
    fn test_param_count() {
        let mut rng = Rng::new(42);
        let input_dim = 10;
        let weights = GruWeights::new(input_dim, &mut rng);
        let count = weights.param_count();
        // 3 * (32*10 + 32*32 + 32) = 3 * (320 + 1024 + 32) = 3 * 1376 = 4128
        assert_eq!(count, 3 * (HIDDEN_DIM * input_dim + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM));
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
