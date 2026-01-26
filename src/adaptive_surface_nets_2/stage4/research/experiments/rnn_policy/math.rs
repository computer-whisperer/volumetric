//! Math utilities for RNN policy.
//!
//! Provides basic vector and matrix operations needed for neural network computations.

/// Vector dot product (slice version).
pub fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector multiplication: result[i] = sum_j(matrix[i][j] * vec[j]).
/// Matrix is stored in row-major order as a flat array.
pub fn matvec(matrix: &[f64], vec: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), rows * cols);
    debug_assert_eq!(vec.len(), cols);

    let mut result = vec![0.0; rows];
    for i in 0..rows {
        let row_start = i * cols;
        for j in 0..cols {
            result[i] += matrix[row_start + j] * vec[j];
        }
    }
    result
}

/// Outer product: result[i][j] = a[i] * b[j].
/// Returns flat array in row-major order.
pub fn outer_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; a.len() * b.len()];
    for (i, &ai) in a.iter().enumerate() {
        let row_start = i * b.len();
        for (j, &bj) in b.iter().enumerate() {
            result[row_start + j] = ai * bj;
        }
    }
    result
}

/// Element-wise vector addition: result[i] = a[i] + b[i].
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise vector subtraction: result[i] = a[i] - b[i].
pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Element-wise vector multiplication (Hadamard product): result[i] = a[i] * b[i].
pub fn vec_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Scale vector by scalar: result[i] = vec[i] * scalar.
pub fn vec_scale(vec: &[f64], scalar: f64) -> Vec<f64> {
    vec.iter().map(|x| x * scalar).collect()
}

/// Sigmoid activation function.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x)).
/// Takes sigmoid output as input for efficiency.
pub fn sigmoid_derivative(sig_x: f64) -> f64 {
    sig_x * (1.0 - sig_x)
}

/// Tanh activation function.
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Tanh derivative: 1 - tanh(x)^2.
/// Takes tanh output as input for efficiency.
pub fn tanh_derivative(tanh_x: f64) -> f64 {
    1.0 - tanh_x * tanh_x
}

/// Softmax function: exp(x[i]) / sum(exp(x[j])).
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-12 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|v| v / sum).collect()
}

/// Element-wise apply sigmoid to vector.
pub fn vec_sigmoid(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| sigmoid(x)).collect()
}

/// Element-wise apply tanh to vector.
pub fn vec_tanh(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| tanh(x)).collect()
}

/// Euclidean distance between two 3D points.
pub fn distance_3d(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dz = a.2 - b.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Length of a 3D vector.
pub fn length_3d(v: (f64, f64, f64)) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

/// Normalize a 3D vector.
pub fn normalize_3d(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = length_3d(v);
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 1.0)
    }
}

/// Add two 3D vectors.
pub fn add_3d(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

/// Subtract two 3D vectors.
pub fn sub_3d(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

// Aliases for classifier.rs compatibility
pub fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    sub_3d(a, b)
}

pub fn length(v: (f64, f64, f64)) -> f64 {
    length_3d(v)
}

pub fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    normalize_3d(v)
}

/// Dot product of two 3D vectors (tuple version).
pub fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

/// Cross product of two 3D vectors.
pub fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

/// Scale a 3D vector.
pub fn scale_3d(v: (f64, f64, f64), s: f64) -> (f64, f64, f64) {
    (v.0 * s, v.1 * s, v.2 * s)
}

/// Argmax: returns index of maximum value.
pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Clip a value to a range.
pub fn clip(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec() {
        // 2x3 matrix times 3-vector
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vec = vec![1.0, 2.0, 3.0];
        let result = matvec(&matrix, &vec, 2, 3);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-10); // 1*1 + 2*2 + 3*3
        assert!((result[1] - 32.0).abs() < 1e-10); // 4*1 + 5*2 + 6*3
    }

    #[test]
    fn test_outer_product() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let result = outer_product(&a, &b);
        assert_eq!(result.len(), 6);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
        assert!((result[3] - 6.0).abs() < 1e-10);
        assert!((result[4] - 8.0).abs() < 1e-10);
        assert!((result[5] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_sigmoid_tanh() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!((tanh(0.0) - 0.0).abs() < 1e-10);

        let sig = sigmoid(2.0);
        let dsig = sigmoid_derivative(sig);
        assert!(dsig > 0.0 && dsig < 0.25);

        let th = tanh(1.0);
        let dth = tanh_derivative(th);
        assert!(dth > 0.0 && dth < 1.0);
    }
}
