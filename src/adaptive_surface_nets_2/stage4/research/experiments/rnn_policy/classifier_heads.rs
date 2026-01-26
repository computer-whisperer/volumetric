//! Neural classification heads for geometry type and normal prediction.
//!
//! These heads replace the RANSAC-based classifier, learning to directly predict
//! geometry type (face/edge/corner) and normals from the RNN hidden state.
//!
//! ## Architecture
//!
//! Three parallel heads from the shared RNN hidden state:
//! - Face head: confidence + 1 normal (4 outputs)
//! - Edge head: confidence + 2 normals + edge direction (10 outputs)
//! - Corner head: confidence + 3 normals (10 outputs)
//!
//! ## Training Loss
//!
//! For the oracle-selected correct head:
//! - Maximize confidence (cross-entropy style)
//! - Minimize normal error (cosine similarity loss)
//!
//! For wrong heads:
//! - Minimize confidence (penalize false positives)
//! - Ignore normal predictions

use super::gru::{Rng, HIDDEN_DIM};
use super::math::{dot, matvec, vec_add};

/// Number of outputs for each head.
pub const FACE_HEAD_OUTPUTS: usize = 4;  // confidence + 1 normal
pub const EDGE_HEAD_OUTPUTS: usize = 10; // confidence + 2 normals + direction
pub const CORNER_HEAD_OUTPUTS: usize = 10; // confidence + 3 normals

/// Weights for a single classification head (linear layer).
#[derive(Clone)]
pub struct HeadWeights {
    /// Weight matrix [output_dim x hidden_dim].
    pub w: Vec<f64>,
    /// Bias vector [output_dim].
    pub b: Vec<f64>,
    /// Output dimension.
    pub output_dim: usize,
}

impl HeadWeights {
    /// Create new head weights with small random initialization.
    pub fn new(output_dim: usize, rng: &mut Rng) -> Self {
        let scale = 0.1;
        let w: Vec<f64> = (0..output_dim * HIDDEN_DIM)
            .map(|_| rng.next_range(scale))
            .collect();
        let b = vec![0.0; output_dim];

        Self { w, b, output_dim }
    }

    /// Forward pass: hidden -> outputs.
    pub fn forward(&self, hidden: &[f64]) -> Vec<f64> {
        debug_assert_eq!(hidden.len(), HIDDEN_DIM);
        let raw = matvec(&self.w, hidden, self.output_dim, HIDDEN_DIM);
        vec_add(&raw, &self.b)
    }

    /// Parameter count.
    pub fn param_count(&self) -> usize {
        self.w.len() + self.b.len()
    }
}

/// All three classification heads.
#[derive(Clone)]
pub struct ClassifierHeads {
    pub face: HeadWeights,
    pub edge: HeadWeights,
    pub corner: HeadWeights,
}

impl ClassifierHeads {
    /// Create new classifier heads with random initialization.
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            face: HeadWeights::new(FACE_HEAD_OUTPUTS, rng),
            edge: HeadWeights::new(EDGE_HEAD_OUTPUTS, rng),
            corner: HeadWeights::new(CORNER_HEAD_OUTPUTS, rng),
        }
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.face.param_count() + self.edge.param_count() + self.corner.param_count()
    }

    /// Run all three heads and return predictions.
    pub fn forward(&self, hidden: &[f64]) -> ClassifierPredictions {
        let face_raw = self.face.forward(hidden);
        let edge_raw = self.edge.forward(hidden);
        let corner_raw = self.corner.forward(hidden);

        ClassifierPredictions {
            face: FacePrediction::from_raw(&face_raw),
            edge: EdgePrediction::from_raw(&edge_raw),
            corner: CornerPrediction::from_raw(&corner_raw),
        }
    }
}

/// Predictions from all three heads.
#[derive(Clone, Debug)]
pub struct ClassifierPredictions {
    pub face: FacePrediction,
    pub edge: EdgePrediction,
    pub corner: CornerPrediction,
}

impl ClassifierPredictions {
    /// Get confidences as array [face, edge, corner].
    pub fn confidences(&self) -> [f64; 3] {
        [self.face.confidence, self.edge.confidence, self.corner.confidence]
    }

    /// Get the predicted geometry type (highest confidence).
    pub fn predicted_type(&self) -> GeometryType {
        let confs = self.confidences();
        if confs[0] >= confs[1] && confs[0] >= confs[2] {
            GeometryType::Face
        } else if confs[1] >= confs[2] {
            GeometryType::Edge
        } else {
            GeometryType::Corner
        }
    }
}

/// Geometry type enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryType {
    Face,
    Edge,
    Corner,
}

/// Face head prediction.
#[derive(Clone, Debug)]
pub struct FacePrediction {
    /// Confidence (0-1 after sigmoid).
    pub confidence: f64,
    /// Predicted normal (normalized).
    pub normal: (f64, f64, f64),
    /// Raw outputs before activation (for gradient computation).
    pub raw_confidence: f64,
    pub raw_normal: (f64, f64, f64),
}

impl FacePrediction {
    fn from_raw(raw: &[f64]) -> Self {
        debug_assert_eq!(raw.len(), FACE_HEAD_OUTPUTS);

        let raw_confidence = raw[0];
        let confidence = sigmoid(raw_confidence);

        let raw_normal = (raw[1], raw[2], raw[3]);
        let normal = normalize_vec(raw_normal);

        Self {
            confidence,
            normal,
            raw_confidence,
            raw_normal,
        }
    }
}

/// Edge head prediction.
#[derive(Clone, Debug)]
pub struct EdgePrediction {
    /// Confidence (0-1 after sigmoid).
    pub confidence: f64,
    /// First face normal (normalized).
    pub normal_a: (f64, f64, f64),
    /// Second face normal (normalized).
    pub normal_b: (f64, f64, f64),
    /// Edge direction (normalized).
    pub direction: (f64, f64, f64),
    /// Raw outputs.
    pub raw_confidence: f64,
    pub raw_normal_a: (f64, f64, f64),
    pub raw_normal_b: (f64, f64, f64),
    pub raw_direction: (f64, f64, f64),
}

impl EdgePrediction {
    fn from_raw(raw: &[f64]) -> Self {
        debug_assert_eq!(raw.len(), EDGE_HEAD_OUTPUTS);

        let raw_confidence = raw[0];
        let confidence = sigmoid(raw_confidence);

        let raw_normal_a = (raw[1], raw[2], raw[3]);
        let raw_normal_b = (raw[4], raw[5], raw[6]);
        let raw_direction = (raw[7], raw[8], raw[9]);

        Self {
            confidence,
            normal_a: normalize_vec(raw_normal_a),
            normal_b: normalize_vec(raw_normal_b),
            direction: normalize_vec(raw_direction),
            raw_confidence,
            raw_normal_a,
            raw_normal_b,
            raw_direction,
        }
    }
}

/// Corner head prediction.
#[derive(Clone, Debug)]
pub struct CornerPrediction {
    /// Confidence (0-1 after sigmoid).
    pub confidence: f64,
    /// Three face normals (normalized).
    pub normals: [(f64, f64, f64); 3],
    /// Raw outputs.
    pub raw_confidence: f64,
    pub raw_normals: [(f64, f64, f64); 3],
}

impl CornerPrediction {
    fn from_raw(raw: &[f64]) -> Self {
        debug_assert_eq!(raw.len(), CORNER_HEAD_OUTPUTS);

        let raw_confidence = raw[0];
        let confidence = sigmoid(raw_confidence);

        let raw_normals = [
            (raw[1], raw[2], raw[3]),
            (raw[4], raw[5], raw[6]),
            (raw[7], raw[8], raw[9]),
        ];

        Self {
            confidence,
            normals: [
                normalize_vec(raw_normals[0]),
                normalize_vec(raw_normals[1]),
                normalize_vec(raw_normals[2]),
            ],
            raw_confidence,
            raw_normals,
        }
    }
}

/// Expected classification from oracle (for loss computation).
#[derive(Clone, Debug)]
pub enum ExpectedGeometry {
    Face {
        normal: (f64, f64, f64),
    },
    Edge {
        normal_a: (f64, f64, f64),
        normal_b: (f64, f64, f64),
        direction: (f64, f64, f64),
    },
    Corner {
        normals: [(f64, f64, f64); 3],
    },
}

impl ExpectedGeometry {
    pub fn geometry_type(&self) -> GeometryType {
        match self {
            ExpectedGeometry::Face { .. } => GeometryType::Face,
            ExpectedGeometry::Edge { .. } => GeometryType::Edge,
            ExpectedGeometry::Corner { .. } => GeometryType::Corner,
        }
    }
}

/// Loss configuration.
#[derive(Clone, Debug)]
pub struct ClassifierLossConfig {
    /// Weight for correct head confidence loss.
    pub w_correct_conf: f64,
    /// Weight for correct head normal accuracy loss.
    pub w_normal: f64,
    /// Weight for penalizing wrong head confidences.
    pub w_wrong_conf: f64,
    /// Per-class weight multipliers to handle class imbalance.
    /// Order: [face, edge, corner]
    pub class_weights: [f64; 3],
}

impl Default for ClassifierLossConfig {
    fn default() -> Self {
        Self {
            w_correct_conf: 1.0,
            w_normal: 5.0,
            w_wrong_conf: 2.0, // Increased from 0.5 to penalize wrong predictions more
            // Balance class weights inversely to frequency
            // Face: 12/52 = 0.23, Edge: 24/52 = 0.46, Corner: 16/52 = 0.31
            // Inverse weights normalized: Face: 2.0, Edge: 1.0, Corner: 1.5
            class_weights: [2.0, 1.0, 1.5],
        }
    }
}

/// Compute classification loss.
///
/// Returns (total_loss, correct_conf_loss, normal_loss, wrong_conf_loss).
pub fn compute_classifier_loss(
    predictions: &ClassifierPredictions,
    expected: &ExpectedGeometry,
    config: &ClassifierLossConfig,
) -> (f64, f64, f64, f64) {
    let correct_type = expected.geometry_type();

    // Get class weight for the correct type
    let class_weight = match correct_type {
        GeometryType::Face => config.class_weights[0],
        GeometryType::Edge => config.class_weights[1],
        GeometryType::Corner => config.class_weights[2],
    };

    // Correct head confidence loss: -log(confidence)
    let correct_conf = match correct_type {
        GeometryType::Face => predictions.face.confidence,
        GeometryType::Edge => predictions.edge.confidence,
        GeometryType::Corner => predictions.corner.confidence,
    };
    let correct_conf_loss = -correct_conf.max(1e-10).ln();

    // Normal loss for correct head (cosine similarity: 1 - |dot|)
    let normal_loss = match expected {
        ExpectedGeometry::Face { normal } => {
            normal_cosine_loss(predictions.face.normal, *normal)
        }
        ExpectedGeometry::Edge { normal_a, normal_b, direction } => {
            // Best matching of predicted normals to expected
            let (loss_a, loss_b) = best_normal_pair_loss(
                (predictions.edge.normal_a, predictions.edge.normal_b),
                (*normal_a, *normal_b),
            );
            let dir_loss = normal_cosine_loss(predictions.edge.direction, *direction);
            (loss_a + loss_b + dir_loss) / 3.0
        }
        ExpectedGeometry::Corner { normals } => {
            // Greedy matching of predicted normals to expected
            let losses = best_corner_normal_losses(&predictions.corner.normals, normals);
            losses.iter().sum::<f64>() / 3.0
        }
    };

    // Wrong head confidence penalty: log(1 - confidence) for wrong heads
    // We want wrong heads to have LOW confidence, so penalize high confidence
    let wrong_conf_loss = match correct_type {
        GeometryType::Face => {
            -((1.0 - predictions.edge.confidence).max(1e-10).ln()
                + (1.0 - predictions.corner.confidence).max(1e-10).ln()) / 2.0
        }
        GeometryType::Edge => {
            -((1.0 - predictions.face.confidence).max(1e-10).ln()
                + (1.0 - predictions.corner.confidence).max(1e-10).ln()) / 2.0
        }
        GeometryType::Corner => {
            -((1.0 - predictions.face.confidence).max(1e-10).ln()
                + (1.0 - predictions.edge.confidence).max(1e-10).ln()) / 2.0
        }
    };

    // Apply class weight to balance minority classes
    let total = class_weight * (
        config.w_correct_conf * correct_conf_loss
        + config.w_normal * normal_loss
        + config.w_wrong_conf * wrong_conf_loss
    );

    (total, correct_conf_loss, normal_loss, wrong_conf_loss)
}

/// Cosine similarity loss for normals: 1 - |dot(a, b)|.
/// Handles sign ambiguity (normal can point either direction).
fn normal_cosine_loss(predicted: (f64, f64, f64), expected: (f64, f64, f64)) -> f64 {
    let d = dot(predicted, expected).abs();
    1.0 - d.min(1.0)
}

/// Find best assignment of 2 predicted normals to 2 expected normals.
fn best_normal_pair_loss(
    predicted: ((f64, f64, f64), (f64, f64, f64)),
    expected: ((f64, f64, f64), (f64, f64, f64)),
) -> (f64, f64) {
    // Assignment 1: pred.0 -> exp.0, pred.1 -> exp.1
    let loss1_a = normal_cosine_loss(predicted.0, expected.0);
    let loss1_b = normal_cosine_loss(predicted.1, expected.1);

    // Assignment 2: pred.0 -> exp.1, pred.1 -> exp.0
    let loss2_a = normal_cosine_loss(predicted.0, expected.1);
    let loss2_b = normal_cosine_loss(predicted.1, expected.0);

    if loss1_a + loss1_b < loss2_a + loss2_b {
        (loss1_a, loss1_b)
    } else {
        (loss2_a, loss2_b)
    }
}

/// Find best greedy assignment of 3 predicted normals to 3 expected normals.
fn best_corner_normal_losses(
    predicted: &[(f64, f64, f64); 3],
    expected: &[(f64, f64, f64); 3],
) -> [f64; 3] {
    let mut losses = [0.0; 3];
    let mut used = [false; 3];

    for pred in predicted {
        let mut best_loss = f64::MAX;
        let mut best_idx = 0;

        for (i, exp) in expected.iter().enumerate() {
            if !used[i] {
                let loss = normal_cosine_loss(*pred, *exp);
                if loss < best_loss {
                    best_loss = loss;
                    best_idx = i;
                }
            }
        }

        used[best_idx] = true;
        losses[best_idx] = best_loss;
    }

    losses
}

// Helper functions

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn normalize_vec(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    if len > 1e-10 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 1.0) // Default direction
    }
}

/// Gradients for classifier heads.
#[derive(Clone)]
pub struct ClassifierHeadGradients {
    pub face_dw: Vec<f64>,
    pub face_db: Vec<f64>,
    pub edge_dw: Vec<f64>,
    pub edge_db: Vec<f64>,
    pub corner_dw: Vec<f64>,
    pub corner_db: Vec<f64>,
}

impl ClassifierHeadGradients {
    /// Create zero gradients.
    pub fn zeros() -> Self {
        Self {
            face_dw: vec![0.0; FACE_HEAD_OUTPUTS * HIDDEN_DIM],
            face_db: vec![0.0; FACE_HEAD_OUTPUTS],
            edge_dw: vec![0.0; EDGE_HEAD_OUTPUTS * HIDDEN_DIM],
            edge_db: vec![0.0; EDGE_HEAD_OUTPUTS],
            corner_dw: vec![0.0; CORNER_HEAD_OUTPUTS * HIDDEN_DIM],
            corner_db: vec![0.0; CORNER_HEAD_OUTPUTS],
        }
    }

    /// Add another gradient to this one.
    pub fn add(&mut self, other: &ClassifierHeadGradients) {
        for (a, b) in self.face_dw.iter_mut().zip(other.face_dw.iter()) {
            *a += b;
        }
        for (a, b) in self.face_db.iter_mut().zip(other.face_db.iter()) {
            *a += b;
        }
        for (a, b) in self.edge_dw.iter_mut().zip(other.edge_dw.iter()) {
            *a += b;
        }
        for (a, b) in self.edge_db.iter_mut().zip(other.edge_db.iter()) {
            *a += b;
        }
        for (a, b) in self.corner_dw.iter_mut().zip(other.corner_dw.iter()) {
            *a += b;
        }
        for (a, b) in self.corner_db.iter_mut().zip(other.corner_db.iter()) {
            *a += b;
        }
    }

    /// Scale gradients by a factor.
    pub fn scale(&mut self, factor: f64) {
        for v in self.face_dw.iter_mut() { *v *= factor; }
        for v in self.face_db.iter_mut() { *v *= factor; }
        for v in self.edge_dw.iter_mut() { *v *= factor; }
        for v in self.edge_db.iter_mut() { *v *= factor; }
        for v in self.corner_dw.iter_mut() { *v *= factor; }
        for v in self.corner_db.iter_mut() { *v *= factor; }
    }

    /// Clip gradient magnitudes.
    pub fn clip(&mut self, max_norm: f64) {
        clip_vec(&mut self.face_dw, max_norm);
        clip_vec(&mut self.face_db, max_norm);
        clip_vec(&mut self.edge_dw, max_norm);
        clip_vec(&mut self.edge_db, max_norm);
        clip_vec(&mut self.corner_dw, max_norm);
        clip_vec(&mut self.corner_db, max_norm);
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

/// Compute gradients for classifier heads using supervised loss.
///
/// Returns (head_gradients, dL/d_hidden) for backprop through the network.
pub fn compute_classifier_gradients(
    heads: &ClassifierHeads,
    hidden: &[f64],
    predictions: &ClassifierPredictions,
    expected: &ExpectedGeometry,
    config: &ClassifierLossConfig,
) -> (ClassifierHeadGradients, Vec<f64>) {
    let mut grads = ClassifierHeadGradients::zeros();
    let mut dh = vec![0.0; HIDDEN_DIM];

    let correct_type = expected.geometry_type();

    // Compute gradients for each head
    // For the correct head: gradient from confidence loss + normal loss
    // For wrong heads: gradient from wrong confidence penalty

    // Face head gradients
    let face_grad = compute_face_head_gradient(
        &heads.face,
        hidden,
        predictions,
        expected,
        correct_type == GeometryType::Face,
        config,
    );
    grads.face_dw = face_grad.dw;
    grads.face_db = face_grad.db;
    for (i, dh_i) in face_grad.dh.into_iter().enumerate() {
        dh[i] += dh_i;
    }

    // Edge head gradients
    let edge_grad = compute_edge_head_gradient(
        &heads.edge,
        hidden,
        predictions,
        expected,
        correct_type == GeometryType::Edge,
        config,
    );
    grads.edge_dw = edge_grad.dw;
    grads.edge_db = edge_grad.db;
    for (i, dh_i) in edge_grad.dh.into_iter().enumerate() {
        dh[i] += dh_i;
    }

    // Corner head gradients
    let corner_grad = compute_corner_head_gradient(
        &heads.corner,
        hidden,
        predictions,
        expected,
        correct_type == GeometryType::Corner,
        config,
    );
    grads.corner_dw = corner_grad.dw;
    grads.corner_db = corner_grad.db;
    for (i, dh_i) in corner_grad.dh.into_iter().enumerate() {
        dh[i] += dh_i;
    }

    (grads, dh)
}

/// Per-head gradient result.
struct HeadGradResult {
    dw: Vec<f64>,
    db: Vec<f64>,
    dh: Vec<f64>,
}

/// Compute gradients for face head.
fn compute_face_head_gradient(
    head: &HeadWeights,
    hidden: &[f64],
    predictions: &ClassifierPredictions,
    expected: &ExpectedGeometry,
    is_correct: bool,
    config: &ClassifierLossConfig,
) -> HeadGradResult {
    let pred = &predictions.face;
    let mut d_raw = vec![0.0; FACE_HEAD_OUTPUTS];

    if is_correct {
        // Correct head: gradient from -log(conf) => d/d_raw_conf = -sigmoid'(x)/sigmoid(x) = -(1-sigmoid(x))
        // Since loss = -log(sigmoid(x)), dL/dx = sigmoid(x) - 1 = conf - 1
        d_raw[0] = config.w_correct_conf * (pred.confidence - 1.0);

        // Normal loss gradient (for correct head only)
        if let ExpectedGeometry::Face { normal } = expected {
            let d_normal = normal_loss_gradient(pred.raw_normal, pred.normal, *normal);
            d_raw[1] += config.w_normal * d_normal.0;
            d_raw[2] += config.w_normal * d_normal.1;
            d_raw[3] += config.w_normal * d_normal.2;
        }
    } else {
        // Wrong head: gradient from -log(1-conf) => d/d_raw_conf = sigmoid(x)/(1-sigmoid(x)) * sigmoid'(x)
        // = sigmoid(x) = conf
        d_raw[0] = config.w_wrong_conf * pred.confidence * 0.5; // /2 for averaging two wrong heads
    }

    linear_backward(head, hidden, &d_raw)
}

/// Compute gradients for edge head.
fn compute_edge_head_gradient(
    head: &HeadWeights,
    hidden: &[f64],
    predictions: &ClassifierPredictions,
    expected: &ExpectedGeometry,
    is_correct: bool,
    config: &ClassifierLossConfig,
) -> HeadGradResult {
    let pred = &predictions.edge;
    let mut d_raw = vec![0.0; EDGE_HEAD_OUTPUTS];

    if is_correct {
        d_raw[0] = config.w_correct_conf * (pred.confidence - 1.0);

        if let ExpectedGeometry::Edge { normal_a, normal_b, direction } = expected {
            // Find best assignment for normals
            let loss1 = normal_cosine_loss(pred.normal_a, *normal_a)
                + normal_cosine_loss(pred.normal_b, *normal_b);
            let loss2 = normal_cosine_loss(pred.normal_a, *normal_b)
                + normal_cosine_loss(pred.normal_b, *normal_a);

            let (exp_a, exp_b) = if loss1 <= loss2 {
                (*normal_a, *normal_b)
            } else {
                (*normal_b, *normal_a)
            };

            let d_normal_a = normal_loss_gradient(pred.raw_normal_a, pred.normal_a, exp_a);
            let d_normal_b = normal_loss_gradient(pred.raw_normal_b, pred.normal_b, exp_b);
            let d_dir = normal_loss_gradient(pred.raw_direction, pred.direction, *direction);

            let scale = config.w_normal / 3.0;
            d_raw[1] += scale * d_normal_a.0;
            d_raw[2] += scale * d_normal_a.1;
            d_raw[3] += scale * d_normal_a.2;
            d_raw[4] += scale * d_normal_b.0;
            d_raw[5] += scale * d_normal_b.1;
            d_raw[6] += scale * d_normal_b.2;
            d_raw[7] += scale * d_dir.0;
            d_raw[8] += scale * d_dir.1;
            d_raw[9] += scale * d_dir.2;
        }
    } else {
        d_raw[0] = config.w_wrong_conf * pred.confidence * 0.5;
    }

    linear_backward(head, hidden, &d_raw)
}

/// Compute gradients for corner head.
fn compute_corner_head_gradient(
    head: &HeadWeights,
    hidden: &[f64],
    predictions: &ClassifierPredictions,
    expected: &ExpectedGeometry,
    is_correct: bool,
    config: &ClassifierLossConfig,
) -> HeadGradResult {
    let pred = &predictions.corner;
    let mut d_raw = vec![0.0; CORNER_HEAD_OUTPUTS];

    if is_correct {
        d_raw[0] = config.w_correct_conf * (pred.confidence - 1.0);

        if let ExpectedGeometry::Corner { normals: exp_normals } = expected {
            // Greedy matching (same as loss computation)
            let mut used = [false; 3];
            let mut assignments = [(0usize, 0usize); 3];

            for (pred_idx, pred_n) in pred.normals.iter().enumerate() {
                let mut best_loss = f64::MAX;
                let mut best_exp_idx = 0;

                for (exp_idx, exp_n) in exp_normals.iter().enumerate() {
                    if !used[exp_idx] {
                        let loss = normal_cosine_loss(*pred_n, *exp_n);
                        if loss < best_loss {
                            best_loss = loss;
                            best_exp_idx = exp_idx;
                        }
                    }
                }
                used[best_exp_idx] = true;
                assignments[pred_idx] = (pred_idx, best_exp_idx);
            }

            let scale = config.w_normal / 3.0;
            for (pred_idx, exp_idx) in assignments {
                let d_n = normal_loss_gradient(
                    pred.raw_normals[pred_idx],
                    pred.normals[pred_idx],
                    exp_normals[exp_idx],
                );
                let base = 1 + pred_idx * 3;
                d_raw[base] += scale * d_n.0;
                d_raw[base + 1] += scale * d_n.1;
                d_raw[base + 2] += scale * d_n.2;
            }
        }
    } else {
        d_raw[0] = config.w_wrong_conf * pred.confidence * 0.5;
    }

    linear_backward(head, hidden, &d_raw)
}

/// Backward pass through a linear layer: y = Wx + b.
/// Given dL/dy, compute dL/dW, dL/db, dL/dx (dL/d_hidden).
fn linear_backward(head: &HeadWeights, hidden: &[f64], d_output: &[f64]) -> HeadGradResult {
    let out_dim = head.output_dim;
    let in_dim = HIDDEN_DIM;

    // dL/db = dL/dy
    let db = d_output.to_vec();

    // dL/dW[i,j] = dL/dy[i] * x[j]
    let mut dw = vec![0.0; out_dim * in_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            dw[i * in_dim + j] = d_output[i] * hidden[j];
        }
    }

    // dL/dx[j] = sum_i(dL/dy[i] * W[i,j])
    let mut dh = vec![0.0; in_dim];
    for j in 0..in_dim {
        for i in 0..out_dim {
            dh[j] += d_output[i] * head.w[i * in_dim + j];
        }
    }

    HeadGradResult { dw, db, dh }
}

/// Gradient of normal cosine loss w.r.t. raw (pre-normalized) vector.
///
/// Loss = 1 - |dot(normalize(raw), expected)|
fn normal_loss_gradient(
    raw: (f64, f64, f64),
    normalized: (f64, f64, f64),
    expected: (f64, f64, f64),
) -> (f64, f64, f64) {
    let len_sq = raw.0 * raw.0 + raw.1 * raw.1 + raw.2 * raw.2;
    let len = len_sq.sqrt();

    if len < 1e-10 {
        return (0.0, 0.0, 0.0);
    }

    // d(normalize(v))/dv = (I - n*n^T) / |v|
    // where n = normalized vector
    let d = dot(normalized, expected);
    let sign = if d >= 0.0 { -1.0 } else { 1.0 }; // Loss = 1 - |d|, so dL/d(|d|) = -1, d|d|/dd = sign(d)

    // dL/d_raw = sign * d(dot(norm, exp))/d_raw
    //          = sign * exp^T * d(norm)/d_raw
    //          = sign * (exp - (exp.norm)*norm) / len

    let exp_dot_norm = dot(expected, normalized);
    let dx = sign * (expected.0 - exp_dot_norm * normalized.0) / len;
    let dy = sign * (expected.1 - exp_dot_norm * normalized.1) / len;
    let dz = sign * (expected.2 - exp_dot_norm * normalized.2) / len;

    (dx, dy, dz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_head_forward_shapes() {
        let mut rng = Rng::new(42);
        let heads = ClassifierHeads::new(&mut rng);

        let hidden = vec![0.1; HIDDEN_DIM];
        let preds = heads.forward(&hidden);

        // Check confidences are in valid range
        assert!(preds.face.confidence >= 0.0 && preds.face.confidence <= 1.0);
        assert!(preds.edge.confidence >= 0.0 && preds.edge.confidence <= 1.0);
        assert!(preds.corner.confidence >= 0.0 && preds.corner.confidence <= 1.0);

        // Check normals are normalized
        let face_len = vec_len(preds.face.normal);
        assert!((face_len - 1.0).abs() < 1e-6, "Face normal not unit: {}", face_len);
    }

    #[test]
    fn test_cosine_loss() {
        // Same direction: loss = 0
        let loss1 = normal_cosine_loss((1.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        assert!(loss1 < 1e-6, "Same direction should have zero loss");

        // Opposite direction: loss = 0 (sign ambiguity)
        let loss2 = normal_cosine_loss((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0));
        assert!(loss2 < 1e-6, "Opposite direction should have zero loss");

        // Perpendicular: loss = 1
        let loss3 = normal_cosine_loss((1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
        assert!((loss3 - 1.0).abs() < 1e-6, "Perpendicular should have loss 1");
    }

    #[test]
    fn test_classifier_loss() {
        let mut rng = Rng::new(42);
        let heads = ClassifierHeads::new(&mut rng);
        let hidden = vec![0.1; HIDDEN_DIM];
        let preds = heads.forward(&hidden);

        let expected = ExpectedGeometry::Face {
            normal: (0.0, 0.0, 1.0),
        };

        let config = ClassifierLossConfig::default();
        let (total, conf_loss, normal_loss, wrong_loss) =
            compute_classifier_loss(&preds, &expected, &config);

        // All losses should be non-negative
        assert!(total >= 0.0);
        assert!(conf_loss >= 0.0);
        assert!(normal_loss >= 0.0);
        assert!(wrong_loss >= 0.0);
    }

    fn vec_len(v: (f64, f64, f64)) -> f64 {
        (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
    }
}
