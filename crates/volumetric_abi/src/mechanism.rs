//! Articulated assemblies: the `Mechanism` and `Assembly` values.
//!
//! A [`Mechanism`] is a tree of rigid parts joined by fixed, revolute or
//! prismatic joints. Parts are authored in the world frame at the rest
//! state, as they are measured and built, and every joint's axis is given
//! in world coordinates at rest; the pose of a part is the product of its
//! chain's joint motions, root first, so no local frames are ever
//! authored and the rest state is the state as photographed. Design:
//! `ASSEMBLY_PLAN.md` at the repository root.
//!
//! The state is an [`F64Map`] keyed by joint name: degrees for a revolute
//! joint, metres for a prismatic one. A joint with a `drive` follows
//! another joint at a ratio and offset (a synchro-tilt) and is not a state.
//! Missing keys take the joint's default; unknown keys and out-of-range
//! values are errors.
//!
//! An [`Assembly`] is a mechanism together with its part models and one
//! state: explicit data like a view set, self-contained, never fed to the
//! model executor as a whole. Its part bytes are the unposed inputs, so a
//! part's mesh-cache key is the same at every state.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::annotations::ParameterSpec;
use crate::f64_map::F64Map;

/// The schema this module writes.
pub const MECHANISM_SCHEMA: u32 = 1;

/// The name of the fixed root every chain hangs from.
pub const WORLD: &str = "world";

/// What a joint does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointKind {
    /// Rigidly attached; no state.
    Fixed,
    /// Rotation about the axis, in degrees.
    Revolute,
    /// Translation along the axis, in metres.
    Prismatic,
}

impl JointKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Revolute => "revolute",
            Self::Prismatic => "prismatic",
        }
    }
}

/// A joint's axis: a point on it and its direction, both in world
/// coordinates at the rest state. The direction need not be unit length.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Axis {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
}

impl Axis {
    /// The axis with its direction normalised; `None` when the direction
    /// is zero or not finite.
    pub fn unit(&self) -> Option<Axis> {
        let d = self.direction;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !(len.is_finite() && len > 1e-12) || self.origin.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(Axis {
            origin: self.origin,
            direction: d.map(|c| c / len),
        })
    }
}

/// A coupling: the joint's value is `ratio * value(joint) + offset`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Drive {
    pub joint: String,
    pub ratio: f64,
    #[serde(default)]
    pub offset: f64,
}

/// One joint of the tree: it attaches `child` to `parent` (a part name or
/// [`WORLD`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    pub name: String,
    pub kind: JointKind,
    pub parent: String,
    pub child: String,
    /// Required for revolute and prismatic joints; ignored by a fixed one.
    #[serde(default)]
    pub axis: Option<Axis>,
    #[serde(default)]
    pub min: f64,
    #[serde(default)]
    pub max: f64,
    #[serde(default)]
    pub default: f64,
    #[serde(default)]
    pub drive: Option<Drive>,
}

impl Joint {
    /// Whether the joint's value is a state of the mechanism (moving and
    /// not driven by another joint).
    pub fn is_state(&self) -> bool {
        self.kind != JointKind::Fixed && self.drive.is_none()
    }
}

/// A tree of parts joined by joints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mechanism {
    pub schema: u32,
    /// Part names, in the order an assembly's part models are given.
    pub parts: Vec<String>,
    pub joints: Vec<Joint>,
}

/// A rigid (or affine) map of 3-space: `p' = linear * p + offset`, row
/// major. The same shape as `model_wrap_core::Affine`, kept here so the
/// ABI stays free of the wasm rewriting crates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid {
    pub linear: [[f64; 3]; 3],
    pub offset: [f64; 3],
}

impl Rigid {
    pub const IDENTITY: Rigid = Rigid {
        linear: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        offset: [0.0; 3],
    };

    pub fn translation(offset: [f64; 3]) -> Rigid {
        Rigid {
            linear: Rigid::IDENTITY.linear,
            offset,
        }
    }

    /// Rotation by `radians` about the unit `axis` (Rodrigues), so that
    /// the axis's points stay put.
    pub fn rotation_about(axis: &Axis, radians: f64) -> Rigid {
        let [x, y, z] = axis.direction;
        let (s, c) = radians.sin_cos();
        let t = 1.0 - c;
        let linear = [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ];
        let o = axis.origin;
        let ro = mul_vec(&linear, o);
        Rigid {
            linear,
            offset: [o[0] - ro[0], o[1] - ro[1], o[2] - ro[2]],
        }
    }

    /// `self` followed by `next`: `next(self(p))`.
    pub fn then(&self, next: &Rigid) -> Rigid {
        let mut linear = [[0.0; 3]; 3];
        for (i, row) in linear.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| next.linear[i][k] * self.linear[k][j]).sum();
            }
        }
        let o = mul_vec(&next.linear, self.offset);
        Rigid {
            linear,
            offset: [
                o[0] + next.offset[0],
                o[1] + next.offset[1],
                o[2] + next.offset[2],
            ],
        }
    }

    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let q = mul_vec(&self.linear, p);
        [
            q[0] + self.offset[0],
            q[1] + self.offset[1],
            q[2] + self.offset[2],
        ]
    }

    /// The linear part applied to a direction.
    pub fn apply_vector(&self, v: [f64; 3]) -> [f64; 3] {
        mul_vec(&self.linear, v)
    }

    /// Column-major 4x4 (the layout `glam::Mat4::from_cols_array` reads).
    pub fn to_cols_array_f32(&self) -> [f32; 16] {
        let l = &self.linear;
        let o = &self.offset;
        [
            l[0][0] as f32,
            l[1][0] as f32,
            l[2][0] as f32,
            0.0,
            l[0][1] as f32,
            l[1][1] as f32,
            l[2][1] as f32,
            0.0,
            l[0][2] as f32,
            l[1][2] as f32,
            l[2][2] as f32,
            0.0,
            o[0] as f32,
            o[1] as f32,
            o[2] as f32,
            1.0,
        ]
    }
}

fn mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Mechanism {
    pub fn new(parts: Vec<String>, joints: Vec<Joint>) -> Self {
        Self {
            schema: MECHANISM_SCHEMA,
            parts,
            joints,
        }
    }

    /// Structural validation: unique names, a tree rooted at the world
    /// reaching every part once, usable axes, ranges containing defaults,
    /// drives that follow a moving, undriven joint.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MECHANISM_SCHEMA {
            return Err(format!(
                "mechanism schema {} is not {MECHANISM_SCHEMA}",
                self.schema
            ));
        }
        if self.parts.is_empty() {
            return Err("a mechanism needs at least one part".to_string());
        }
        let mut part_names = BTreeSet::new();
        for part in &self.parts {
            if part.is_empty() || part == WORLD {
                return Err(format!("`{part}` is not a usable part name"));
            }
            if !part_names.insert(part.as_str()) {
                return Err(format!("part `{part}` is listed twice"));
            }
        }
        let mut joint_names = BTreeSet::new();
        let mut attached: BTreeMap<&str, &Joint> = BTreeMap::new();
        for joint in &self.joints {
            if joint.name.is_empty() {
                return Err("a joint has an empty name".to_string());
            }
            if !joint_names.insert(joint.name.as_str()) {
                return Err(format!("joint `{}` is listed twice", joint.name));
            }
            if joint.parent != WORLD && !part_names.contains(joint.parent.as_str()) {
                return Err(format!(
                    "joint `{}`: parent `{}` is neither a part nor `{WORLD}`",
                    joint.name, joint.parent
                ));
            }
            if !part_names.contains(joint.child.as_str()) {
                return Err(format!(
                    "joint `{}`: child `{}` is not a part",
                    joint.name, joint.child
                ));
            }
            if joint.parent == joint.child {
                return Err(format!("joint `{}` attaches a part to itself", joint.name));
            }
            if attached.insert(joint.child.as_str(), joint).is_some() {
                return Err(format!(
                    "part `{}` is attached by more than one joint",
                    joint.child
                ));
            }
            if joint.kind != JointKind::Fixed {
                let axis = joint.axis.ok_or_else(|| {
                    format!(
                        "joint `{}` is {} and needs an axis",
                        joint.name,
                        joint.kind.name()
                    )
                })?;
                if axis.unit().is_none() {
                    return Err(format!(
                        "joint `{}`: axis direction {:?} at {:?} is degenerate",
                        joint.name, axis.direction, axis.origin
                    ));
                }
                for (what, value) in [
                    ("min", joint.min),
                    ("max", joint.max),
                    ("default", joint.default),
                ] {
                    if !value.is_finite() {
                        return Err(format!("joint `{}`: {what} must be finite", joint.name));
                    }
                }
                if !(joint.min <= joint.default && joint.default <= joint.max) {
                    return Err(format!(
                        "joint `{}`: default {} is outside [{}, {}]",
                        joint.name, joint.default, joint.min, joint.max
                    ));
                }
            }
            if let Some(drive) = &joint.drive {
                if joint.kind == JointKind::Fixed {
                    return Err(format!(
                        "joint `{}` is fixed and cannot be driven",
                        joint.name
                    ));
                }
                if !(drive.ratio.is_finite() && drive.offset.is_finite()) {
                    return Err(format!(
                        "joint `{}`: drive ratio and offset must be finite",
                        joint.name
                    ));
                }
                let Some(target) = self.joints.iter().find(|j| j.name == drive.joint) else {
                    return Err(format!(
                        "joint `{}` is driven by `{}`, which is not a joint",
                        joint.name, drive.joint
                    ));
                };
                if !target.is_state() {
                    return Err(format!(
                        "joint `{}` is driven by `{}`, which is not a state (fixed or driven itself)",
                        joint.name, drive.joint
                    ));
                }
            }
        }
        for part in &self.parts {
            // Walk to the world; a cycle or a loose part fails within the
            // part count.
            let mut at = part.as_str();
            for _ in 0..=self.parts.len() {
                let Some(joint) = attached.get(at) else {
                    return Err(format!(
                        "part `{at}` hangs from nothing (no joint attaches it)"
                    ));
                };
                if joint.parent == WORLD {
                    break;
                }
                at = joint.parent.as_str();
                if at == part {
                    return Err(format!("part `{part}` is attached in a cycle"));
                }
            }
        }
        Ok(())
    }

    /// The joint attaching `part`, or `None` for an unknown part.
    pub fn joint_of(&self, part: &str) -> Option<&Joint> {
        self.joints.iter().find(|j| j.child == part)
    }

    /// The joints on `part`'s chain, root first (indices into `joints`).
    pub fn chain(&self, part: &str) -> Vec<usize> {
        let mut out = Vec::new();
        let mut at = part;
        while let Some(idx) = self.joints.iter().position(|j| j.child == at) {
            out.push(idx);
            at = self.joints[idx].parent.as_str();
            if at == WORLD || out.len() > self.joints.len() {
                break;
            }
        }
        out.reverse();
        out
    }

    /// The names of the mechanism's states: its moving, undriven joints in
    /// joint order. This is the order `velocity` reports in.
    pub fn state_keys(&self) -> Vec<&str> {
        self.joints
            .iter()
            .filter(|j| j.is_state())
            .map(|j| j.name.as_str())
            .collect()
    }

    /// The default state: every state key at its joint's default.
    pub fn default_state(&self) -> F64Map {
        self.joints
            .iter()
            .filter(|j| j.is_state())
            .map(|j| (j.name.clone(), j.default))
            .collect()
    }

    /// One [`ParameterSpec`] per state, for the host's inline F64Map form;
    /// `source_line` is the joint's one-based index.
    pub fn parameter_specs(&self) -> Vec<ParameterSpec> {
        self.joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.is_state())
            .map(|(i, j)| ParameterSpec {
                source_line: i + 1,
                binding_name: j.name.clone(),
                key: j.name.clone(),
                default: j.default,
                min: Some(j.min),
                max: Some(j.max),
            })
            .collect()
    }

    /// Every joint's value under `state`: states from the map (their
    /// default when absent, an error when out of range), driven joints
    /// from their drive, fixed joints 0. Unknown keys are an error.
    pub fn joint_values(&self, state: &F64Map) -> Result<Vec<f64>, String> {
        for key in state.keys() {
            match self.joints.iter().find(|j| j.name == *key) {
                Some(joint) if joint.is_state() => {}
                Some(_) => {
                    return Err(format!("`{key}` is not a state (fixed or driven joint)"));
                }
                None => return Err(format!("`{key}` is not a joint of the mechanism")),
            }
        }
        let own: Vec<f64> = self
            .joints
            .iter()
            .map(|joint| {
                if !joint.is_state() {
                    return Ok(0.0);
                }
                let value = state.get(&joint.name).copied().unwrap_or(joint.default);
                if !value.is_finite() {
                    return Err(format!("state `{}` must be finite", joint.name));
                }
                if value < joint.min || value > joint.max {
                    return Err(format!(
                        "state `{}` = {value} is outside [{}, {}]",
                        joint.name, joint.min, joint.max
                    ));
                }
                Ok(value)
            })
            .collect::<Result<_, _>>()?;
        Ok(self
            .joints
            .iter()
            .enumerate()
            .map(|(i, joint)| match &joint.drive {
                Some(drive) => {
                    let target = self
                        .joints
                        .iter()
                        .position(|j| j.name == drive.joint)
                        .expect("validated drive target");
                    drive.ratio * own[target] + drive.offset
                }
                None => own[i],
            })
            .collect())
    }

    /// The motion one joint contributes at `value` (degrees or metres).
    fn motion(joint: &Joint, value: f64) -> Rigid {
        match joint.kind {
            JointKind::Fixed => Rigid::IDENTITY,
            JointKind::Revolute => {
                let axis = joint.axis.and_then(|a| a.unit()).expect("validated axis");
                Rigid::rotation_about(&axis, value.to_radians())
            }
            JointKind::Prismatic => {
                let axis = joint.axis.and_then(|a| a.unit()).expect("validated axis");
                Rigid::translation(axis.direction.map(|c| c * value))
            }
        }
    }

    /// The pose of every part (world <- part, in `parts` order) under
    /// `state`: the product of each chain's joint motions, root first.
    pub fn pose(&self, state: &F64Map) -> Result<Vec<Rigid>, String> {
        let values = self.joint_values(state)?;
        Ok(self
            .parts
            .iter()
            .map(|part| {
                self.chain(part)
                    .into_iter()
                    .fold(Rigid::IDENTITY, |acc, idx| {
                        // acc is the parent's pose; the joint moves in the
                        // parent's frame, so its motion is applied first.
                        Self::motion(&self.joints[idx], values[idx]).then(&acc)
                    })
            })
            .collect())
    }

    /// The world velocity of the point `point` (given in world coordinates
    /// at `state`) on `part`, per unit rate of each state in
    /// [`state_keys`](Self::state_keys) order: for a revolute joint on the
    /// chain `ω × (x − o)` about its posed axis, per degree; for a
    /// prismatic one its posed direction; zero for joints off the chain.
    /// Driven joints add their ratio times the same to the joint they
    /// follow. This is the Jacobian a drag needs.
    pub fn velocity(
        &self,
        state: &F64Map,
        part: &str,
        point: [f64; 3],
    ) -> Result<Vec<[f64; 3]>, String> {
        if !self.parts.iter().any(|p| p == part) {
            return Err(format!("`{part}` is not a part of the mechanism"));
        }
        let values = self.joint_values(state)?;
        let keys = self.state_keys();
        let mut out = vec![[0.0; 3]; keys.len()];
        let mut parent_pose = Rigid::IDENTITY;
        for idx in self.chain(part) {
            let joint = &self.joints[idx];
            let (key, ratio) = match &joint.drive {
                Some(drive) => (drive.joint.as_str(), drive.ratio),
                None => (joint.name.as_str(), 1.0),
            };
            if let Some(slot) = keys.iter().position(|k| *k == key) {
                let axis = joint.axis.and_then(|a| a.unit()).expect("validated axis");
                let direction = parent_pose.apply_vector(axis.direction);
                let v = match joint.kind {
                    JointKind::Fixed => [0.0; 3],
                    JointKind::Revolute => {
                        let origin = parent_pose.apply(axis.origin);
                        let r = [
                            point[0] - origin[0],
                            point[1] - origin[1],
                            point[2] - origin[2],
                        ];
                        cross(direction, r).map(|c| c.to_radians())
                    }
                    JointKind::Prismatic => direction,
                };
                for (o, c) in out[slot].iter_mut().zip(v) {
                    *o += ratio * c;
                }
            }
            parent_pose = Self::motion(joint, values[idx]).then(&parent_pose);
        }
        Ok(out)
    }
}

impl Mechanism {
    /// Every state clamped into its joint's range.
    pub fn clamp_state(&self, state: &F64Map) -> F64Map {
        state
            .iter()
            .map(|(key, value)| {
                let clamped = match self.joints.iter().find(|j| j.name == *key) {
                    Some(joint) => value.clamp(joint.min, joint.max),
                    None => *value,
                };
                (key.clone(), clamped)
            })
            .collect()
    }

    /// The state that brings `local`, a point of `part` in the part's own
    /// (rest) coordinates, as near `target` (world) as the joints allow,
    /// starting from `state`: Levenberg-Marquardt on [`velocity`]
    /// (`velocity`) with per-joint damping, `iterations` steps at most,
    /// every step clamped to the ranges.
    /// This is what a drag solves each time the pointer moves.
    pub fn pull(
        &self,
        state: &F64Map,
        part: &str,
        local: [f64; 3],
        target: [f64; 3],
        iterations: usize,
    ) -> Result<F64Map, String> {
        let index = self
            .parts
            .iter()
            .position(|p| p == part)
            .ok_or_else(|| format!("`{part}` is not a part of the mechanism"))?;
        let keys: Vec<String> = self.state_keys().iter().map(|k| k.to_string()).collect();
        let mut state = self.clamp_state(state);
        for key in &keys {
            let default = self.joints.iter().find(|j| j.name == *key).unwrap().default;
            state.entry(key.clone()).or_insert(default);
        }
        for _ in 0..iterations {
            let x = self.pose(&state)?[index].apply(local);
            let r = [target[0] - x[0], target[1] - x[1], target[2] - x[2]];
            let miss = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            if miss < 1e-9 {
                break;
            }
            let j = self.velocity(&state, part, x)?;
            // Normal equations in the joints, (JᵀJ + Λ) Δq = Jᵀ r, with
            // Marquardt's diagonal damping Λ = μ diag(JᵀJ) so a joint in
            // degrees and one in metres are damped alike (a shared scalar
            // would crush whichever has the smaller unit).
            let n = keys.len();
            let mut a = vec![0.0; n * n];
            let mut b = vec![0.0; n];
            for (p, jp) in j.iter().enumerate() {
                b[p] = jp[0] * r[0] + jp[1] * r[1] + jp[2] * r[2];
                for (q, jq) in j.iter().enumerate() {
                    a[p * n + q] = jp[0] * jq[0] + jp[1] * jq[1] + jp[2] * jq[2];
                }
            }
            let largest = (0..n).map(|p| a[p * n + p]).fold(0.0, f64::max);
            if largest <= 0.0 {
                break;
            }
            for p in 0..n {
                a[p * n + p] += 1e-3 * a[p * n + p] + 1e-12 * largest;
            }
            let Some(dq) = solve_dense(a, b) else {
                break;
            };
            for (key, dq) in keys.iter().zip(dq) {
                if let Some(value) = state.get_mut(key) {
                    *value += dq;
                }
            }
            state = self.clamp_state(&state);
        }
        Ok(state)
    }
}

/// `a x = b` for a dense n x n system (row major) by Gaussian elimination
/// with partial pivoting; `None` when singular.
fn solve_dense(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&p, &q| a[p * n + col].abs().total_cmp(&a[q * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-300 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(pivot * n + k, col * n + k);
            }
            b.swap(pivot, col);
        }
        for row in col + 1..n {
            let f = a[row * n + col] / a[col * n + col];
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in row + 1..n {
            sum -= a[row * n + k] * x[k];
        }
        x[row] = sum / a[row * n + row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// CBOR-encode a mechanism.
pub fn encode_mechanism(mechanism: &Mechanism) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(mechanism, &mut out)
        .expect("mechanism CBOR serialization should not fail");
    out
}

/// Decode and validate a mechanism payload.
pub fn decode_mechanism(bytes: &[u8]) -> Result<Mechanism, String> {
    let mechanism: Mechanism = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode mechanism CBOR: {e}"))?;
    mechanism.validate()?;
    Ok(mechanism)
}

/// One part of an assembly: its name and its unposed model module.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssemblyPart {
    pub name: String,
    #[serde(with = "serde_bytes")]
    pub model: Vec<u8>,
}

/// A mechanism with its part models and one state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Assembly {
    pub schema: u32,
    pub mechanism: Mechanism,
    /// In the mechanism's part order.
    pub parts: Vec<AssemblyPart>,
    /// The state the assembly is posed at; every state key present.
    pub state: F64Map,
}

impl Assembly {
    /// Build and validate: `parts` must follow the mechanism's part order
    /// and every model must be a wasm module; the state is completed with
    /// defaults and range-checked.
    pub fn new(
        mechanism: Mechanism,
        parts: Vec<AssemblyPart>,
        state: &F64Map,
    ) -> Result<Self, String> {
        let mut state = state.clone();
        for (key, value) in mechanism.default_state() {
            state.entry(key).or_insert(value);
        }
        let assembly = Self {
            schema: MECHANISM_SCHEMA,
            mechanism,
            parts,
            state,
        };
        assembly.validate()?;
        Ok(assembly)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MECHANISM_SCHEMA {
            return Err(format!(
                "assembly schema {} is not {MECHANISM_SCHEMA}",
                self.schema
            ));
        }
        self.mechanism.validate()?;
        if self.parts.len() != self.mechanism.parts.len() {
            return Err(format!(
                "the mechanism names {} parts but {} models were given",
                self.mechanism.parts.len(),
                self.parts.len()
            ));
        }
        for (part, name) in self.parts.iter().zip(&self.mechanism.parts) {
            if part.name != *name {
                return Err(format!(
                    "part `{}` is where the mechanism expects `{name}`",
                    part.name
                ));
            }
            if !part.model.starts_with(b"\0asm") {
                return Err(format!("part `{name}`: the model is not a wasm module"));
            }
        }
        self.mechanism.joint_values(&self.state)?;
        Ok(())
    }

    /// The pose of every part at the assembly's state.
    pub fn poses(&self) -> Vec<Rigid> {
        self.mechanism
            .pose(&self.state)
            .expect("a validated assembly poses")
    }
}

/// CBOR-encode an assembly.
pub fn encode_assembly(assembly: &Assembly) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(assembly, &mut out)
        .expect("assembly CBOR serialization should not fail");
    out
}

/// Decode and validate an assembly payload.
pub fn decode_assembly(bytes: &[u8]) -> Result<Assembly, String> {
    let assembly: Assembly = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode assembly CBOR: {e}"))?;
    assembly.validate()?;
    Ok(assembly)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint(name: &str, kind: JointKind, parent: &str, child: &str, axis: Axis) -> Joint {
        Joint {
            name: name.to_string(),
            kind,
            parent: parent.to_string(),
            child: child.to_string(),
            axis: Some(axis),
            min: -180.0,
            max: 180.0,
            default: 0.0,
            drive: None,
        }
    }

    fn z_axis(origin: [f64; 3]) -> Axis {
        Axis {
            origin,
            direction: [0.0, 0.0, 2.0],
        }
    }

    /// A column that swivels about z through (1, 0, 0), a lift that
    /// slides along it, and a seat that tilts about the x axis 1 m up on
    /// the lift; the back follows the seat at half the angle.
    fn chair() -> Mechanism {
        let mut lift = joint(
            "lift",
            JointKind::Prismatic,
            "column",
            "piston",
            z_axis([1.0, 0.0, 0.0]),
        );
        lift.min = 0.0;
        lift.max = 0.2;
        let mut back = joint(
            "back_tilt",
            JointKind::Revolute,
            "seat",
            "back",
            Axis {
                origin: [1.0, 0.0, 1.0],
                direction: [1.0, 0.0, 0.0],
            },
        );
        back.drive = Some(Drive {
            joint: "tilt".to_string(),
            ratio: 0.5,
            offset: 0.0,
        });
        Mechanism::new(
            ["column", "piston", "seat", "back"]
                .map(str::to_string)
                .to_vec(),
            vec![
                joint(
                    "swivel",
                    JointKind::Revolute,
                    WORLD,
                    "column",
                    z_axis([1.0, 0.0, 0.0]),
                ),
                lift,
                joint(
                    "tilt",
                    JointKind::Revolute,
                    "piston",
                    "seat",
                    Axis {
                        origin: [1.0, 0.0, 1.0],
                        direction: [1.0, 0.0, 0.0],
                    },
                ),
                back,
            ],
        )
    }

    fn state(entries: &[(&str, f64)]) -> F64Map {
        entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn the_chair_validates_and_round_trips() {
        let mech = chair();
        mech.validate().unwrap();
        assert_eq!(mech.state_keys(), ["swivel", "lift", "tilt"]);
        assert_eq!(mech.chain("back"), [0, 1, 2, 3]);
        assert_eq!(mech.chain("column"), [0]);
        let specs = mech.parameter_specs();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[1].key, "lift");
        assert_eq!(specs[1].max, Some(0.2));
        let decoded = decode_mechanism(&encode_mechanism(&mech)).unwrap();
        assert_eq!(decoded, mech);
    }

    #[test]
    fn structural_faults_are_named() {
        let mut loose = chair();
        loose.parts.push("arm".to_string());
        assert!(loose.validate().unwrap_err().contains("hangs from nothing"));

        let mut cycle = chair();
        cycle.joints[0].parent = "back".to_string();
        assert!(cycle.validate().unwrap_err().contains("cycle"));

        let mut twice = chair();
        twice.joints.push(joint(
            "again",
            JointKind::Fixed,
            WORLD,
            "seat",
            z_axis([0.0; 3]),
        ));
        assert!(
            twice
                .validate()
                .unwrap_err()
                .contains("more than one joint")
        );

        let mut flat = chair();
        flat.joints[0].axis = Some(Axis {
            origin: [0.0; 3],
            direction: [0.0; 3],
        });
        assert!(flat.validate().unwrap_err().contains("degenerate"));

        let mut outside = chair();
        outside.joints[1].default = 1.0;
        assert!(outside.validate().unwrap_err().contains("outside"));

        let mut chained_drive = chair();
        chained_drive.joints[2].drive = Some(Drive {
            joint: "back_tilt".to_string(),
            ratio: 1.0,
            offset: 0.0,
        });
        assert!(
            chained_drive
                .validate()
                .unwrap_err()
                .contains("not a state")
        );
    }

    #[test]
    fn states_complete_with_defaults_and_refuse_strangers() {
        let mech = chair();
        assert_eq!(
            mech.joint_values(&state(&[("tilt", 10.0)])).unwrap(),
            [0.0, 0.0, 10.0, 5.0]
        );
        assert!(
            mech.joint_values(&state(&[("lift", 0.3)]))
                .unwrap_err()
                .contains("outside")
        );
        assert!(
            mech.joint_values(&state(&[("wheel", 0.0)]))
                .unwrap_err()
                .contains("not a joint")
        );
        assert!(
            mech.joint_values(&state(&[("back_tilt", 1.0)]))
                .unwrap_err()
                .contains("not a state")
        );
    }

    #[test]
    fn poses_compose_root_first_about_rest_axes() {
        let mech = chair();
        // Rest: every part at the identity.
        for pose in mech.pose(&F64Map::new()).unwrap() {
            assert_eq!(pose, Rigid::IDENTITY);
        }
        // Lift only: the piston, seat and back rise; the column stays.
        let poses = mech.pose(&state(&[("lift", 0.1)])).unwrap();
        assert_eq!(poses[0], Rigid::IDENTITY);
        assert!(close(
            poses[3].apply([5.0, 5.0, 5.0]),
            [5.0, 5.0, 5.1],
            1e-12
        ));
        // Swivel 90° about z through (1, 0, 0): a seat point at (2, 0, 1)
        // goes to (1, 1, 1); the column's own axis is unmoved.
        let poses = mech.pose(&state(&[("swivel", 90.0)])).unwrap();
        assert!(close(
            poses[2].apply([2.0, 0.0, 1.0]),
            [1.0, 1.0, 1.0],
            1e-12
        ));
        assert!(close(
            poses[0].apply([1.0, 0.0, 7.0]),
            [1.0, 0.0, 7.0],
            1e-12
        ));
        // Swivel then lift then tilt: the tilt axis rides on the swivelled,
        // lifted piston, so a back point 1 m behind the tilt axis at
        // (1, 1, 1) at rest ends up rotated about the moved axis.
        let poses = mech
            .pose(&state(&[("swivel", 90.0), ("lift", 0.1), ("tilt", 90.0)]))
            .unwrap();
        // The tilt axis at rest: through (1, 0, 1) along x. After the
        // swivel it runs along y through (1, 0, 1); after the lift through
        // (1, 0, 1.1). Point (1, 1, 1) on the seat: the swivel takes it to
        // (0, 0, 1), the lift to (0, 0, 1.1), and the tilt (90° about y
        // through (1, 0, 1.1)) takes (0, 0, 1.1) - (1, 0, 1.1) = (-1, 0, 0)
        // to (0, 0, 1) so the point lands at (1, 0, 2.1).
        assert!(close(
            poses[2].apply([1.0, 1.0, 1.0]),
            [1.0, 0.0, 2.1],
            1e-12
        ));
        // The back rides on the seat and adds half the tilt of its own:
        // 135° about the same moved axis.
        let back = poses[3].apply([1.0, 1.0, 1.0]);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!(close(back, [1.0 + h, 0.0, 1.1 + h], 1e-12), "{back:?}");
    }

    #[test]
    fn velocity_matches_finite_differences() {
        let mech = chair();
        let base = state(&[("swivel", 30.0), ("lift", 0.05), ("tilt", 20.0)]);
        let rest_point = [1.4, 0.3, 1.2];
        for part in ["column", "piston", "seat", "back"] {
            let index = mech.parts.iter().position(|p| p == part).unwrap();
            let posed = mech.pose(&base).unwrap()[index].apply(rest_point);
            let velocity = mech.velocity(&base, part, posed).unwrap();
            for (k, key) in mech.state_keys().iter().enumerate() {
                let h = 1e-6;
                let mut plus = base.clone();
                *plus.get_mut(*key).unwrap() += h;
                let mut minus = base.clone();
                *minus.get_mut(*key).unwrap() -= h;
                let a = mech.pose(&plus).unwrap()[index].apply(rest_point);
                let b = mech.pose(&minus).unwrap()[index].apply(rest_point);
                let fd: [f64; 3] = std::array::from_fn(|i| (a[i] - b[i]) / (2.0 * h));
                assert!(
                    close(velocity[k], fd, 1e-6),
                    "{part} d/d{key}: {:?} vs {fd:?}",
                    velocity[k]
                );
            }
        }
        // A joint off the chain contributes nothing to its ancestors.
        let column = mech.velocity(&base, "column", [1.0, 1.0, 0.0]).unwrap();
        assert_eq!(column[1], [0.0; 3]);
        assert_eq!(column[2], [0.0; 3]);
    }

    #[test]
    fn pull_turns_the_joints_that_reach_the_target_and_stays_in_range() {
        let mech = chair();
        // A seat point 1 m out along x at rest, pulled to +y: the swivel
        // turns a quarter turn (the tilt axis is along x and cannot help).
        let pulled = mech
            .pull(&F64Map::new(), "seat", [2.0, 0.0, 1.0], [1.0, 1.0, 1.0], 20)
            .unwrap();
        assert!((pulled["swivel"] - 90.0).abs() < 1e-6, "{pulled:?}");
        assert!(pulled["tilt"].abs() < 1e-6, "{pulled:?}");
        // Pulled straight up beyond the lift's 0.2 m range: the lift stops
        // at its maximum and the rest of the miss stays.
        let pulled = mech
            .pull(&F64Map::new(), "seat", [2.0, 0.0, 1.0], [2.0, 0.0, 2.0], 20)
            .unwrap();
        assert!((pulled["lift"] - 0.2).abs() < 1e-9, "{pulled:?}");
        // The rest point of the column can be pulled nowhere useful: the
        // column has only the swivel, and a point on its axis has no
        // velocity, so the state stays put.
        let pulled = mech
            .pull(
                &F64Map::new(),
                "column",
                [1.0, 0.0, 0.5],
                [1.0, 0.0, 3.0],
                5,
            )
            .unwrap();
        assert_eq!(pulled["swivel"], 0.0);
        assert!(
            mech.pull(&F64Map::new(), "wheel", [0.0; 3], [0.0; 3], 1)
                .unwrap_err()
                .contains("not a part")
        );
    }

    #[test]
    fn assemblies_pair_parts_with_the_mechanism() {
        let mech = chair();
        let wasm = || b"\0asm\x01\0\0\0".to_vec();
        let parts: Vec<AssemblyPart> = mech
            .parts
            .iter()
            .map(|name| AssemblyPart {
                name: name.clone(),
                model: wasm(),
            })
            .collect();
        let assembly =
            Assembly::new(mech.clone(), parts.clone(), &state(&[("tilt", 5.0)])).unwrap();
        assert_eq!(assembly.state["tilt"], 5.0);
        assert_eq!(assembly.state["swivel"], 0.0);
        assert_eq!(assembly.poses().len(), 4);
        let decoded = decode_assembly(&encode_assembly(&assembly)).unwrap();
        assert_eq!(decoded, assembly);

        let mut swapped = parts.clone();
        swapped.swap(0, 1);
        assert!(
            Assembly::new(mech.clone(), swapped, &F64Map::new())
                .unwrap_err()
                .contains("expects")
        );
        let mut short = parts.clone();
        short.pop();
        assert!(
            Assembly::new(mech.clone(), short, &F64Map::new())
                .unwrap_err()
                .contains("models were given")
        );
        let mut text = parts;
        text[0].model = b"not wasm".to_vec();
        assert!(
            Assembly::new(mech, text, &F64Map::new())
                .unwrap_err()
                .contains("not a wasm module")
        );
    }
}
