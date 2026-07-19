//! The builder-side in-memory representation. Importers construct this,
//! then [`crate::payload::build_payload`] serializes it. Internal to the
//! brep pipeline — deliberately not an ABI value type.

use crate::math::{Affine, Frame};

/// A boundary-representation model: distinct solid bodies plus the placed
/// occurrences that exist in model space.
#[derive(Clone, Debug, Default)]
pub struct BRepModel {
    pub solids: Vec<Solid>,
    pub instances: Vec<Instance>,
}

/// One placed occurrence of a solid.
#[derive(Clone, Debug)]
pub struct Instance {
    /// Index into [`BRepModel::solids`].
    pub solid: usize,
    /// Solid-local to model-space; rigid plus optional uniform scale.
    pub local_to_world: Affine,
    /// Human-readable name ("U2", "board 1"); empty if unnamed. Used for
    /// import-time filtering only — never serialized into the payload.
    pub label: String,
}

/// A solid body: a set of trimmed faces forming (nominally) a closed
/// boundary. Like `TriMesh`, closedness is a property of the data, not
/// the type; parity classification gives open shells parity's literal
/// behavior.
#[derive(Clone, Debug, Default)]
pub struct Solid {
    pub faces: Vec<Face>,
}

/// A trimmed face: an exact surface plus the UV region of it that
/// belongs to the solid boundary.
#[derive(Clone, Debug)]
pub struct Face {
    pub surface: Surface,
    /// Trim loops as UV polylines, implicitly closed. The trim region is
    /// the even-odd fill over all loops, so loop orientation and nesting
    /// order don't matter. On periodic surfaces the loops are stored
    /// *unwrapped*: consecutive points continue past the seam (u may run
    /// beyond one period; matching happens modulo the period).
    pub trims: Vec<Vec<[f64; 2]>>,
}

/// An exact surface with its natural UV parameterization.
#[derive(Clone, Debug)]
pub enum Surface {
    /// `P(u,v) = origin + u*x + v*y`; normal `z`.
    Plane { frame: Frame },
    /// Axis along `z`. `P(u,v) = origin + r*(cos u * x + sin u * y) + v*z`;
    /// u-period 2π.
    Cylinder { frame: Frame, radius: f64 },
    /// Radius grows with v: `r(v) = radius + v*tan(half_angle)`, apex on
    /// the axis where `r(v) = 0`.
    /// `P(u,v) = origin + r(v)*(cos u * x + sin u * y) + v*z`; u-period 2π.
    Cone {
        frame: Frame,
        radius: f64,
        half_angle: f64,
    },
    /// `P(u,v) = origin + r*(cos v * (cos u * x + sin u * y) + sin v * z)`;
    /// u azimuth (period 2π), v latitude in [-π/2, π/2].
    Sphere { frame: Frame, radius: f64 },
    /// `P(u,v) = origin + (R + r*cos v)*(cos u * x + sin u * y) + r*sin v * z`;
    /// both u and v have period 2π.
    Torus {
        frame: Frame,
        major: f64,
        minor: f64,
    },
    /// A polyline profile in the frame's XY plane swept along the frame z
    /// axis: `P(u,v) = profile(u) + v*z` where u is the polyline
    /// parameter (segment index plus fraction, so u spans
    /// `[0, profile.len()-1]`). Importers lower exact line/circle
    /// profiles to Plane/Cylinder and flatten everything else to a
    /// polyline at the import tolerance — the one place a surface is
    /// approximated rather than exact.
    ExtrusionPolyline {
        frame: Frame,
        profile: Vec<[f64; 2]>,
    },
    /// A (possibly rational) B-spline surface. Control points are xyzw;
    /// w = 1 everywhere for polynomial surfaces.
    Nurbs(NurbsSurface),
}

/// B-spline surface data, clamped knot vectors, control net indexed
/// `[i * nctrl_v + j]` (v-fastest).
#[derive(Clone, Debug)]
pub struct NurbsSurface {
    pub degree_u: usize,
    pub degree_v: usize,
    pub nctrl_u: usize,
    pub nctrl_v: usize,
    /// Length `nctrl_u + degree_u + 1`.
    pub knots_u: Vec<f64>,
    /// Length `nctrl_v + degree_v + 1`.
    pub knots_v: Vec<f64>,
    /// `nctrl_u * nctrl_v` xyzw control points, v-fastest.
    pub ctrl: Vec<[f64; 4]>,
}

impl NurbsSurface {
    pub fn validate(&self) -> Result<(), String> {
        if self.degree_u == 0 && self.degree_v == 0 {
            return Err("nurbs surface with degree 0 in both directions".into());
        }
        if self.nctrl_u <= self.degree_u || self.nctrl_v <= self.degree_v {
            return Err(format!(
                "nurbs control net {}x{} too small for degrees {}x{}",
                self.nctrl_u, self.nctrl_v, self.degree_u, self.degree_v
            ));
        }
        if self.knots_u.len() != self.nctrl_u + self.degree_u + 1 {
            return Err(format!(
                "knots_u length {} != nctrl_u {} + degree_u {} + 1",
                self.knots_u.len(),
                self.nctrl_u,
                self.degree_u
            ));
        }
        if self.knots_v.len() != self.nctrl_v + self.degree_v + 1 {
            return Err(format!(
                "knots_v length {} != nctrl_v {} + degree_v {} + 1",
                self.knots_v.len(),
                self.nctrl_v,
                self.degree_v
            ));
        }
        if self.ctrl.len() != self.nctrl_u * self.nctrl_v {
            return Err(format!(
                "control net length {} != {}x{}",
                self.ctrl.len(),
                self.nctrl_u,
                self.nctrl_v
            ));
        }
        for k in self.knots_u.windows(2).chain(self.knots_v.windows(2)) {
            if !(k[1] >= k[0]) {
                return Err("knot vector not non-decreasing".into());
            }
        }
        for c in &self.ctrl {
            if !(c[3] > 0.0) || c.iter().any(|v| !v.is_finite()) {
                return Err("control point non-finite or weight <= 0".into());
            }
        }
        Ok(())
    }

    /// The valid parameter domain `[u_min, u_max, v_min, v_max]`.
    pub fn domain(&self) -> [f64; 4] {
        [
            self.knots_u[self.degree_u],
            self.knots_u[self.nctrl_u],
            self.knots_v[self.degree_v],
            self.knots_v[self.nctrl_v],
        ]
    }
}

impl Surface {
    /// `(u_period, v_period)`, 0.0 meaning aperiodic. Periods describe the
    /// parameterization (a full cylinder repeats in u every 2π) — trim
    /// loops still restrict the face to whatever region they enclose.
    pub fn periods(&self) -> (f64, f64) {
        use core::f64::consts::TAU;
        match self {
            Surface::Plane { .. } | Surface::ExtrusionPolyline { .. } | Surface::Nurbs(_) => {
                (0.0, 0.0)
            }
            Surface::Cylinder { .. } | Surface::Cone { .. } | Surface::Sphere { .. } => (TAU, 0.0),
            Surface::Torus { .. } => (TAU, TAU),
        }
    }
}
