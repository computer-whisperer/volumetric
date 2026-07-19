//! STEP entities -> `brep_core` IR: solids from manifold BReps (faces
//! lowered to exact surfaces, edge curves flattened and projected into
//! UV trim loops) and instances from the AP214 assembly graph.

use crate::entities::{Ctx, Curve, StepEdge, StepSurface, length_unit_scale};
use crate::p21::{Arg, DataSection};
use brep_core::ir::{BRepModel, Face, Instance, Solid, Surface};
use brep_core::math::{Affine, Frame, Vec3, cross, dot, norm, normalize, sub};
use brep_core::nurbs::CurveData;
use brep_core::project;
use core::f64::consts::TAU;
use std::collections::HashMap;

/// Import tuning; `chord_tol` is the 3D flattening tolerance (mm) for
/// trim polylines and lowered extrusion profiles.
pub struct Options {
    pub chord_tol: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options { chord_tol: 0.005 }
    }
}

/// Convert a parsed STEP file into a BRep model, walking the assembly
/// graph from its roots. Fails loudly: any face of any referenced solid
/// that cannot be converted fails the import with the entity id (a
/// silently missing face would flip parity for its whole solid).
pub fn build_model(data: &DataSection, opts: &Options) -> Result<BRepModel, String> {
    let scale = length_unit_scale(data)?;
    let ctx = Ctx { data, scale };
    let asm = AssemblyGraph::build(data)?;

    let mut model = BRepModel::default();
    let mut solid_index: HashMap<u64, usize> = HashMap::new();

    for root in &asm.roots {
        let label = asm.pd_name.get(root).cloned().unwrap_or_default();
        walk(
            &ctx,
            &asm,
            *root,
            Affine::IDENTITY,
            &label,
            opts,
            &mut model,
            &mut solid_index,
        )?;
    }
    if model.instances.is_empty() {
        return Err("no solid geometry found in the file".into());
    }
    Ok(model)
}

/// Keep only instances whose label passes the include/exclude lists
/// (case-sensitive exact match; empty include list means "everything"),
/// then drop unreferenced solids.
pub fn filter_instances(model: &mut BRepModel, include: &[String], exclude: &[String]) {
    model.instances.retain(|inst| {
        let inc = include.is_empty() || include.iter().any(|l| *l == inst.label);
        inc && !exclude.iter().any(|l| *l == inst.label)
    });
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut solids = Vec::new();
    for inst in &mut model.instances {
        let new = *remap.entry(inst.solid).or_insert_with(|| {
            solids.push(model.solids[inst.solid].clone());
            solids.len() - 1
        });
        inst.solid = new;
    }
    model.solids = solids;
}

// -------------------------------------------------------------------
// Assembly graph
// -------------------------------------------------------------------

struct Nauo {
    child_pd: u64,
    /// The NAUO's name — KiCad puts the reference designator here.
    label: String,
    /// Child-to-parent placement.
    transform: Affine,
}

struct AssemblyGraph {
    /// PRODUCT_DEFINITION -> its shape representation.
    pd_rep: HashMap<u64, u64>,
    /// Child links per parent PD.
    children: HashMap<u64, Vec<usize>>,
    nauos: Vec<Nauo>,
    /// PDs that are no NAUO's child.
    roots: Vec<u64>,
    /// PD -> product name (fallback label).
    pd_name: HashMap<u64, String>,
}

impl AssemblyGraph {
    fn build(data: &DataSection) -> Result<AssemblyGraph, String> {
        // Angle-preserving accessors only; no unit scaling is needed for
        // the graph itself, but transforms carry translations, so use a
        // scaled context.
        let scale = length_unit_scale(data)?;
        let ctx = Ctx { data, scale };

        let mut pd_rep = HashMap::new();
        let mut pd_name = HashMap::new();
        let mut nauo_ids = Vec::new();
        let mut cdsr_ids = Vec::new();
        for (&id, e) in &data.entities {
            if let Some(r) = e.simple() {
                match r.name.as_str() {
                    "SHAPE_DEFINITION_REPRESENTATION" => {
                        // (definition: PDS, used_representation)
                        let pds = data.deref(&r.args[0])?;
                        let pds_r = pds
                            .simple()
                            .filter(|r| r.name == "PRODUCT_DEFINITION_SHAPE")
                            .ok_or_else(|| format!("#{id}: SDR without a PDS"))?;
                        let def = pds_r.args[2]
                            .as_ref_id()
                            .ok_or_else(|| format!("#{id}: PDS definition not a reference"))?;
                        // Only PDS of product definitions matter here
                        // (placement PDSes point at NAUOs).
                        if data.get(def)?.is("PRODUCT_DEFINITION") {
                            let rep = r.args[1]
                                .as_ref_id()
                                .ok_or_else(|| format!("#{id}: SDR rep not a reference"))?;
                            pd_rep.insert(def, rep);
                        }
                    }
                    "NEXT_ASSEMBLY_USAGE_OCCURRENCE" => nauo_ids.push(id),
                    "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" => cdsr_ids.push(id),
                    "PRODUCT_DEFINITION" => {
                        // formation -> product -> name.
                        let name = (|| {
                            let formation = data.deref(r.args.get(2)?).ok()?;
                            let product =
                                data.deref(formation.simple()?.args.get(2)?).ok()?;
                            let pr = product.simple()?;
                            let id_field = pr.args.first()?.as_str()?;
                            let name_field = pr.args.get(1).and_then(Arg::as_str);
                            Some(if id_field.is_empty() {
                                name_field.unwrap_or_default().to_string()
                            } else {
                                id_field.to_string()
                            })
                        })()
                        .unwrap_or_default();
                        pd_name.insert(id, name);
                    }
                    _ => {}
                }
            }
        }

        // CDSR: NAUO id -> (rep_1, rep_2, transform item pair).
        let mut nauo_cdsr: HashMap<u64, (u64, u64, u64, u64)> = HashMap::new();
        for id in cdsr_ids {
            let r = data.get(id)?.simple().ok_or("CDSR must be simple")?.clone();
            let rr = data.deref(&r.args[0])?;
            let rel = rr
                .record("REPRESENTATION_RELATIONSHIP")
                .ok_or_else(|| format!("#{id}: no REPRESENTATION_RELATIONSHIP"))?;
            let rep_1 = rel.args[2]
                .as_ref_id()
                .ok_or_else(|| format!("#{id}: rep_1 not a reference"))?;
            let rep_2 = rel.args[3]
                .as_ref_id()
                .ok_or_else(|| format!("#{id}: rep_2 not a reference"))?;
            let idt_id = rr
                .record("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
                .and_then(|t| t.args.first())
                .and_then(Arg::as_ref_id)
                .ok_or_else(|| format!("#{id}: no transformation"))?;
            let idt = data.get(idt_id)?.simple().ok_or("IDT must be simple")?;
            let axis_1 = idt.args[2]
                .as_ref_id()
                .ok_or_else(|| format!("#{idt_id}: IDT item 1 not a reference"))?;
            let axis_2 = idt.args[3]
                .as_ref_id()
                .ok_or_else(|| format!("#{idt_id}: IDT item 2 not a reference"))?;
            // The PDS this CDSR describes points at the NAUO.
            let pds = data.deref(&r.args[1])?;
            let nauo = pds
                .simple()
                .filter(|p| p.name == "PRODUCT_DEFINITION_SHAPE")
                .and_then(|p| p.args[2].as_ref_id())
                .ok_or_else(|| format!("#{id}: CDSR without a placement PDS"))?;
            nauo_cdsr.insert(nauo, (rep_1, rep_2, axis_1, axis_2));
        }

        let mut nauos = Vec::new();
        let mut children: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut child_pds: Vec<u64> = Vec::new();
        for id in nauo_ids {
            let e = data.get(id)?;
            let r = e.simple().ok_or("NAUO must be simple")?;
            let parent_pd = r.args[3]
                .as_ref_id()
                .ok_or_else(|| format!("#{id}: NAUO relating not a reference"))?;
            let child_pd = r.args[4]
                .as_ref_id()
                .ok_or_else(|| format!("#{id}: NAUO related not a reference"))?;
            let label = r.args[1].as_str().unwrap_or_default().to_string();
            let (rep_1, _rep_2, axis_1, axis_2) = *nauo_cdsr.get(&id).ok_or_else(|| {
                format!("#{id}: assembly occurrence has no placement (missing CDSR)")
            })?;
            // Child-to-parent: the child-side axis maps onto the
            // parent-side axis. Which IDT item is child-side follows
            // from which representation is the child's.
            let child_rep = pd_rep.get(&child_pd).copied();
            let (child_axis, parent_axis) = if child_rep == Some(rep_1) {
                (axis_1, axis_2)
            } else {
                (axis_2, axis_1)
            };
            let fc = frame_affine(&ctx.axis2(child_axis)?);
            let fp = frame_affine(&ctx.axis2(parent_axis)?);
            let transform = fp.compose(&fc.rigid_inverse()?);
            nauos.push(Nauo {
                child_pd,
                label,
                transform,
            });
            children
                .entry(parent_pd)
                .or_default()
                .push(nauos.len() - 1);
            child_pds.push(child_pd);
        }

        let roots: Vec<u64> = pd_rep
            .keys()
            .copied()
            .filter(|pd| !child_pds.contains(pd))
            .collect();
        if roots.is_empty() && !pd_rep.is_empty() {
            return Err("assembly graph has a cycle (no root product)".into());
        }
        Ok(AssemblyGraph {
            pd_rep,
            children,
            nauos,
            roots,
            pd_name,
        })
    }
}

/// Frame -> local-to-world affine.
fn frame_affine(f: &Frame) -> Affine {
    Affine([
        f.x[0], f.y[0], f.z[0], f.origin[0], //
        f.x[1], f.y[1], f.z[1], f.origin[1], //
        f.x[2], f.y[2], f.z[2], f.origin[2],
    ])
}

#[allow(clippy::too_many_arguments)]
fn walk(
    ctx: &Ctx,
    asm: &AssemblyGraph,
    pd: u64,
    transform: Affine,
    label: &str,
    opts: &Options,
    model: &mut BRepModel,
    solid_index: &mut HashMap<u64, usize>,
) -> Result<(), String> {
    if let Some(&rep) = asm.pd_rep.get(&pd) {
        // Any MANIFOLD_SOLID_BREP item of the representation is a body
        // of this product.
        let rep_entity = ctx.data.get(rep)?;
        let items = rep_entity
            .records
            .iter()
            .find_map(|r| r.args.get(1).and_then(Arg::as_list))
            .unwrap_or(&[]);
        for item in items {
            let Some(item_id) = item.as_ref_id() else {
                continue;
            };
            if !ctx.data.get(item_id)?.is("MANIFOLD_SOLID_BREP") {
                continue;
            }
            let solid = match solid_index.get(&item_id) {
                Some(&idx) => idx,
                None => {
                    let converted = convert_solid(ctx, item_id, opts)
                        .map_err(|e| format!("solid #{item_id}: {e}"))?;
                    model.solids.push(converted);
                    let idx = model.solids.len() - 1;
                    solid_index.insert(item_id, idx);
                    idx
                }
            };
            model.instances.push(Instance {
                solid,
                local_to_world: transform,
                label: label.to_string(),
            });
        }
    }
    if let Some(kids) = asm.children.get(&pd) {
        for &k in kids {
            let nauo = &asm.nauos[k];
            let child_transform = transform.compose(&nauo.transform);
            // The occurrence name is the reference designator ("J7") in
            // KiCad exports; OCCT-generated names ("=>[0:1:1:3]") and
            // empty names inherit the enclosing label.
            let child_label = if !nauo.label.is_empty() && !nauo.label.starts_with("=>") {
                nauo.label.as_str()
            } else {
                label
            };
            walk(
                ctx,
                asm,
                nauo.child_pd,
                child_transform,
                child_label,
                opts,
                model,
                solid_index,
            )?;
        }
    }
    Ok(())
}

// -------------------------------------------------------------------
// Solid and face conversion
// -------------------------------------------------------------------

fn convert_solid(ctx: &Ctx, msb_id: u64, opts: &Options) -> Result<Solid, String> {
    let mut faces = Vec::new();
    for face_id in ctx.solid_faces(msb_id)? {
        faces.push(
            convert_face(ctx, face_id, opts).map_err(|e| format!("face #{face_id}: {e}"))?,
        );
    }
    Ok(Solid { faces })
}

fn convert_face(ctx: &Ctx, face_id: u64, opts: &Options) -> Result<Face, String> {
    let (surface_id, loops) = ctx.advanced_face(face_id)?;
    let step_surface = ctx
        .surface(surface_id)
        .map_err(|e| format!("surface: {e}"))?;
    let surface = lower_surface(step_surface, opts)?;

    let proj_tol = opts.chord_tol * 4.0 + 1e-6;
    let mut trims = Vec::new();
    for (li, lp) in loops.iter().enumerate() {
        let pts = loop_points(ctx, lp, opts).map_err(|e| format!("loop {li}: {e}"))?;
        let mut uv: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
        let mut hint = None;
        for p in &pts {
            let q = project::project_point(&surface, *p, hint, proj_tol)
                .map_err(|e| format!("loop {li}: {e}"))?;
            uv.push(q);
            hint = Some(q);
        }
        project::verify_loop(&surface, &pts, &uv, proj_tol * 2.0)
            .map_err(|e| format!("loop {li}: {e}"))?;
        trims.push(uv);
    }
    Ok(Face { surface, trims })
}

/// Lower a STEP surface to the exact `brep_core` vocabulary. Extrusions
/// of lines and axis-aligned circles become planes and cylinders; other
/// profiles flatten to a polyline prism at the chord tolerance — the
/// only approximation in the pipeline.
fn lower_surface(s: StepSurface, opts: &Options) -> Result<Surface, String> {
    Ok(match s {
        StepSurface::Plane(frame) => Surface::Plane { frame },
        StepSurface::Cylinder(frame, radius) => Surface::Cylinder { frame, radius },
        StepSurface::Cone(frame, radius, half_angle) => Surface::Cone {
            frame,
            radius,
            half_angle,
        },
        StepSurface::Sphere(frame, radius) => Surface::Sphere { frame, radius },
        StepSurface::Torus(frame, major, minor) => Surface::Torus {
            frame,
            major,
            minor,
        },
        StepSurface::Nurbs(n) => Surface::Nurbs(n),
        StepSurface::Extrusion(profile, dir) => {
            let dir_n = normalize(dir);
            match *profile {
                Curve::Line { point, dir: line_dir } => {
                    let normal = cross(normalize(line_dir), dir_n);
                    if norm(normal) < 1e-9 {
                        return Err("extrusion of a line along itself".into());
                    }
                    Surface::Plane {
                        frame: Frame::from_axis_ref(point, normal, line_dir),
                    }
                }
                Curve::Circle { frame, radius } if dot(frame.z, dir_n).abs() > 1.0 - 1e-9 => {
                    Surface::Cylinder { frame, radius }
                }
                other => {
                    // General profile: flatten in 3D, project onto the
                    // plane perpendicular to the extrusion direction.
                    let pts3 = match &other {
                        Curve::Circle { frame, radius } => {
                            project::flatten_circle(frame, *radius, 0.0, TAU, opts.chord_tol)
                        }
                        Curve::Bspline(c) => {
                            let dom = c.domain();
                            project::flatten_bspline(c, dom[0], dom[1], opts.chord_tol)
                        }
                        Curve::Line { .. } => unreachable!(),
                    };
                    let origin = pts3[0];
                    let first_seg = sub(*pts3.last().unwrap(), pts3[0]);
                    let seg = if norm(first_seg) > 1e-9 {
                        first_seg
                    } else {
                        sub(pts3[pts3.len() / 2], pts3[0])
                    };
                    let frame = Frame::from_axis_ref(origin, dir_n, seg);
                    let profile: Vec<[f64; 2]> = pts3
                        .iter()
                        .map(|p| {
                            let l = frame.to_local(*p);
                            [l[0], l[1]]
                        })
                        .collect();
                    Surface::ExtrusionPolyline { frame, profile }
                }
            }
        }
    })
}

/// Flatten one loop of oriented edges into a chained, closed 3D
/// polyline (the closing point is omitted).
fn loop_points(ctx: &Ctx, edges: &[StepEdge], opts: &Options) -> Result<Vec<Vec3>, String> {
    let tol = opts.chord_tol;
    let join_tol = (tol * 10.0).max(1e-6);
    let mut out: Vec<Vec3> = Vec::new();
    for (i, e) in edges.iter().enumerate() {
        let pts = edge_points(ctx, e, tol).map_err(|err| format!("edge {i}: {err}"))?;
        if let Some(last) = out.last() {
            if norm(sub(*last, pts[0])) > join_tol {
                return Err(format!(
                    "edge {i} does not chain: gap {:.3e}",
                    norm(sub(*last, pts[0]))
                ));
            }
        }
        let skip = usize::from(!out.is_empty());
        out.extend(&pts[skip..]);
    }
    if out.len() > 1 {
        let gap = norm(sub(*out.last().unwrap(), out[0]));
        if gap > join_tol {
            return Err(format!("loop does not close: gap {gap:.3e}"));
        }
        out.pop();
    }
    if out.len() < 3 {
        return Err("degenerate loop with fewer than 3 points".into());
    }
    Ok(out)
}

/// Flatten one oriented edge into 3D points from its start vertex to
/// its end vertex.
fn edge_points(ctx: &Ctx, e: &StepEdge, tol: f64) -> Result<Vec<Vec3>, String> {
    let curve = ctx.curve(e.curve_id)?;
    let mut pts = match curve {
        Curve::Line { .. } => vec![e.start, e.end],
        Curve::Circle { frame, radius } => {
            let a_of = |p: Vec3| {
                let l = frame.to_local(p);
                l[1].atan2(l[0])
            };
            let a_start = a_of(e.start);
            let full = norm(sub(e.start, e.end)) < tol * 1e-3 + 1e-9;
            // Sweep CCW in the circle frame when the curve runs forward
            // along the traversal, CW otherwise.
            let (a0, a1) = if full {
                (a_start, a_start + TAU)
            } else {
                let mut a_end = a_of(e.end);
                if e.forward {
                    if a_end <= a_start {
                        a_end += TAU;
                    }
                    (a_start, a_end)
                } else {
                    if a_end >= a_start {
                        a_end -= TAU;
                    }
                    (a_start, a_end)
                }
            };
            let (lo, hi, reversed) = if full && !e.forward {
                (a_start - TAU, a_start, true)
            } else if a1 >= a0 {
                (a0, a1, false)
            } else {
                (a1, a0, true)
            };
            let mut pts = project::flatten_circle(&frame, radius, lo, hi, tol);
            if reversed {
                pts.reverse();
            }
            pts
        }
        Curve::Bspline(c) => {
            let dom = c.domain();
            let mut pts = project::flatten_bspline(&c, dom[0], dom[1], tol);
            if !e.forward {
                pts.reverse();
            }
            pts
        }
    };
    // Snap the flattened ends onto the topological vertices so chaining
    // and closure tests see exact joins.
    let join_tol = (tol * 10.0).max(1e-6);
    let d_start = norm(sub(pts[0], e.start));
    let d_end = norm(sub(*pts.last().unwrap(), e.end));
    if d_start > join_tol || d_end > join_tol {
        return Err(format!(
            "flattened curve #{} misses its vertices by {d_start:.3e}/{d_end:.3e}",
            e.curve_id
        ));
    }
    let n = pts.len();
    pts[0] = e.start;
    pts[n - 1] = e.end;
    Ok(pts)
}
