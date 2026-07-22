//! Field-aware editing primitives for explicit meshes ([`FeaMesh`] of any
//! element kind): element filtering, Bar2 short-strut welding, and
//! connected-component pruning. Shared by the operators that generate
//! strut networks (`strut_pattern_operator`) and the ones that edit
//! meshes mid-DAG (`mesh_clip_operator` and friends), so the subtle
//! algorithms — welding in particular, whose absence makes FEA
//! conditioning explode — exist exactly once.
//!
//! All functions are pure `&FeaMesh -> FeaMesh` transforms that preserve
//! named node/element fields: surviving entries keep their values, merged
//! nodes average theirs (consistent with the centroid position), and
//! dropped entries take their field values with them.

use volumetric_abi::fea::{FeaField, FeaMesh};

/// Keep exactly the flagged elements (`keep[e]`, one per element), in
/// order. Nodes no longer referenced by any kept element are dropped and
/// the survivors compacted; node and element fields slice to match.
pub fn filter_elements(mesh: &FeaMesh, keep: &[bool]) -> Result<FeaMesh, String> {
    if keep.len() != mesh.element_count() {
        return Err(format!(
            "keep flags cover {} elements, mesh has {}",
            keep.len(),
            mesh.element_count()
        ));
    }
    let per_element = mesh.element_kind.node_count();
    let mut node_remap = vec![u32::MAX; mesh.node_count()];
    let mut kept_nodes: Vec<u32> = Vec::new();
    let mut connectivity = Vec::new();
    let mut kept_elements: Vec<u32> = Vec::new();
    for (e, kept) in keep.iter().enumerate() {
        if !kept {
            continue;
        }
        kept_elements.push(e as u32);
        for &node in &mesh.connectivity[e * per_element..(e + 1) * per_element] {
            if node_remap[node as usize] == u32::MAX {
                node_remap[node as usize] = kept_nodes.len() as u32;
                kept_nodes.push(node);
            }
            connectivity.push(node_remap[node as usize]);
        }
    }

    let node_positions = gather(&mesh.node_positions, 3, &kept_nodes);
    let mesh = FeaMesh {
        element_kind: mesh.element_kind,
        node_positions,
        connectivity,
        node_fields: gather_fields(&mesh.node_fields, &kept_nodes),
        element_fields: gather_fields(&mesh.element_fields, &kept_elements),
    };
    mesh.validate()?;
    Ok(mesh)
}

/// Weld away Bar2 struts shorter than their weld length: union-find
/// clusters over the short struts, each cluster's nodes merge at their
/// centroid (node fields averaging alongside), the contracted struts
/// vanish, and struts left connecting the same pair of joints keep only
/// the first copy (and its element fields). Repeats until no strut is
/// short — merging joints can pull previously-long struts under their
/// threshold.
///
/// `weld_length(e)` gives the threshold per *original* element index, so
/// per-strut thresholds (e.g. `factor * radius[e]`) follow a strut
/// through the passes; a constant closure is a uniform weld. A strut
/// shorter than its own radius is a joint blob, not a beam: leaving such
/// struts in makes the FEA solver's conditioning explode (measured: an
/// unwelded foam fails CG at 3e8 stiffness contrast; welded at 1 radius
/// it converges in ~10k iterations).
pub fn weld_short_bars(
    mesh: &FeaMesh,
    weld_length: &dyn Fn(usize) -> f64,
) -> Result<FeaMesh, String> {
    if mesh.element_kind != volumetric_abi::fea::FeaElementKind::Bar2 {
        return Err(format!(
            "short-strut welding needs a Bar2 mesh, got {:?} elements",
            mesh.element_kind
        ));
    }
    let mut positions = mesh.node_positions.clone();
    let mut connectivity = mesh.connectivity.clone();
    let mut node_data: Vec<Vec<f64>> = mesh.node_fields.iter().map(|f| f.data.clone()).collect();
    // Which original element each surviving strut came from (for
    // per-element thresholds and final field slicing).
    let mut origin: Vec<u32> = (0..mesh.element_count() as u32).collect();

    loop {
        let node_count = positions.len() / 3;
        let mut parent: Vec<u32> = (0..node_count as u32).collect();
        let mut short = 0usize;
        for (pair, &org) in connectivity.chunks_exact(2).zip(&origin) {
            let (a, b) = (pair[0] as usize * 3, pair[1] as usize * 3);
            let len2: f64 = (0..3)
                .map(|c| (positions[a + c] - positions[b + c]).powi(2))
                .sum();
            let threshold = weld_length(org as usize);
            if len2 < threshold * threshold {
                short += 1;
                let (ra, rb) = (find(&mut parent, pair[0]), find(&mut parent, pair[1]));
                if ra != rb {
                    let (lo, hi) = (ra.min(rb), ra.max(rb));
                    parent[hi as usize] = lo;
                }
            }
        }
        if short == 0 {
            break;
        }

        // Cluster centroids, for positions and node fields alike.
        let mut cluster_size = vec![0usize; node_count];
        let mut centroid = vec![0.0f64; positions.len()];
        let mut field_centroid: Vec<Vec<f64>> =
            node_data.iter().map(|d| vec![0.0; d.len()]).collect();
        for node in 0..node_count {
            let root = find(&mut parent, node as u32) as usize;
            cluster_size[root] += 1;
            for c in 0..3 {
                centroid[root * 3 + c] += positions[node * 3 + c];
            }
            for (sums, data) in field_centroid.iter_mut().zip(&node_data) {
                let components = data.len() / node_count;
                for c in 0..components {
                    sums[root * components + c] += data[node * components + c];
                }
            }
        }

        let mut remap = vec![u32::MAX; node_count];
        let mut new_positions: Vec<f64> = Vec::new();
        let mut new_node_data: Vec<Vec<f64>> = node_data.iter().map(|_| Vec::new()).collect();
        let mut new_connectivity: Vec<u32> = Vec::new();
        let mut new_origin: Vec<u32> = Vec::new();
        let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for (pair, &org) in connectivity.chunks_exact(2).zip(&origin) {
            let (ra, rb) = (find(&mut parent, pair[0]), find(&mut parent, pair[1]));
            if ra == rb {
                continue; // contracted into a joint
            }
            if !seen.insert((ra.min(rb), ra.max(rb))) {
                continue; // parallel duplicate of a kept strut
            }
            for root in [ra, rb] {
                if remap[root as usize] == u32::MAX {
                    remap[root as usize] = (new_positions.len() / 3) as u32;
                    let size = cluster_size[root as usize] as f64;
                    for c in 0..3 {
                        new_positions.push(centroid[root as usize * 3 + c] / size);
                    }
                    for (out, sums) in new_node_data.iter_mut().zip(&field_centroid) {
                        let components = sums.len() / node_count;
                        for c in 0..components {
                            out.push(sums[root as usize * components + c] / size);
                        }
                    }
                }
                new_connectivity.push(remap[root as usize]);
            }
            new_origin.push(org);
        }
        positions = new_positions;
        connectivity = new_connectivity;
        node_data = new_node_data;
        origin = new_origin;
        if connectivity.is_empty() {
            break;
        }
    }

    let node_fields = mesh
        .node_fields
        .iter()
        .zip(node_data)
        .map(|(f, data)| FeaField {
            name: f.name.clone(),
            components: f.components,
            data,
        })
        .collect();
    let mesh = FeaMesh {
        element_kind: mesh.element_kind,
        node_positions: positions,
        connectivity,
        node_fields,
        element_fields: gather_fields(&mesh.element_fields, &origin),
    };
    mesh.validate()?;
    Ok(mesh)
}

/// Keep only the largest connected component of a Bar2 mesh (by strut
/// count; ties break toward the component containing the lowest node
/// index). Floating fragments make FEA solves singular; editing
/// workflows that *want* disconnected pieces simply skip this.
pub fn largest_bar_component(mesh: &FeaMesh) -> Result<FeaMesh, String> {
    if mesh.element_kind != volumetric_abi::fea::FeaElementKind::Bar2 {
        return Err(format!(
            "component pruning needs a Bar2 mesh, got {:?} elements",
            mesh.element_kind
        ));
    }
    let node_count = mesh.node_count();
    let mut parent: Vec<u32> = (0..node_count as u32).collect();
    for pair in mesh.connectivity.chunks_exact(2) {
        let (ra, rb) = (find(&mut parent, pair[0]), find(&mut parent, pair[1]));
        if ra != rb {
            // Union toward the smaller root: component roots stay the
            // lowest node index they contain.
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            parent[hi as usize] = lo;
        }
    }
    let mut strut_count: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for pair in mesh.connectivity.chunks_exact(2) {
        *strut_count.entry(find(&mut parent, pair[0])).or_default() += 1;
    }
    let Some(keep_root) = strut_count
        .iter()
        .max_by_key(|(root, count)| (**count, std::cmp::Reverse(**root)))
        .map(|(root, _)| *root)
    else {
        return Ok(mesh.clone());
    };
    let keep: Vec<bool> = mesh
        .connectivity
        .chunks_exact(2)
        .map(|pair| find(&mut parent, pair[0]) == keep_root)
        .collect();
    filter_elements(mesh, &keep)
}

/// Concatenate same-kind meshes: nodes, elements, and fields in input
/// order (connectivity offset per part). Only fields present in *every*
/// part — same name, same component count, same container — survive;
/// the rest are dropped rather than padded with invented values. Field
/// order follows the first part.
pub fn concat_meshes(meshes: &[&FeaMesh]) -> Result<FeaMesh, String> {
    let Some(first) = meshes.first() else {
        return Err("nothing to merge (no meshes)".to_string());
    };
    let kind = first.element_kind;
    if let Some(other) = meshes.iter().find(|m| m.element_kind != kind) {
        return Err(format!(
            "cannot merge meshes of different element kinds ({kind:?} and {:?})",
            other.element_kind
        ));
    }

    let mut node_positions = Vec::new();
    let mut connectivity = Vec::new();
    for m in meshes {
        let offset = (node_positions.len() / 3) as u32;
        node_positions.extend_from_slice(&m.node_positions);
        connectivity.extend(m.connectivity.iter().map(|&n| n + offset));
    }

    let common = |pick: fn(&FeaMesh) -> &Vec<FeaField>| -> Vec<FeaField> {
        pick(first)
            .iter()
            .filter(|f| {
                meshes.iter().all(|m| {
                    pick(m)
                        .iter()
                        .any(|g| g.name == f.name && g.components == f.components)
                })
            })
            .map(|f| FeaField {
                name: f.name.clone(),
                components: f.components,
                data: meshes
                    .iter()
                    .flat_map(|m| {
                        pick(m)
                            .iter()
                            .find(|g| g.name == f.name)
                            .expect("filtered to common fields")
                            .data
                            .iter()
                            .copied()
                    })
                    .collect(),
            })
            .collect()
    };

    let mesh = FeaMesh {
        element_kind: kind,
        node_positions,
        connectivity,
        node_fields: common(|m| &m.node_fields),
        element_fields: common(|m| &m.element_fields),
    };
    mesh.validate()?;
    Ok(mesh)
}

/// Weld nodes lying within `tolerance` of each other into their first
/// occurrence (positions and node fields both take the representative's
/// values — the weld is for stitching coincident duplicates, not for
/// moving geometry). Elements are remapped; Bar2 self-loops and exact
/// duplicate elements (same node set) are dropped, with their element
/// fields, and unreferenced nodes compact away.
pub fn weld_coincident_nodes(mesh: &FeaMesh, tolerance: f64) -> Result<FeaMesh, String> {
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(format!("weld tolerance must be positive, got {tolerance}"));
    }
    // Representative per node: the first node within tolerance, via a
    // tolerance-sized grid hash (27 buckets cover every candidate).
    let mut buckets: std::collections::HashMap<[i64; 3], Vec<u32>> =
        std::collections::HashMap::new();
    let mut rep = vec![u32::MAX; mesh.node_count()];
    for n in 0..mesh.node_count() {
        let p = mesh.node_position(n);
        let key: [i64; 3] = core::array::from_fn(|a| (p[a] / tolerance).floor() as i64);
        'search: for dz in -1..=1i64 {
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let k = [key[0] + dx, key[1] + dy, key[2] + dz];
                    if let Some(ids) = buckets.get(&k) {
                        for &id in ids {
                            let q = mesh.node_position(id as usize);
                            let d2: f64 = (0..3).map(|a| (p[a] - q[a]).powi(2)).sum();
                            if d2 < tolerance * tolerance {
                                rep[n] = id;
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        if rep[n] == u32::MAX {
            rep[n] = n as u32;
            buckets.entry(key).or_default().push(n as u32);
        }
    }

    // Surviving elements: remapped, minus Bar2 self-loops and exact
    // duplicates by (sorted) node set.
    let mut seen: std::collections::HashSet<Vec<u32>> = std::collections::HashSet::new();
    let mut keep_elements: Vec<u32> = Vec::new();
    let mut remapped: Vec<u32> = Vec::new();
    for e in 0..mesh.element_count() {
        let nodes: Vec<u32> = mesh.element(e).iter().map(|&n| rep[n as usize]).collect();
        if mesh.element_kind == volumetric_abi::fea::FeaElementKind::Bar2 && nodes[0] == nodes[1] {
            continue;
        }
        let mut key = nodes.clone();
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        keep_elements.push(e as u32);
        remapped.extend(nodes);
    }

    // Compact representative nodes on first use.
    let mut node_remap = vec![u32::MAX; mesh.node_count()];
    let mut kept_nodes: Vec<u32> = Vec::new();
    let mut connectivity = Vec::with_capacity(remapped.len());
    for node in remapped {
        if node_remap[node as usize] == u32::MAX {
            node_remap[node as usize] = kept_nodes.len() as u32;
            kept_nodes.push(node);
        }
        connectivity.push(node_remap[node as usize]);
    }
    let mesh = FeaMesh {
        element_kind: mesh.element_kind,
        node_positions: gather(&mesh.node_positions, 3, &kept_nodes),
        connectivity,
        node_fields: gather_fields(&mesh.node_fields, &kept_nodes),
        element_fields: gather_fields(&mesh.element_fields, &keep_elements),
    };
    mesh.validate()?;
    Ok(mesh)
}

/// Path-halving union-find lookup.
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Entry-major gather: the `width`-wide rows of `data` selected by
/// `indices`, in order.
fn gather(data: &[f64], width: usize, indices: &[u32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(indices.len() * width);
    for &i in indices {
        let base = i as usize * width;
        out.extend_from_slice(&data[base..base + width]);
    }
    out
}

fn gather_fields(fields: &[FeaField], indices: &[u32]) -> Vec<FeaField> {
    fields
        .iter()
        .map(|f| FeaField {
            name: f.name.clone(),
            components: f.components,
            data: gather(&f.data, f.components, indices),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::fea::FeaElementKind;

    fn bar_mesh(positions: Vec<f64>, connectivity: Vec<u32>) -> FeaMesh {
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: positions,
            connectivity,
            node_fields: vec![],
            element_fields: vec![],
        }
    }

    #[test]
    fn filtering_compacts_nodes_and_slices_fields() {
        let mut mesh = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            connectivity: vec![0, 1, 2],
            node_fields: vec![FeaField {
                name: "weight".to_string(),
                components: 1,
                data: vec![10.0, 20.0, 30.0],
            }],
            element_fields: vec![FeaField {
                name: "tag".to_string(),
                components: 1,
                data: vec![1.0, 2.0, 3.0],
            }],
        };
        let kept = filter_elements(&mesh, &[true, false, true]).unwrap();
        assert_eq!(kept.node_count(), 2);
        assert_eq!(kept.element_count(), 2);
        assert_eq!(kept.node_position(1), [2.0, 0.0, 0.0]);
        assert_eq!(kept.node_fields[0].data, vec![10.0, 30.0]);
        assert_eq!(kept.element_fields[0].data, vec![1.0, 3.0]);

        // Bar2: dropping an element drops its now-unreferenced node.
        mesh.element_kind = FeaElementKind::Bar2;
        mesh.connectivity = vec![0, 1, 1, 2];
        mesh.element_fields[0].data = vec![1.0, 2.0];
        let kept = filter_elements(&mesh, &[true, false]).unwrap();
        assert_eq!(kept.node_count(), 2);
        assert_eq!(kept.connectivity, vec![0, 1]);
        assert_eq!(kept.node_fields[0].data, vec![10.0, 20.0]);

        assert!(filter_elements(&mesh, &[true]).is_err(), "wrong flag count");
    }

    #[test]
    fn welding_collapses_short_struts_into_joints() {
        // A "dumbbell": two long struts joined by a tiny middle edge.
        //   0 --(1.0)-- 1 --(0.01)-- 2 --(1.0)-- 3
        let mesh = bar_mesh(
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.01, 0.0, 0.0, //
                2.01, 0.0, 0.0,
            ],
            vec![0, 1, 1, 2, 2, 3],
        );
        let welded = weld_short_bars(&mesh, &|_| 0.05).unwrap();
        assert_eq!(welded.element_count(), 2, "tiny middle edge contracts");
        assert_eq!(welded.node_count(), 3, "nodes 1 and 2 merge");
        // The joint sits at the pair's centroid.
        assert!(
            (0..welded.node_count())
                .map(|n| welded.node_position(n))
                .any(|q| (q[0] - 1.005).abs() < 1e-12 && q[1] == 0.0 && q[2] == 0.0)
        );
    }

    #[test]
    fn welding_dedupes_parallel_struts() {
        // A sliver triangle: welding its short edge leaves the two long
        // edges connecting the same pair of joints; only one survives.
        let mesh = bar_mesh(
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 0.01, 0.0,
            ],
            vec![0, 1, 1, 2, 2, 0],
        );
        let welded = weld_short_bars(&mesh, &|_| 0.05).unwrap();
        assert_eq!(welded.element_count(), 1, "parallel duplicates dedup");
        assert_eq!(welded.node_count(), 2);
    }

    #[test]
    fn welding_cascades_through_chains() {
        // Three 0.04 edges chain into a 0.12 cluster: each merge pulls the
        // next edge under the threshold, so the whole chain becomes one
        // joint even though welding is threshold-per-pass.
        let mesh = bar_mesh(
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.04, 0.0, 0.0, //
                1.08, 0.0, 0.0, //
                1.12, 0.0, 0.0, //
                2.12, 0.0, 0.0,
            ],
            vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        );
        let welded = weld_short_bars(&mesh, &|_| 0.05).unwrap();
        assert_eq!(
            welded.element_count(),
            2,
            "the chain contracts to one joint"
        );
        assert_eq!(welded.node_count(), 3);
    }

    #[test]
    fn welding_carries_fields_and_per_element_thresholds() {
        // The dumbbell again, now with a node field and per-strut radii:
        // only the middle strut's radius makes it weldable.
        let mut mesh = bar_mesh(
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.01, 0.0, 0.0, //
                2.01, 0.0, 0.0,
            ],
            vec![0, 1, 1, 2, 2, 3],
        );
        mesh.node_fields.push(FeaField {
            name: "weight".to_string(),
            components: 1,
            data: vec![1.0, 2.0, 4.0, 8.0],
        });
        mesh.element_fields.push(FeaField {
            name: "radius".to_string(),
            components: 1,
            data: vec![0.001, 0.05, 0.001],
        });
        let radii = mesh.element_fields[0].data.clone();
        let welded = weld_short_bars(&mesh, &|e| radii[e]).unwrap();
        assert_eq!(welded.element_count(), 2);
        assert_eq!(welded.node_count(), 3);
        // The merged joint averages the merged nodes' field values...
        let weights = &welded.node_fields[0].data;
        assert!(weights.iter().any(|&w| (w - 3.0).abs() < 1e-12));
        // ...and the surviving struts keep their own radii.
        assert_eq!(welded.element_fields[0].data, vec![0.001, 0.001]);
    }

    #[test]
    fn concatenation_offsets_and_intersects_fields() {
        let a = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            connectivity: vec![0, 1],
            node_fields: vec![
                FeaField {
                    name: "weight".to_string(),
                    components: 1,
                    data: vec![1.0, 2.0],
                },
                FeaField {
                    name: "only_in_a".to_string(),
                    components: 1,
                    data: vec![9.0, 9.0],
                },
            ],
            element_fields: vec![],
        };
        let b = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: vec![5.0, 0.0, 0.0],
            connectivity: vec![0],
            node_fields: vec![FeaField {
                name: "weight".to_string(),
                components: 1,
                data: vec![3.0],
            }],
            element_fields: vec![],
        };
        let merged = concat_meshes(&[&a, &b]).unwrap();
        assert_eq!(merged.node_count(), 3);
        assert_eq!(merged.connectivity, vec![0, 1, 2]);
        assert_eq!(
            merged.node_fields.len(),
            1,
            "only the common field survives"
        );
        assert_eq!(merged.node_fields[0].data, vec![1.0, 2.0, 3.0]);

        let bars = bar_mesh(vec![0.0; 6], vec![0, 1]);
        let err = concat_meshes(&[&a, &bars]).unwrap_err();
        assert!(err.contains("different element kinds"), "{err}");
        assert!(concat_meshes(&[]).is_err());
    }

    #[test]
    fn coincident_welding_stitches_and_dedupes() {
        // Two 2-strut chains sharing a coincident middle joint (within
        // tolerance), plus an exact duplicate strut.
        let mesh = bar_mesh(
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 1e-9, 0.0, // coincides with node 1
                2.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, // coincides with node 0
            ],
            vec![0, 1, 2, 3, 4, 1],
        );
        let welded = weld_coincident_nodes(&mesh, 1e-6).unwrap();
        // Node 2 -> 1 and node 4 -> 0; strut (4,1) becomes duplicate of
        // (0,1) and drops.
        assert_eq!(welded.node_count(), 3);
        assert_eq!(welded.element_count(), 2);

        // Point clouds dedupe coincident points the same way.
        let cloud = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: vec![0.0, 0.0, 0.0, 5e-7, 0.0, 0.0, 3.0, 0.0, 0.0],
            connectivity: vec![0, 1, 2],
            node_fields: vec![FeaField {
                name: "weight".to_string(),
                components: 1,
                data: vec![1.0, 2.0, 3.0],
            }],
            element_fields: vec![],
        };
        let welded = weld_coincident_nodes(&cloud, 1e-6).unwrap();
        assert_eq!(welded.element_count(), 2);
        // First occurrence wins, fields follow it.
        assert_eq!(welded.node_fields[0].data, vec![1.0, 3.0]);

        assert!(weld_coincident_nodes(&cloud, 0.0).is_err());
    }

    #[test]
    fn component_pruning_keeps_the_biggest_piece() {
        // A 3-strut path and a disjoint single strut.
        let mesh = bar_mesh(
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                2.0, 0.0, 0.0, //
                3.0, 0.0, 0.0, //
                10.0, 0.0, 0.0, //
                11.0, 0.0, 0.0,
            ],
            vec![0, 1, 1, 2, 2, 3, 4, 5],
        );
        let pruned = largest_bar_component(&mesh).unwrap();
        assert_eq!(pruned.element_count(), 3);
        assert_eq!(pruned.node_count(), 4);
        assert!(
            (0..pruned.node_count()).all(|n| pruned.node_position(n)[0] < 5.0),
            "island survived"
        );
    }
}
