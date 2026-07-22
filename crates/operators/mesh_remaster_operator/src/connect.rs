//! The `connectivity` requirement: no floating fragments.
//!
//! The surface pass already re-drapes dropped arcs (see `drape.rs`);
//! this pass handles what remains at the mesh level — pieces severed
//! upstream (a clip, a trim) or split by `support: "drop"`, where no
//! dropped arc exists to reuse. Fix `"reconnect"` synthesizes straight
//! ties between the nearest node pairs of separate components, shortest
//! first (a spanning forest over components), as long as a tie stays
//! within `max_new_strut` x the median strut length; components nothing
//! can reach are pruned. Fix `"prune"` keeps the largest component only.
//! Synthesized ties copy their element fields from a strut incident to
//! the orphan-side node (`skin` forced to 0 — a tie runs through the
//! bulk, not along the surface) and are flagged 1.0 in the `tie` field.

use std::collections::HashMap;

use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh};

use crate::{ConnectivityConfig, ConnectivityFix, dist, uf_find, uf_union};

pub(crate) fn enforce(mesh: &FeaMesh, config: &ConnectivityConfig) -> Result<FeaMesh, String> {
    if mesh.element_kind != FeaElementKind::Bar2 {
        return Err(format!(
            "the connectivity requirement needs a Bar2 strut mesh, got {:?} elements",
            mesh.element_kind
        ));
    }
    let node_count = mesh.node_count();
    let mut parent: Vec<u32> = (0..node_count as u32).collect();
    for pair in mesh.connectivity.chunks_exact(2) {
        uf_union(&mut parent, pair[0], pair[1]);
    }
    let mut strut_count: HashMap<u32, usize> = HashMap::new();
    for pair in mesh.connectivity.chunks_exact(2) {
        *strut_count
            .entry(uf_find(&mut parent, pair[0]))
            .or_default() += 1;
    }
    if strut_count.len() <= 1 {
        return Ok(mesh.clone());
    }
    if config.fix == ConnectivityFix::Prune {
        return mesh_edit_core::largest_bar_component(mesh);
    }

    let mut out = mesh.clone();
    if config.max_new_strut > 0.0 {
        let ties = plan_ties(&out, &mut parent, &strut_count, config.max_new_strut);
        if !ties.is_empty() {
            append_ties(&mut out, &ties)?;
            out.validate()?;
        }
    }

    // Whatever no tie could reach is pruned — a floating fragment is
    // singular for FEA and falls off a print either way.
    let mut check: Vec<u32> = (0..out.node_count() as u32).collect();
    let mut roots = std::collections::HashSet::new();
    for pair in out.connectivity.chunks_exact(2) {
        uf_union(&mut check, pair[0], pair[1]);
    }
    for pair in out.connectivity.chunks_exact(2) {
        roots.insert(uf_find(&mut check, pair[0]));
    }
    if roots.len() > 1 {
        out = mesh_edit_core::largest_bar_component(&out)?;
    }
    Ok(out)
}

/// Pick the ties to synthesize: for every node outside the largest
/// component, find its nearest node in any other component (spatial
/// hash, cells the size of the reach limit), then join components
/// shortest-first — and repeat until a pass plans nothing (Borůvka
/// phases). One pass is not enough: two orphans that merge with each
/// other still need the *merged* component's best outgoing edge, which
/// only the rescan finds. Returns (orphan-side node, other node) pairs.
fn plan_ties(
    mesh: &FeaMesh,
    parent: &mut [u32],
    strut_count: &HashMap<u32, usize>,
    max_new_strut: f64,
) -> Vec<(u32, u32)> {
    let limit = max_new_strut * median_strut_length(mesh);
    if !(limit.is_finite() && limit > 0.0) {
        return Vec::new();
    }
    let mut main_root = strut_count
        .iter()
        .max_by_key(|(root, count)| (**count, std::cmp::Reverse(**root)))
        .map(|(root, _)| *root)
        .unwrap();

    // Only element-referenced nodes participate (stray nodes have no
    // donor strut for the tie's fields).
    let mut used = vec![false; mesh.node_count()];
    for &n in &mesh.connectivity {
        used[n as usize] = true;
    }
    let positions: Vec<[f64; 3]> = (0..mesh.node_count())
        .map(|n| mesh.node_position(n))
        .collect();
    let key = |p: [f64; 3]| -> [i64; 3] { core::array::from_fn(|c| (p[c] / limit).floor() as i64) };
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for n in 0..mesh.node_count() as u32 {
        if used[n as usize] {
            grid.entry(key(positions[n as usize])).or_default().push(n);
        }
    }

    let mut ties = Vec::new();
    loop {
        // The main component's root can change as orphans union into it.
        main_root = uf_find(parent, main_root);
        // Each orphan node's nearest foreign neighbor within the limit;
        // every current component's globally best outgoing edge is
        // among these, so shortest-first joining is one Borůvka phase.
        let mut candidates: Vec<(f64, u32, u32)> = Vec::new();
        for n in 0..mesh.node_count() as u32 {
            if !used[n as usize] || uf_find(parent, n) == main_root {
                continue;
            }
            let root = uf_find(parent, n);
            let cell = key(positions[n as usize]);
            let mut best: Option<(f64, u32)> = None;
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        let Some(bucket) = grid.get(&[cell[0] + dx, cell[1] + dy, cell[2] + dz])
                        else {
                            continue;
                        };
                        for &m in bucket {
                            if uf_find(parent, m) == root {
                                continue;
                            }
                            let d = dist(positions[n as usize], positions[m as usize]);
                            if d <= limit && best.is_none_or(|(bd, _)| d < bd) {
                                best = Some((d, m));
                            }
                        }
                    }
                }
            }
            if let Some((d, m)) = best {
                candidates.push((d, n, m));
            }
        }
        candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
        let planned = ties.len();
        for &(_, n, m) in &candidates {
            if uf_union(parent, n, m) {
                ties.push((n, m));
            }
        }
        if ties.len() == planned {
            return ties;
        }
    }
}

/// Append synthesized tie struts, copying element fields from a strut
/// incident to the orphan-side node; `skin` is forced to 0 and `tie`
/// (created if absent) to 1.
fn append_ties(mesh: &mut FeaMesh, ties: &[(u32, u32)]) -> Result<(), String> {
    let element_count = mesh.element_count();
    let mut first_elem = vec![u32::MAX; mesh.node_count()];
    for e in 0..element_count {
        for &n in mesh.element(e) {
            if first_elem[n as usize] == u32::MAX {
                first_elem[n as usize] = e as u32;
            }
        }
    }
    match mesh.element_fields.iter().find(|f| f.name == "tie") {
        Some(f) if f.components != 1 => {
            return Err("the input already has a non-scalar 'tie' element field".to_string());
        }
        Some(_) => {}
        None => mesh.element_fields.push(FeaField {
            name: "tie".to_string(),
            components: 1,
            data: vec![0.0; element_count],
        }),
    }
    for &(n, m) in ties {
        mesh.connectivity.extend([n, m]);
        let donor = first_elem[n as usize] as usize;
        for f in &mut mesh.element_fields {
            if f.components == 1 && (f.name == "tie" || f.name == "skin") {
                f.data.push(if f.name == "tie" { 1.0 } else { 0.0 });
            } else {
                let base = donor * f.components;
                for k in 0..f.components {
                    let v = f.data[base + k];
                    f.data.push(v);
                }
            }
        }
    }
    Ok(())
}

fn median_strut_length(mesh: &FeaMesh) -> f64 {
    let mut lens: Vec<f64> = (0..mesh.element_count())
        .map(|e| {
            let pair = mesh.element(e);
            dist(
                mesh.node_position(pair[0] as usize),
                mesh.node_position(pair[1] as usize),
            )
        })
        .collect();
    if lens.is_empty() {
        return 0.0;
    }
    let k = lens.len() / 2;
    lens.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    lens[k]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConnectivityConfig;
    use crate::tests::{bar_mesh, component_count};

    #[test]
    fn connected_meshes_pass_through() {
        let mesh = bar_mesh(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            &[[0, 1], [1, 2]],
            0.05,
        );
        let out = enforce(&mesh, &ConnectivityConfig::default()).unwrap();
        assert_eq!(out.element_count(), 2);
        assert!(
            out.element_fields.iter().all(|f| f.name != "tie"),
            "nothing to tie, no field added"
        );
    }

    #[test]
    fn synthesis_ties_nearby_components_and_copies_fields() {
        // Two two-strut chains, 0.15 apart; median strut 0.1, default
        // limit 1.5 x 0.1 = 0.15 reaches exactly.
        let nodes = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.35, 0.0, 0.0],
        ];
        let mesh = bar_mesh(&nodes, &[[0, 1], [2, 3]], 0.05);
        let out = enforce(&mesh, &ConnectivityConfig::default()).unwrap();

        assert_eq!(component_count(&out), 1);
        assert_eq!(out.element_count(), 3, "one synthesized tie");
        let tie = out.element_fields.iter().find(|f| f.name == "tie").unwrap();
        assert_eq!(tie.data, vec![0.0, 0.0, 1.0]);
        let radius = out
            .element_fields
            .iter()
            .find(|f| f.name == "radius")
            .unwrap();
        assert_eq!(radius.data[2], 0.05, "tie copies the donor radius");
        // The tie spans the gap: nodes 1 and 2.
        let pair = out.element(2);
        let (a, b) = (pair[0].min(pair[1]), pair[0].max(pair[1]));
        assert_eq!((a, b), (1, 2));
    }

    #[test]
    fn unreachable_components_are_pruned() {
        // The far chain sits 10 units away — beyond any tie — and the
        // small nearby fragment reconnects; the far one is pruned.
        let nodes = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.32, 0.0, 0.0],
            [0.42, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
        ];
        let mesh = bar_mesh(&nodes, &[[0, 1], [1, 2], [3, 4], [5, 6]], 0.05);
        let out = enforce(&mesh, &ConnectivityConfig::default()).unwrap();

        assert_eq!(component_count(&out), 1);
        assert_eq!(out.element_count(), 4, "chain + fragment + tie, far pruned");
        assert!(
            (0..out.node_count()).all(|n| out.node_position(n)[0] < 1.0),
            "the far chain is gone"
        );
    }

    /// Regression: one planning pass starves chains — after two orphans
    /// merge with each other, the merged component's best outgoing edge
    /// toward the main body only exists in a rescan. (Found in review:
    /// the single-phase version pruned reachable geometry here.)
    #[test]
    fn orphan_chains_reach_the_main_body_across_phases() {
        // Main: four unit struts along y. Orphan A at x=0.9, orphan B at
        // x=1.7; with limit 1.0, A's nearest foreign node is B (0.8),
        // not main (0.9), so phase one merges A+B and only phase two
        // ties the pair back to main.
        let nodes = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0],
            [1.7, 0.0, 0.0],
            [1.7, 1.0, 0.0],
        ];
        let bars = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [7, 8]];
        let mesh = bar_mesh(&nodes, &bars, 0.05);
        let config = ConnectivityConfig {
            max_new_strut: 1.0,
            ..ConnectivityConfig::default()
        };
        let out = enforce(&mesh, &config).unwrap();

        assert_eq!(component_count(&out), 1);
        assert_eq!(
            out.element_count(),
            8,
            "all six struts plus two ties — nothing reachable may be pruned"
        );
    }

    #[test]
    fn prune_keeps_the_largest_component_only() {
        let nodes = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.35, 0.0, 0.0],
        ];
        let mesh = bar_mesh(&nodes, &[[0, 1], [1, 2], [3, 4]], 0.05);
        let config = ConnectivityConfig {
            fix: ConnectivityFix::Prune,
            ..ConnectivityConfig::default()
        };
        let out = enforce(&mesh, &config).unwrap();
        assert_eq!(out.element_count(), 2, "only the two-strut chain survives");
        assert_eq!(component_count(&out), 1);
    }

    #[test]
    fn zero_reach_never_synthesizes() {
        let nodes = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ];
        let mesh = bar_mesh(&nodes, &[[0, 1], [2, 3]], 0.05);
        let config = ConnectivityConfig {
            max_new_strut: 0.0,
            ..ConnectivityConfig::default()
        };
        let out = enforce(&mesh, &config).unwrap();
        assert_eq!(out.element_count(), 1, "no tie, the tied component pruned");
        assert_eq!(component_count(&out), 1);
    }
}
