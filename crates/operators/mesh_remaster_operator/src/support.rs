//! The `support` requirement: printable along the build axis on a
//! printer that handles overhangs but not hooks (resin).
//!
//! A print fails exactly where a region appears in a slice unattached to
//! already-cured material of its own piece. On the strut graph that is
//! sub-level-set connectivity of the height function: sweeping the build
//! plane upward, a node is attached iff it connects to the bed through
//! material at or below its own height. An ascending strut transmits
//! support upward at any angle (its lower end prints first); a
//! descending strut cannot — its far tip appears in an earlier slice
//! than its anchor and cures floating. `max_descent` degrees of
//! per-strut slack model in-slice cohesion (a quasi-horizontal strut
//! appears attached to its supported end in effectively one slice); the
//! slack deliberately does not compound along chains — a chain of
//! slightly-descending struts is a hook, not a fan.
//!
//! Formally each node gets a *support time* `tau` — the height at which
//! its attachment to the bed completes — propagated Dijkstra-style from
//! the bed seeds in ascending `tau` order (the relaxation is monotone in
//! `tau`, so a binary heap gives the exact minimum):
//!
//! - fix `"raise"`: crossing a strut from a supported node `m`, the far
//!   node may sit no lower than `tau(m) - tan(max_descent) * reach`
//!   (its horizontal reach), so its height becomes
//!   `z' = max(z, tau(m) - tan(max_descent) * reach)` and its support
//!   time `tau = max(tau(m), z')`. Nodes rise straight up (x/y kept —
//!   the descent is projected out of the strut, the strut's horizontal
//!   footprint stays); a hanging hook flattens into a fan anchored where
//!   it meets supported material. The raise distances land in a `raise`
//!   node field.
//! - fix `"drop"`: heights are fixed; a strut transmits support only
//!   when the far node already satisfies the descent window
//!   (`z >= tau(m) - tan(max_descent) * reach`). Everything that never
//!   acquires a support time is removed — transitively: geometry resting
//!   on removed geometry is removed too.
//!
//! Bed seeds are the nodes within `bed_tolerance` of the mesh's extreme
//! along the build axis (`extreme: "max"` prints hanging from the top).
//! Pieces with no path to any seed cannot be raised into validity — with
//! fix `"raise"` that is an error pointing at the `connectivity`
//! requirement; fix `"drop"` removes them like the rest.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh};

use crate::{Extreme, SupportConfig, SupportFix};

/// Heap key: support time, then height (ties prefer the lower landing),
/// then node for a total order.
#[derive(PartialEq)]
struct Key {
    tau: f64,
    z: f64,
    node: u32,
}

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.tau
            .total_cmp(&other.tau)
            .then(self.z.total_cmp(&other.z))
            .then(self.node.cmp(&other.node))
    }
}

/// Enforce the requirement; returns the fixed mesh and whether any
/// element was removed (the caller re-runs connectivity if so).
pub(crate) fn enforce(
    mesh: &FeaMesh,
    config: &SupportConfig,
    has_connectivity: bool,
) -> Result<(FeaMesh, bool), String> {
    if mesh.element_kind != FeaElementKind::Bar2 {
        return Err(format!(
            "the support requirement needs a Bar2 strut mesh, got {:?} elements",
            mesh.element_kind
        ));
    }
    let axis = config.axis.index();
    let sign = match config.extreme {
        Extreme::Min => 1.0,
        Extreme::Max => -1.0,
    };
    let n = mesh.node_count();
    let positions: Vec<[f64; 3]> = (0..n).map(|i| mesh.node_position(i)).collect();
    let h: Vec<f64> = positions.iter().map(|p| sign * p[axis]).collect();

    let mut used = vec![false; n];
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];
    for pair in mesh.connectivity.chunks_exact(2) {
        adjacency[pair[0] as usize].push(pair[1]);
        adjacency[pair[1] as usize].push(pair[0]);
        used[pair[0] as usize] = true;
        used[pair[1] as usize] = true;
    }

    let hmin = (0..n)
        .filter(|&i| used[i])
        .map(|i| h[i])
        .fold(f64::INFINITY, f64::min);
    let hmax = (0..n)
        .filter(|&i| used[i])
        .map(|i| h[i])
        .fold(f64::NEG_INFINITY, f64::max);
    if !(hmin.is_finite() && hmax.is_finite()) {
        return Err("the mesh has no usable nodes along the build axis".to_string());
    }
    let bed_tolerance = if config.bed_tolerance > 0.0 {
        config.bed_tolerance
    } else {
        (hmax - hmin) * 1e-4
    };
    let slope = config.max_descent.to_radians().tan();
    let raise_mode = config.fix == SupportFix::Raise;

    // Horizontal reach of a strut: distance in the two non-axis coords.
    let reach = |a: usize, b: usize| -> f64 {
        let mut s = 0.0;
        for c in 0..3 {
            if c != axis {
                let d = positions[a][c] - positions[b][c];
                s += d * d;
            }
        }
        s.sqrt()
    };

    let mut tau = vec![f64::INFINITY; n];
    let mut z = vec![f64::INFINITY; n];
    let mut heap: BinaryHeap<Reverse<Key>> = BinaryHeap::new();
    for i in 0..n {
        if used[i] && h[i] <= hmin + bed_tolerance {
            tau[i] = h[i];
            z[i] = h[i];
            heap.push(Reverse(Key {
                tau: h[i],
                z: h[i],
                node: i as u32,
            }));
        }
    }
    while let Some(Reverse(key)) = heap.pop() {
        let i = key.node as usize;
        if key.tau > tau[i] || (key.tau == tau[i] && key.z > z[i]) {
            continue; // stale
        }
        for &j in &adjacency[i] {
            let j = j as usize;
            let floor = key.tau - slope * reach(i, j);
            let z_cand = if raise_mode {
                h[j].max(floor)
            } else {
                if h[j] < floor {
                    continue; // the strut cannot transmit support here
                }
                h[j]
            };
            let tau_cand = key.tau.max(z_cand);
            if tau_cand < tau[j] || (tau_cand == tau[j] && z_cand < z[j]) {
                tau[j] = tau_cand;
                z[j] = z_cand;
                heap.push(Reverse(Key {
                    tau: tau_cand,
                    z: z_cand,
                    node: j as u32,
                }));
            }
        }
    }

    match config.fix {
        SupportFix::Raise => {
            let unreachable = (0..n).filter(|&i| used[i] && !tau[i].is_finite()).count();
            if unreachable > 0 {
                let hint = if has_connectivity {
                    "they sit farther than connectivity.max_new_strut from the body"
                } else {
                    "add a connectivity requirement to tie the pieces together, or use \
                     support fix \"drop\""
                };
                return Err(format!(
                    "{unreachable} nodes have no path to the build bed at all — {hint}"
                ));
            }
            let mut out = mesh.clone();
            let mut raise = vec![0.0; n];
            for i in 0..n {
                if used[i] && z[i] > h[i] {
                    raise[i] = z[i] - h[i];
                    out.node_positions[i * 3 + axis] = sign * z[i];
                }
            }
            match out.node_fields.iter_mut().find(|f| f.name == "raise") {
                Some(f) if f.components == 1 => f.data = raise,
                Some(_) => {
                    return Err(
                        "the input already has a non-scalar 'raise' node field".to_string()
                    );
                }
                None => out.node_fields.push(FeaField {
                    name: "raise".to_string(),
                    components: 1,
                    data: raise,
                }),
            }
            out.validate()?;
            Ok((out, false))
        }
        SupportFix::Drop => {
            // A kept strut must transmit support in at least one
            // direction — an edge unusable both ways hangs outside its
            // own cohesion window even when both endpoints are held up
            // through other paths. (At max_descent 0 the disjunct is
            // free: tau == h on supported nodes, and one end is always
            // the lower one.)
            let usable = |a: usize, b: usize| h[b] >= tau[a] - slope * reach(a, b);
            let keep: Vec<bool> = (0..mesh.element_count())
                .map(|e| {
                    let pair = mesh.element(e);
                    let (a, b) = (pair[0] as usize, pair[1] as usize);
                    tau[a].is_finite()
                        && tau[b].is_finite()
                        && (usable(a, b) || usable(b, a))
                })
                .collect();
            if keep.iter().all(|&k| k) {
                return Ok((mesh.clone(), false));
            }
            if !keep.contains(&true) {
                return Err(
                    "the support requirement removed every strut — is the bed on the right \
                     extreme of the build axis?"
                        .to_string(),
                );
            }
            Ok((mesh_edit_core::filter_elements(mesh, &keep)?, true))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::bar_mesh;
    use crate::{BuildAxis, SupportConfig};

    fn raise_field(mesh: &FeaMesh) -> &[f64] {
        &mesh
            .node_fields
            .iter()
            .find(|f| f.name == "raise")
            .expect("raise field")
            .data
    }

    /// Bed - column - hook - riser: the hook node hangs below its
    /// neighbors and must rise to the lower one's height.
    fn hook_chain() -> FeaMesh {
        bar_mesh(
            &[
                [0.0, 0.0, 0.0], // bed
                [0.0, 0.0, 1.0], // top of column
                [1.0, 0.0, 0.4], // hook: descends off the column top
                [2.0, 0.0, 1.5], // riser continuing up from the hook
            ],
            &[[0, 1], [1, 2], [2, 3]],
            0.05,
        )
    }

    #[test]
    fn hooks_rise_to_their_support_time() {
        let (out, dropped) =
            enforce(&hook_chain(), &SupportConfig::default(), false).unwrap();
        assert!(!dropped);
        // The hook rises to the column top's height; x/y stay.
        let p = out.node_position(2);
        assert_eq!(p[0], 1.0);
        assert!((p[2] - 1.0).abs() < 1e-12, "hook raised to 1.0: {p:?}");
        // The riser was already above its (raised) support: untouched.
        assert_eq!(out.node_position(3)[2], 1.5);
        assert_eq!(raise_field(&out), &[0.0, 0.0, 0.6, 0.0]);
    }

    #[test]
    fn descent_slack_tolerates_shallow_hooks() {
        // tan(45) x reach 1.0 allows a full unit of descent: the hook at
        // 0.4 is within 1.0 - 1.0 = 0.0 <= 0.4, so nothing moves.
        let config = SupportConfig {
            max_descent: 45.0,
            ..SupportConfig::default()
        };
        let (out, dropped) = enforce(&hook_chain(), &config, false).unwrap();
        assert!(!dropped);
        assert_eq!(out.node_position(2)[2], 0.4, "within the slack window");
        assert!(raise_field(&out).iter().all(|&r| r == 0.0));
    }

    #[test]
    fn slack_does_not_compound_along_chains() {
        // Two struts each descending within their own window; the second
        // measures its window from the first's support TIME (the column
        // top), not its raised height, so the chain cannot stairstep
        // down indefinitely.
        let mesh = bar_mesh(
            &[
                [0.0, 0.0, 0.0], // bed
                [0.0, 0.0, 1.0], // column top
                [1.0, 0.0, 0.7], // first descent: within tan(20) x 1.0
                [2.0, 0.0, 0.4], // second descent: window still anchors at 1.0
            ],
            &[[0, 1], [1, 2], [2, 3]],
            0.05,
        );
        let config = SupportConfig {
            max_descent: 20.0,
            ..SupportConfig::default()
        };
        let (out, _) = enforce(&mesh, &config, false).unwrap();
        let slack = 20.0f64.to_radians().tan();
        assert_eq!(out.node_position(2)[2], 0.7, "first hop within slack");
        // tau(2) = 1.0 (the anchor's support completes there), so node 3
        // may sit no lower than 1.0 - slack, NOT 0.7 - slack.
        let z3 = out.node_position(3)[2];
        assert!(
            (z3 - (1.0 - slack)).abs() < 1e-12,
            "expected {} got {z3}",
            1.0 - slack
        );
    }

    #[test]
    fn extreme_max_builds_downward() {
        let mesh = bar_mesh(
            &[
                [0.0, 0.0, 2.0], // "bed" at the top
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.6], // hook (in flipped orientation)
            ],
            &[[0, 1], [1, 2]],
            0.05,
        );
        let config = SupportConfig {
            extreme: crate::Extreme::Max,
            ..SupportConfig::default()
        };
        let (out, _) = enforce(&mesh, &config, false).unwrap();
        let p = out.node_position(2);
        assert!(
            (p[2] - 1.0).abs() < 1e-12,
            "the hook sinks to its neighbor in max mode: {p:?}"
        );
    }

    #[test]
    fn build_axis_is_configurable() {
        // The same hook chain rotated onto the x axis.
        let mesh = bar_mesh(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.4, 1.0, 0.0], // hook along x
            ],
            &[[0, 1], [1, 2]],
            0.05,
        );
        let config = SupportConfig {
            axis: BuildAxis::X,
            ..SupportConfig::default()
        };
        let (out, _) = enforce(&mesh, &config, false).unwrap();
        let p = out.node_position(2);
        assert!((p[0] - 1.0).abs() < 1e-12, "raised along x: {p:?}");
        assert_eq!(p[1], 1.0, "other coords untouched");
    }

    #[test]
    fn drop_removes_unsupported_geometry_transitively() {
        let (out, dropped) = enforce(
            &hook_chain(),
            &SupportConfig {
                fix: SupportFix::Drop,
                ..SupportConfig::default()
            },
            false,
        )
        .unwrap();
        assert!(dropped);
        // The hook fails at original heights, and the riser above it
        // dies transitively even though it ascends.
        assert_eq!(out.element_count(), 1, "only the bed column survives");
        assert_eq!(out.node_count(), 2);
    }

    /// Regression: with nonzero slack, a strut can be unusable in BOTH
    /// directions while both its endpoints stay supported through other
    /// paths — it hangs outside its own cohesion window and must drop.
    /// (Found in review: the endpoint-finiteness filter kept it.)
    #[test]
    fn drop_removes_struts_unusable_in_both_directions() {
        // Two bed columns hold up long shallow arms; the arms' far ends
        // j (tau 0.9, h 0.3) and a (tau 1.2, h 0.5) are both supported,
        // but the short steep strut j-a (reach 0.1) violates the window
        // both ways at 20 degrees: h(a) < tau(j) - slack and
        // h(j) < tau(a) - slack.
        let nodes = [
            [4.0, 0.0, 0.0],  // bed 1
            [4.0, 0.0, 0.9],  // top of column 1
            [0.1, 0.0, 0.3],  // j: supported via the shallow arm from 1
            [0.0, 0.0, 0.5],  // a: supported via the shallow arm from 4
            [-4.0, 0.0, 1.2], // top of column 2
            [-4.0, 0.0, 0.0], // bed 2
        ];
        let bars = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]];
        let mesh = bar_mesh(&nodes, &bars, 0.05);
        let config = SupportConfig {
            max_descent: 20.0,
            fix: SupportFix::Drop,
            ..SupportConfig::default()
        };
        let (out, dropped) = enforce(&mesh, &config, false).unwrap();

        assert!(dropped);
        assert_eq!(out.element_count(), 4, "exactly the j-a strut drops");
        // Both arms and both columns survive; only the steep short strut
        // between the two late-supported nodes is gone.
        for e in 0..out.element_count() {
            let pair = out.element(e);
            let (pa, pb) = (
                out.node_position(pair[0] as usize),
                out.node_position(pair[1] as usize),
            );
            let short = crate::dist(pa, pb) < 0.5;
            assert!(!short, "the j-a strut must not survive: {pa:?} {pb:?}");
        }
        // Raise mode agrees the input was invalid: it moves j upward.
        let raise_config = SupportConfig {
            max_descent: 20.0,
            ..SupportConfig::default()
        };
        let (raised, _) = enforce(&mesh, &raise_config, false).unwrap();
        assert!(
            raised.node_position(3)[2] > 0.5,
            "raise mode agrees the strut violated its window: it lifts a \
             into it instead (the short path's tau 0.9 beats the arm's 1.2)"
        );
    }

    #[test]
    fn unreachable_pieces_error_in_raise_mode() {
        let mesh = bar_mesh(
            &[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [5.0, 0.0, 3.0], // floating fragment, no path to bed
                [5.0, 0.0, 4.0],
            ],
            &[[0, 1], [2, 3]],
            0.05,
        );
        let err = enforce(&mesh, &SupportConfig::default(), false).unwrap_err();
        assert!(err.contains("connectivity"), "unexpected error: {err}");
        // Drop mode removes them instead.
        let (out, dropped) = enforce(
            &mesh,
            &SupportConfig {
                fix: SupportFix::Drop,
                ..SupportConfig::default()
            },
            false,
        )
        .unwrap();
        assert!(dropped);
        assert_eq!(out.element_count(), 1);
    }

    #[test]
    fn raising_is_idempotent() {
        let (once, _) = enforce(&hook_chain(), &SupportConfig::default(), false).unwrap();
        let (twice, dropped) = enforce(&once, &SupportConfig::default(), false).unwrap();
        assert!(!dropped);
        assert!(
            raise_field(&twice).iter().all(|&r| r == 0.0),
            "a raised mesh is already valid: {:?}",
            raise_field(&twice)
        );
        assert_eq!(once.node_positions, twice.node_positions);
    }

    #[test]
    fn flat_meshes_seed_everywhere() {
        // Every node at the same height: hmax == hmin, everything seeds,
        // nothing rises, nothing drops.
        let mesh = bar_mesh(
            &[[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
            &[[0, 1], [1, 2]],
            0.05,
        );
        let (out, dropped) = enforce(&mesh, &SupportConfig::default(), false).unwrap();
        assert!(!dropped);
        assert!(raise_field(&out).iter().all(|&r| r == 0.0));
    }
}
