//! Two-level additive Schwarz preconditioner for Bar2 frame lattices —
//! the scale/parallelism prototype.
//!
//! Motivation (measured on the strut example at 1000x stiffness contrast):
//! Jacobi needs ~2k CG iterations per contact solve and block-Jacobi only
//! shaves 1.5x, because the conditioning pathology is *global* — soft and
//! stiff column regions in series create long-wavelength error modes no
//! local preconditioner can touch. The fine SpMV is also memory-bandwidth
//! bound, so threading it tops out around 1.2x. This preconditioner attacks
//! both at once:
//!
//! - **Subdomains** (spatial boxes of ~`target_nodes` nodes, one strut layer
//!   of overlap): dense-Cholesky local solves, embarrassingly parallel with
//!   coarse granularity — cache-resident factors instead of bandwidth-bound
//!   streaming.
//! - **Coarse space** (6 rigid-body modes per subdomain, GDSW-style): a
//!   small dense solve that propagates corrections globally per iteration,
//!   killing the contrast-induced low-frequency modes.
//!
//! Prototype scope: built once per solve against the glued-face constraints
//! only; contact constraints are handled by masking the preconditioned
//! residual (a preconditioner need not be exact — CG stays correct because
//! the operator side is masked exactly). Subdomain factors are dense, which
//! bounds practical subdomain size; a sparse local solver is the production
//! path beyond ~10^6 struts.

use crate::frame::FrameModel;
use crate::{StiffnessModel, cholesky_in_place, cholesky_solve_in_place};
use rayon::prelude::*;
use volumetric_abi::fea::FeaMesh;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchwarzParams {
    /// Target nodes per subdomain box before overlap.
    pub target_nodes: usize,
    /// Fine-level solver: dense Cholesky per overlapping subdomain (true) —
    /// stronger, O(m^2) per apply and O(m^2) memory — or block-Jacobi
    /// (false) — O(n) apply, pairing the coarse correction with the
    /// cheapest local smoother.
    pub dense_local: bool,
}

impl Default for SchwarzParams {
    fn default() -> Self {
        Self {
            target_nodes: 64,
            dense_local: true,
        }
    }
}

struct Subdomain {
    /// Global dof indices covered by this (overlapping) subdomain.
    dofs: Vec<u32>,
    /// Dense Cholesky lower factor of the masked local stiffness.
    factor: Vec<f64>,
}

/// The fine (local) level of the two-level preconditioner.
enum LocalSolves {
    Dense(Vec<Subdomain>),
    /// Per-node inverted 6x6 blocks (same construction as Precond::Block,
    /// masked against the build-time constraints).
    BlockJacobi(Vec<f64>),
}

pub(crate) struct SchwarzPrecond {
    local: LocalSolves,
    /// Non-overlapping owner subdomain per node (coarse partition of unity).
    owner: Vec<u32>,
    /// Per-node rigid-body-mode matrix N (build-time constraint rows baked
    /// in — the apply-side Z must match the Z that assembled A_c).
    modes: Vec<[[f64; 6]; 6]>,
    /// Dense Cholesky factor of the coarse operator Z^T K Z (6 dofs per
    /// subdomain: rigid-body modes about the subdomain centroid).
    coarse_factor: Vec<f64>,
    coarse_n: usize,
}

/// The 6x6 map N(p, c) from a subdomain's rigid-body-mode coefficients to a
/// node's (translation, rotation) dofs: columns 0..3 are unit translations,
/// column 3+k is rotation about axis k around the centroid c — translation
/// part `e_k x (p - c)`, rotation part `e_k`. Constrained node dofs get
/// zeroed rows so the coarse space stays inside the constrained subspace.
fn node_modes(p: [f64; 3], c: [f64; 3], constrained_rows: &[bool; 6]) -> [[f64; 6]; 6] {
    let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    let mut n = [[0.0f64; 6]; 6];
    for t in 0..3 {
        n[t][t] = 1.0;
    }
    // e_k x d for k = 0, 1, 2.
    let cross = [
        [0.0, -d[2], d[1]],
        [d[2], 0.0, -d[0]],
        [-d[1], d[0], 0.0],
    ];
    for k in 0..3 {
        for row in 0..3 {
            n[row][3 + k] = cross[k][row];
        }
        n[3 + k][3 + k] = 1.0;
    }
    for row in 0..6 {
        if constrained_rows[row] {
            for col in 0..6 {
                n[row][col] = 0.0;
            }
        }
    }
    n
}

impl SchwarzPrecond {
    pub(crate) fn build(
        mesh: &FeaMesh,
        model: &FrameModel,
        constrained: &[bool],
        params: SchwarzParams,
    ) -> Option<Self> {
        let node_count = mesh.node_count();
        if node_count == 0 {
            return None;
        }

        // --- Partition: uniform boxes sized so the average box holds about
        // target_nodes nodes, then compacted to non-empty boxes.
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for node in 0..node_count {
            let p = mesh.node_position(node);
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let volume: f64 = (0..3).map(|a| (hi[a] - lo[a]).max(1e-9)).product();
        let cell = (volume * params.target_nodes.max(1) as f64 / node_count as f64)
            .cbrt()
            .max(1e-9);
        let dims: [usize; 3] =
            std::array::from_fn(|a| (((hi[a] - lo[a]) / cell).floor() as usize + 1).max(1));
        let box_of = |p: [f64; 3]| -> usize {
            let mut idx = 0;
            for a in 0..3 {
                let i = (((p[a] - lo[a]) / cell).floor() as usize).min(dims[a] - 1);
                idx = idx * dims[a] + i;
            }
            idx
        };
        let mut box_ids: Vec<usize> = (0..node_count)
            .map(|node| box_of(mesh.node_position(node)))
            .collect();
        // Compact to dense subdomain ids.
        let mut remap = std::collections::HashMap::new();
        for id in box_ids.iter_mut() {
            let next = remap.len();
            *id = *remap.entry(*id).or_insert(next);
        }
        let nsub = remap.len();
        let owner: Vec<u32> = box_ids.iter().map(|&b| b as u32).collect();

        let mut centroids = vec![[0.0f64; 3]; nsub];
        let mut counts = vec![0usize; nsub];
        for node in 0..node_count {
            let p = mesh.node_position(node);
            let c = &mut centroids[owner[node] as usize];
            for a in 0..3 {
                c[a] += p[a];
            }
            counts[owner[node] as usize] += 1;
        }
        for (c, n) in centroids.iter_mut().zip(&counts) {
            for a in 0..3 {
                c[a] /= (*n).max(1) as f64;
            }
        }

        let local = if params.dense_local {
            build_dense_local(mesh, model, constrained, &owner, nsub)?
        } else {
            LocalSolves::BlockJacobi(crate::invert_node_blocks(
                &model.node_blocks()?,
                6,
                constrained,
            ))
        };

        // --- Coarse operator A_c = Z^T K Z, assembled per strut from the
        // element stiffness and each endpoint's owner modes. The per-node
        // mode matrices are kept for the apply side (Z must be the same Z).
        let modes: Vec<[[f64; 6]; 6]> = (0..node_count)
            .map(|node| {
                let rows: [bool; 6] = std::array::from_fn(|c| constrained[node * 6 + c]);
                node_modes(
                    mesh.node_position(node),
                    centroids[owner[node] as usize],
                    &rows,
                )
            })
            .collect();
        let coarse_n = nsub * 6;
        let mut coarse = vec![0.0f64; coarse_n * coarse_n];
        for e in 0..model.strut_count() {
            let [n1, n2] = model.strut_nodes(e);
            let ke = model.element_stiffness(e);
            let ends = [n1 as usize, n2 as usize];
            let n_mats: Vec<[[f64; 6]; 6]> =
                ends.iter().map(|&node| modes[node]).collect();
            for (ei, &node_i) in ends.iter().enumerate() {
                for (ej, &node_j) in ends.iter().enumerate() {
                    let (si, sj) = (owner[node_i] as usize, owner[node_j] as usize);
                    // H = N_i^T K_e[6ei.., 6ej..] N_j, a 6x6 into (si, sj).
                    for a in 0..6 {
                        for b in 0..6 {
                            let mut sum = 0.0;
                            for r in 0..6 {
                                for c in 0..6 {
                                    sum += n_mats[ei][r][a]
                                        * ke[ei * 6 + r][ej * 6 + c]
                                        * n_mats[ej][c][b];
                                }
                            }
                            coarse[(si * 6 + a) * coarse_n + sj * 6 + b] += sum;
                        }
                    }
                }
            }
        }
        // Guard degenerate coarse dofs (fully glued subdomains).
        for i in 0..coarse_n {
            if coarse[i * coarse_n + i] <= 1e-12 {
                for j in 0..coarse_n {
                    coarse[i * coarse_n + j] = 0.0;
                    coarse[j * coarse_n + i] = 0.0;
                }
                coarse[i * coarse_n + i] = 1.0;
            }
        }
        if !cholesky_in_place(&mut coarse, coarse_n) {
            return None;
        }

        Some(Self {
            local,
            owner,
            modes,
            coarse_factor: coarse,
            coarse_n,
        })
    }
}

/// Assemble and factor the dense overlapping-subdomain solves.
fn build_dense_local(
    mesh: &FeaMesh,
    model: &FrameModel,
    constrained: &[bool],
    owner: &[u32],
    nsub: usize,
) -> Option<LocalSolves> {
    let node_count = mesh.node_count();
    // Overlap: each subdomain takes its own nodes plus every node one strut
    // away, collected per subdomain from the element list.
    let mut extended: Vec<std::collections::BTreeSet<u32>> = vec![Default::default(); nsub];
    for node in 0..node_count {
        extended[owner[node] as usize].insert(node as u32);
    }
    for e in 0..model.strut_count() {
        let [a, b] = model.strut_nodes(e);
        let (sa, sb) = (owner[a as usize] as usize, owner[b as usize] as usize);
        if sa != sb {
            extended[sa].insert(b);
            extended[sb].insert(a);
        }
    }

    // Dense assembly + factorization (parallel: this is the expensive part
    // and each subdomain is independent).
    let element_of_node: Vec<Vec<u32>> = {
            let mut incidence = vec![Vec::new(); node_count];
            for e in 0..model.strut_count() {
                for node in model.strut_nodes(e) {
                    incidence[node as usize].push(e as u32);
                }
            }
            incidence
        };
        let subdomains: Vec<Option<Subdomain>> = extended
            .par_iter()
            .map(|nodes| {
                let nodes: Vec<u32> = nodes.iter().copied().collect();
                let m = nodes.len() * 6;
                let index_of: std::collections::HashMap<u32, usize> = nodes
                    .iter()
                    .enumerate()
                    .map(|(i, &node)| (node, i))
                    .collect();
                let mut k = vec![0.0f64; m * m];
                let mut seen = std::collections::HashSet::new();
                for &node in &nodes {
                    for &e in &element_of_node[node as usize] {
                        if !seen.insert(e) {
                            continue;
                        }
                        let [n1, n2] = model.strut_nodes(e as usize);
                        // R_s K R_s^T: keep every block whose row AND column
                        // node are inside. A strut crossing the subdomain
                        // boundary still contributes its inside node's
                        // diagonal block — that grounding is what keeps
                        // interior subdomains SPD (skipping cut struts
                        // leaves a pure-Neumann singular local problem).
                        let ends = [index_of.get(&n1), index_of.get(&n2)];
                        let ke = model.element_stiffness(e as usize);
                        for bi in 0..4 {
                            let Some(&row_node) = ends[bi / 2] else {
                                continue;
                            };
                            for bj in 0..4 {
                                let Some(&col_node) = ends[bj / 2] else {
                                    continue;
                                };
                                let row_base = row_node * 6 + (bi % 2) * 3;
                                let col_base = col_node * 6 + (bj % 2) * 3;
                                for r in 0..3 {
                                    for c in 0..3 {
                                        k[(row_base + r) * m + col_base + c] +=
                                            ke[bi * 3 + r][bj * 3 + c];
                                    }
                                }
                            }
                        }
                    }
                }
                // Mask constrained (and floating) dofs to identity.
                let dofs: Vec<u32> = nodes
                    .iter()
                    .flat_map(|&node| (0..6).map(move |c| node * 6 + c))
                    .collect();
                for (local, &dof) in dofs.iter().enumerate() {
                    if constrained[dof as usize] || k[local * m + local] <= 0.0 {
                        for j in 0..m {
                            k[local * m + j] = 0.0;
                            k[j * m + local] = 0.0;
                        }
                        k[local * m + local] = 1.0;
                    }
                }
                if !cholesky_in_place(&mut k, m) {
                    return None;
                }
                Some(Subdomain { dofs, factor: k })
            })
            .collect();
    let subdomains: Vec<Subdomain> = subdomains.into_iter().collect::<Option<Vec<_>>>()?;
    Some(LocalSolves::Dense(subdomains))
}

impl SchwarzPrecond {
    /// z = mask(local solves + Z A_c^{-1} Z^T r). The mask
    /// uses the CURRENT constraint set (contact constraints appear after
    /// this preconditioner was built) so the search direction never leaves
    /// the constrained subspace.
    pub(crate) fn apply(&self, r: &[f64], z: &mut [f64], constrained: &[bool]) {
        match &self.local {
            // Subdomain solves in parallel, each producing its local
            // contribution; scatter-add serially afterwards (contributions
            // overlap, so the scatter must not race).
            LocalSolves::Dense(subdomains) => {
                let locals: Vec<Vec<f64>> = subdomains
                    .par_iter()
                    .map(|s| {
                        let mut local: Vec<f64> =
                            s.dofs.iter().map(|&dof| r[dof as usize]).collect();
                        cholesky_solve_in_place(&s.factor, local.len(), &mut local);
                        local
                    })
                    .collect();
                z.fill(0.0);
                for (s, local) in subdomains.iter().zip(&locals) {
                    for (&dof, value) in s.dofs.iter().zip(local) {
                        z[dof as usize] += value;
                    }
                }
            }
            LocalSolves::BlockJacobi(inverses) => {
                for node in 0..r.len() / 6 {
                    let inv = &inverses[node * 36..(node + 1) * 36];
                    let rn = &r[node * 6..(node + 1) * 6];
                    let zn = &mut z[node * 6..(node + 1) * 6];
                    for i in 0..6 {
                        zn[i] = (0..6).map(|j| inv[i * 6 + j] * rn[j]).sum();
                    }
                }
            }
        }

        // Coarse correction.
        let mut rc = vec![0.0f64; self.coarse_n];
        let node_count = self.owner.len();
        for node in 0..node_count {
            let n = &self.modes[node];
            let base = self.owner[node] as usize * 6;
            for col in 0..6 {
                let mut sum = 0.0;
                for row in 0..6 {
                    sum += n[row][col] * r[node * 6 + row];
                }
                rc[base + col] += sum;
            }
        }
        cholesky_solve_in_place(&self.coarse_factor, self.coarse_n, &mut rc);
        for node in 0..node_count {
            let n = &self.modes[node];
            let base = self.owner[node] as usize * 6;
            for row in 0..6 {
                let mut sum = 0.0;
                for col in 0..6 {
                    sum += n[row][col] * rc[base + col];
                }
                z[node * 6 + row] += sum;
            }
        }

        for (v, &c) in z.iter_mut().zip(constrained) {
            if c {
                *v = 0.0;
            }
        }
    }
}
