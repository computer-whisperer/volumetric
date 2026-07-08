//! Two-level additive Schwarz preconditioner for Bar2 frame lattices — the
//! scale/parallelism path for big solves.
//!
//! Motivation (measured on the strut example at 1000x stiffness contrast):
//! Jacobi needs ~2k CG iterations per contact solve and block-Jacobi only
//! shaves 1.5x, because the conditioning pathology is *global* — soft and
//! stiff column regions in series create long-wavelength error modes no
//! local preconditioner can touch, and iteration counts grow with mesh size
//! on top. The fine SpMV is memory-bandwidth bound, so threading it tops
//! out around 1.2x. This preconditioner attacks both at once:
//!
//! - **Subdomains** (spatial boxes of ~`target_nodes` nodes, one strut layer
//!   of overlap): f32 sparse-Cholesky local solves (faer), embarrassingly
//!   parallel with coarse granularity.
//! - **Coarse space** (6 rigid-body modes per subdomain, GDSW-style),
//!   assembled and factored sparse — subdomains only couple to neighbors:
//!   propagates corrections globally each iteration, killing the
//!   contrast-induced low-frequency modes and pinning iteration counts
//!   nearly flat in mesh size (204 -> 325 CG iterations from 4.7k to 1M
//!   struts, where block-Jacobi grows 380 -> 1810).
//!
//! At 1M struts / 24 threads this solves 1.8x faster than block-Jacobi
//! (40s vs 71s, phase split ~17s local solves / ~7s coarse / ~10s CG SpMV);
//! below ~100k struts block-Jacobi's lower constant wins — hence the
//! opt-in via [`crate::PrecondChoice`].
//!
//! Built once per solve against the glued-face constraints only; contact
//! constraints are handled by masking the preconditioned residual (a
//! preconditioner need not be exact — CG stays correct because the operator
//! side is masked exactly). Remaining headroom, in measured order: the
//! serial Z^T r / Z x_c coarse products, the serial overlap scatter-add,
//! and the ~1.2x-ceiling parallel SpMV.

use crate::frame::FrameModel;
use crate::StiffnessModel;
use rayon::prelude::*;
use volumetric_abi::fea::FeaMesh;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchwarzParams {
    /// Target nodes per subdomain box before overlap.
    pub target_nodes: usize,
    /// Fine-level solver: sparse-Cholesky direct solves per overlapping
    /// subdomain (true; faer factorization, the real preconditioner) or
    /// block-Jacobi (false; O(n) apply — measured to interfere with the
    /// coarse correction, kept for comparison runs).
    pub direct_local: bool,
}

impl Default for SchwarzParams {
    fn default() -> Self {
        Self {
            target_nodes: 128,
            direct_local: true,
        }
    }
}

struct Subdomain {
    /// Global dof indices covered by this (overlapping) subdomain.
    dofs: Vec<u32>,
    /// Sparse Cholesky factorization of the masked local stiffness, in f32:
    /// the triangular solves stream the factor from memory every iteration
    /// (way past cache), so halving the bytes halves the dominant cost. The
    /// preconditioner only steers CG — the Krylov iteration itself stays
    /// f64, so converged answers are unaffected.
    solver: faer::sparse::linalg::solvers::Llt<usize, f32>,
}

/// The fine (local) level of the two-level preconditioner.
enum LocalSolves {
    Direct(Vec<Subdomain>),
    /// Per-node inverted 6x6 blocks (same construction as Precond::Block,
    /// masked against the build-time constraints).
    BlockJacobi(Vec<f64>),
}

// Phase timers (nanos) for the scaling bench: negligible cost (a handful of
// atomic adds per apply), and the breakdown they print is what has driven
// every tuning decision here so far.
pub(crate) static T_LOCAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static T_SCATTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static T_COARSE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static T_BUILD_LOCAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static T_BUILD_COARSE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(crate) struct SchwarzPrecond {
    local: LocalSolves,
    /// Non-overlapping owner subdomain per node (coarse partition of unity).
    owner: Vec<u32>,
    /// Per-node rigid-body-mode matrix N (build-time constraint rows baked
    /// in — the apply-side Z must match the Z that assembled A_c).
    modes: Vec<[[f64; 6]; 6]>,
    /// Sparse Cholesky factorization of the coarse operator Z^T K Z (6 dofs
    /// per subdomain: rigid-body modes about the subdomain centroid). Sparse
    /// because subdomains only couple to spatial neighbors — dense coarse
    /// handling was measured to dominate everything past ~1k subdomains.
    coarse_solver: faer::sparse::linalg::solvers::Llt<usize, f64>,
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

        let build_timer = std::time::Instant::now();
        let local = if params.direct_local {
            build_direct_local(mesh, model, constrained, &owner, nsub)?
        } else {
            LocalSolves::BlockJacobi(crate::invert_node_blocks(
                &model.node_blocks()?,
                6,
                constrained,
            ))
        };

        T_BUILD_LOCAL.fetch_add(
            build_timer.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let build_timer = std::time::Instant::now();

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
        // Accumulate per-subdomain-pair 6x6 blocks (canonical si >= sj; the
        // symmetric counterpart is implied) — A_c is sparse with box-graph
        // adjacency, so a block map then lower-triangle triplets.
        let coarse_n = nsub * 6;
        let mut blocks: std::collections::HashMap<(u32, u32), [f64; 36]> =
            std::collections::HashMap::new();
        for e in 0..model.strut_count() {
            let [n1, n2] = model.strut_nodes(e);
            let ke = model.element_stiffness(e);
            let ends = [n1 as usize, n2 as usize];
            let n_mats: Vec<[[f64; 6]; 6]> =
                ends.iter().map(|&node| modes[node]).collect();
            for (ei, &node_i) in ends.iter().enumerate() {
                for (ej, &node_j) in ends.iter().enumerate() {
                    let (si, sj) = (owner[node_i], owner[node_j]);
                    if si < sj {
                        continue; // the (ej, ei) pass covers the transpose
                    }
                    let block = blocks.entry((si, sj)).or_insert([0.0; 36]);
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
                            block[a * 6 + b] += sum;
                        }
                    }
                }
            }
        }
        // Degenerate coarse dofs (fully glued subdomains) become identity;
        // detect them from the diagonal blocks.
        let mut degenerate = vec![false; coarse_n];
        for s in 0..nsub as u32 {
            if let Some(block) = blocks.get(&(s, s)) {
                for a in 0..6 {
                    degenerate[s as usize * 6 + a] = block[a * 6 + a] <= 1e-12;
                }
            } else {
                degenerate[s as usize * 6..s as usize * 6 + 6].fill(true);
            }
        }
        let mut triplets: Vec<faer::sparse::Triplet<usize, usize, f64>> = Vec::new();
        for (&(si, sj), block) in &blocks {
            for a in 0..6 {
                for b in 0..6 {
                    let (row, col) = (si as usize * 6 + a, sj as usize * 6 + b);
                    if row < col || degenerate[row] || degenerate[col] {
                        continue;
                    }
                    triplets.push(faer::sparse::Triplet::new(row, col, block[a * 6 + b]));
                }
            }
        }
        for (i, &degen) in degenerate.iter().enumerate() {
            if degen {
                triplets.push(faer::sparse::Triplet::new(i, i, 1.0));
            }
        }
        let coarse_mat =
            faer::sparse::SparseColMat::try_new_from_triplets(coarse_n, coarse_n, &triplets)
                .ok()?;
        let coarse_solver = coarse_mat.sp_cholesky(faer::Side::Lower).ok()?;
        T_BUILD_COARSE.fetch_add(
            build_timer.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        Some(Self {
            local,
            owner,
            modes,
            coarse_solver,
            coarse_n,
        })
    }
}

/// Assemble and factor the sparse overlapping-subdomain solves.
fn build_direct_local(
    mesh: &FeaMesh,
    model: &FrameModel,
    constrained: &[bool],
    owner: &[u32],
    nsub: usize,
) -> Option<LocalSolves> {
    use faer::sparse::{SparseColMat, Triplet};

    // faer ops run inside the outer rayon fan-out below (and per-subdomain
    // problems are small), so keep faer itself sequential.
    faer::set_global_parallelism(faer::Par::Seq);

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

    // Sparse assembly + factorization (parallel: each subdomain is
    // independent; this is the one-time cost per solve).
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
            // Lower-triangle triplets (duplicates sum); constrained dofs are
            // masked by skipping their entries, and get identity diagonals
            // together with floating (no-stiffness) dofs afterwards.
            // f32: see the Subdomain::solver docs.
            let mut triplets: Vec<Triplet<usize, usize, f32>> = Vec::new();
            let mut has_diagonal = vec![false; m];
            let mut seen = std::collections::HashSet::new();
            for &node in &nodes {
                for &e in &element_of_node[node as usize] {
                    if !seen.insert(e) {
                        continue;
                    }
                    let [n1, n2] = model.strut_nodes(e as usize);
                    // R_s K R_s^T: keep every block whose row AND column
                    // node are inside. A strut crossing the subdomain
                    // boundary still contributes its inside node's diagonal
                    // block — that grounding is what keeps interior
                    // subdomains SPD (skipping cut struts leaves a
                    // pure-Neumann singular local problem).
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
                                    let (row, col) = (row_base + r, col_base + c);
                                    if row < col {
                                        continue;
                                    }
                                    let dof_row = nodes[row / 6] * 6 + (row % 6) as u32;
                                    let dof_col = nodes[col / 6] * 6 + (col % 6) as u32;
                                    if constrained[dof_row as usize]
                                        || constrained[dof_col as usize]
                                    {
                                        continue;
                                    }
                                    let value = ke[bi * 3 + r][bj * 3 + c] as f32;
                                    if row == col {
                                        has_diagonal[row] = true;
                                    }
                                    triplets.push(Triplet::new(row, col, value));
                                }
                            }
                        }
                    }
                }
            }
            let dofs: Vec<u32> = nodes
                .iter()
                .flat_map(|&node| (0..6).map(move |c| node * 6 + c))
                .collect();
            for (local, &dof) in dofs.iter().enumerate() {
                if constrained[dof as usize] || !has_diagonal[local] {
                    triplets.push(Triplet::new(local, local, 1.0));
                }
            }
            let matrix = SparseColMat::try_new_from_triplets(m, m, &triplets).ok()?;
            let solver = matrix.sp_cholesky(faer::Side::Lower).ok()?;
            Some(Subdomain { dofs, solver })
        })
        .collect();
    let subdomains: Vec<Subdomain> = subdomains.into_iter().collect::<Option<Vec<_>>>()?;
    Some(LocalSolves::Direct(subdomains))
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
            LocalSolves::Direct(subdomains) => {
                use faer::linalg::solvers::Solve;
                let timer = std::time::Instant::now();
                let locals: Vec<Vec<f32>> = subdomains
                    .par_iter()
                    .map(|s| {
                        let mut local: Vec<f32> =
                            s.dofs.iter().map(|&dof| r[dof as usize] as f32).collect();
                        let m = local.len();
                        s.solver.solve_in_place(
                            faer::MatMut::from_column_major_slice_mut(&mut local, m, 1),
                        );
                        local
                    })
                    .collect();
                T_LOCAL.fetch_add(
                    timer.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                let timer = std::time::Instant::now();
                z.fill(0.0);
                for (s, local) in subdomains.iter().zip(&locals) {
                    for (&dof, &value) in s.dofs.iter().zip(local) {
                        z[dof as usize] += value as f64;
                    }
                }
                T_SCATTER.fetch_add(
                    timer.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
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
        let coarse_timer = std::time::Instant::now();
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
        {
            use faer::linalg::solvers::Solve;
            self.coarse_solver.solve_in_place(
                faer::MatMut::from_column_major_slice_mut(&mut rc, self.coarse_n, 1),
            );
        }
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

        T_COARSE.fetch_add(
            coarse_timer.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        for (v, &c) in z.iter_mut().zip(constrained) {
            if c {
                *v = 0.0;
            }
        }
    }
}
