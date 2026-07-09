//! Mesh connectivity queries: vertex adjacency and k-ring neighborhoods.

/// Per-vertex neighbor lists built from a triangle index buffer, stored in
/// compressed (CSR) form: one shared buffer plus per-vertex extents, instead
/// of millions of tiny per-vertex allocations.
pub struct MeshAdjacency {
    starts: Vec<u32>,
    data: Vec<u32>,
}

impl MeshAdjacency {
    pub fn build(vertex_count: usize, indices: &[u32]) -> Self {
        // Directed edges packed as (vertex << 32 | neighbor): one (parallel)
        // sort groups the buffer by vertex with neighbors ordered within each
        // group, and dedup collapses the repeats from the faces sharing each
        // edge — the same result as per-vertex sort + dedup.
        let mut pairs: Vec<u64> = Vec::with_capacity(indices.len() * 2);
        for tri in indices.chunks_exact(3) {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                pairs.push(((a as u64) << 32) | b as u64);
                pairs.push(((b as u64) << 32) | a as u64);
            }
        }
        crate::parallel_iter::sort_unstable(&mut pairs);
        pairs.dedup();

        let mut starts = vec![0u32; vertex_count + 1];
        let mut data = Vec::with_capacity(pairs.len());
        for &pair in &pairs {
            starts[(pair >> 32) as usize + 1] += 1;
            data.push(pair as u32);
        }
        for i in 1..starts.len() {
            starts[i] += starts[i - 1];
        }
        Self { starts, data }
    }

    pub fn neighbors(&self, v: u32) -> &[u32] {
        &self.data[self.starts[v as usize] as usize..self.starts[v as usize + 1] as usize]
    }

    /// All vertices within `k` mesh edges of `v`, including `v` itself.
    pub fn k_ring(&self, v: u32, k: usize) -> Vec<u32> {
        let mut visited = vec![v];
        let mut frontier = vec![v];
        for _ in 0..k {
            let mut next = Vec::new();
            for &u in &frontier {
                for &w in self.neighbors(u) {
                    if !visited.contains(&w) {
                        visited.push(w);
                        next.push(w);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        visited
    }
}

/// Unique undirected mesh edges as (min_index, max_index) pairs.
pub fn unique_edges(indices: &[u32]) -> Vec<(u32, u32)> {
    let mut edges: Vec<(u32, u32)> = indices
        .chunks_exact(3)
        .flat_map(|tri| {
            [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
                .map(|(a, b)| (a.min(b), a.max(b)))
        })
        .collect();
    edges.sort_unstable();
    edges.dedup();
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two triangles sharing edge 1-2: (0,1,2) and (1,3,2).
    const INDICES: [u32; 6] = [0, 1, 2, 1, 3, 2];

    #[test]
    fn adjacency_and_rings() {
        let adj = MeshAdjacency::build(4, &INDICES);
        assert_eq!(adj.neighbors(0), &[1, 2]);
        assert_eq!(adj.neighbors(1), &[0, 2, 3]);
        let mut ring1 = adj.k_ring(0, 1);
        ring1.sort_unstable();
        assert_eq!(ring1, vec![0, 1, 2]);
        let mut ring2 = adj.k_ring(0, 2);
        ring2.sort_unstable();
        assert_eq!(ring2, vec![0, 1, 2, 3]);
    }

    #[test]
    fn unique_edge_list() {
        let edges = unique_edges(&INDICES);
        assert_eq!(edges, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
    }
}
