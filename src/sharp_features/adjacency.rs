//! Mesh connectivity queries: vertex adjacency and k-ring neighborhoods.

/// Per-vertex neighbor lists built from a triangle index buffer.
pub struct MeshAdjacency {
    neighbors: Vec<Vec<u32>>,
}

impl MeshAdjacency {
    pub fn build(vertex_count: usize, indices: &[u32]) -> Self {
        let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
        for tri in indices.chunks_exact(3) {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                neighbors[a as usize].push(b);
                neighbors[b as usize].push(a);
            }
        }
        for list in &mut neighbors {
            list.sort_unstable();
            list.dedup();
        }
        Self { neighbors }
    }

    pub fn neighbors(&self, v: u32) -> &[u32] {
        &self.neighbors[v as usize]
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
