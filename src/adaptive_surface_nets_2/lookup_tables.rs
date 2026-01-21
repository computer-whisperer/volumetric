//! Canonical lookup tables for marching cubes triangulation.
//!
//! Contains corner and edge offset tables, as well as the marching cubes
//! triangulation tables adapted to our corner numbering convention.

// =============================================================================
// CANONICAL LOOKUP TABLES
// =============================================================================

/// Corner offset table: CORNER_OFFSETS[i] = (dx, dy, dz) for corner index i.
/// Uses the canonical bit encoding: index = (z << 2) | (y << 1) | x
pub const CORNER_OFFSETS: [(i32, i32, i32); 8] = [
    (0, 0, 0), // 0: ---
    (1, 0, 0), // 1: +--
    (0, 1, 0), // 2: -+-
    (1, 1, 0), // 3: ++-
    (0, 0, 1), // 4: --+
    (1, 0, 1), // 5: +-+
    (0, 1, 1), // 6: -++
    (1, 1, 1), // 7: +++
];

/// Edge definition table: EDGE_TABLE[i] = (corner_a, corner_b, axis, dx, dy, dz)
/// - corner_a, corner_b: the two corner indices this edge connects
/// - axis: 0=X, 1=Y, 2=Z
/// - (dx, dy, dz): offset from cell origin to edge's minimum corner (in cell units)
pub const EDGE_TABLE: [(usize, usize, u8, i32, i32, i32); 12] = [
    // X-axis edges (axis=0)
    (0, 1, 0, 0, 0, 0), // edge 0: corners 0-1
    (2, 3, 0, 0, 1, 0), // edge 1: corners 2-3
    (4, 5, 0, 0, 0, 1), // edge 2: corners 4-5
    (6, 7, 0, 0, 1, 1), // edge 3: corners 6-7
    // Y-axis edges (axis=1)
    (0, 2, 1, 0, 0, 0), // edge 4: corners 0-2
    (1, 3, 1, 1, 0, 0), // edge 5: corners 1-3
    (4, 6, 1, 0, 0, 1), // edge 6: corners 4-6
    (5, 7, 1, 1, 0, 1), // edge 7: corners 5-7
    // Z-axis edges (axis=2)
    (0, 4, 2, 0, 0, 0), // edge 8: corners 0-4
    (1, 5, 2, 1, 0, 0), // edge 9: corners 1-5
    (2, 6, 2, 0, 1, 0), // edge 10: corners 2-6
    (3, 7, 2, 1, 1, 0), // edge 11: corners 3-7
];

// =============================================================================
// MARCHING CUBES CONVENTION TRANSLATION
// =============================================================================
//
// Our corner numbering uses: index = (z << 2) | (y << 1) | x
//   0:(0,0,0) 1:(1,0,0) 2:(0,1,0) 3:(1,1,0) 4:(0,0,1) 5:(1,0,1) 6:(0,1,1) 7:(1,1,1)
//
// Standard MC corner numbering (Paul Bourke):
//   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
//
// Mapping: MC corner N corresponds to our corner:
//   MC 0 -> Our 0, MC 1 -> Our 1, MC 2 -> Our 3, MC 3 -> Our 2
//   MC 4 -> Our 4, MC 5 -> Our 5, MC 6 -> Our 7, MC 7 -> Our 6
//
// Similarly for edges (MC edge -> Our edge):
//   0->0, 1->5, 2->1, 3->4, 4->2, 5->7, 6->3, 7->6, 8->8, 9->9, 10->11, 11->10

/// Convert our corner mask to standard MC corner mask for table lookup.
/// Our bits: 7 6 5 4 3 2 1 0 (our corners 7,6,5,4,3,2,1,0)
/// MC bits:  7 6 5 4 3 2 1 0 (MC corners 7,6,5,4,3,2,1,0)
///
/// Mapping: MC bit N should contain our corner that maps to MC corner N
///   MC bit 0 <- our bit 0 (our corner 0 -> MC corner 0)
///   MC bit 1 <- our bit 1 (our corner 1 -> MC corner 1)
///   MC bit 2 <- our bit 3 (our corner 3 -> MC corner 2)
///   MC bit 3 <- our bit 2 (our corner 2 -> MC corner 3)
///   MC bit 4 <- our bit 4 (our corner 4 -> MC corner 4)
///   MC bit 5 <- our bit 5 (our corner 5 -> MC corner 5)
///   MC bit 6 <- our bit 7 (our corner 7 -> MC corner 6)
///   MC bit 7 <- our bit 6 (our corner 6 -> MC corner 7)
#[inline]
pub fn our_mask_to_mc_mask(our_mask: u8) -> u8 {
    (our_mask & 0b00000011) |           // bits 0,1 stay
    ((our_mask & 0b00000100) << 1) |    // our bit 2 -> MC bit 3
    ((our_mask & 0b00001000) >> 1) |    // our bit 3 -> MC bit 2
    (our_mask & 0b00110000) |           // bits 4,5 stay
    ((our_mask & 0b01000000) << 1) |    // our bit 6 -> MC bit 7
    ((our_mask & 0b10000000) >> 1)      // our bit 7 -> MC bit 6
}

/// Convert MC edge index to our edge index.
/// Standard MC edges connect MC corners, we need to map to our edge numbering.
pub const MC_EDGE_TO_OUR_EDGE: [usize; 12] = [
    0,  // MC edge 0 (MC corners 0-1) -> our edge 0 (our corners 0-1)
    5,  // MC edge 1 (MC corners 1-2) -> our edge 5 (our corners 1-3)
    1,  // MC edge 2 (MC corners 2-3) -> our edge 1 (our corners 2-3, but MC 2=our 3, MC 3=our 2)
    4,  // MC edge 3 (MC corners 3-0) -> our edge 4 (our corners 0-2)
    2,  // MC edge 4 (MC corners 4-5) -> our edge 2 (our corners 4-5)
    7,  // MC edge 5 (MC corners 5-6) -> our edge 7 (our corners 5-7)
    3,  // MC edge 6 (MC corners 6-7) -> our edge 3 (our corners 6-7, but MC 6=our 7, MC 7=our 6)
    6,  // MC edge 7 (MC corners 7-4) -> our edge 6 (our corners 4-6)
    8,  // MC edge 8 (MC corners 0-4) -> our edge 8 (our corners 0-4)
    9,  // MC edge 9 (MC corners 1-5) -> our edge 9 (our corners 1-5)
    11, // MC edge 10 (MC corners 2-6) -> our edge 11 (MC 2=our 3, MC 6=our 7 -> corners 3-7)
    10, // MC edge 11 (MC corners 3-7) -> our edge 10 (MC 3=our 2, MC 7=our 6 -> corners 2-6)
];

/// Marching Cubes edge flags: MC_EDGE_FLAGS[corner_mask] = bitmask of active edges
/// Bit N is set if edge N has a sign change (crosses the surface).
/// This uses the standard Marching Cubes table adapted to our corner indexing.
#[allow(dead_code)]
pub const MC_EDGE_FLAGS: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// Marching Cubes triangle table: MC_TRI_TABLE[corner_mask] = list of edge indices forming triangles
/// Each entry is a slice of edge indices, grouped in threes (each triple forms one triangle).
/// -1 marks the end of the list. Maximum 5 triangles (15 edges) per configuration.
/// Winding is set so normals point from inside (1) to outside (0).
pub const MC_TRI_TABLE: [[i8; 16]; 256] = include!("../mc_tri_table.inc");
