//! STL Import Operator.
//!
//! Host ABI:
//! - `host.get_input_len(i32) -> u32`
//! - `host.get_input_data(i32, ptr, len)`
//! - `host.post_output(i32, ptr, len)`
//!
//! Operator ABI:
//! - `get_metadata() -> i64` returning `(ptr: u32, len: u32)` packed as `ptr | (len << 32)`
//!
//! Behavior:
//! - Reads STL data (binary or ASCII) from input 0
//! - Reads CBOR configuration from input 1 (schema declared in metadata)
//! - Produces a WASM model with BVH-accelerated point-in-mesh testing

use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ExportKind, ExportSection, Function, FunctionSection,
    Instruction, MemorySection, MemoryType, Module, TypeSection, ValType,
};

// ============================================================================
// Operator Metadata Types
// ============================================================================

// NOTE: Order must match the OperatorMetadataInput enum in src/lib.rs for CBOR compatibility
#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataInput {
    ModelWASM,
    CBORConfiguration(String),
    LuaSource(String),
    Blob,
}

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataOutput {
    ModelWASM,
}

#[derive(Clone, Debug, serde::Serialize)]
struct OperatorMetadata {
    name: String,
    version: String,
    inputs: Vec<OperatorMetadataInput>,
    outputs: Vec<OperatorMetadataOutput>,
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Clone, Debug, serde::Deserialize)]
struct StlImportConfig {
    #[serde(default = "default_scale")]
    scale: f64,
    #[serde(default)]
    translate: [f64; 3],
    #[serde(default)]
    center: bool,
}

fn default_scale() -> f64 {
    1.0
}

impl Default for StlImportConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            translate: [0.0, 0.0, 0.0],
            center: false,
        }
    }
}

// ============================================================================
// Host ABI
// ============================================================================

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

// ============================================================================
// STL Data Structures
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct Triangle {
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
}

#[allow(dead_code)]
struct ParsedMesh {
    triangles: Vec<Triangle>,
    aabb_min: [f64; 3],
    aabb_max: [f64; 3],
}

// ============================================================================
// BVH Data Structures
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct BvhNode {
    aabb_min: [f64; 3],
    aabb_max: [f64; 3],
    left_or_start: u32,  // left child index (internal) or first triangle index (leaf)
    right_or_count: u32, // right child index (internal) or triangle count (leaf)
    is_leaf: bool,
}

struct Bvh {
    nodes: Vec<BvhNode>,
    triangle_indices: Vec<u32>,
}

// ============================================================================
// STL Parsing
// ============================================================================

fn is_ascii_stl(data: &[u8]) -> bool {
    if data.len() < 6 {
        return false;
    }
    // ASCII STL starts with "solid " (with space) and should contain "facet"
    // Binary STL can also start with "solid" in its header, so we need more checks
    if !data.starts_with(b"solid ") {
        return false;
    }
    // Check if it looks like ASCII by searching for "facet" keyword
    // within the first few KB
    let check_len = data.len().min(1024);
    for window in data[..check_len].windows(5) {
        if window == b"facet" {
            return true;
        }
    }
    // If no "facet" found, it's likely binary with "solid" in header
    false
}

fn parse_binary_stl(data: &[u8]) -> Result<Vec<Triangle>, &'static str> {
    if data.len() < 84 {
        return Err("Binary STL too short");
    }

    // Skip 80-byte header
    let triangle_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;

    // Each triangle is 50 bytes: 12 bytes normal + 36 bytes vertices + 2 bytes attribute
    let expected_size = 84 + triangle_count * 50;
    if data.len() < expected_size {
        return Err("Binary STL truncated");
    }

    let mut triangles = Vec::with_capacity(triangle_count);
    let mut offset = 84;

    for _ in 0..triangle_count {
        // Skip normal (12 bytes)
        offset += 12;

        // Read 3 vertices, each is 3 f32 values
        let mut vertices = [[0.0f64; 3]; 3];
        for v in &mut vertices {
            for coord in v.iter_mut() {
                if offset + 4 > data.len() {
                    return Err("Unexpected end of binary STL");
                }
                let bytes = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
                *coord = f32::from_le_bytes(bytes) as f64;
                offset += 4;
            }
        }

        // Skip attribute byte count (2 bytes)
        offset += 2;

        triangles.push(Triangle {
            v0: vertices[0],
            v1: vertices[1],
            v2: vertices[2],
        });
    }

    Ok(triangles)
}

fn parse_ascii_stl(data: &[u8]) -> Result<Vec<Triangle>, &'static str> {
    let text = core::str::from_utf8(data).map_err(|_| "Invalid UTF-8 in ASCII STL")?;
    let mut triangles = Vec::new();
    let mut current_vertices: Vec<[f64; 3]> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("vertex ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let x = parts[1].parse::<f64>().unwrap_or(0.0);
                let y = parts[2].parse::<f64>().unwrap_or(0.0);
                let z = parts[3].parse::<f64>().unwrap_or(0.0);
                current_vertices.push([x, y, z]);

                if current_vertices.len() == 3 {
                    triangles.push(Triangle {
                        v0: current_vertices[0],
                        v1: current_vertices[1],
                        v2: current_vertices[2],
                    });
                    current_vertices.clear();
                }
            }
        }
    }

    Ok(triangles)
}

fn parse_stl(data: &[u8]) -> Result<Vec<Triangle>, &'static str> {
    if is_ascii_stl(data) {
        parse_ascii_stl(data)
    } else {
        parse_binary_stl(data)
    }
}

fn compute_aabb(triangles: &[Triangle]) -> ([f64; 3], [f64; 3]) {
    if triangles.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }

    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];

    for tri in triangles {
        for v in [tri.v0, tri.v1, tri.v2] {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
        }
    }

    (min, max)
}

fn apply_transforms(triangles: &mut [Triangle], config: &StlImportConfig, aabb_min: &[f64; 3], aabb_max: &[f64; 3]) {
    // Compute center if centering is requested
    let center_offset = if config.center {
        [
            -(aabb_min[0] + aabb_max[0]) / 2.0,
            -(aabb_min[1] + aabb_max[1]) / 2.0,
            -(aabb_min[2] + aabb_max[2]) / 2.0,
        ]
    } else {
        [0.0, 0.0, 0.0]
    };

    for tri in triangles.iter_mut() {
        for v in [&mut tri.v0, &mut tri.v1, &mut tri.v2] {
            for i in 0..3 {
                // Apply: center -> scale -> translate
                v[i] = (v[i] + center_offset[i]) * config.scale + config.translate[i];
            }
        }
    }
}

fn process_stl(data: &[u8], config: &StlImportConfig) -> Result<ParsedMesh, &'static str> {
    let mut triangles = parse_stl(data)?;

    if triangles.is_empty() {
        return Err("No triangles in STL");
    }

    // Compute initial AABB for centering
    let (init_min, init_max) = compute_aabb(&triangles);

    // Apply transforms
    apply_transforms(&mut triangles, config, &init_min, &init_max);

    // Compute final AABB
    let (aabb_min, aabb_max) = compute_aabb(&triangles);

    Ok(ParsedMesh {
        triangles,
        aabb_min,
        aabb_max,
    })
}

// ============================================================================
// BVH Construction
// ============================================================================

fn triangle_centroid(tri: &Triangle) -> [f64; 3] {
    [
        (tri.v0[0] + tri.v1[0] + tri.v2[0]) / 3.0,
        (tri.v0[1] + tri.v1[1] + tri.v2[1]) / 3.0,
        (tri.v0[2] + tri.v1[2] + tri.v2[2]) / 3.0,
    ]
}

fn triangle_aabb(tri: &Triangle) -> ([f64; 3], [f64; 3]) {
    let mut min = tri.v0;
    let mut max = tri.v0;

    for v in [tri.v1, tri.v2] {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }

    (min, max)
}

fn aabb_surface_area(min: &[f64; 3], max: &[f64; 3]) -> f64 {
    let dx = (max[0] - min[0]).max(0.0);
    let dy = (max[1] - min[1]).max(0.0);
    let dz = (max[2] - min[2]).max(0.0);
    2.0 * (dx * dy + dy * dz + dz * dx)
}

fn merge_aabb(a_min: &[f64; 3], a_max: &[f64; 3], b_min: &[f64; 3], b_max: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let min = [
        a_min[0].min(b_min[0]),
        a_min[1].min(b_min[1]),
        a_min[2].min(b_min[2]),
    ];
    let max = [
        a_max[0].max(b_max[0]),
        a_max[1].max(b_max[1]),
        a_max[2].max(b_max[2]),
    ];
    (min, max)
}

const MAX_LEAF_TRIANGLES: usize = 4;
const SAH_BINS: usize = 12;

fn build_bvh(triangles: &[Triangle]) -> Bvh {
    let n = triangles.len();
    if n == 0 {
        return Bvh {
            nodes: vec![BvhNode {
                aabb_min: [0.0; 3],
                aabb_max: [0.0; 3],
                left_or_start: 0,
                right_or_count: 0,
                is_leaf: true,
            }],
            triangle_indices: vec![],
        };
    }

    // Compute centroids and AABBs for all triangles
    let centroids: Vec<[f64; 3]> = triangles.iter().map(triangle_centroid).collect();
    let tri_aabbs: Vec<([f64; 3], [f64; 3])> = triangles.iter().map(triangle_aabb).collect();

    // Initialize indices
    let mut indices: Vec<u32> = (0..n as u32).collect();

    // Build BVH recursively
    let mut nodes = Vec::new();
    build_bvh_recursive(
        &mut nodes,
        &mut indices,
        triangles,
        &centroids,
        &tri_aabbs,
        0,
        n,
    );

    Bvh {
        nodes,
        triangle_indices: indices,
    }
}

fn build_bvh_recursive(
    nodes: &mut Vec<BvhNode>,
    indices: &mut [u32],
    triangles: &[Triangle],
    centroids: &[[f64; 3]],
    tri_aabbs: &[([f64; 3], [f64; 3])],
    start: usize,
    end: usize,
) -> u32 {
    let count = end - start;

    // Compute AABB for this node
    let (mut node_min, mut node_max) = tri_aabbs[indices[start] as usize];
    for i in (start + 1)..end {
        let (tri_min, tri_max) = tri_aabbs[indices[i] as usize];
        (node_min, node_max) = merge_aabb(&node_min, &node_max, &tri_min, &tri_max);
    }

    // Create leaf if few triangles
    if count <= MAX_LEAF_TRIANGLES {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            aabb_min: node_min,
            aabb_max: node_max,
            left_or_start: start as u32,
            right_or_count: count as u32,
            is_leaf: true,
        });
        return node_idx;
    }

    // Find best split using SAH
    let (best_axis, best_split_idx) = find_best_split(
        &indices[start..end],
        centroids,
        tri_aabbs,
        &node_min,
        &node_max,
    );

    // If no good split found, create leaf
    if best_split_idx == 0 || best_split_idx >= count {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            aabb_min: node_min,
            aabb_max: node_max,
            left_or_start: start as u32,
            right_or_count: count as u32,
            is_leaf: true,
        });
        return node_idx;
    }

    // Partition indices by centroid along best axis
    partition_by_axis(&mut indices[start..end], centroids, best_axis, best_split_idx);

    let mid = start + best_split_idx;

    // Reserve space for this node
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        aabb_min: node_min,
        aabb_max: node_max,
        left_or_start: 0,  // Will be filled in
        right_or_count: 0, // Will be filled in
        is_leaf: false,
    });

    // Build children
    let left_idx = build_bvh_recursive(nodes, indices, triangles, centroids, tri_aabbs, start, mid);
    let right_idx = build_bvh_recursive(nodes, indices, triangles, centroids, tri_aabbs, mid, end);

    // Update node with child indices
    nodes[node_idx as usize].left_or_start = left_idx;
    nodes[node_idx as usize].right_or_count = right_idx;

    node_idx
}

fn find_best_split(
    indices: &[u32],
    centroids: &[[f64; 3]],
    tri_aabbs: &[([f64; 3], [f64; 3])],
    node_min: &[f64; 3],
    node_max: &[f64; 3],
) -> (usize, usize) {
    let count = indices.len();
    let node_sa = aabb_surface_area(node_min, node_max);
    if node_sa <= 0.0 {
        return (0, count / 2);
    }

    let mut best_cost = f64::MAX;
    let mut best_axis = 0;
    let mut best_split = count / 2;

    // Try each axis
    for axis in 0..3 {
        let extent = node_max[axis] - node_min[axis];
        if extent <= 0.0 {
            continue;
        }

        // Bin triangles
        let mut bins = [(0usize, [f64::MAX; 3], [f64::MIN; 3]); SAH_BINS];
        for &idx in indices {
            let c = centroids[idx as usize][axis];
            let bin_idx = ((c - node_min[axis]) / extent * SAH_BINS as f64) as usize;
            let bin_idx = bin_idx.min(SAH_BINS - 1);
            bins[bin_idx].0 += 1;
            let (tri_min, tri_max) = tri_aabbs[idx as usize];
            for i in 0..3 {
                bins[bin_idx].1[i] = bins[bin_idx].1[i].min(tri_min[i]);
                bins[bin_idx].2[i] = bins[bin_idx].2[i].max(tri_max[i]);
            }
        }

        // Compute prefix sums for left side
        let mut left_count = [0usize; SAH_BINS];
        let mut left_aabb_min = [[f64::MAX; 3]; SAH_BINS];
        let mut left_aabb_max = [[f64::MIN; 3]; SAH_BINS];
        let mut running_count = 0;
        let mut running_min = [f64::MAX; 3];
        let mut running_max = [f64::MIN; 3];

        for i in 0..SAH_BINS {
            running_count += bins[i].0;
            left_count[i] = running_count;
            if bins[i].0 > 0 {
                for j in 0..3 {
                    running_min[j] = running_min[j].min(bins[i].1[j]);
                    running_max[j] = running_max[j].max(bins[i].2[j]);
                }
            }
            left_aabb_min[i] = running_min;
            left_aabb_max[i] = running_max;
        }

        // Compute suffix sums for right side and evaluate SAH
        running_count = 0;
        running_min = [f64::MAX; 3];
        running_max = [f64::MIN; 3];

        for i in (1..SAH_BINS).rev() {
            running_count += bins[i].0;
            if bins[i].0 > 0 {
                for j in 0..3 {
                    running_min[j] = running_min[j].min(bins[i].1[j]);
                    running_max[j] = running_max[j].max(bins[i].2[j]);
                }
            }

            let left_n = left_count[i - 1];
            let right_n = running_count;

            if left_n == 0 || right_n == 0 {
                continue;
            }

            let left_sa = aabb_surface_area(&left_aabb_min[i - 1], &left_aabb_max[i - 1]);
            let right_sa = aabb_surface_area(&running_min, &running_max);

            // SAH cost: traverse_cost + (left_sa/node_sa * left_n + right_sa/node_sa * right_n) * intersect_cost
            let cost = 1.0 + (left_sa * left_n as f64 + right_sa * right_n as f64) / node_sa;

            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = left_n;
            }
        }
    }

    (best_axis, best_split)
}

fn partition_by_axis(indices: &mut [u32], centroids: &[[f64; 3]], axis: usize, split_count: usize) {
    // Simple nth_element-like partition using selection
    // Sort indices by centroid on the given axis
    indices.sort_by(|&a, &b| {
        centroids[a as usize][axis]
            .partial_cmp(&centroids[b as usize][axis])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    // The first split_count elements are now on the left
    let _ = split_count; // split_count determines where we stop, sorting handles it
}

// ============================================================================
// WASM Generation
// ============================================================================

// Memory layout constants
const HEADER_SIZE: u32 = 16;
const NODE_SIZE: u32 = 56; // 6 × f64 (48) + 2 × u32 (8)
const TRIANGLE_SIZE: u32 = 72; // 9 × f64
const STACK_SIZE: u32 = 256; // 64 × u32

// Function indices (these must match the order we add functions)
const FN_IS_INSIDE: u32 = 0;
const FN_BOUNDS_MIN_X: u32 = 1;
const FN_BOUNDS_MIN_Y: u32 = 2;
const FN_BOUNDS_MIN_Z: u32 = 3;
const FN_BOUNDS_MAX_X: u32 = 4;
const FN_BOUNDS_MAX_Y: u32 = 5;
const FN_BOUNDS_MAX_Z: u32 = 6;

// Type indices
const TYPE_IS_INSIDE: u32 = 0; // (f64, f64, f64) -> f32
const TYPE_BOUNDS: u32 = 1; // () -> f64

fn generate_wasm(mesh: &ParsedMesh, bvh: &Bvh) -> Vec<u8> {
    let node_count = bvh.nodes.len() as u32;
    let triangle_count = mesh.triangles.len() as u32;

    // Calculate offsets
    let nodes_offset = HEADER_SIZE;
    let triangles_offset = nodes_offset + node_count * NODE_SIZE;
    let stack_offset = triangles_offset + triangle_count * TRIANGLE_SIZE;
    let total_data_size = stack_offset + STACK_SIZE;

    // Calculate required memory pages (64KB each)
    let memory_pages = ((total_data_size + 65535) / 65536) as u64;

    // Serialize data section
    let data_bytes = serialize_data(mesh, bvh, nodes_offset, triangles_offset);

    // Build WASM module
    let mut module = Module::new();

    // Type section
    let mut types = TypeSection::new();
    // Type 0: is_inside(f64, f64, f64) -> f32
    types.ty().function([ValType::F64, ValType::F64, ValType::F64], [ValType::F32]);
    // Type 1: bounds() -> f64
    types.ty().function([], [ValType::F64]);
    module.section(&types);

    // Function section
    let mut funcs = FunctionSection::new();
    funcs.function(TYPE_IS_INSIDE); // is_inside
    funcs.function(TYPE_BOUNDS); // get_bounds_min_x
    funcs.function(TYPE_BOUNDS); // get_bounds_min_y
    funcs.function(TYPE_BOUNDS); // get_bounds_min_z
    funcs.function(TYPE_BOUNDS); // get_bounds_max_x
    funcs.function(TYPE_BOUNDS); // get_bounds_max_y
    funcs.function(TYPE_BOUNDS); // get_bounds_max_z
    module.section(&funcs);

    // Memory section
    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: memory_pages,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    module.section(&memories);

    // Export section
    let mut exports = ExportSection::new();
    exports.export("is_inside", ExportKind::Func, FN_IS_INSIDE);
    exports.export("get_bounds_min_x", ExportKind::Func, FN_BOUNDS_MIN_X);
    exports.export("get_bounds_min_y", ExportKind::Func, FN_BOUNDS_MIN_Y);
    exports.export("get_bounds_min_z", ExportKind::Func, FN_BOUNDS_MIN_Z);
    exports.export("get_bounds_max_x", ExportKind::Func, FN_BOUNDS_MAX_X);
    exports.export("get_bounds_max_y", ExportKind::Func, FN_BOUNDS_MAX_Y);
    exports.export("get_bounds_max_z", ExportKind::Func, FN_BOUNDS_MAX_Z);
    module.section(&exports);

    // Code section
    let mut code = CodeSection::new();

    // is_inside function
    code.function(&generate_is_inside_function(
        nodes_offset,
        triangles_offset,
        stack_offset,
        &bvh.triangle_indices,
    ));

    // Bounds getter functions - read from root node AABB
    // Root node is at nodes_offset, AABB min is at offset 0, max at offset 24
    code.function(&generate_bounds_getter(nodes_offset, 0)); // min_x
    code.function(&generate_bounds_getter(nodes_offset, 8)); // min_y
    code.function(&generate_bounds_getter(nodes_offset, 16)); // min_z
    code.function(&generate_bounds_getter(nodes_offset, 24)); // max_x
    code.function(&generate_bounds_getter(nodes_offset, 32)); // max_y
    code.function(&generate_bounds_getter(nodes_offset, 40)); // max_z

    module.section(&code);

    // Data section
    let mut data = DataSection::new();
    data.active(0, &ConstExpr::i32_const(0), data_bytes);
    module.section(&data);

    module.finish()
}

fn serialize_data(
    mesh: &ParsedMesh,
    bvh: &Bvh,
    nodes_offset: u32,
    triangles_offset: u32,
) -> Vec<u8> {
    let node_count = bvh.nodes.len() as u32;
    let triangle_count = mesh.triangles.len() as u32;

    let total_size = (triangles_offset + triangle_count * TRIANGLE_SIZE + STACK_SIZE) as usize;
    let mut data = vec![0u8; total_size];

    // Write header
    data[0..4].copy_from_slice(&node_count.to_le_bytes());
    data[4..8].copy_from_slice(&triangle_count.to_le_bytes());
    data[8..12].copy_from_slice(&nodes_offset.to_le_bytes());
    data[12..16].copy_from_slice(&triangles_offset.to_le_bytes());

    // Write BVH nodes
    for (i, node) in bvh.nodes.iter().enumerate() {
        let offset = nodes_offset as usize + i * NODE_SIZE as usize;
        // AABB min (24 bytes)
        for j in 0..3 {
            let bytes = node.aabb_min[j].to_le_bytes();
            data[offset + j * 8..offset + j * 8 + 8].copy_from_slice(&bytes);
        }
        // AABB max (24 bytes)
        for j in 0..3 {
            let bytes = node.aabb_max[j].to_le_bytes();
            data[offset + 24 + j * 8..offset + 24 + j * 8 + 8].copy_from_slice(&bytes);
        }
        // left_or_start (4 bytes)
        data[offset + 48..offset + 52].copy_from_slice(&node.left_or_start.to_le_bytes());
        // right_or_count (4 bytes)
        // For leaf nodes, set MSB to distinguish from internal nodes
        let right_or_count = if node.is_leaf {
            node.right_or_count | 0x80000000
        } else {
            node.right_or_count
        };
        data[offset + 52..offset + 56].copy_from_slice(&right_or_count.to_le_bytes());
    }

    // Write triangles (reordered by BVH indices)
    for (i, &tri_idx) in bvh.triangle_indices.iter().enumerate() {
        let tri = &mesh.triangles[tri_idx as usize];
        let offset = triangles_offset as usize + i * TRIANGLE_SIZE as usize;

        // Write v0, v1, v2 (each 24 bytes)
        for (vi, v) in [tri.v0, tri.v1, tri.v2].iter().enumerate() {
            for j in 0..3 {
                let bytes = v[j].to_le_bytes();
                data[offset + vi * 24 + j * 8..offset + vi * 24 + j * 8 + 8].copy_from_slice(&bytes);
            }
        }
    }

    data
}

fn generate_bounds_getter(nodes_offset: u32, field_offset: u32) -> Function {
    let mut f = Function::new([]);
    // Load f64 from memory at nodes_offset + field_offset
    f.instruction(&Instruction::I32Const((nodes_offset + field_offset) as i32));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 0,
        align: 3, // 2^3 = 8 byte alignment
        memory_index: 0,
    }));
    f.instruction(&Instruction::End);
    f
}

fn generate_is_inside_function(
    nodes_offset: u32,
    triangles_offset: u32,
    stack_offset: u32,
    _triangle_indices: &[u32],
) -> Function {
    // Local variables:
    // 0-2: x, y, z (parameters)
    // 3: hit_count (i32)
    // 4: stack_ptr (i32) - index into stack
    // 5: current_node_idx (i32)
    // 6: node_offset (i32)
    // 7-12: node AABB (f64) - min_x, min_y, min_z, max_x, max_y, max_z
    // 13: left_or_start (i32)
    // 14: right_or_count (i32)
    // 15: tri_idx (i32)
    // 16: tri_offset (i32)
    // 17-25: triangle vertices (f64) - v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z
    // 26-28: edge1 (f64)
    // 29-31: edge2 (f64)
    // 32-34: h (f64) - cross product result
    // 35: a (f64)
    // 36: f (f64)
    // 37-39: s (f64)
    // 40: u (f64)
    // 41-43: q (f64)
    // 44: v_param (f64)
    // 45: t (f64)

    let locals = vec![
        (1, ValType::I32),  // hit_count
        (1, ValType::I32),  // stack_ptr
        (1, ValType::I32),  // current_node_idx
        (1, ValType::I32),  // node_offset
        (6, ValType::F64),  // node AABB: min_x, min_y, min_z, max_x, max_y, max_z
        (1, ValType::I32),  // left_or_start
        (1, ValType::I32),  // right_or_count
        (1, ValType::I32),  // tri_idx
        (1, ValType::I32),  // tri_offset
        (9, ValType::F64),  // triangle vertices
        (3, ValType::F64),  // edge1
        (3, ValType::F64),  // edge2
        (3, ValType::F64),  // h
        (1, ValType::F64),  // a
        (1, ValType::F64),  // f
        (3, ValType::F64),  // s
        (1, ValType::F64),  // u
        (3, ValType::F64),  // q
        (1, ValType::F64),  // v_param
        (1, ValType::F64),  // t
    ];

    let mut f = Function::new(locals);

    // Local indices (after parameters x=0, y=1, z=2)
    let hit_count: u32 = 3;
    let stack_ptr: u32 = 4;
    let current_node_idx: u32 = 5;
    let node_offset: u32 = 6;
    let node_min_x: u32 = 7;
    let node_min_y: u32 = 8;
    let node_min_z: u32 = 9;
    let node_max_x: u32 = 10;
    let node_max_y: u32 = 11;
    let node_max_z: u32 = 12;
    let left_or_start: u32 = 13;
    let right_or_count: u32 = 14;
    let tri_idx: u32 = 15;
    let tri_offset: u32 = 16;
    let v0x: u32 = 17;
    let v0y: u32 = 18;
    let v0z: u32 = 19;
    let v1x: u32 = 20;
    let v1y: u32 = 21;
    let v1z: u32 = 22;
    let v2x: u32 = 23;
    let v2y: u32 = 24;
    let v2z: u32 = 25;
    let edge1_x: u32 = 26;
    let edge1_y: u32 = 27;
    let edge1_z: u32 = 28;
    let edge2_x: u32 = 29;
    let edge2_y: u32 = 30;
    let edge2_z: u32 = 31;
    let h_x: u32 = 32;
    let h_y: u32 = 33;
    let h_z: u32 = 34;
    let a: u32 = 35;
    let ff: u32 = 36;
    let s_x: u32 = 37;
    let s_y: u32 = 38;
    let s_z: u32 = 39;
    let u: u32 = 40;
    let q_x: u32 = 41;
    let q_y: u32 = 42;
    let q_z: u32 = 43;
    let v_param: u32 = 44;
    let t: u32 = 45;

    // Add small jitter to Y and Z to avoid axis-aligned edge intersections
    // This prevents the ray from passing exactly through mesh edges where
    // triangles meet, which causes double-counting or missed intersections.
    // Use different jitter values for Y and Z to also avoid diagonal edges.
    // The jitter must be tiny (1e-9) to avoid affecting thin-walled geometry
    // like gyroids, while still being large enough to avoid floating-point
    // coincidence at triangle edges.
    const RAY_JITTER_Y: f64 = 1.234567e-9;
    const RAY_JITTER_Z: f64 = 2.345678e-9;
    f.instruction(&Instruction::LocalGet(1)); // y
    f.instruction(&Instruction::F64Const(RAY_JITTER_Y));
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(1)); // y = y + jitter_y

    f.instruction(&Instruction::LocalGet(2)); // z
    f.instruction(&Instruction::F64Const(RAY_JITTER_Z));
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(2)); // z = z + jitter_z

    // Initialize hit_count = 0
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::LocalSet(hit_count));

    // Initialize stack_ptr = 0
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::LocalSet(stack_ptr));

    // Push root node (index 0) onto stack
    // stack[stack_ptr] = 0
    f.instruction(&Instruction::I32Const(stack_offset as i32));
    f.instruction(&Instruction::I32Const(0)); // root node index
    f.instruction(&Instruction::I32Store(wasm_encoder::MemArg {
        offset: 0,
        align: 2,
        memory_index: 0,
    }));

    // stack_ptr = 1
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::LocalSet(stack_ptr));

    // Main traversal loop
    f.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty)); // outer block for break
    f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty)); // main loop

    // if stack_ptr == 0, break
    f.instruction(&Instruction::LocalGet(stack_ptr));
    f.instruction(&Instruction::I32Eqz);
    f.instruction(&Instruction::BrIf(1)); // break to outer block

    // Pop node from stack: stack_ptr--; current_node_idx = stack[stack_ptr]
    f.instruction(&Instruction::LocalGet(stack_ptr));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Sub);
    f.instruction(&Instruction::LocalTee(stack_ptr));
    f.instruction(&Instruction::I32Const(4)); // 4 bytes per u32
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(stack_offset as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::I32Load(wasm_encoder::MemArg {
        offset: 0,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(current_node_idx));

    // Compute node_offset = nodes_offset + current_node_idx * 56
    f.instruction(&Instruction::LocalGet(current_node_idx));
    f.instruction(&Instruction::I32Const(NODE_SIZE as i32));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(nodes_offset as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(node_offset));

    // Load node AABB
    // min_x
    f.instruction(&Instruction::LocalGet(node_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 0,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(node_min_x));
    // min_y
    f.instruction(&Instruction::LocalGet(node_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 8,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(node_min_y));
    // min_z
    f.instruction(&Instruction::LocalGet(node_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 16,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(node_min_z));
    // max_x
    f.instruction(&Instruction::LocalGet(node_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 24,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(node_max_x));
    // max_y
    f.instruction(&Instruction::LocalGet(node_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 32,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(node_max_y));
    // max_z
    f.instruction(&Instruction::LocalGet(node_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 40,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(node_max_z));

    // Load left_or_start and right_or_count
    f.instruction(&Instruction::LocalGet(node_offset));
    f.instruction(&Instruction::I32Load(wasm_encoder::MemArg {
        offset: 48,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(left_or_start));

    f.instruction(&Instruction::LocalGet(node_offset));
    f.instruction(&Instruction::I32Load(wasm_encoder::MemArg {
        offset: 52,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(right_or_count));

    // Ray-AABB test for ray in +X direction from (x, y, z)
    // For +X ray: check if y in [min_y, max_y] and z in [min_z, max_z] and max_x >= x
    // (y >= min_y) && (y <= max_y) && (z >= min_z) && (z <= max_z) && (max_x >= x)
    f.instruction(&Instruction::LocalGet(1)); // y
    f.instruction(&Instruction::LocalGet(node_min_y));
    f.instruction(&Instruction::F64Ge);
    f.instruction(&Instruction::LocalGet(1)); // y
    f.instruction(&Instruction::LocalGet(node_max_y));
    f.instruction(&Instruction::F64Le);
    f.instruction(&Instruction::I32And);
    f.instruction(&Instruction::LocalGet(2)); // z
    f.instruction(&Instruction::LocalGet(node_min_z));
    f.instruction(&Instruction::F64Ge);
    f.instruction(&Instruction::I32And);
    f.instruction(&Instruction::LocalGet(2)); // z
    f.instruction(&Instruction::LocalGet(node_max_z));
    f.instruction(&Instruction::F64Le);
    f.instruction(&Instruction::I32And);
    f.instruction(&Instruction::LocalGet(node_max_x));
    f.instruction(&Instruction::LocalGet(0)); // x
    f.instruction(&Instruction::F64Ge);
    f.instruction(&Instruction::I32And);

    // If ray doesn't hit AABB, continue to next iteration
    f.instruction(&Instruction::I32Eqz);
    f.instruction(&Instruction::BrIf(0)); // continue loop

    // Check if leaf node by testing MSB (set for leaves in serialize_data)
    // (right_or_count & 0x80000000) == 0 means internal node
    f.instruction(&Instruction::LocalGet(right_or_count));
    f.instruction(&Instruction::I32Const(0x80000000_u32 as i32));
    f.instruction(&Instruction::I32And);
    f.instruction(&Instruction::I32Eqz);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));

    // Internal node: push both children onto stack
    // Push left child (stored in left_or_start)
    f.instruction(&Instruction::LocalGet(stack_ptr));
    f.instruction(&Instruction::I32Const(4));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(stack_offset as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalGet(left_or_start));
    f.instruction(&Instruction::I32Store(wasm_encoder::MemArg {
        offset: 0,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalGet(stack_ptr));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(stack_ptr));

    // Push right child (stored in right_or_count, no MSB for internal nodes)
    f.instruction(&Instruction::LocalGet(stack_ptr));
    f.instruction(&Instruction::I32Const(4));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(stack_offset as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalGet(right_or_count));
    f.instruction(&Instruction::I32Store(wasm_encoder::MemArg {
        offset: 0,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalGet(stack_ptr));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(stack_ptr));

    f.instruction(&Instruction::Else);

    // Leaf node: test triangles
    // right_or_count has MSB set, clear it to get count
    // Actually we need to mask it: count = right_or_count & 0x7FFFFFFF
    f.instruction(&Instruction::LocalGet(right_or_count));
    f.instruction(&Instruction::I32Const(0x7FFFFFFF_u32 as i32));
    f.instruction(&Instruction::I32And);
    f.instruction(&Instruction::LocalSet(right_or_count)); // Now it's the actual count

    // tri_idx = left_or_start (first triangle index in reordered array)
    f.instruction(&Instruction::LocalGet(left_or_start));
    f.instruction(&Instruction::LocalSet(tri_idx));

    // Loop over triangles in leaf
    f.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty)); // break target
    f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));

    // if tri_idx >= left_or_start + count, break
    f.instruction(&Instruction::LocalGet(tri_idx));
    f.instruction(&Instruction::LocalGet(left_or_start));
    f.instruction(&Instruction::LocalGet(right_or_count));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::I32GeU);
    f.instruction(&Instruction::BrIf(1)); // break

    // Load triangle
    // tri_offset = triangles_offset + tri_idx * 72
    f.instruction(&Instruction::LocalGet(tri_idx));
    f.instruction(&Instruction::I32Const(TRIANGLE_SIZE as i32));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(triangles_offset as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(tri_offset));

    // Load v0
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 0, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v0x));
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 8, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v0y));
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 16, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v0z));

    // Load v1
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 24, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v1x));
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 32, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v1y));
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 40, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v1z));

    // Load v2
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 48, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v2x));
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 56, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v2y));
    f.instruction(&Instruction::LocalGet(tri_offset));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg { offset: 64, align: 3, memory_index: 0 }));
    f.instruction(&Instruction::LocalSet(v2z));

    // Möller-Trumbore ray-triangle intersection
    // Ray: origin = (x, y, z), direction = (1, 0, 0)

    // edge1 = v1 - v0
    f.instruction(&Instruction::LocalGet(v1x));
    f.instruction(&Instruction::LocalGet(v0x));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(edge1_x));
    f.instruction(&Instruction::LocalGet(v1y));
    f.instruction(&Instruction::LocalGet(v0y));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(edge1_y));
    f.instruction(&Instruction::LocalGet(v1z));
    f.instruction(&Instruction::LocalGet(v0z));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(edge1_z));

    // edge2 = v2 - v0
    f.instruction(&Instruction::LocalGet(v2x));
    f.instruction(&Instruction::LocalGet(v0x));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(edge2_x));
    f.instruction(&Instruction::LocalGet(v2y));
    f.instruction(&Instruction::LocalGet(v0y));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(edge2_y));
    f.instruction(&Instruction::LocalGet(v2z));
    f.instruction(&Instruction::LocalGet(v0z));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(edge2_z));

    // h = ray_dir × edge2 = (1,0,0) × edge2 = (0*edge2_z - 0*edge2_y, 0*edge2_x - 1*edge2_z, 1*edge2_y - 0*edge2_x)
    //   = (0, -edge2_z, edge2_y)
    f.instruction(&Instruction::F64Const(0.0));
    f.instruction(&Instruction::LocalSet(h_x));
    f.instruction(&Instruction::LocalGet(edge2_z));
    f.instruction(&Instruction::F64Neg);
    f.instruction(&Instruction::LocalSet(h_y));
    f.instruction(&Instruction::LocalGet(edge2_y));
    f.instruction(&Instruction::LocalSet(h_z));

    // a = edge1 · h
    f.instruction(&Instruction::LocalGet(edge1_x));
    f.instruction(&Instruction::LocalGet(h_x));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(edge1_y));
    f.instruction(&Instruction::LocalGet(h_y));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalGet(edge1_z));
    f.instruction(&Instruction::LocalGet(h_z));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(a));

    // if |a| < epsilon, skip (ray parallel to triangle)
    f.instruction(&Instruction::LocalGet(a));
    f.instruction(&Instruction::F64Abs);
    f.instruction(&Instruction::F64Const(1e-10));
    f.instruction(&Instruction::F64Lt);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    // Skip to next triangle
    f.instruction(&Instruction::LocalGet(tri_idx));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(tri_idx));
    f.instruction(&Instruction::Br(1)); // continue triangle loop
    f.instruction(&Instruction::End);

    // f = 1/a
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalGet(a));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::LocalSet(ff));

    // s = ray_origin - v0
    f.instruction(&Instruction::LocalGet(0)); // x
    f.instruction(&Instruction::LocalGet(v0x));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(s_x));
    f.instruction(&Instruction::LocalGet(1)); // y
    f.instruction(&Instruction::LocalGet(v0y));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(s_y));
    f.instruction(&Instruction::LocalGet(2)); // z
    f.instruction(&Instruction::LocalGet(v0z));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(s_z));

    // u = f * (s · h)
    f.instruction(&Instruction::LocalGet(ff));
    f.instruction(&Instruction::LocalGet(s_x));
    f.instruction(&Instruction::LocalGet(h_x));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(s_y));
    f.instruction(&Instruction::LocalGet(h_y));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalGet(s_z));
    f.instruction(&Instruction::LocalGet(h_z));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalSet(u));

    // if u < -epsilon or u > 1+epsilon, skip (with tolerance for edge hits)
    const BARY_EPSILON: f64 = 1e-9;
    f.instruction(&Instruction::LocalGet(u));
    f.instruction(&Instruction::F64Const(-BARY_EPSILON));
    f.instruction(&Instruction::F64Lt);
    f.instruction(&Instruction::LocalGet(u));
    f.instruction(&Instruction::F64Const(1.0 + BARY_EPSILON));
    f.instruction(&Instruction::F64Gt);
    f.instruction(&Instruction::I32Or);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::LocalGet(tri_idx));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(tri_idx));
    f.instruction(&Instruction::Br(1));
    f.instruction(&Instruction::End);

    // q = s × edge1
    f.instruction(&Instruction::LocalGet(s_y));
    f.instruction(&Instruction::LocalGet(edge1_z));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(s_z));
    f.instruction(&Instruction::LocalGet(edge1_y));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(q_x));

    f.instruction(&Instruction::LocalGet(s_z));
    f.instruction(&Instruction::LocalGet(edge1_x));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(s_x));
    f.instruction(&Instruction::LocalGet(edge1_z));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(q_y));

    f.instruction(&Instruction::LocalGet(s_x));
    f.instruction(&Instruction::LocalGet(edge1_y));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(s_y));
    f.instruction(&Instruction::LocalGet(edge1_x));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(q_z));

    // v = f * (ray_dir · q) = f * (1*q_x + 0*q_y + 0*q_z) = f * q_x
    f.instruction(&Instruction::LocalGet(ff));
    f.instruction(&Instruction::LocalGet(q_x));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalSet(v_param));

    // if v < -epsilon or u + v > 1+epsilon, skip (with tolerance for edge hits)
    f.instruction(&Instruction::LocalGet(v_param));
    f.instruction(&Instruction::F64Const(-BARY_EPSILON));
    f.instruction(&Instruction::F64Lt);
    f.instruction(&Instruction::LocalGet(u));
    f.instruction(&Instruction::LocalGet(v_param));
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::F64Const(1.0 + BARY_EPSILON));
    f.instruction(&Instruction::F64Gt);
    f.instruction(&Instruction::I32Or);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::LocalGet(tri_idx));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(tri_idx));
    f.instruction(&Instruction::Br(1));
    f.instruction(&Instruction::End);

    // t = f * (edge2 · q)
    f.instruction(&Instruction::LocalGet(ff));
    f.instruction(&Instruction::LocalGet(edge2_x));
    f.instruction(&Instruction::LocalGet(q_x));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(edge2_y));
    f.instruction(&Instruction::LocalGet(q_y));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalGet(edge2_z));
    f.instruction(&Instruction::LocalGet(q_z));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalSet(t));

    // if t > epsilon, we have a hit
    f.instruction(&Instruction::LocalGet(t));
    f.instruction(&Instruction::F64Const(1e-10));
    f.instruction(&Instruction::F64Gt);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    // hit_count++
    f.instruction(&Instruction::LocalGet(hit_count));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(hit_count));
    f.instruction(&Instruction::End);

    // tri_idx++
    f.instruction(&Instruction::LocalGet(tri_idx));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(tri_idx));

    // Continue triangle loop
    f.instruction(&Instruction::Br(0));
    f.instruction(&Instruction::End); // end loop
    f.instruction(&Instruction::End); // end block

    f.instruction(&Instruction::End); // end if/else for leaf

    // Continue main BVH traversal loop
    f.instruction(&Instruction::Br(0));
    f.instruction(&Instruction::End); // end loop
    f.instruction(&Instruction::End); // end block

    // Return (hit_count & 1) as f32
    // Odd count = inside (1.0), even count = outside (0.0)
    f.instruction(&Instruction::LocalGet(hit_count));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32And);
    f.instruction(&Instruction::F32ConvertI32S);
    f.instruction(&Instruction::End);

    f
}

fn process_and_generate_wasm(stl_data: &[u8], config: &StlImportConfig) -> Result<Vec<u8>, &'static str> {
    let mesh = process_stl(stl_data, config)?;
    let bvh = build_bvh(&mesh.triangles);
    Ok(generate_wasm(&mesh, &bvh))
}

// ============================================================================
// Operator Entry Points
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    // Read STL data from input 0
    let stl_len = unsafe { get_input_len(0) } as usize;
    let mut stl_buf = vec![0u8; stl_len];
    if stl_len > 0 {
        unsafe { get_input_data(0, stl_buf.as_mut_ptr() as i32, stl_len as i32) };
    }

    // Read config from input 1
    let config = {
        let cfg_len = unsafe { get_input_len(1) } as usize;
        if cfg_len == 0 {
            StlImportConfig::default()
        } else {
            let mut cfg_buf = vec![0u8; cfg_len];
            unsafe { get_input_data(1, cfg_buf.as_mut_ptr() as i32, cfg_len as i32) };
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            ciborium::de::from_reader::<StlImportConfig, _>(&mut cursor).unwrap_or_default()
        }
    };

    // Process STL and generate WASM
    let output = match process_and_generate_wasm(&stl_buf, &config) {
        Ok(wasm) => wasm,
        Err(_) => {
            // Return empty/minimal WASM on error
            // This shouldn't happen in normal use
            Vec::new()
        }
    };

    unsafe {
        post_output(0, output.as_ptr() as i32, output.len() as i32);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let schema = "{ scale: float .default 1.0, translate: [float, float, float] .default [0,0,0], center: bool .default false }".to_string();
        let metadata = OperatorMetadata {
            name: "stl_import_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };

        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out)
            .expect("stl_import_operator metadata CBOR serialization should not fail");
        out
    });

    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}
