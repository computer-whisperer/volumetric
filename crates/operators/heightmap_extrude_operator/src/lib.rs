//! Heightmap Extrude Operator.
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
//! - Reads CBOR configuration from input 0 (width, depth, height, clip)
//! - Reads PNG/JPEG/BMP/GIF image data from input 1 (16-bit grayscale supported)
//! - Produces a WASM model with bilinear-interpolated heightmap extrusion
//! - Pixels with normalized height < clip threshold produce no geometry
//!
//! Memory Layout in Generated WASM:
//! - Offset 0-3:   width_pixels: u32
//! - Offset 4-7:   height_pixels: u32
//! - Offset 8-15:  config.width: f64 (model width in meters)
//! - Offset 16-23: config.depth: f64 (model depth in meters)
//! - Offset 24-31: config.height: f64 (max height for full white)
//! - Offset 32-39: config.clip: f64 (clip threshold, 0.0-1.0)
//! - Offset 40+:   Height data (width_pixels * height_pixels * f32, row-major, normalized 0-1)

use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ExportKind, ExportSection, Function, FunctionSection,
    Instruction, MemorySection, MemoryType, Module, TypeSection, ValType,
};

// ============================================================================
// Operator Metadata Types
// ============================================================================

#[derive(Clone, Debug, serde::Serialize)]
#[allow(dead_code)]
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
struct HeightmapConfig {
    #[serde(default = "default_dimension")]
    width: f64,
    #[serde(default = "default_dimension")]
    depth: f64,
    #[serde(default = "default_dimension")]
    height: f64,
    /// Clip threshold: pixels with normalized height < clip produce no geometry
    #[serde(default)]
    clip: f64,
}

fn default_dimension() -> f64 {
    1.0
}

impl Default for HeightmapConfig {
    fn default() -> Self {
        Self {
            width: 1.0,
            depth: 1.0,
            height: 1.0,
            clip: 0.0,
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
// Image Processing
// ============================================================================

struct Heightmap {
    width: u32,
    height: u32,
    /// Normalized height values [0.0, 1.0], row-major order
    data: Vec<f32>,
}

fn parse_image_to_heightmap(image_data: &[u8]) -> Result<Heightmap, &'static str> {
    let img = image::load_from_memory(image_data).map_err(|_| "Failed to decode image")?;

    let width = img.width();
    let height = img.height();

    if width == 0 || height == 0 {
        return Err("Image has zero dimensions");
    }

    // Convert to 16-bit grayscale to preserve full precision for 16-bit PNGs
    // 8-bit images are upconverted without loss
    let gray = img.to_luma16();
    let data: Vec<f32> = gray.pixels().map(|p| p.0[0] as f32 / 65535.0).collect();

    Ok(Heightmap {
        width,
        height,
        data,
    })
}

// ============================================================================
// WASM Generation
// ============================================================================

// Memory layout constants
// Header: 2 × u32 (8) + 4 × f64 (32) = 40 bytes
// - Offset 0-3: width_pixels (u32)
// - Offset 4-7: height_pixels (u32)
// - Offset 8-15: config.width (f64)
// - Offset 16-23: config.depth (f64)
// - Offset 24-31: config.height (f64)
// - Offset 32-39: config.clip (f64)
const HEADER_SIZE: u32 = 40;
const DATA_OFFSET: u32 = HEADER_SIZE;

// Function indices
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

fn generate_wasm(heightmap: &Heightmap, config: &HeightmapConfig) -> Vec<u8> {
    let pixel_count = heightmap.width * heightmap.height;
    let data_size = pixel_count * 4; // f32 per pixel
    let total_data_size = HEADER_SIZE + data_size;

    // Calculate required memory pages (64KB each)
    let memory_pages = ((total_data_size + 65535) / 65536) as u64;

    // Serialize data section
    let data_bytes = serialize_data(heightmap, config);

    // Build WASM module
    let mut module = Module::new();

    // Type section
    let mut types = TypeSection::new();
    // Type 0: is_inside(f64, f64, f64) -> f32
    types
        .ty()
        .function([ValType::F64, ValType::F64, ValType::F64], [ValType::F32]);
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

    // is_inside function with bilinear interpolation
    code.function(&generate_is_inside_function(heightmap.width, heightmap.height));

    // Bounds getter functions
    // Centered origin: x spans [-width/2, +width/2], z spans [-depth/2, +depth/2]
    code.function(&generate_bounds_const(-config.width / 2.0)); // min_x
    code.function(&generate_bounds_const(0.0)); // min_y
    code.function(&generate_bounds_const(-config.depth / 2.0)); // min_z
    code.function(&generate_bounds_const(config.width / 2.0)); // max_x
    code.function(&generate_bounds_const(config.height)); // max_y
    code.function(&generate_bounds_const(config.depth / 2.0)); // max_z

    module.section(&code);

    // Data section
    let mut data = DataSection::new();
    data.active(0, &ConstExpr::i32_const(0), data_bytes);
    module.section(&data);

    module.finish()
}

fn serialize_data(heightmap: &Heightmap, config: &HeightmapConfig) -> Vec<u8> {
    let pixel_count = heightmap.width * heightmap.height;
    let total_size = (HEADER_SIZE + pixel_count * 4) as usize;
    let mut data = vec![0u8; total_size];

    // Write header
    // Offset 0-3: width_pixels (u32)
    data[0..4].copy_from_slice(&heightmap.width.to_le_bytes());
    // Offset 4-7: height_pixels (u32)
    data[4..8].copy_from_slice(&heightmap.height.to_le_bytes());
    // Offset 8-15: config.width (f64)
    data[8..16].copy_from_slice(&config.width.to_le_bytes());
    // Offset 16-23: config.depth (f64)
    data[16..24].copy_from_slice(&config.depth.to_le_bytes());
    // Offset 24-31: config.height (f64)
    data[24..32].copy_from_slice(&config.height.to_le_bytes());
    // Offset 32-39: config.clip (f64)
    data[32..40].copy_from_slice(&config.clip.to_le_bytes());

    // Write heightmap data (row-major, f32)
    for (i, &h) in heightmap.data.iter().enumerate() {
        let offset = DATA_OFFSET as usize + i * 4;
        data[offset..offset + 4].copy_from_slice(&h.to_le_bytes());
    }

    data
}

fn generate_bounds_const(value: f64) -> Function {
    let mut f = Function::new([]);
    f.instruction(&Instruction::F64Const(value));
    f.instruction(&Instruction::End);
    f
}

fn generate_is_inside_function(width_pixels: u32, height_pixels: u32) -> Function {
    // Parameters: x (0), y (1), z (2)
    // Local variables:
    // 3: config_width (f64)
    // 4: config_depth (f64)
    // 5: config_height (f64)
    // 6: config_clip (f64)
    // 7: nx (f64) - normalized x coordinate [0, 1]
    // 8: nz (f64) - normalized z coordinate [0, 1]
    // 9: px (f64) - pixel x coordinate (floating point)
    // 10: pz (f64) - pixel z coordinate (floating point)
    // 11: px_floor (i32) - integer part of px
    // 12: pz_floor (i32) - integer part of pz
    // 13: px_frac (f64) - fractional part of px
    // 14: pz_frac (f64) - fractional part of pz
    // 15: px1 (i32) - clamped px_floor + 1
    // 16: pz1 (i32) - clamped pz_floor + 1
    // 17: h00 (f64) - height at (px_floor, pz_floor)
    // 18: h10 (f64) - height at (px1, pz_floor)
    // 19: h01 (f64) - height at (px_floor, pz1)
    // 20: h11 (f64) - height at (px1, pz1)
    // 21: h_interp (f64) - bilinearly interpolated height
    // 22: surface_height (f64) - scaled interpolated height

    let locals = vec![
        (4, ValType::F64), // config_width, config_depth, config_height, config_clip
        (2, ValType::F64), // nx, nz
        (2, ValType::F64), // px, pz
        (2, ValType::I32), // px_floor, pz_floor
        (2, ValType::F64), // px_frac, pz_frac
        (2, ValType::I32), // px1, pz1
        (4, ValType::F64), // h00, h10, h01, h11
        (2, ValType::F64), // h_interp, surface_height
    ];

    let mut f = Function::new(locals);

    // Local indices
    let x: u32 = 0;
    let y: u32 = 1;
    let z: u32 = 2;
    let config_width: u32 = 3;
    let config_depth: u32 = 4;
    let config_height: u32 = 5;
    let config_clip: u32 = 6;
    let nx: u32 = 7;
    let nz: u32 = 8;
    let px: u32 = 9;
    let pz: u32 = 10;
    let px_floor: u32 = 11;
    let pz_floor: u32 = 12;
    let px_frac: u32 = 13;
    let pz_frac: u32 = 14;
    let px1: u32 = 15;
    let pz1: u32 = 16;
    let h00: u32 = 17;
    let h10: u32 = 18;
    let h01: u32 = 19;
    let h11: u32 = 20;
    let h_interp: u32 = 21;
    let surface_height: u32 = 22;

    // Load config from memory
    // config_width at offset 8
    f.instruction(&Instruction::I32Const(8));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 0,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(config_width));

    // config_depth at offset 16
    f.instruction(&Instruction::I32Const(16));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 0,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(config_depth));

    // config_height at offset 24
    f.instruction(&Instruction::I32Const(24));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 0,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(config_height));

    // config_clip at offset 32
    f.instruction(&Instruction::I32Const(32));
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 0,
        align: 3,
        memory_index: 0,
    }));
    f.instruction(&Instruction::LocalSet(config_clip));

    // Check if y < 0: return 0 (below model)
    f.instruction(&Instruction::LocalGet(y));
    f.instruction(&Instruction::F64Const(0.0));
    f.instruction(&Instruction::F64Lt);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::F32Const(0.0));
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);

    // Compute normalized coordinates (centered origin):
    // nx = (x + width/2) / width
    f.instruction(&Instruction::LocalGet(x));
    f.instruction(&Instruction::LocalGet(config_width));
    f.instruction(&Instruction::F64Const(2.0));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalGet(config_width));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::LocalSet(nx));

    // nz = (z + depth/2) / depth
    f.instruction(&Instruction::LocalGet(z));
    f.instruction(&Instruction::LocalGet(config_depth));
    f.instruction(&Instruction::F64Const(2.0));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalGet(config_depth));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::LocalSet(nz));

    // Check bounds: if nx < 0 or nx > 1 or nz < 0 or nz > 1, return 0
    f.instruction(&Instruction::LocalGet(nx));
    f.instruction(&Instruction::F64Const(0.0));
    f.instruction(&Instruction::F64Lt);
    f.instruction(&Instruction::LocalGet(nx));
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::F64Gt);
    f.instruction(&Instruction::I32Or);
    f.instruction(&Instruction::LocalGet(nz));
    f.instruction(&Instruction::F64Const(0.0));
    f.instruction(&Instruction::F64Lt);
    f.instruction(&Instruction::I32Or);
    f.instruction(&Instruction::LocalGet(nz));
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::F64Gt);
    f.instruction(&Instruction::I32Or);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::F32Const(0.0));
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);

    // Map to pixel coordinates:
    // px = nx * (width_pixels - 1)
    // pz = nz * (height_pixels - 1)
    f.instruction(&Instruction::LocalGet(nx));
    f.instruction(&Instruction::F64Const((width_pixels - 1) as f64));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalSet(px));

    f.instruction(&Instruction::LocalGet(nz));
    f.instruction(&Instruction::F64Const((height_pixels - 1) as f64));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalSet(pz));

    // Compute floor and fractional parts
    // px_floor = floor(px)
    f.instruction(&Instruction::LocalGet(px));
    f.instruction(&Instruction::F64Floor);
    f.instruction(&Instruction::I32TruncF64S);
    f.instruction(&Instruction::LocalSet(px_floor));

    // pz_floor = floor(pz)
    f.instruction(&Instruction::LocalGet(pz));
    f.instruction(&Instruction::F64Floor);
    f.instruction(&Instruction::I32TruncF64S);
    f.instruction(&Instruction::LocalSet(pz_floor));

    // px_frac = px - floor(px)
    f.instruction(&Instruction::LocalGet(px));
    f.instruction(&Instruction::LocalGet(px_floor));
    f.instruction(&Instruction::F64ConvertI32S);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(px_frac));

    // pz_frac = pz - floor(pz)
    f.instruction(&Instruction::LocalGet(pz));
    f.instruction(&Instruction::LocalGet(pz_floor));
    f.instruction(&Instruction::F64ConvertI32S);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(pz_frac));

    // Clamp px_floor to [0, width_pixels - 1]
    // if px_floor < 0: px_floor = 0
    f.instruction(&Instruction::LocalGet(px_floor));
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::I32LtS);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::LocalSet(px_floor));
    f.instruction(&Instruction::End);
    // if px_floor >= width_pixels: px_floor = width_pixels - 1
    f.instruction(&Instruction::LocalGet(px_floor));
    f.instruction(&Instruction::I32Const((width_pixels - 1) as i32));
    f.instruction(&Instruction::I32GtS);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::I32Const((width_pixels - 1) as i32));
    f.instruction(&Instruction::LocalSet(px_floor));
    f.instruction(&Instruction::End);

    // Clamp pz_floor to [0, height_pixels - 1]
    f.instruction(&Instruction::LocalGet(pz_floor));
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::I32LtS);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::LocalSet(pz_floor));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::LocalGet(pz_floor));
    f.instruction(&Instruction::I32Const((height_pixels - 1) as i32));
    f.instruction(&Instruction::I32GtS);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::I32Const((height_pixels - 1) as i32));
    f.instruction(&Instruction::LocalSet(pz_floor));
    f.instruction(&Instruction::End);

    // px1 = min(px_floor + 1, width_pixels - 1)
    f.instruction(&Instruction::LocalGet(px_floor));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(px1));
    f.instruction(&Instruction::LocalGet(px1));
    f.instruction(&Instruction::I32Const((width_pixels - 1) as i32));
    f.instruction(&Instruction::I32GtS);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::I32Const((width_pixels - 1) as i32));
    f.instruction(&Instruction::LocalSet(px1));
    f.instruction(&Instruction::End);

    // pz1 = min(pz_floor + 1, height_pixels - 1)
    f.instruction(&Instruction::LocalGet(pz_floor));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(pz1));
    f.instruction(&Instruction::LocalGet(pz1));
    f.instruction(&Instruction::I32Const((height_pixels - 1) as i32));
    f.instruction(&Instruction::I32GtS);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::I32Const((height_pixels - 1) as i32));
    f.instruction(&Instruction::LocalSet(pz1));
    f.instruction(&Instruction::End);

    // Load height values at four corners
    // Memory offset for pixel (x, z) = DATA_OFFSET + (z * width_pixels + x) * 4

    // h00 at (px_floor, pz_floor)
    // offset = DATA_OFFSET + (pz_floor * width_pixels + px_floor) * 4
    f.instruction(&Instruction::LocalGet(pz_floor));
    f.instruction(&Instruction::I32Const(width_pixels as i32));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::LocalGet(px_floor));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::I32Const(4));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(DATA_OFFSET as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::F32Load(wasm_encoder::MemArg {
        offset: 0,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::F64PromoteF32);
    f.instruction(&Instruction::LocalSet(h00));

    // h10 at (px1, pz_floor)
    f.instruction(&Instruction::LocalGet(pz_floor));
    f.instruction(&Instruction::I32Const(width_pixels as i32));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::LocalGet(px1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::I32Const(4));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(DATA_OFFSET as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::F32Load(wasm_encoder::MemArg {
        offset: 0,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::F64PromoteF32);
    f.instruction(&Instruction::LocalSet(h10));

    // h01 at (px_floor, pz1)
    f.instruction(&Instruction::LocalGet(pz1));
    f.instruction(&Instruction::I32Const(width_pixels as i32));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::LocalGet(px_floor));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::I32Const(4));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(DATA_OFFSET as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::F32Load(wasm_encoder::MemArg {
        offset: 0,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::F64PromoteF32);
    f.instruction(&Instruction::LocalSet(h01));

    // h11 at (px1, pz1)
    f.instruction(&Instruction::LocalGet(pz1));
    f.instruction(&Instruction::I32Const(width_pixels as i32));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::LocalGet(px1));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::I32Const(4));
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Const(DATA_OFFSET as i32));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::F32Load(wasm_encoder::MemArg {
        offset: 0,
        align: 2,
        memory_index: 0,
    }));
    f.instruction(&Instruction::F64PromoteF32);
    f.instruction(&Instruction::LocalSet(h11));

    // Bilinear interpolation:
    // h_interp = (1-px_frac)*(1-pz_frac)*h00 + px_frac*(1-pz_frac)*h10
    //          + (1-px_frac)*pz_frac*h01 + px_frac*pz_frac*h11

    // (1-px_frac)*(1-pz_frac)*h00
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalGet(px_frac));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalGet(pz_frac));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(h00));
    f.instruction(&Instruction::F64Mul);

    // + px_frac*(1-pz_frac)*h10
    f.instruction(&Instruction::LocalGet(px_frac));
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalGet(pz_frac));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(h10));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);

    // + (1-px_frac)*pz_frac*h01
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalGet(px_frac));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalGet(pz_frac));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(h01));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);

    // + px_frac*pz_frac*h11
    f.instruction(&Instruction::LocalGet(px_frac));
    f.instruction(&Instruction::LocalGet(pz_frac));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(h11));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);

    f.instruction(&Instruction::LocalSet(h_interp));

    // Clip check: if h_interp < config_clip, return 0.0 (no geometry)
    f.instruction(&Instruction::LocalGet(h_interp));
    f.instruction(&Instruction::LocalGet(config_clip));
    f.instruction(&Instruction::F64Lt);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::F32Const(0.0));
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);

    // surface_height = h_interp * config_height
    f.instruction(&Instruction::LocalGet(h_interp));
    f.instruction(&Instruction::LocalGet(config_height));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalSet(surface_height));

    // Return (y <= surface_height) ? 1.0 : 0.0
    f.instruction(&Instruction::LocalGet(y));
    f.instruction(&Instruction::LocalGet(surface_height));
    f.instruction(&Instruction::F64Le);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Result(ValType::F32)));
    f.instruction(&Instruction::F32Const(1.0));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::F32Const(0.0));
    f.instruction(&Instruction::End);

    f.instruction(&Instruction::End);
    f
}

fn process_and_generate_wasm(
    image_data: &[u8],
    config: &HeightmapConfig,
) -> Result<Vec<u8>, &'static str> {
    let heightmap = parse_image_to_heightmap(image_data)?;
    Ok(generate_wasm(&heightmap, config))
}

// ============================================================================
// Operator Entry Points
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    // Read config from input 0
    let config = {
        let cfg_len = unsafe { get_input_len(0) } as usize;
        if cfg_len == 0 {
            HeightmapConfig::default()
        } else {
            let mut cfg_buf = vec![0u8; cfg_len];
            unsafe { get_input_data(0, cfg_buf.as_mut_ptr() as i32, cfg_len as i32) };
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            ciborium::de::from_reader::<HeightmapConfig, _>(&mut cursor).unwrap_or_default()
        }
    };

    // Read image data from input 1
    let img_len = unsafe { get_input_len(1) } as usize;
    let mut img_buf = vec![0u8; img_len];
    if img_len > 0 {
        unsafe { get_input_data(1, img_buf.as_mut_ptr() as i32, img_len as i32) };
    }

    // Process image and generate WASM
    let output = match process_and_generate_wasm(&img_buf, &config) {
        Ok(wasm) => wasm,
        Err(_) => {
            // Return empty WASM on error
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
        let schema =
            "{ width: float .default 1.0, depth: float .default 1.0, height: float .default 1.0, clip: float .default 0.0 }"
                .to_string();
        let metadata = OperatorMetadata {
            name: "heightmap_extrude_operator".to_string(),
            version: "0.1.0".to_string(),
            inputs: vec![
                OperatorMetadataInput::CBORConfiguration(schema),
                OperatorMetadataInput::Blob,
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };

        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out)
            .expect("heightmap_extrude_operator metadata CBOR serialization should not fail");
        out
    });

    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}
