//! Module-merge plumbing for operators that combine WASM models.
//!
//! Merging two import-free modules works by appending every index-space
//! section of module B after module A's, with B's internal references
//! rebased by A's section counts ([`OffsetReencoder`]); each module keeps
//! its own memory (the output is multi-memory). The consuming operator then
//! emits its own glue/wrapper functions and the export section.
//!
//! Extracted from `boolean_operator`, shared with `lattice_operator`.

use wasm_encoder::{CodeSection, FunctionSection, TypeSection};

/// Per-index-space entry counts of a module, used to rebase a second
/// module's indices when appending it.
#[derive(Default)]
pub struct SectionCounts {
    pub types: u32,
    pub funcs: u32,
    pub globals: u32,
    pub tables: u32,
    pub memories: u32,
    pub elements: u32,
    pub data: u32,
    pub has_data_count: bool,
}

pub fn count_sections(wasm: &[u8]) -> Result<SectionCounts, String> {
    let mut counts = SectionCounts::default();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            wasmparser::Payload::TypeSection(s) => {
                counts.types = counts.types.saturating_add(s.count())
            }
            wasmparser::Payload::FunctionSection(s) => {
                counts.funcs = counts.funcs.saturating_add(s.count())
            }
            wasmparser::Payload::GlobalSection(s) => {
                counts.globals = counts.globals.saturating_add(s.count())
            }
            wasmparser::Payload::TableSection(s) => {
                counts.tables = counts.tables.saturating_add(s.count())
            }
            wasmparser::Payload::MemorySection(s) => {
                counts.memories = counts.memories.saturating_add(s.count())
            }
            wasmparser::Payload::ElementSection(s) => {
                counts.elements = counts.elements.saturating_add(s.count())
            }
            wasmparser::Payload::DataSection(s) => {
                counts.data = counts.data.saturating_add(s.count())
            }
            wasmparser::Payload::DataCountSection { .. } => counts.has_data_count = true,
            wasmparser::Payload::ImportSection(s) => {
                if s.count() > 0 {
                    return Err("Model must not import anything".to_string());
                }
            }
            wasmparser::Payload::StartSection { .. } => {
                return Err("Model must not define a start function".to_string());
            }
            _ => {}
        }
    }
    Ok(counts)
}

/// Rebase every index a reencoded item references by fixed per-space
/// offsets (the section counts of the modules appended before it).
pub struct OffsetReencoder {
    pub type_offset: u32,
    pub func_offset: u32,
    pub global_offset: u32,
    pub table_offset: u32,
    pub memory_offset: u32,
}

impl OffsetReencoder {
    /// Identity mapping — for the first module of a merge.
    pub fn identity() -> Self {
        Self {
            type_offset: 0,
            func_offset: 0,
            global_offset: 0,
            table_offset: 0,
            memory_offset: 0,
        }
    }

    /// Offsets that place a module after `counts` (a previously appended
    /// module's section counts).
    pub fn after(counts: &SectionCounts) -> Self {
        Self {
            type_offset: counts.types,
            func_offset: counts.funcs,
            global_offset: counts.globals,
            table_offset: counts.tables,
            memory_offset: counts.memories,
        }
    }
}

impl wasm_encoder::reencode::Reencode for OffsetReencoder {
    type Error = std::convert::Infallible;

    fn type_index(&mut self, ty: u32) -> Result<u32, wasm_encoder::reencode::Error<Self::Error>> {
        Ok(self.type_offset + ty)
    }

    fn function_index(
        &mut self,
        func: u32,
    ) -> Result<u32, wasm_encoder::reencode::Error<Self::Error>> {
        Ok(self.func_offset + func)
    }

    fn global_index(
        &mut self,
        global: u32,
    ) -> Result<u32, wasm_encoder::reencode::Error<Self::Error>> {
        Ok(self.global_offset + global)
    }

    fn table_index(
        &mut self,
        table: u32,
    ) -> Result<u32, wasm_encoder::reencode::Error<Self::Error>> {
        Ok(self.table_offset + table)
    }

    fn memory_index(
        &mut self,
        memory: u32,
    ) -> Result<u32, wasm_encoder::reencode::Error<Self::Error>> {
        Ok(self.memory_offset + memory)
    }

    fn data_index(&mut self, data: u32) -> Result<u32, wasm_encoder::reencode::Error<Self::Error>> {
        Ok(data)
    }

    fn element_index(
        &mut self,
        element: u32,
    ) -> Result<u32, wasm_encoder::reencode::Error<Self::Error>> {
        Ok(element)
    }
}

/// The section accumulators one merge builds up; consumers append modules
/// into it, emit glue into `types`/`funcs`/`code`, and finish with their
/// own export section via [`MergeSections::finish`].
#[derive(Default)]
pub struct MergeSections {
    pub types: TypeSection,
    pub funcs: FunctionSection,
    pub code: CodeSection,
    pub globals: wasm_encoder::GlobalSection,
    pub tables: wasm_encoder::TableSection,
    pub memories: wasm_encoder::MemorySection,
    pub elements: wasm_encoder::ElementSection,
    pub data: wasm_encoder::DataSection,
}

impl MergeSections {
    /// Appends every index-space section of `wasm`, rebased through `re`.
    pub fn append_module(&mut self, wasm: &[u8], re: &mut OffsetReencoder) -> Result<(), String> {
        for payload in wasmparser::Parser::new(0).parse_all(wasm) {
            match payload.map_err(|e| e.to_string())? {
                wasmparser::Payload::TypeSection(s) => {
                    wasm_encoder::reencode::utils::parse_type_section(re, &mut self.types, s)
                        .map_err(|e| format!("type section reencode failed: {e}"))?;
                }
                wasmparser::Payload::FunctionSection(s) => {
                    wasm_encoder::reencode::utils::parse_function_section(re, &mut self.funcs, s)
                        .map_err(|e| format!("function section reencode failed: {e}"))?;
                }
                wasmparser::Payload::CodeSectionEntry(body) => {
                    wasm_encoder::reencode::utils::parse_function_body(re, &mut self.code, body)
                        .map_err(|e| format!("code reencode failed: {e}"))?;
                }
                wasmparser::Payload::GlobalSection(s) => {
                    wasm_encoder::reencode::utils::parse_global_section(re, &mut self.globals, s)
                        .map_err(|e| format!("global section reencode failed: {e}"))?;
                }
                wasmparser::Payload::TableSection(s) => {
                    wasm_encoder::reencode::utils::parse_table_section(re, &mut self.tables, s)
                        .map_err(|e| format!("table section reencode failed: {e}"))?;
                }
                wasmparser::Payload::MemorySection(s) => {
                    wasm_encoder::reencode::utils::parse_memory_section(re, &mut self.memories, s)
                        .map_err(|e| format!("memory section reencode failed: {e}"))?;
                }
                wasmparser::Payload::ElementSection(s) => {
                    wasm_encoder::reencode::utils::parse_element_section(re, &mut self.elements, s)
                        .map_err(|e| format!("element section reencode failed: {e}"))?;
                }
                wasmparser::Payload::DataSection(s) => {
                    wasm_encoder::reencode::utils::parse_data_section(re, &mut self.data, s)
                        .map_err(|e| format!("data section reencode failed: {e}"))?;
                }
                wasmparser::Payload::ImportSection(s) => {
                    if s.count() > 0 {
                        return Err("Model must not import anything".to_string());
                    }
                }
                wasmparser::Payload::ExportSection(_)
                | wasmparser::Payload::DataCountSection { .. }
                | wasmparser::Payload::StartSection { .. }
                | wasmparser::Payload::CustomSection(_) => {}
                wasmparser::Payload::End(_) => break,
                _ => {}
            }
        }
        Ok(())
    }

    /// Assembles the final module in canonical section order with the
    /// consumer's export section. `data_count` emits a DataCount section
    /// when any merged module had one (bulk-memory requirement).
    pub fn finish(self, exports: &wasm_encoder::ExportSection, data_count: Option<u32>) -> Vec<u8> {
        let mut out = wasm_encoder::Module::new();
        if self.types.len() > 0 {
            out.section(&self.types);
        }
        if self.funcs.len() > 0 {
            out.section(&self.funcs);
        }
        if self.tables.len() > 0 {
            out.section(&self.tables);
        }
        if self.memories.len() > 0 {
            out.section(&self.memories);
        }
        if self.globals.len() > 0 {
            out.section(&self.globals);
        }
        out.section(exports);
        if self.elements.len() > 0 {
            out.section(&self.elements);
        }
        if let Some(count) = data_count {
            out.section(&wasm_encoder::DataCountSection { count });
        }
        out.section(&self.code);
        if self.data.len() > 0 {
            out.section(&self.data);
        }
        out.finish()
    }
}

/// The export indices of an N-dimensional Model ABI module, plus the
/// optional typed-channel extension exports.
#[derive(Debug)]
pub struct ModelExports {
    pub get_dimensions: u32,
    pub get_io_ptr: u32,
    pub get_bounds: u32,
    pub sample: u32,
    pub memory: u32,
    /// `sample_channels(pos_ptr, out_ptr)`, when the model declares typed
    /// channels.
    pub sample_channels: Option<u32>,
    /// `get_sample_format() -> i64`, when the model declares typed channels.
    pub get_sample_format: Option<u32>,
}

pub fn parse_model_exports(wasm: &[u8]) -> Result<ModelExports, String> {
    const FUNCTIONS: &[&str] = &[
        "get_dimensions",
        "get_io_ptr",
        "get_bounds",
        "sample",
        "sample_channels",
        "get_sample_format",
    ];
    let mut map: std::collections::HashMap<String, (wasmparser::ExternalKind, u32)> =
        std::collections::HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            wasmparser::Payload::ImportSection(s) => {
                if s.count() > 0 {
                    return Err("Model must not import anything".to_string());
                }
            }
            wasmparser::Payload::ExportSection(s) => {
                for e in s {
                    let e = e.map_err(|e| e.to_string())?;
                    if FUNCTIONS.contains(&e.name) || e.name == "memory" {
                        map.insert(e.name.to_string(), (e.kind, e.index));
                    }
                }
            }
            _ => {}
        }
    }

    let get_func = |name: &str| -> Result<u32, String> {
        map.get(name)
            .filter(|(k, _)| *k == wasmparser::ExternalKind::Func)
            .map(|(_, i)| *i)
            .ok_or_else(|| format!("Model missing function export `{name}`"))
    };
    let opt_func = |name: &str| -> Option<u32> {
        map.get(name)
            .filter(|(k, _)| *k == wasmparser::ExternalKind::Func)
            .map(|(_, i)| *i)
    };
    let memory = map
        .get("memory")
        .filter(|(k, _)| *k == wasmparser::ExternalKind::Memory)
        .map(|(_, i)| *i)
        .ok_or_else(|| "Model missing memory export `memory`".to_string())?;

    Ok(ModelExports {
        get_dimensions: get_func("get_dimensions")?,
        get_io_ptr: get_func("get_io_ptr")?,
        get_bounds: get_func("get_bounds")?,
        sample: get_func("sample")?,
        memory,
        sample_channels: opt_func("sample_channels"),
        get_sample_format: opt_func("get_sample_format"),
    })
}

/// The memory export index of a non-Model module (e.g. a template whose
/// memory the merge glue must address for cross-memory stores).
pub fn find_memory_export(wasm: &[u8]) -> Result<u32, String> {
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let wasmparser::Payload::ExportSection(s) = payload.map_err(|e| e.to_string())? {
            for e in s {
                let e = e.map_err(|e| e.to_string())?;
                if e.name == "memory" && e.kind == wasmparser::ExternalKind::Memory {
                    return Ok(e.index);
                }
            }
        }
    }
    Err("module missing memory export `memory`".to_string())
}

/// The function index of a named function export, for non-Model modules
/// (e.g. a lattice template exporting its evaluator).
pub fn find_function_export(wasm: &[u8], name: &str) -> Result<u32, String> {
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let wasmparser::Payload::ExportSection(s) = payload.map_err(|e| e.to_string())? {
            for e in s {
                let e = e.map_err(|e| e.to_string())?;
                if e.name == name && e.kind == wasmparser::ExternalKind::Func {
                    return Ok(e.index);
                }
            }
        }
    }
    Err(format!("module missing function export `{name}`"))
}
