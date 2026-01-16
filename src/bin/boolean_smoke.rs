use anyhow::{Context, Result};
use std::fs;
use wasmtime::*;
use wasmparser::Payload;

fn print_section_summary(label: &str, wasm: &[u8]) {
    let mut types = 0u32;
    let mut funcs = 0u32;
    let mut globals = 0u32;
    let mut tables = 0u32;
    let mut memories = 0u32;
    let mut elements = 0u32;
    let mut data = 0u32;
    let mut imports = 0u32;
    let mut has_start = false;
    let mut exports = 0u32;
    let mut custom = 0u32;

    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload {
            Ok(Payload::TypeSection(s)) => types = types.saturating_add(s.count()),
            Ok(Payload::FunctionSection(s)) => funcs = funcs.saturating_add(s.count()),
            Ok(Payload::GlobalSection(s)) => globals = globals.saturating_add(s.count()),
            Ok(Payload::TableSection(s)) => tables = tables.saturating_add(s.count()),
            Ok(Payload::MemorySection(s)) => memories = memories.saturating_add(s.count()),
            Ok(Payload::ElementSection(s)) => elements = elements.saturating_add(s.count()),
            Ok(Payload::DataSection(s)) => data = data.saturating_add(s.count()),
            Ok(Payload::ImportSection(s)) => imports = imports.saturating_add(s.count()),
            Ok(Payload::ExportSection(s)) => exports = exports.saturating_add(s.count()),
            Ok(Payload::StartSection { .. }) => has_start = true,
            Ok(Payload::CustomSection(_)) => custom = custom.saturating_add(1),
            _ => {}
        }
    }

    println!(
        "{label}: bytes={}, types={types}, funcs={funcs}, globals={globals}, tables={tables}, memories={memories}, elements={elements}, data={data}, imports={imports}, exports={exports}, start={has_start}, custom_sections={custom}",
        wasm.len()
    );
}

fn run_operator(engine: &Engine, op_wasm: &[u8], inputs: Vec<Vec<u8>>) -> Result<Vec<u8>> {
    // Minimal host that exposes inputs and captures the single output.
    #[derive(Default)]
    struct HostState {
        inputs: Vec<Vec<u8>>,
        output: Vec<u8>,
    }

    let mut store = Store::new(
        engine,
        HostState {
            inputs,
            output: Vec::new(),
        },
    );

    let mut linker = Linker::new(engine);
    linker.func_wrap("host", "get_input_len", |caller: Caller<'_, HostState>, idx: i32| -> u32 {
        caller.data().inputs[idx as usize].len() as u32
    })?;

    linker.func_wrap(
        "host",
        "get_input_data",
        |mut caller: Caller<'_, HostState>, idx: i32, ptr: i32, len: i32| {
            let len = len as usize;
            let ptr = ptr as usize;
            // Avoid borrow conflicts by copying input bytes out of the store state first.
            let tmp: Vec<u8> = caller.data().inputs[idx as usize][..len].to_vec();
            let mem = caller
                .get_export("memory")
                .and_then(|e| e.into_memory())
                .unwrap();
            mem.write(&mut caller, ptr, &tmp).unwrap();
        },
    )?;

    linker.func_wrap(
        "host",
        "post_output",
        |mut caller: Caller<'_, HostState>, _out_idx: i32, ptr: i32, len: i32| {
            let len = len as usize;
            let ptr = ptr as usize;
            let mem = caller
                .get_export("memory")
                .and_then(|e| e.into_memory())
                .unwrap();
            let mut tmp = vec![0u8; len];
            mem.read(&mut caller, ptr, &mut tmp).unwrap();
            caller.data_mut().output = tmp;
        },
    )?;

    let module = Module::new(engine, op_wasm)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let run = instance.get_typed_func::<(), ()>(&mut store, "run")?;
    run.call(&mut store, ())?;

    Ok(store.data().output.clone())
}

fn main() -> Result<()> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("target").join("wasm32-unknown-unknown").join("debug");

    let torus = fs::read(dir.join("simple_torus_model.wasm")).context("read torus model")?;
    let translate_op = fs::read(dir.join("translate_operator.wasm")).context("read translate operator")?;
    let boolean_op = fs::read(dir.join("boolean_operator.wasm")).context("read boolean operator")?;

    // CBOR for `{ dx: 1.0, dy: 0.0, dz: 0.0 }` (the operator's default), encoded explicitly.
    // This keeps the smoke binary self-contained.
    let translate_cfg = vec![
        0xA3, // map(3)
        0x62, 0x64, 0x78, // "dx"
        0xFA, 0x3F, 0x80, 0x00, 0x00, // f32 1.0
        0x62, 0x64, 0x79, // "dy"
        0xFA, 0x00, 0x00, 0x00, 0x00, // f32 0.0
        0x62, 0x64, 0x7A, // "dz"
        0xFA, 0x00, 0x00, 0x00, 0x00, // f32 0.0
    ];

    let engine = Engine::default();

    let translated_torus = run_operator(&engine, &translate_op, vec![torus.clone(), translate_cfg])
        .context("run translate operator")?;
    print_section_summary("torus", &torus);
    print_section_summary("translated_torus", &translated_torus);
    println!(
        "Input torus size: {} bytes, translated torus size: {} bytes",
        torus.len(),
        translated_torus.len()
    );

    let merged = run_operator(&engine, &boolean_op, vec![torus, translated_torus, Vec::new()])
        .context("run boolean operator")?;
    println!("Merged model size: {} bytes", merged.len());

    // Validate the merged model can be instantiated and queried.
    let mut store = Store::new(&engine, ());
    let merged_module = Module::new(&engine, merged)?;
    let merged_instance = Instance::new(&mut store, &merged_module, &[])?;
    let is_inside =
        merged_instance.get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")?;
    let v0 = is_inside.call(&mut store, (0.0, 0.0, 0.0))?;
    let v1 = is_inside.call(&mut store, (5.0, 5.0, 5.0))?;
    println!("density(0,0,0) = {v0}, density(5,5,5) = {v1}");

    Ok(())
}
