//! Cancellation interrupts a project run mid-operator.
//!
//! Long-running operator steps (an FEA inverse solve can take tens of
//! minutes) must observe cancellation while executing, not just between
//! timeline steps — the executor uses wasmtime epoch interruption to trap
//! the running wasm. The operator here spins forever, so only a mid-operator
//! interrupt can make the run return at all.

#![cfg(feature = "native")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use volumetric::{Environment, ExecutionError, ExecutionStep, ImportedAsset, Project};

#[test]
fn cancel_interrupts_a_project_run_mid_operator() {
    let spinning_operator = r#"(module
        (memory (export "memory") 1)
        (func (export "run") (loop $spin (br $spin))))"#
        .as_bytes()
        .to_vec();
    let project = Project {
        version: 2,
        imports: vec![ImportedAsset::operator("op".to_string(), spinning_operator)],
        timeline: vec![ExecutionStep {
            operator_id: "op".to_string(),
            inputs: vec![],
            outputs: vec!["out".to_string()],
        }],
        exports: vec![],
        baked: None,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        flag.store(true, Ordering::Relaxed);
    });

    let result = project.run_cancellable(&mut Environment::new(), &cancel);
    assert!(
        matches!(result, Err(ExecutionError::Cancelled)),
        "{result:?}"
    );
}
