//! Editing a project from a host: the coercions and conventions the CLI's
//! `project-add-*` commands and the Python bindings share. One copy, so an
//! input that the CLI accepts is one the bindings accept, and the value
//! `project-run --json` prints is the one `Asset.value` returns.

use crate::{
    AssetTypeHint, ExecutionInput, ImportedAsset, LoadedAsset, OperatorMetadata,
    OperatorMetadataInput, Project,
};
use anyhow::{Context, Result};

/// An operator input as a host hands it over, before it is checked
/// against the operator's declared slot type.
#[derive(Clone, Debug)]
pub enum InputValue {
    /// A reference to an asset in the project, resolved at run time.
    Asset(String),
    /// A JSON value, coerced by the slot type (an array for `VecF64`, an
    /// object for a CBOR configuration or an `F64Map`, a string for a
    /// script source).
    Json(serde_json::Value),
    /// Raw bytes, passed through (validated for the typed slots).
    Bytes(Vec<u8>),
    /// The slot stays unwired: empty inline bytes, which is what the GUI
    /// stores for a slot it has nothing to wire and what operators with
    /// optional inputs test for. Any slot type accepts it; a slot the
    /// operator requires fails at run time with the operator's message.
    Unwired,
}

/// Short human label for a declared operator input slot type.
pub fn input_type_label(input: &OperatorMetadataInput) -> String {
    match input {
        OperatorMetadataInput::ModelWASM => "ModelWASM".to_string(),
        OperatorMetadataInput::CBORConfiguration(_) => "CBOR configuration".to_string(),
        OperatorMetadataInput::LuaSource(_) => "Lua source".to_string(),
        OperatorMetadataInput::WgslSource(_) => "WGSL source".to_string(),
        OperatorMetadataInput::F64Map => "F64Map".to_string(),
        OperatorMetadataInput::Blob => "Blob".to_string(),
        OperatorMetadataInput::VecF64(dim) => format!("VecF64({dim})"),
        OperatorMetadataInput::FeaMesh => "FeaMesh".to_string(),
        OperatorMetadataInput::TriMesh => "TriMesh".to_string(),
        OperatorMetadataInput::Subspace => "Subspace".to_string(),
        OperatorMetadataInput::ViewSet => "ViewSet".to_string(),
        OperatorMetadataInput::Splat => "Splat".to_string(),
        OperatorMetadataInput::Mechanism => "Mechanism".to_string(),
        OperatorMetadataInput::Assembly => "Assembly".to_string(),
    }
}

/// One line per declared input slot, for count-mismatch errors.
pub fn describe_declared_inputs(metadata: &OperatorMetadata) -> String {
    metadata
        .inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let name = metadata
                .input_name(i)
                .map(|n| format!("{n} "))
                .unwrap_or_default();
            let arity = if metadata.variadic_slot() == Some(i) {
                ", one or more"
            } else {
                ""
            };
            format!("  [{i}] {name}({}{arity})", input_type_label(input))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// "3" or "at least 3": how many inputs the operator's declaration takes.
pub fn expected_input_count(metadata: &OperatorMetadata) -> String {
    match metadata.variadic_slot() {
        Some(_) => format!("at least {}", metadata.inputs.len()),
        None => metadata.inputs.len().to_string(),
    }
}

/// Check an input against its declared slot type, coercing where the
/// intent is unambiguous (JSON array -> VecF64 raw bytes, JSON object ->
/// CBOR or F64Map, JSON string -> script source bytes). Asset references
/// always pass — they resolve at run time.
pub fn coerce_input(
    value: InputValue,
    slot: &OperatorMetadataInput,
    slot_desc: &str,
) -> Result<ExecutionInput> {
    let inline = |bytes| Ok(ExecutionInput::Inline(bytes));
    match (value, slot) {
        (InputValue::Asset(id), _) => Ok(ExecutionInput::AssetRef(id)),
        (InputValue::Unwired, _) => inline(Vec::new()),

        // VecF64: raw little-endian f64s. Accept a JSON array of the right
        // arity, or raw bytes of exactly the right length.
        (InputValue::Json(serde_json::Value::Array(items)), OperatorMetadataInput::VecF64(dim)) => {
            if items.len() != *dim {
                anyhow::bail!(
                    "{slot_desc} expects VecF64({dim}) but the JSON array has {} element(s)",
                    items.len()
                );
            }
            let mut bytes = Vec::with_capacity(dim * 8);
            for (i, item) in items.iter().enumerate() {
                let v = item.as_f64().with_context(|| {
                    format!("{slot_desc}: JSON array element {i} ({item}) is not a number")
                })?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            inline(bytes)
        }
        (InputValue::Json(other), OperatorMetadataInput::VecF64(dim)) => {
            anyhow::bail!(
                "{slot_desc} expects VecF64({dim}): pass a JSON array [x,y,..] with {dim} \
                 numbers, got JSON {other}"
            );
        }
        (InputValue::Bytes(bytes), OperatorMetadataInput::VecF64(dim)) => {
            if bytes.len() != dim * 8 {
                anyhow::bail!(
                    "{slot_desc} expects VecF64({dim}) = {} raw little-endian bytes, but the \
                     raw input has {} bytes (tip: a JSON array [x,y,..] also works)",
                    dim * 8,
                    bytes.len()
                );
            }
            inline(bytes)
        }

        // CBOR configuration: JSON converts, raw bytes pass through as
        // pre-encoded CBOR.
        (InputValue::Json(value), OperatorMetadataInput::CBORConfiguration(_)) => {
            let mut cbor_bytes = Vec::new();
            ciborium::into_writer(&value, &mut cbor_bytes)
                .context("Failed to convert JSON to CBOR")?;
            inline(cbor_bytes)
        }

        (InputValue::Json(value), OperatorMetadataInput::F64Map) => {
            inline(encode_json_f64_map(&value, slot_desc)?)
        }

        // Script source: a JSON string is the script text; raw bytes pass.
        (
            InputValue::Json(serde_json::Value::String(source)),
            OperatorMetadataInput::LuaSource(_) | OperatorMetadataInput::WgslSource(_),
        ) => inline(source.into_bytes()),

        (InputValue::Bytes(bytes), OperatorMetadataInput::F64Map) => {
            volumetric_abi::f64_map::decode(&bytes)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("{slot_desc} is not a valid F64Map"))?;
            inline(bytes)
        }

        (InputValue::Bytes(bytes), OperatorMetadataInput::ViewSet) => {
            volumetric_abi::viewset::decode_viewset(&bytes)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("{slot_desc} is not a valid view set"))?;
            inline(bytes)
        }

        (InputValue::Bytes(bytes), OperatorMetadataInput::Splat) => {
            volumetric_abi::splat::decode_splat(&bytes)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("{slot_desc} is not a valid splat"))?;
            inline(bytes)
        }

        // Binary slot types can't be built from JSON literals.
        (InputValue::Json(_), slot_type) => {
            anyhow::bail!(
                "{slot_desc} expects {} — pass an asset reference or raw bytes, not JSON",
                input_type_label(slot_type)
            );
        }

        (InputValue::Bytes(bytes), _) => inline(bytes),
    }
}

/// A JSON object of finite numbers as an encoded `F64Map`.
pub fn encode_json_f64_map(value: &serde_json::Value, context: &str) -> Result<Vec<u8>> {
    let serde_json::Value::Object(entries) = value else {
        anyhow::bail!("{context} expects a JSON object whose values are finite numbers");
    };
    let mut values = volumetric_abi::f64_map::F64Map::new();
    for (key, value) in entries {
        let number = value
            .as_f64()
            .with_context(|| format!("{context}: value for `{key}` is not a number"))?;
        values.insert(key.clone(), number);
    }
    volumetric_abi::f64_map::encode(&values)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("{context} is invalid"))
}

/// The names a host may call an imported asset's kind by: the CLI's
/// `--type` spellings.
pub const ASSET_KINDS: &[(&str, AssetTypeHint)] = &[
    ("lua", AssetTypeHint::LuaSource),
    ("wgsl", AssetTypeHint::WgslSource),
    ("config", AssetTypeHint::Config),
    ("f64map", AssetTypeHint::F64Map),
    ("blob", AssetTypeHint::Binary),
    ("viewset", AssetTypeHint::ViewSet),
    ("splat", AssetTypeHint::Splat),
    ("mechanism", AssetTypeHint::Mechanism),
    ("assembly", AssetTypeHint::Assembly),
];

/// The kind an asset file of this extension imports as by default.
pub fn asset_kind_for_extension(extension: &str) -> AssetTypeHint {
    match extension.to_ascii_lowercase().as_str() {
        "lua" => AssetTypeHint::LuaSource,
        "wgsl" => AssetTypeHint::WgslSource,
        "cbor" => AssetTypeHint::Config,
        "vviews" => AssetTypeHint::ViewSet,
        "vsplat" => AssetTypeHint::Splat,
        "vmech" => AssetTypeHint::Mechanism,
        "vasm" => AssetTypeHint::Assembly,
        _ => AssetTypeHint::Binary,
    }
}

/// Parse one of [`ASSET_KINDS`]' names.
pub fn parse_asset_kind(name: &str) -> Result<AssetTypeHint> {
    let lower = name.to_ascii_lowercase();
    ASSET_KINDS
        .iter()
        .find(|(kind, _)| *kind == lower)
        .map(|(_, hint)| *hint)
        .with_context(|| {
            let names: Vec<&str> = ASSET_KINDS.iter().map(|(k, _)| *k).collect();
            format!("unknown asset kind `{name}`; one of {}", names.join(", "))
        })
}

/// Check that `bytes` are a valid asset of `kind` (the typed kinds decode;
/// a wasm module under any other kind is a mistyped command).
pub fn validate_asset_bytes(kind: AssetTypeHint, bytes: &[u8]) -> Result<()> {
    match kind {
        AssetTypeHint::ViewSet => {
            volumetric_abi::viewset::decode_viewset(bytes)
                .map_err(anyhow::Error::msg)
                .context("Invalid view set asset")?;
        }
        AssetTypeHint::Splat => {
            volumetric_abi::splat::decode_splat(bytes)
                .map_err(anyhow::Error::msg)
                .context("Invalid splat asset")?;
        }
        AssetTypeHint::Mechanism => {
            volumetric_abi::mechanism::decode_mechanism(bytes)
                .map_err(anyhow::Error::msg)
                .context("Invalid mechanism asset")?;
        }
        AssetTypeHint::Assembly => {
            volumetric_abi::mechanism::decode_assembly(bytes)
                .map_err(anyhow::Error::msg)
                .context("Invalid assembly asset")?;
        }
        AssetTypeHint::F64Map => {
            volumetric_abi::f64_map::decode(bytes)
                .map_err(anyhow::Error::msg)
                .context("Invalid F64Map asset")?;
        }
        _ => {}
    }
    if bytes.starts_with(b"\0asm") {
        anyhow::bail!("the data is a WASM module; add it as a model or an operator instead");
    }
    Ok(())
}

/// Import `bytes` as an asset of `kind` under a unique id from `id_base`
/// and return the id.
pub fn add_asset(
    project: &mut Project,
    id_base: &str,
    kind: AssetTypeHint,
    bytes: Vec<u8>,
) -> Result<String> {
    validate_asset_bytes(kind, &bytes)?;
    let id = project.unique_asset_id(id_base);
    project
        .imports_mut()
        .push(ImportedAsset::new(id.clone(), bytes, Some(kind)));
    Ok(id)
}

/// What [`add_operation`] appended.
#[derive(Clone, Debug)]
pub struct AddedOperation {
    /// The operator import's id (shared with earlier steps of the same
    /// operator).
    pub import_id: String,
    /// One output id per declared output, slot 0 first.
    pub output_ids: Vec<String>,
}

/// Append a step running the operator `op_bytes` (named `op_name` for ids
/// and messages) on `inputs`, checked and coerced against its metadata.
/// `output_id` names output slot 0 (the rest take its declared-name
/// suffixes); `None` derives one from the operator and its first asset
/// input. The outputs are exported unless `export` is false.
pub fn add_operation(
    project: &mut Project,
    op_name: &str,
    op_bytes: Vec<u8>,
    inputs: Vec<InputValue>,
    output_id: Option<String>,
    export: bool,
) -> Result<AddedOperation> {
    let metadata = crate::operator_metadata_from_wasm_bytes(&op_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to read operator metadata: {e}"))?;
    let count = inputs.len();
    if !metadata.accepts_input_count(count) {
        anyhow::bail!(
            "{op_name} expects {} input(s), got {count}:\n{}\n(leave an optional slot \
             unwired to skip it)",
            expected_input_count(&metadata),
            describe_declared_inputs(&metadata)
        );
    }

    let inputs: Vec<ExecutionInput> = inputs
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            let slot = metadata
                .input_type(idx, count)
                .expect("input count was checked against the declaration");
            let name = metadata
                .input_label(idx, count)
                .map(|n| format!(" ({n})"))
                .unwrap_or_default();
            let slot_desc = format!("input [{idx}]{name}");
            coerce_input(value, slot, &slot_desc)
        })
        .collect::<Result<_>>()?;

    let primary_input = inputs.iter().find_map(|i| match i {
        ExecutionInput::AssetRef(id) => Some(id.as_str()),
        _ => None,
    });
    let output_id =
        output_id.unwrap_or_else(|| project.default_output_name(op_name, primary_input));
    let output_ids = project.output_ids_for(output_id, &metadata);
    let import_id = project.insert_operation(op_name, op_bytes, inputs, output_ids.clone());
    if !export {
        project.exports_mut().retain(|id| !output_ids.contains(id));
    }
    Ok(AddedOperation {
        import_id,
        output_ids,
    })
}

/// The scalar-ish value of an asset as JSON, for kinds that have one:
/// `Subspace` (dimensions, rank, origin, basis rows), `F64Map` (object)
/// and `VecF64` (array). Models, meshes, view sets and blobs have none.
pub fn asset_value_json(asset: &LoadedAsset) -> Option<serde_json::Value> {
    match asset.type_hint()? {
        AssetTypeHint::Subspace => {
            let subspace = volumetric_abi::subspace::decode_subspace(asset.data()).ok()?;
            let basis: Vec<&[f64]> = subspace.basis.chunks(subspace.ambient().max(1)).collect();
            Some(serde_json::json!({
                "dimensions": subspace.dimensions,
                "rank": subspace.rank(),
                "origin": subspace.origin,
                "basis": basis,
            }))
        }
        AssetTypeHint::F64Map => {
            let map = volumetric_abi::f64_map::decode(asset.data()).ok()?;
            serde_json::to_value(map).ok()
        }
        AssetTypeHint::VecF64(_) => Some(serde_json::Value::from(vec_f64(asset.data()))),
        _ => None,
    }
}

/// The little-endian f64s of a `VecF64` asset.
pub fn vec_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
        .collect()
}

/// Select a timeline step by 0-based index or operator-id substring (which
/// must match exactly one step).
pub fn select_step(project: &Project, selector: &str) -> Result<usize> {
    if let Ok(idx) = selector.parse::<usize>() {
        anyhow::ensure!(
            idx < project.timeline.len(),
            "step index {idx} out of range; the timeline has {} step(s)",
            project.timeline.len()
        );
        return Ok(idx);
    }
    let matches: Vec<usize> = project
        .timeline
        .iter()
        .enumerate()
        .filter(|(_, s)| s.operator_id.contains(selector))
        .map(|(i, _)| i)
        .collect();
    match matches.as_slice() {
        [only] => Ok(*only),
        [] => anyhow::bail!(
            "no timeline step's operator id contains {selector:?}; steps: {}",
            project
                .timeline
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{i}:{}", s.operator_id))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        many => anyhow::bail!(
            "{selector:?} matches {} steps ({}); use an index",
            many.len(),
            many.iter()
                .map(|i| format!("{i}:{}", project.timeline[*i].operator_id))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Coerce a JSON literal to a schema-typed config value. Integers promote to
/// floats for `Float` fields — the raw JSON→CBOR path can't do this (it has
/// no schema), and operators reject CBOR ints in f64 fields.
pub fn json_config_value(
    field: &crate::operator_config::ConfigField,
    value: &serde_json::Value,
    path: &str,
) -> Result<crate::operator_config::ConfigValue> {
    use crate::operator_config::{ConfigFieldType, ConfigValue};

    fn scalar(ty: &ConfigFieldType, value: &serde_json::Value, path: &str) -> Result<ConfigValue> {
        match ty {
            ConfigFieldType::Bool => value
                .as_bool()
                .map(ConfigValue::Bool)
                .with_context(|| format!("{path}: expected a bool, got {value}")),
            ConfigFieldType::Int => value
                .as_i64()
                .map(ConfigValue::Int)
                .with_context(|| format!("{path}: expected an integer, got {value}")),
            ConfigFieldType::Float => value
                .as_f64()
                .map(ConfigValue::Float)
                .with_context(|| format!("{path}: expected a number, got {value}")),
            ConfigFieldType::Text => value
                .as_str()
                .map(|s| ConfigValue::Text(s.to_string()))
                .with_context(|| format!("{path}: expected a string, got {value}")),
            ConfigFieldType::Enum(options) => {
                let s = value
                    .as_str()
                    .with_context(|| format!("{path}: expected a string, got {value}"))?;
                anyhow::ensure!(
                    options.iter().any(|o| o == s),
                    "{path}: {s:?} is not one of {}",
                    options.join("/")
                );
                Ok(ConfigValue::Text(s.to_string()))
            }
            ConfigFieldType::List { element, min_len } => {
                let items = value
                    .as_array()
                    .with_context(|| format!("{path}: expected an array, got {value}"))?;
                anyhow::ensure!(
                    items.len() >= *min_len,
                    "{path}: needs at least {min_len} element(s), got {}",
                    items.len()
                );
                items
                    .iter()
                    .map(|item| scalar(element, item, path))
                    .collect::<Result<Vec<_>>>()
                    .map(ConfigValue::List)
            }
            ConfigFieldType::Group(_) => {
                // A bool toggles an optional group's enablement marker;
                // sub-fields are set through their dotted paths.
                value.as_bool().map(ConfigValue::Bool).with_context(|| {
                    format!(
                        "{path} is a config group: pass true/false to toggle it, \
                         or set sub-fields via dotted paths ({path}.<field>)"
                    )
                })
            }
        }
    }

    let value = scalar(&field.ty, value, path)?;
    anyhow::ensure!(
        field.in_bounds(&value),
        "{path}: {value:?} is outside the declared bounds [{}, {}]",
        field.min.map_or("-inf".into(), |v| v.to_string()),
        field.max.map_or("+inf".into(), |v| v.to_string()),
    );
    Ok(value)
}

/// One field a [`set_config`] call changed.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigChange {
    pub path: String,
    /// The value before, as its debug form; None when unset.
    pub previous: Option<String>,
    pub value: String,
}

/// Merge `updates` (field path → JSON value, checked against the
/// operator's declared schema) into the configuration input of the step
/// `selector` picks. Returns the step index and what changed.
pub fn set_config(
    project: &mut Project,
    selector: &str,
    updates: &serde_json::Map<String, serde_json::Value>,
) -> Result<(usize, Vec<ConfigChange>)> {
    use crate::operator_config;

    let step_index = select_step(project, selector)?;
    let operator_id = project.timeline[step_index].operator_id.clone();
    let op_bytes = project
        .imports
        .iter()
        .find(|a| a.id == operator_id)
        .map(|a| a.data.clone())
        .with_context(|| format!("operator asset {operator_id:?} not found in project imports"))?;
    let metadata = crate::operator_metadata_from_wasm_bytes(&op_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to read operator metadata: {e}"))?;

    let (slot, cddl) = metadata
        .inputs
        .iter()
        .enumerate()
        .find_map(|(i, input)| match input {
            OperatorMetadataInput::CBORConfiguration(schema) => Some((i, schema.clone())),
            _ => None,
        })
        .with_context(|| format!("{operator_id} declares no configuration input"))?;
    let fields = operator_config::parse_schema(&cddl)
        .map_err(|e| anyhow::anyhow!("{operator_id}'s config schema failed to parse: {e}"))?;

    let step = &mut project.timeline[step_index];
    anyhow::ensure!(
        metadata.accepts_input_count(step.inputs.len()),
        "step {step_index} has {} input(s) but {operator_id} declares {}; \
         the project predates the operator version — re-add the step first",
        step.inputs.len(),
        expected_input_count(&metadata)
    );
    let slot = metadata.input_of_slot(slot, step.inputs.len());
    let mut values = match &step.inputs[slot] {
        ExecutionInput::Inline(bytes) if !bytes.is_empty() => operator_config::decode(bytes),
        _ => operator_config::default_values(&fields),
    };
    anyhow::ensure!(
        !updates.is_empty(),
        "config object is empty; nothing to set"
    );

    let mut changes = Vec::with_capacity(updates.len());
    for (path, json_value) in updates {
        let field = operator_config::find_field(&fields, path).with_context(|| {
            format!(
                "{operator_id} has no config field {path:?}; fields: {}",
                fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let value = json_config_value(field, json_value, path)?;
        let previous = values.insert(path.clone(), value.clone());
        changes.push(ConfigChange {
            path: path.clone(),
            previous: previous.map(|p| format!("{p:?}")),
            value: format!("{value:?}"),
        });
    }
    step.inputs[slot] = ExecutionInput::Inline(operator_config::encode(&fields, &values));
    Ok((step_index, changes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3_slot() -> OperatorMetadataInput {
        OperatorMetadataInput::VecF64(3)
    }

    fn json(text: &str) -> InputValue {
        InputValue::Json(serde_json::from_str(text).unwrap())
    }

    #[test]
    fn json_arrays_coerce_to_vecf64_bytes() {
        let ExecutionInput::Inline(bytes) =
            coerce_input(json("[0.25,-0.26,0.25]"), &vec3_slot(), "input [1]").unwrap()
        else {
            panic!("expected inline bytes");
        };
        assert_eq!(bytes.len(), 24);
        assert_eq!(vec_f64(&bytes), vec![0.25, -0.26, 0.25]);
    }

    #[test]
    fn vecf64_inputs_reject_shape_mismatches() {
        let err = coerce_input(json("[1,2]"), &vec3_slot(), "input [1]")
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 element(s)"), "{err}");

        assert!(coerce_input(json("{\"x\":1}"), &vec3_slot(), "input [1]").is_err());

        let err = coerce_input(InputValue::Bytes(vec![0u8; 23]), &vec3_slot(), "input [1]")
            .unwrap_err()
            .to_string();
        assert!(err.contains("23 bytes"), "{err}");
    }

    #[test]
    fn unwired_leaves_any_slot_empty() {
        for slot in [
            OperatorMetadataInput::Subspace,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::Blob,
            OperatorMetadataInput::CBORConfiguration(String::new()),
        ] {
            let input = coerce_input(InputValue::Unwired, &slot, "input [1]").unwrap();
            assert!(
                matches!(&input, ExecutionInput::Inline(bytes) if bytes.is_empty()),
                "{slot:?} -> {input:?}"
            );
        }
    }

    #[test]
    fn json_is_rejected_for_binary_slots() {
        let err = coerce_input(
            json("{\"op\":\"union\"}"),
            &OperatorMetadataInput::ModelWASM,
            "input [0]",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("ModelWASM"), "{err}");
    }

    #[test]
    fn config_and_script_slots_accept_json() {
        let ExecutionInput::Inline(cbor) = coerce_input(
            json("{\"op\":\"intersect\"}"),
            &OperatorMetadataInput::CBORConfiguration(String::new()),
            "c",
        )
        .unwrap() else {
            panic!("expected inline");
        };
        let value: serde_json::Value = ciborium::from_reader(cbor.as_slice()).unwrap();
        assert_eq!(value, serde_json::json!({"op": "intersect"}));

        for slot in [
            OperatorMetadataInput::LuaSource(String::new()),
            OperatorMetadataInput::WgslSource(String::new()),
        ] {
            let ExecutionInput::Inline(source) =
                coerce_input(json("\"return 1\""), &slot, "l").unwrap()
            else {
                panic!("expected inline");
            };
            assert_eq!(source, b"return 1");
        }
    }

    #[test]
    fn f64_map_slots_accept_numeric_json_objects() {
        let ExecutionInput::Inline(bytes) = coerce_input(
            json("{\"spinner.bearing_pitch\":0.04,\"global.scale\":2}"),
            &OperatorMetadataInput::F64Map,
            "parameters",
        )
        .unwrap() else {
            panic!("expected inline F64Map");
        };
        let values = volumetric_abi::f64_map::decode(&bytes).unwrap();
        assert_eq!(values["spinner.bearing_pitch"], 0.04);
        assert_eq!(values["global.scale"], 2.0);

        assert!(
            coerce_input(
                json("{\"x\":\"not numeric\"}"),
                &OperatorMetadataInput::F64Map,
                "parameters"
            )
            .is_err()
        );
    }

    /// `project-run --json` and `Asset.value` carry the numbers of small
    /// values so a reader never needs a CBOR decoder: a Subspace as origin
    /// plus basis rows, an F64Map as its entries, a VecF64 as its
    /// components; bulk values stay size-only.
    #[test]
    fn small_values_decode_to_json() {
        let asset = |type_hint, data: Vec<u8>| {
            LoadedAsset::from_parts("v".to_string(), data, Some(type_hint), vec![])
        };

        let plane = volumetric_abi::subspace::Subspace {
            dimensions: 3,
            origin: vec![1.0, 2.0, 3.0],
            basis: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        };
        let value = asset_value_json(&asset(
            AssetTypeHint::Subspace,
            volumetric_abi::subspace::encode_subspace(&plane),
        ))
        .expect("subspace decodes");
        assert_eq!(value["dimensions"], 3);
        assert_eq!(value["rank"], 2);
        assert_eq!(value["origin"], serde_json::json!([1.0, 2.0, 3.0]));
        assert_eq!(
            value["basis"],
            serde_json::json!([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        );

        let mut map = volumetric_abi::f64_map::F64Map::new();
        map.insert("radius".to_string(), 0.0235);
        map.insert("inliers".to_string(), 6189.0);
        let value = asset_value_json(&asset(
            AssetTypeHint::F64Map,
            volumetric_abi::f64_map::encode(&map).unwrap(),
        ))
        .expect("f64 map decodes");
        assert_eq!(
            value,
            serde_json::json!({"inliers": 6189.0, "radius": 0.0235})
        );

        let mut bytes = Vec::new();
        for v in [0.5f64, -1.0, 2.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let value = asset_value_json(&asset(AssetTypeHint::VecF64(3), bytes)).expect("vec decodes");
        assert_eq!(value, serde_json::json!([0.5, -1.0, 2.0]));

        assert!(asset_value_json(&asset(AssetTypeHint::Binary, vec![1, 2, 3])).is_none());
        assert!(asset_value_json(&asset(AssetTypeHint::Subspace, vec![0xff])).is_none());
    }

    /// Count-mismatch errors name the variadic slot and the "at least"
    /// arity, so the hint matches what repeating an input means.
    #[test]
    fn variadic_declarations_describe_their_arity() {
        let metadata = OperatorMetadata {
            name: "nary".to_string(),
            version: "0.0.0".to_string(),
            display_name: String::new(),
            description: String::new(),
            category: String::new(),
            icon_svg: String::new(),
            docs: String::new(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration("{ op: tstr }".to_string()),
            ],
            variadic_input: Some(0),
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![],
            output_names: vec![],
        };
        assert_eq!(expected_input_count(&metadata), "at least 2");
        let described = describe_declared_inputs(&metadata);
        assert!(
            described.contains("[0] Model (ModelWASM, one or more)"),
            "{described}"
        );
        assert!(
            described.contains("[1] Config (CBOR configuration)"),
            "{described}"
        );
    }

    #[test]
    fn asset_kinds_parse_and_validate() {
        assert_eq!(parse_asset_kind("ViewSet").unwrap(), AssetTypeHint::ViewSet);
        assert!(parse_asset_kind("mesh").is_err());
        assert_eq!(asset_kind_for_extension("VVIEWS"), AssetTypeHint::ViewSet);
        assert!(validate_asset_bytes(AssetTypeHint::Binary, b"\0asm\x01").is_err());
        assert!(validate_asset_bytes(AssetTypeHint::ViewSet, b"junk").is_err());
        assert!(validate_asset_bytes(AssetTypeHint::Binary, b"anything").is_ok());
    }
}
