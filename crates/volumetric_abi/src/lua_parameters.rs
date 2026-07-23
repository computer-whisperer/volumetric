//! Parameter annotations understood by Lua-model tooling.
//!
//! An annotation lives on the same line as a numeric module constant:
//!
//! ```lua
//! local pitch = 0.035 -- @param key="spinner.pitch" min=0.02 max=0.06
//! ```
//!
//! The annotation is a schema hint and an F64Map binding. It does not change
//! the map wire format or make unannotated Lua constants externally mutable.
//! Option grammar and validation are shared with the WGSL dialect via
//! [`crate::annotations`].

use crate::annotations;

/// One annotated Lua module constant.
#[derive(Clone, Debug, PartialEq)]
pub struct LuaParameter {
    /// One-based source line of the annotated declaration. Consumers use this
    /// to bind the annotation to that exact module constant rather than to a
    /// same-named local in a nested scope.
    pub source_line: usize,
    pub local_name: String,
    pub key: String,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl LuaParameter {
    /// Validate one routed value against the annotation's finite/range rules.
    pub fn validate_value(&self, value: f64) -> Result<(), String> {
        self.as_spec().validate_value(value)
    }

    fn as_spec(&self) -> annotations::ParameterSpec {
        annotations::ParameterSpec {
            source_line: self.source_line,
            binding_name: self.local_name.clone(),
            key: self.key.clone(),
            default: self.default,
            min: self.min,
            max: self.max,
        }
    }
}

/// Parse every `@param` annotation in source order.
///
/// Annotated declarations deliberately require a single numeric literal on
/// the left side of the comment. Derived values should remain ordinary module
/// constants so both the compiler and the UI agree on the editable default.
pub fn parse(source: &str) -> Result<Vec<LuaParameter>, String> {
    let mut specs = Vec::new();

    for (line_index, line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let Some((code, comment)) = line.split_once("--") else {
            continue;
        };
        let comment = comment.trim();
        let Some(options) = comment.strip_prefix("@param") else {
            continue;
        };

        let declaration = code.trim().strip_prefix("local ").ok_or_else(|| {
            format!("line {line_number}: @param must annotate a module `local name = number`")
        })?;
        let (local_name, default) = declaration
            .split_once('=')
            .ok_or_else(|| format!("line {line_number}: @param declaration is missing `=`"))?;
        let local_name = local_name.trim();
        if !is_lua_identifier(local_name) {
            return Err(format!(
                "line {line_number}: `{local_name}` is not a simple Lua identifier"
            ));
        }
        let default = annotations::parse_finite(default.trim(), line_number, "default")?;
        specs.push(annotations::parse_spec(
            options,
            line_number,
            local_name,
            default,
        )?);
    }

    annotations::check_duplicates(&specs)?;
    Ok(specs
        .into_iter()
        .map(|spec| LuaParameter {
            source_line: spec.source_line,
            local_name: spec.binding_name,
            key: spec.key,
            default: spec.default,
            min: spec.min,
            max: spec.max,
        })
        .collect())
}

/// Render parameters as the existing host configuration CDDL subset. This is
/// a presentation/schema bridge only; F64Map remains an open flat map and may
/// contain keys that this particular Lua consumer does not use.
pub fn schema_cddl(parameters: &[LuaParameter]) -> String {
    let specs: Vec<_> = parameters.iter().map(LuaParameter::as_spec).collect();
    annotations::schema_cddl(&specs)
}

fn is_lua_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bindings_defaults_and_ranges() {
        let source = r#"
local pitch = 0.035 -- @param key="spinner.pitch" min=0.02 max=0.06
local clearance = 0.00015 -- @param min=0.0
local derived = pitch / 2.0
"#;
        let parameters = parse(source).unwrap();
        assert_eq!(parameters.len(), 2);
        assert_eq!(parameters[0].source_line, 2);
        assert_eq!(parameters[0].local_name, "pitch");
        assert_eq!(parameters[0].key, "spinner.pitch");
        assert_eq!(parameters[0].default, 0.035);
        assert_eq!(parameters[0].min, Some(0.02));
        assert_eq!(parameters[0].max, Some(0.06));
        assert_eq!(parameters[1].key, "clearance");
    }

    #[test]
    fn schema_uses_external_keys_and_existing_config_annotations() {
        let parameters =
            parse("local pitch = 0.035 -- @param key=spinner.pitch min=0.02 max=0.06").unwrap();
        assert_eq!(
            schema_cddl(&parameters),
            "{ spinner.pitch: float .default 0.035 .ge 0.02 .le 0.06 }"
        );
    }

    #[test]
    fn rejects_ambiguous_or_invalid_declarations() {
        assert!(
            parse("local x = y -- @param")
                .unwrap_err()
                .contains("not a number")
        );
        assert!(
            parse("local x = 2 -- @param min=0 max=1")
                .unwrap_err()
                .contains("above maximum")
        );
        assert!(
            parse("local x = 1 -- @param key=a\nlocal y = 2 -- @param key=a")
                .unwrap_err()
                .contains("duplicate parameter key")
        );
    }
}
