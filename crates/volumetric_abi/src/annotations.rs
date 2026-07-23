//! Shared machinery for `@param` source annotations.
//!
//! Both scripting frontends bind routed [`crate::f64_map::F64Map`] values to
//! source declarations through a same-line comment annotation:
//!
//! ```text
//! local pitch = 0.035 -- @param key="spinner.pitch" min=0.02 max=0.06   (Lua)
//! override pitch: f64 = 0.035; // @param key="spinner.pitch" min=0.02  (WGSL)
//! ```
//!
//! The language modules ([`crate::lua_parameters`], [`crate::wgsl_parameters`])
//! own declaration syntax; the option grammar, range validation, and CDDL
//! schema rendering live here so the two dialects cannot drift.

/// The language-independent payload of one `@param` annotation.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterSpec {
    /// One-based source line of the annotated declaration.
    pub source_line: usize,
    /// The identifier declared on the annotated line.
    pub binding_name: String,
    /// External F64Map key (defaults to the binding name).
    pub key: String,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl ParameterSpec {
    /// Validate one routed value against the annotation's finite/range rules.
    pub fn validate_value(&self, value: f64) -> Result<(), String> {
        if !value.is_finite() {
            return Err(format!("parameter `{}` must be finite", self.key));
        }
        if self.min.is_some_and(|min| value < min) {
            return Err(format!(
                "parameter `{}` value {value} is below minimum {}",
                self.key,
                self.min.unwrap()
            ));
        }
        if self.max.is_some_and(|max| value > max) {
            return Err(format!(
                "parameter `{}` value {value} is above maximum {}",
                self.key,
                self.max.unwrap()
            ));
        }
        Ok(())
    }
}

/// Parse the option list that follows `@param`, producing a full spec.
/// `binding_name` and `default` come from the language-specific declaration
/// parse; options may override the external key and add range bounds.
pub fn parse_spec(
    options: &str,
    source_line: usize,
    binding_name: &str,
    default: f64,
) -> Result<ParameterSpec, String> {
    let mut key = binding_name.to_string();
    let mut min = None;
    let mut max = None;
    for option in options.split_whitespace() {
        let (name, value) = option
            .split_once('=')
            .ok_or_else(|| format!("line {source_line}: invalid @param option `{option}`"))?;
        match name {
            "key" => {
                let value = value
                    .strip_prefix('"')
                    .and_then(|value| value.strip_suffix('"'))
                    .unwrap_or(value);
                if value.is_empty() {
                    return Err(format!(
                        "line {source_line}: parameter key must not be empty"
                    ));
                }
                key = value.to_string();
            }
            "min" => min = Some(parse_finite(value, source_line, "minimum")?),
            "max" => max = Some(parse_finite(value, source_line, "maximum")?),
            _ => {
                return Err(format!(
                    "line {source_line}: unknown @param option `{name}`"
                ));
            }
        }
    }

    if min.zip(max).is_some_and(|(min, max)| min > max) {
        return Err(format!(
            "line {source_line}: parameter `{key}` minimum exceeds maximum"
        ));
    }
    let spec = ParameterSpec {
        source_line,
        binding_name: binding_name.to_string(),
        key,
        default,
        min,
        max,
    };
    spec.validate_value(default)
        .map_err(|error| format!("line {source_line}: {error}"))?;
    Ok(spec)
}

/// Enforce unique binding names and unique external keys across a script.
pub fn check_duplicates(specs: &[ParameterSpec]) -> Result<(), String> {
    let mut names = std::collections::BTreeSet::new();
    let mut keys = std::collections::BTreeSet::new();
    for spec in specs {
        if !names.insert(spec.binding_name.as_str()) {
            return Err(format!(
                "line {}: duplicate parameter local `{}`",
                spec.source_line, spec.binding_name
            ));
        }
        if !keys.insert(spec.key.as_str()) {
            return Err(format!(
                "line {}: duplicate parameter key `{}`",
                spec.source_line, spec.key
            ));
        }
    }
    Ok(())
}

/// Render parameters as the host configuration CDDL subset. A presentation
/// and schema bridge only; F64Map remains an open flat map.
pub fn schema_cddl(specs: &[ParameterSpec]) -> String {
    let fields = specs
        .iter()
        .map(|spec| {
            let mut field = format!(
                "{}: float .default {}",
                spec.key,
                format_number(spec.default)
            );
            if let Some(min) = spec.min {
                field.push_str(&format!(" .ge {}", format_number(min)));
            }
            if let Some(max) = spec.max {
                field.push_str(&format!(" .le {}", format_number(max)));
            }
            field
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ {fields} }}")
}

pub fn parse_finite(value: &str, line: usize, role: &str) -> Result<f64, String> {
    let value = value
        .parse::<f64>()
        .map_err(|_| format!("line {line}: parameter {role} `{value}` is not a number"))?;
    if !value.is_finite() {
        return Err(format!("line {line}: parameter {role} must be finite"));
    }
    Ok(value)
}

fn format_number(value: f64) -> String {
    if value == 0.0 {
        "0.0".to_string()
    } else {
        value.to_string()
    }
}
