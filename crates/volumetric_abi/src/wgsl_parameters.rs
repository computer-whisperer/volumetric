//! Parameter annotations understood by WGSL-model tooling.
//!
//! An annotation lives on the same line as an `override` declaration:
//!
//! ```wgsl
//! override pitch: f64 = 0.035; // @param key="spinner.pitch" min=0.02 max=0.06
//! ```
//!
//! Only annotated overrides are routed from the F64Map input; unannotated
//! overrides keep their WGSL defaults. The declaration must carry an explicit
//! type and a single numeric-literal default (no suffix) so the compiler and
//! the UI agree on the editable value; the operator separately checks that
//! the named override is scalar f64 in the parsed module. Option grammar and
//! validation are shared with the Lua dialect via [`crate::annotations`].

use crate::annotations;

/// One annotated WGSL override declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct WgslParameter {
    /// One-based source line of the annotated declaration.
    pub source_line: usize,
    /// The override's identifier, which the operator resolves against the
    /// parsed module's override arena.
    pub override_name: String,
    pub key: String,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl WgslParameter {
    /// Validate one routed value against the annotation's finite/range rules.
    pub fn validate_value(&self, value: f64) -> Result<(), String> {
        self.as_spec().validate_value(value)
    }

    fn as_spec(&self) -> annotations::ParameterSpec {
        annotations::ParameterSpec {
            source_line: self.source_line,
            binding_name: self.override_name.clone(),
            key: self.key.clone(),
            default: self.default,
            min: self.min,
            max: self.max,
        }
    }
}

/// Parse every `@param` annotation in source order.
pub fn parse(source: &str) -> Result<Vec<WgslParameter>, String> {
    let mut specs = Vec::new();

    for (line_index, line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let Some((code, comment)) = line.split_once("//") else {
            continue;
        };
        let comment = comment.trim();
        let Some(options) = comment.strip_prefix("@param") else {
            continue;
        };

        let declaration = code.trim().strip_prefix("override ").ok_or_else(|| {
            format!(
                "line {line_number}: @param must annotate an `override name: f64 = number;` declaration"
            )
        })?;
        let (name_and_type, default) = declaration.split_once('=').ok_or_else(|| {
            format!("line {line_number}: @param declaration is missing `= default`")
        })?;
        let override_name = name_and_type
            .split_once(':')
            .ok_or_else(|| {
                format!(
                    "line {line_number}: @param override must declare an explicit type \
                     (e.g. `override {}: f64 = ...`)",
                    name_and_type.trim()
                )
            })?
            .0
            .trim();
        if !is_wgsl_identifier(override_name) {
            return Err(format!(
                "line {line_number}: `{override_name}` is not a simple WGSL identifier"
            ));
        }
        let default = default.trim().trim_end_matches(';').trim();
        let default = annotations::parse_finite(default, line_number, "default")?;
        specs.push(annotations::parse_spec(
            options,
            line_number,
            override_name,
            default,
        )?);
    }

    annotations::check_duplicates(&specs)?;
    Ok(specs
        .into_iter()
        .map(|spec| WgslParameter {
            source_line: spec.source_line,
            override_name: spec.binding_name,
            key: spec.key,
            default: spec.default,
            min: spec.min,
            max: spec.max,
        })
        .collect())
}

/// Render parameters as the host configuration CDDL subset (see
/// [`annotations::schema_cddl`]).
pub fn schema_cddl(parameters: &[WgslParameter]) -> String {
    let specs: Vec<_> = parameters.iter().map(WgslParameter::as_spec).collect();
    annotations::schema_cddl(&specs)
}

fn is_wgsl_identifier(value: &str) -> bool {
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
    fn parses_override_bindings() {
        let source = r#"
alias float = f64;
override pitch: f64 = 0.035; // @param key="spinner.pitch" min=0.02 max=0.06
override clearance: float = 0.00015; // @param min=0.0
const derived: f64 = 0.5;
"#;
        let parameters = parse(source).unwrap();
        assert_eq!(parameters.len(), 2);
        assert_eq!(parameters[0].source_line, 3);
        assert_eq!(parameters[0].override_name, "pitch");
        assert_eq!(parameters[0].key, "spinner.pitch");
        assert_eq!(parameters[0].default, 0.035);
        assert_eq!(parameters[0].min, Some(0.02));
        assert_eq!(parameters[0].max, Some(0.06));
        assert_eq!(parameters[1].key, "clearance");
        assert_eq!(parameters[1].default, 0.00015);
    }

    #[test]
    fn schema_matches_lua_dialect_output() {
        let parameters =
            parse("override pitch: f64 = 0.035; // @param key=spinner.pitch min=0.02 max=0.06")
                .unwrap();
        assert_eq!(
            schema_cddl(&parameters),
            "{ spinner.pitch: float .default 0.035 .ge 0.02 .le 0.06 }"
        );
    }

    #[test]
    fn rejects_invalid_declarations() {
        assert!(
            parse("const x: f64 = 1.0; // @param")
                .unwrap_err()
                .contains("must annotate an `override")
        );
        assert!(
            parse("override x = 1.0; // @param")
                .unwrap_err()
                .contains("explicit type")
        );
        assert!(
            parse("override x: f64 = 2.0; // @param min=0 max=1")
                .unwrap_err()
                .contains("above maximum")
        );
        assert!(
            parse("override x: f64 = 1.0; // @param key=a\noverride y: f64 = 2.0; // @param key=a")
                .unwrap_err()
                .contains("duplicate parameter key")
        );
    }
}
