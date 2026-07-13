//! Operator configuration schemas.
//!
//! Operators describe a config input as a CDDL snippet via
//! [`crate::OperatorMetadataInput::CBORConfiguration`]. The host UI parses that
//! snippet into a list of [`ConfigField`]s, renders an editor, and encodes the
//! edited values back into the CBOR map the operator expects.
//!
//! Supported CDDL subset (a single record/map):
//!
//! ```cddl
//! { dx: float, dy: float, dz: float }
//! { scale: float .default 1.0, center: bool .default false }
//! { op: "union" / "subtract" / "intersect" }
//! { ? bed_position: float }
//! ```
//!
//! Leaf types: `bool`, `int`, `float`, `tstr`, and small string-enum unions.
//! Default values are read from either the RFC 8610 `.default X` control
//! operator or a legacy `(default X)` annotation; string defaults may be
//! quoted (`tstr .default "hi"`, enum `.default "z"`).
//!
//! A `?` occurrence marker makes a field optional: [`encode`] omits it from
//! the config map unless a value was explicitly set, so the operator's own
//! absent-field behavior applies (e.g. brim's auto bed placement).

use std::collections::BTreeMap;

use ciborium::value::Value as CborValue;

/// The type of a single configuration field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigFieldType {
    Bool,
    Int,
    Float,
    Text,
    /// A string enum; the vector holds the allowed values in declaration order.
    Enum(Vec<String>),
}

/// A concrete configuration value.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

/// One field in an operator's configuration schema.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigField {
    pub name: String,
    pub ty: ConfigFieldType,
    /// Declared default (`.default X` / `(default X)`), if the schema gave one.
    pub default: Option<ConfigValue>,
    /// Declared with the CDDL `?` occurrence marker: omitted from the encoded
    /// config unless a value is explicitly set.
    pub optional: bool,
}

/// Error parsing a config CDDL snippet.
#[derive(Debug, thiserror::Error)]
pub enum ConfigSchemaError {
    #[error("invalid CDDL field (expected `name: type`): `{0}`")]
    MalformedField(String),
    #[error("empty field name in CDDL: `{0}`")]
    EmptyName(String),
    #[error("unsupported CDDL type `{0}` (supported: bool, int, float, tstr, string enums)")]
    UnsupportedType(String),
}

impl ConfigField {
    /// The value to seed an editor with: the declared default, or a type-zero.
    pub fn seed_value(&self) -> ConfigValue {
        self.default.clone().unwrap_or_else(|| self.ty.zero())
    }
}

impl ConfigFieldType {
    /// The neutral value for this type when no default is declared.
    pub fn zero(&self) -> ConfigValue {
        match self {
            ConfigFieldType::Bool => ConfigValue::Bool(false),
            ConfigFieldType::Int => ConfigValue::Int(0),
            ConfigFieldType::Float => ConfigValue::Float(0.0),
            ConfigFieldType::Text => ConfigValue::Text(String::new()),
            ConfigFieldType::Enum(options) => {
                ConfigValue::Text(options.first().cloned().unwrap_or_default())
            }
        }
    }
}

impl ConfigValue {
    /// Render the value as the text an editor field shows.
    pub fn to_display_string(&self) -> String {
        match self {
            ConfigValue::Bool(b) => b.to_string(),
            ConfigValue::Int(i) => i.to_string(),
            ConfigValue::Float(f) => format_float(*f),
            ConfigValue::Text(t) => t.clone(),
        }
    }

    /// Parse editor text back into a value of the given type. Returns `None`
    /// when the text isn't a valid value (empty, non-numeric, or an
    /// out-of-vocabulary enum), so the caller keeps the raw edit buffer without
    /// committing it to the config.
    pub fn parse(ty: &ConfigFieldType, text: &str) -> Option<ConfigValue> {
        match ty {
            ConfigFieldType::Bool => match text.trim() {
                "true" => Some(ConfigValue::Bool(true)),
                "false" => Some(ConfigValue::Bool(false)),
                _ => None,
            },
            ConfigFieldType::Int => text.trim().parse::<i64>().ok().map(ConfigValue::Int),
            ConfigFieldType::Float => text.trim().parse::<f64>().ok().map(ConfigValue::Float),
            ConfigFieldType::Text => Some(ConfigValue::Text(text.to_string())),
            ConfigFieldType::Enum(options) => {
                if options.iter().any(|opt| opt == text) {
                    Some(ConfigValue::Text(text.to_string()))
                } else {
                    None
                }
            }
        }
    }

    fn to_cbor(&self) -> CborValue {
        match self {
            ConfigValue::Bool(b) => CborValue::Bool(*b),
            ConfigValue::Int(i) => CborValue::Integer((*i).into()),
            ConfigValue::Float(f) => CborValue::Float(*f),
            ConfigValue::Text(t) => CborValue::Text(t.clone()),
        }
    }
}

/// Format a float without a trailing `.0`-free integer looking odd but also
/// without scientific notation surprises. Integers keep a single decimal so the
/// field still reads as a float (`1` -> `1`, `1.5` -> `1.5`).
fn format_float(f: f64) -> String {
    if f == f.trunc() && f.is_finite() {
        // Show integers plainly; the field is still parsed back as f64.
        format!("{}", f as i64)
    } else {
        format!("{f}")
    }
}

/// Parse a CDDL config snippet into an ordered list of fields.
pub fn parse_schema(cddl: &str) -> Result<Vec<ConfigField>, ConfigSchemaError> {
    let mut s = cddl.trim();
    s = s.strip_prefix('{').unwrap_or(s).trim();
    s = s.strip_suffix('}').unwrap_or(s).trim();

    if s.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        let (name, rhs) = part
            .split_once(':')
            .ok_or_else(|| ConfigSchemaError::MalformedField(part.to_string()))?;
        let mut name = name.trim();
        // The `?` occurrence marker (RFC 8610): the field is optional.
        let optional = name.starts_with('?');
        if optional {
            name = name[1..].trim_start();
        }
        if name.is_empty() {
            return Err(ConfigSchemaError::EmptyName(part.to_string()));
        }
        let rhs = rhs.trim();

        let (ty_str, default_str) = split_type_and_default(rhs);
        let ty = parse_field_type(ty_str)?;
        let default = default_str.and_then(|d| ConfigValue::parse(&ty, unquote(d.trim())));

        out.push(ConfigField {
            name: name.to_string(),
            ty,
            default,
            optional,
        });
    }

    Ok(out)
}

/// Strip one layer of surrounding double quotes: CDDL writes string and
/// enum defaults as quoted literals (`.default "z"`).
fn unquote(token: &str) -> &str {
    token
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(token)
}

/// Split a field's right-hand side into the type token and an optional raw
/// default token, handling both `.default X` and legacy `(default X)`.
fn split_type_and_default(rhs: &str) -> (&str, Option<&str>) {
    if let Some(idx) = rhs.find(".default") {
        let ty = rhs[..idx].trim();
        let default = rhs[idx + ".default".len()..].trim();
        return (ty, (!default.is_empty()).then_some(default));
    }
    // Legacy `(default X)` — but not when the parens are part of an enum union
    // written with quotes.
    if let Some(paren_idx) = rhs.find('(')
        && !rhs[..paren_idx].contains('"')
    {
        let ty = rhs[..paren_idx].trim();
        let inner = rhs[paren_idx + 1..].trim_end_matches(')').trim();
        let default = inner.strip_prefix("default").map(str::trim);
        return (ty, default.filter(|d| !d.is_empty()));
    }
    (rhs, None)
}

fn parse_field_type(ty: &str) -> Result<ConfigFieldType, ConfigSchemaError> {
    match ty {
        "bool" => Ok(ConfigFieldType::Bool),
        "int" => Ok(ConfigFieldType::Int),
        "float" => Ok(ConfigFieldType::Float),
        "tstr" => Ok(ConfigFieldType::Text),
        other if other.contains('"') && other.contains('/') => {
            let trimmed = other
                .trim()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .trim();
            let options: Vec<String> = trimmed
                .split('/')
                .filter_map(|opt| {
                    opt.trim()
                        .strip_prefix('"')
                        .and_then(|s| s.strip_suffix('"'))
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                })
                .collect();
            if options.is_empty() {
                return Err(ConfigSchemaError::UnsupportedType(other.to_string()));
            }
            Ok(ConfigFieldType::Enum(options))
        }
        other => Err(ConfigSchemaError::UnsupportedType(other.to_string())),
    }
}

/// Encode a set of values into the CBOR map an operator expects. Fields missing
/// from `values` fall back to their declared default (or type-zero); missing
/// *optional* fields are omitted so the operator's absent-field behavior
/// applies.
pub fn encode(fields: &[ConfigField], values: &BTreeMap<String, ConfigValue>) -> Vec<u8> {
    let entries: Vec<(CborValue, CborValue)> = fields
        .iter()
        .filter_map(|field| {
            let value = match values.get(&field.name) {
                Some(value) => value.clone(),
                None if field.optional => return None,
                None => field.seed_value(),
            };
            Some((CborValue::Text(field.name.clone()), value.to_cbor()))
        })
        .collect();

    let mut out = Vec::new();
    // Encoding a plain map to an in-memory buffer does not fail.
    let _ = ciborium::ser::into_writer(&CborValue::Map(entries), &mut out);
    out
}

/// Seed values for a schema: the declared default (or type-zero) per
/// required field. Optional fields start unset.
pub fn default_values(fields: &[ConfigField]) -> BTreeMap<String, ConfigValue> {
    fields
        .iter()
        .filter(|field| !field.optional)
        .map(|field| (field.name.clone(), field.seed_value()))
        .collect()
}

/// Best-effort decode of a CBOR config map into values, keyed by field name.
/// Values whose CBOR type doesn't match a known scalar are skipped.
pub fn decode(bytes: &[u8]) -> BTreeMap<String, ConfigValue> {
    let mut out = BTreeMap::new();
    let Ok(CborValue::Map(entries)) = ciborium::de::from_reader::<CborValue, _>(bytes) else {
        return out;
    };

    for (key, value) in entries {
        let CborValue::Text(name) = key else {
            continue;
        };
        if let Some(config_value) = cbor_to_value(&value) {
            out.insert(name, config_value);
        }
    }
    out
}

fn cbor_to_value(value: &CborValue) -> Option<ConfigValue> {
    match value {
        CborValue::Bool(b) => Some(ConfigValue::Bool(*b)),
        CborValue::Integer(i) => Some(ConfigValue::Int((*i).try_into().ok()?)),
        CborValue::Float(f) => Some(ConfigValue::Float(*f)),
        CborValue::Text(t) => Some(ConfigValue::Text(t.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_record_of_floats() {
        let fields = parse_schema("{ dx: float, dy: float, dz: float }").unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "dx");
        assert_eq!(fields[0].ty, ConfigFieldType::Float);
        assert!(fields[0].default.is_none());
    }

    #[test]
    fn parses_rfc8610_default() {
        let fields =
            parse_schema("{ scale: float .default 1.5, center: bool .default true }").unwrap();
        assert_eq!(fields[0].ty, ConfigFieldType::Float);
        assert_eq!(fields[0].default, Some(ConfigValue::Float(1.5)));
        assert_eq!(fields[1].default, Some(ConfigValue::Bool(true)));
    }

    #[test]
    fn parses_legacy_paren_default() {
        let fields = parse_schema("{ scale: float (default 2.0) }").unwrap();
        assert_eq!(fields[0].ty, ConfigFieldType::Float);
        assert_eq!(fields[0].default, Some(ConfigValue::Float(2.0)));
    }

    #[test]
    fn parses_string_enum() {
        let fields = parse_schema(r#"{ op: "union" / "subtract" / "intersect" }"#).unwrap();
        assert_eq!(
            fields[0].ty,
            ConfigFieldType::Enum(vec![
                "union".to_string(),
                "subtract".to_string(),
                "intersect".to_string()
            ])
        );
    }

    #[test]
    fn parses_quoted_string_defaults() {
        let fields =
            parse_schema(r#"{ axis: "x" / "y" / "z" .default "z", name: tstr .default "hi" }"#)
                .unwrap();
        assert_eq!(fields[0].default, Some(ConfigValue::Text("z".to_string())));
        assert_eq!(fields[1].default, Some(ConfigValue::Text("hi".to_string())));
    }

    #[test]
    fn optional_fields_parse_and_stay_unsent() {
        let fields =
            parse_schema(r#"{ width: float .default 5.0, ? bed_position: float }"#).unwrap();
        assert_eq!(fields[0].name, "width");
        assert!(!fields[0].optional);
        assert_eq!(fields[1].name, "bed_position");
        assert!(fields[1].optional);
        assert!(fields[1].default.is_none());

        // Fresh defaults leave the optional field unset, and the encoded map
        // omits it — the operator's own absent-field behavior applies.
        let values = default_values(&fields);
        assert!(!values.contains_key("bed_position"));
        let decoded = decode(&encode(&fields, &values));
        assert_eq!(decoded.get("width"), Some(&ConfigValue::Float(5.0)));
        assert!(!decoded.contains_key("bed_position"));

        // An explicitly set value is sent; removing it omits it again.
        let mut values = values;
        values.insert("bed_position".to_string(), ConfigValue::Float(1.5));
        let decoded = decode(&encode(&fields, &values));
        assert_eq!(decoded.get("bed_position"), Some(&ConfigValue::Float(1.5)));
        values.remove("bed_position");
        assert!(!decode(&encode(&fields, &values)).contains_key("bed_position"));
    }

    #[test]
    fn empty_schema_is_ok() {
        assert!(parse_schema("{}").unwrap().is_empty());
        assert!(parse_schema("").unwrap().is_empty());
    }

    #[test]
    fn encode_then_decode_round_trips() {
        let fields = parse_schema("{ dx: float, n: int, on: bool, name: tstr }").unwrap();
        let mut values = BTreeMap::new();
        values.insert("dx".to_string(), ConfigValue::Float(1.5));
        values.insert("n".to_string(), ConfigValue::Int(7));
        values.insert("on".to_string(), ConfigValue::Bool(true));
        values.insert("name".to_string(), ConfigValue::Text("hi".to_string()));

        let bytes = encode(&fields, &values);
        let decoded = decode(&bytes);
        assert_eq!(decoded, values);
    }

    #[test]
    fn encode_uses_defaults_for_missing_fields() {
        let fields = parse_schema("{ scale: float .default 1.0 }").unwrap();
        let bytes = encode(&fields, &BTreeMap::new());
        let decoded = decode(&bytes);
        assert_eq!(decoded.get("scale"), Some(&ConfigValue::Float(1.0)));
    }

    #[test]
    fn value_parse_gates_invalid_input() {
        assert_eq!(
            ConfigValue::parse(&ConfigFieldType::Float, "1.5"),
            Some(ConfigValue::Float(1.5))
        );
        assert_eq!(ConfigValue::parse(&ConfigFieldType::Float, ""), None);
        assert_eq!(ConfigValue::parse(&ConfigFieldType::Float, "abc"), None);
        assert_eq!(ConfigValue::parse(&ConfigFieldType::Int, "abc"), None);
        let op = ConfigFieldType::Enum(vec!["union".into(), "subtract".into()]);
        assert_eq!(
            ConfigValue::parse(&op, "union"),
            Some(ConfigValue::Text("union".into()))
        );
        assert_eq!(ConfigValue::parse(&op, "nope"), None);
    }
}
