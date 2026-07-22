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
//! { ? surface: { outside: "project" / "drop" } .default true }
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
//!
//! A braced right-hand side declares a *group*: a named sub-record encoded
//! as a nested CBOR map (e.g. the mesh remaster operator's requirement
//! blocks). In the flat values map a group's sub-fields live under
//! dotted paths (`surface.outside`), and the group's own path holds a
//! `Bool` enablement marker: an optional group is encoded only when its
//! marker is `true` (a `.default true` after the closing brace seeds new
//! steps with the group enabled); a required group is always encoded.

use std::collections::BTreeMap;

use ciborium::value::Value as CborValue;

/// The type of a single configuration field.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigFieldType {
    Bool,
    Int,
    Float,
    Text,
    /// A string enum; the vector holds the allowed values in declaration order.
    Enum(Vec<String>),
    /// A homogeneous array of a scalar element type (CDDL `[* T]` / `[+ T]`).
    /// `min_len` is 1 for `[+ T]` (at least one element) and 0 for `[* T]`.
    /// Elements are one of `Bool`/`Int`/`Float`/`Text` — not nested lists or
    /// enums.
    List {
        element: Box<ConfigFieldType>,
        min_len: usize,
    },
    /// A named sub-record (`name: { ... }`), encoded as a nested CBOR map.
    /// Sub-fields live at dotted paths in the flat values map; the group's
    /// own path holds a `Bool` enablement marker (see the module docs).
    /// A group field's `default` of `Bool(true)` means an *optional* group
    /// starts enabled.
    Group(Vec<ConfigField>),
}

/// A concrete configuration value.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    List(Vec<ConfigValue>),
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
    /// Inclusive lower bound (CDDL `.ge N`; `uint` implies `0`). Only
    /// meaningful for `Int`/`Float` fields.
    pub min: Option<f64>,
    /// Inclusive upper bound (CDDL `.le N`). Only meaningful for `Int`/`Float`
    /// fields.
    pub max: Option<f64>,
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

    /// Parse editor text into this field's value, rejecting numeric values
    /// outside the declared `[min, max]` bounds. Returns `None` when the text
    /// is invalid or out of range, so the caller keeps the raw edit buffer
    /// without committing it.
    pub fn parse(&self, text: &str) -> Option<ConfigValue> {
        let value = ConfigValue::parse(&self.ty, text)?;
        self.in_bounds(&value).then_some(value)
    }

    /// Whether a numeric value satisfies the declared bounds (non-numeric
    /// values are always in bounds).
    pub fn in_bounds(&self, value: &ConfigValue) -> bool {
        let n = match value {
            ConfigValue::Int(i) => *i as f64,
            ConfigValue::Float(f) => *f,
            _ => return true,
        };
        self.min.is_none_or(|lo| n >= lo) && self.max.is_none_or(|hi| n <= hi)
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
            ConfigFieldType::List { .. } => ConfigValue::List(Vec::new()),
            // A group's own value is its enablement marker.
            ConfigFieldType::Group(_) => ConfigValue::Bool(false),
        }
    }
}

impl ConfigValue {
    /// Render the value as the text an editor field shows. A `List` joins its
    /// elements with `, ` (the rich array editor edits elements individually;
    /// this is the flat rendering for logs and single-field fallbacks).
    pub fn to_display_string(&self) -> String {
        match self {
            ConfigValue::Bool(b) => b.to_string(),
            ConfigValue::Int(i) => i.to_string(),
            ConfigValue::Float(f) => format_float(*f),
            ConfigValue::Text(t) => t.clone(),
            ConfigValue::List(items) => items
                .iter()
                .map(ConfigValue::to_display_string)
                .collect::<Vec<_>>()
                .join(", "),
        }
    }

    /// Parse editor text back into a value of the given type. Returns `None`
    /// when the text isn't a valid value (empty, non-numeric, or an
    /// out-of-vocabulary enum), so the caller keeps the raw edit buffer without
    /// committing it to the config. A `List` splits on commas and parses each
    /// non-empty element as the element type (the rich editor builds lists from
    /// separate element buffers, but this keeps `parse` total).
    pub fn parse(ty: &ConfigFieldType, text: &str) -> Option<ConfigValue> {
        match ty {
            // A group parses like a bool: its editable value is the
            // enablement marker driven by the section's switch.
            ConfigFieldType::Bool | ConfigFieldType::Group(_) => match text.trim() {
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
            ConfigFieldType::List { element, min_len } => {
                let mut items = Vec::new();
                for part in text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    items.push(ConfigValue::parse(element, part)?);
                }
                (items.len() >= *min_len).then_some(ConfigValue::List(items))
            }
        }
    }

    fn to_cbor(&self) -> CborValue {
        match self {
            ConfigValue::Bool(b) => CborValue::Bool(*b),
            ConfigValue::Int(i) => CborValue::Integer((*i).into()),
            ConfigValue::Float(f) => CborValue::Float(*f),
            ConfigValue::Text(t) => CborValue::Text(t.clone()),
            ConfigValue::List(items) => {
                CborValue::Array(items.iter().map(ConfigValue::to_cbor).collect())
            }
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
    parse_record_body(s)
}

fn parse_record_body(s: &str) -> Result<Vec<ConfigField>, ConfigSchemaError> {
    if s.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for part in split_record_parts(s) {
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

        // A braced right-hand side is a group: a nested record, optionally
        // followed by `.default true` (start enabled).
        if rhs.starts_with('{') {
            let (ty, default) = parse_group(rhs, part)?;
            out.push(ConfigField {
                name: name.to_string(),
                ty,
                default,
                optional,
                min: None,
                max: None,
            });
            continue;
        }

        let ann = split_type_and_annotations(rhs);
        let ty = parse_field_type(ann.ty)?;
        let default = ann
            .default
            .and_then(|d| ConfigValue::parse(&ty, unquote(d.trim())));
        // `uint` is `int` with an implied lower bound of 0, unless the schema
        // set an explicit `.ge`.
        let min = ann.min.or((ann.ty.trim() == "uint").then_some(0.0));

        out.push(ConfigField {
            name: name.to_string(),
            ty,
            default,
            optional,
            min,
            max: ann.max,
        });
    }

    Ok(out)
}

/// Split a record body on the commas at nesting depth zero: commas inside
/// group braces and array brackets belong to the nested type, not the
/// record. (Quoted strings in this subset never contain braces, brackets,
/// or commas.)
fn split_record_parts(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '{' | '[' => depth += 1,
            '}' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Parse a group right-hand side: `{ ...sub-record... }` plus an optional
/// `.default true` / `.default false` tail (whether an optional group
/// starts enabled).
fn parse_group(
    rhs: &str,
    part: &str,
) -> Result<(ConfigFieldType, Option<ConfigValue>), ConfigSchemaError> {
    let mut depth = 0usize;
    let mut close = None;
    for (i, c) in rhs.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return Err(ConfigSchemaError::MalformedField(part.to_string()));
    };
    let sub = parse_record_body(rhs[1..close].trim())?;
    let default = match rhs[close + 1..].trim() {
        "" => None,
        tail => match tail.strip_prefix(".default").map(str::trim) {
            Some("true") => Some(ConfigValue::Bool(true)),
            Some("false") => Some(ConfigValue::Bool(false)),
            _ => return Err(ConfigSchemaError::UnsupportedType(tail.to_string())),
        },
    };
    Ok((ConfigFieldType::Group(sub), default))
}

/// Look a field up by its (possibly dotted) path, walking group nesting.
/// An exact name match wins at every level: Lua parameter schemas use
/// literal dotted names (`sphere.radius`) as flat fields, so the dot only
/// descends when no field carries the full name itself.
pub fn find_field<'a>(fields: &'a [ConfigField], path: &str) -> Option<&'a ConfigField> {
    if let Some(field) = fields.iter().find(|f| f.name == path) {
        return Some(field);
    }
    let (head, rest) = path.split_once('.')?;
    let field = fields.iter().find(|f| f.name == head)?;
    match &field.ty {
        ConfigFieldType::Group(sub) => find_field(sub, rest),
        _ => None,
    }
}

/// Strip one layer of surrounding double quotes: CDDL writes string and
/// enum defaults as quoted literals (`.default "z"`).
fn unquote(token: &str) -> &str {
    token
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(token)
}

/// A field's right-hand side split into its type token and control operators.
struct FieldAnnotations<'a> {
    ty: &'a str,
    default: Option<&'a str>,
    min: Option<f64>,
    max: Option<f64>,
}

/// Split a field's right-hand side into the type token and its control
/// operators: `.default X`, `.ge N`, `.le N` (in any order), plus the legacy
/// `(default X)` form. The type is everything before the first control
/// operator — array (`[* tstr]`) and enum (`"a" / "b"`) types may contain
/// spaces, so we can't just take the first whitespace token.
fn split_type_and_annotations(rhs: &str) -> FieldAnnotations<'_> {
    const OPS: [&str; 3] = [".default", ".ge", ".le"];
    // Earliest dotted control operator marks the end of the type token. These
    // markers never occur inside a type or a value token (numeric literals like
    // `1e-8` and quoted strings don't contain `.default`/`.ge`/`.le`).
    if let Some(start) = OPS.iter().filter_map(|op| rhs.find(op)).min() {
        let ty = rhs[..start].trim();
        let mut default = None;
        let mut min = None;
        let mut max = None;
        let tokens: Vec<&str> = rhs[start..].split_whitespace().collect();
        let mut i = 0;
        while i < tokens.len() {
            match tokens[i] {
                ".default" => default = tokens.get(i + 1).copied(),
                ".ge" => min = tokens.get(i + 1).and_then(|t| t.parse().ok()),
                ".le" => max = tokens.get(i + 1).and_then(|t| t.parse().ok()),
                _ => {
                    i += 1;
                    continue;
                }
            }
            i += 2;
        }
        return FieldAnnotations {
            ty,
            default,
            min,
            max,
        };
    }
    // Legacy `(default X)` — but not when the parens are part of an enum union
    // written with quotes.
    if let Some(paren_idx) = rhs.find('(')
        && !rhs[..paren_idx].contains('"')
    {
        let ty = rhs[..paren_idx].trim();
        let inner = rhs[paren_idx + 1..].trim_end_matches(')').trim();
        let default = inner
            .strip_prefix("default")
            .map(str::trim)
            .filter(|d| !d.is_empty());
        return FieldAnnotations {
            ty,
            default,
            min: None,
            max: None,
        };
    }
    FieldAnnotations {
        ty: rhs,
        default: None,
        min: None,
        max: None,
    }
}

fn parse_field_type(ty: &str) -> Result<ConfigFieldType, ConfigSchemaError> {
    let ty = ty.trim();
    // Array types: `[* T]` (zero or more) / `[+ T]` (one or more), where the
    // element `T` is a scalar leaf. No internal commas, so the record-level
    // comma split still holds.
    if let Some(inner) = ty.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let inner = inner.trim();
        let mut chars = inner.chars();
        let min_len = match chars.next() {
            Some('*') => 0,
            Some('+') => 1,
            _ => return Err(ConfigSchemaError::UnsupportedType(ty.to_string())),
        };
        let element = parse_field_type(chars.as_str().trim())?;
        if matches!(
            element,
            ConfigFieldType::List { .. } | ConfigFieldType::Enum(_)
        ) {
            return Err(ConfigSchemaError::UnsupportedType(ty.to_string()));
        }
        return Ok(ConfigFieldType::List {
            element: Box::new(element),
            min_len,
        });
    }
    match ty {
        "bool" => Ok(ConfigFieldType::Bool),
        // `uint` is `int` in the widget; the non-negativity bound is applied by
        // the caller.
        "int" | "uint" => Ok(ConfigFieldType::Int),
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
/// applies. Group sub-fields are looked up under dotted paths; an optional
/// group is emitted (as a nested map) only when its marker reads `true`.
pub fn encode(fields: &[ConfigField], values: &BTreeMap<String, ConfigValue>) -> Vec<u8> {
    let entries = encode_entries(fields, values, "");
    let mut out = Vec::new();
    // Encoding a plain map to an in-memory buffer does not fail.
    let _ = ciborium::ser::into_writer(&CborValue::Map(entries), &mut out);
    out
}

fn field_path(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

fn encode_entries(
    fields: &[ConfigField],
    values: &BTreeMap<String, ConfigValue>,
    prefix: &str,
) -> Vec<(CborValue, CborValue)> {
    fields
        .iter()
        .filter_map(|field| {
            let path = field_path(prefix, &field.name);
            if let ConfigFieldType::Group(sub) = &field.ty {
                let enabled = matches!(values.get(&path), Some(ConfigValue::Bool(true)));
                if field.optional && !enabled {
                    return None;
                }
                return Some((
                    CborValue::Text(field.name.clone()),
                    CborValue::Map(encode_entries(sub, values, &path)),
                ));
            }
            let value = match values.get(&path) {
                Some(value) => value.clone(),
                None if field.optional => return None,
                None => field.seed_value(),
            };
            Some((CborValue::Text(field.name.clone()), value.to_cbor()))
        })
        .collect()
}

/// Seed values for a schema: the declared default (or type-zero) per
/// required field, at dotted paths inside groups. Optional fields start
/// unset; an optional group with `.default true` starts enabled (its
/// marker is set and its sub-fields seed).
pub fn default_values(fields: &[ConfigField]) -> BTreeMap<String, ConfigValue> {
    let mut out = BTreeMap::new();
    seed_defaults(fields, "", &mut out);
    out
}

fn seed_defaults(fields: &[ConfigField], prefix: &str, out: &mut BTreeMap<String, ConfigValue>) {
    for field in fields {
        let path = field_path(prefix, &field.name);
        if let ConfigFieldType::Group(sub) = &field.ty {
            let enabled = !field.optional || matches!(field.default, Some(ConfigValue::Bool(true)));
            if !enabled {
                continue;
            }
            if field.optional {
                out.insert(path.clone(), ConfigValue::Bool(true));
            }
            seed_defaults(sub, &path, out);
        } else if !field.optional {
            out.insert(path, field.seed_value());
        }
    }
}

/// Best-effort decode of a CBOR config map into values, keyed by field name.
/// Nested maps flatten to dotted paths, with a `Bool(true)` marker at the
/// map's own path. Values whose CBOR type doesn't match a known scalar are
/// skipped.
pub fn decode(bytes: &[u8]) -> BTreeMap<String, ConfigValue> {
    let mut out = BTreeMap::new();
    if let Ok(CborValue::Map(entries)) = ciborium::de::from_reader::<CborValue, _>(bytes) {
        flatten_entries(&entries, "", &mut out);
    }
    out
}

fn flatten_entries(
    entries: &[(CborValue, CborValue)],
    prefix: &str,
    out: &mut BTreeMap<String, ConfigValue>,
) {
    for (key, value) in entries {
        let CborValue::Text(name) = key else {
            continue;
        };
        let path = field_path(prefix, name);
        if let CborValue::Map(sub) = value {
            out.insert(path.clone(), ConfigValue::Bool(true));
            flatten_entries(sub, &path, out);
        } else if let Some(config_value) = cbor_to_value(value) {
            out.insert(path, config_value);
        }
    }
}

fn cbor_to_value(value: &CborValue) -> Option<ConfigValue> {
    match value {
        CborValue::Bool(b) => Some(ConfigValue::Bool(*b)),
        CborValue::Integer(i) => Some(ConfigValue::Int((*i).try_into().ok()?)),
        CborValue::Float(f) => Some(ConfigValue::Float(*f)),
        CborValue::Text(t) => Some(ConfigValue::Text(t.clone())),
        CborValue::Array(items) => Some(ConfigValue::List(
            items.iter().filter_map(cbor_to_value).collect(),
        )),
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

    fn list(element: ConfigFieldType, min_len: usize) -> ConfigFieldType {
        ConfigFieldType::List {
            element: Box::new(element),
            min_len,
        }
    }

    #[test]
    fn parses_array_types() {
        let fields =
            parse_schema(r#"{ channels: [* tstr], weights: [+ float], counts: [* int] }"#).unwrap();
        assert_eq!(fields[0].ty, list(ConfigFieldType::Text, 0));
        assert_eq!(fields[1].ty, list(ConfigFieldType::Float, 1));
        assert_eq!(fields[2].ty, list(ConfigFieldType::Int, 0));
        // A fresh array field seeds to an empty list.
        assert_eq!(fields[0].seed_value(), ConfigValue::List(Vec::new()));
    }

    #[test]
    fn arrays_of_lists_or_enums_are_unsupported() {
        assert!(parse_schema(r#"{ x: [* [* int]] }"#).is_err());
        assert!(parse_schema(r#"{ x: [* "a" / "b"] }"#).is_err());
        assert!(parse_schema(r#"{ x: [ tstr] }"#).is_err()); // no occurrence marker
    }

    #[test]
    fn array_parse_respects_min_len_and_element_type() {
        let star = list(ConfigFieldType::Text, 0);
        let plus = list(ConfigFieldType::Text, 1);
        // `*` allows empty; `+` does not.
        assert_eq!(
            ConfigValue::parse(&star, ""),
            Some(ConfigValue::List(Vec::new()))
        );
        assert_eq!(ConfigValue::parse(&plus, ""), None);
        // Comma-separated, trimmed, empties dropped.
        assert_eq!(
            ConfigValue::parse(&star, "radius, stiffness_scale ,"),
            Some(ConfigValue::List(vec![
                ConfigValue::Text("radius".into()),
                ConfigValue::Text("stiffness_scale".into()),
            ]))
        );
        // A bad element fails the whole list.
        let nums = list(ConfigFieldType::Float, 0);
        assert_eq!(ConfigValue::parse(&nums, "1.0, nope"), None);
    }

    #[test]
    fn arrays_round_trip_through_cbor() {
        let fields = parse_schema(r#"{ channels: [* tstr] }"#).unwrap();
        let mut values = BTreeMap::new();
        values.insert(
            "channels".to_string(),
            ConfigValue::List(vec![
                ConfigValue::Text("radius".into()),
                ConfigValue::Text("stiffness_scale".into()),
            ]),
        );
        let decoded = decode(&encode(&fields, &values));
        assert_eq!(decoded.get("channels"), values.get("channels"));
        // A missing array field encodes as an empty CBOR array (all-present
        // semantics), not an omitted key.
        let decoded = decode(&encode(&fields, &BTreeMap::new()));
        assert_eq!(
            decoded.get("channels"),
            Some(&ConfigValue::List(Vec::new()))
        );
    }

    const BLOCKS: &str = r#"{ ? surface: { outside: "project" / "drop" .default "project", skin_radius_factor: float .default 1.0 } .default true, ? support: { axis: "auto" / "x" .default "auto", max_descent: float .default 0.0 }, weld: bool .default true }"#;

    #[test]
    fn parses_nested_groups() {
        let fields = parse_schema(BLOCKS).unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "surface");
        assert!(fields[0].optional);
        assert_eq!(fields[0].default, Some(ConfigValue::Bool(true)));
        let ConfigFieldType::Group(sub) = &fields[0].ty else {
            panic!("surface is a group");
        };
        assert_eq!(sub[0].name, "outside");
        assert!(matches!(sub[0].ty, ConfigFieldType::Enum(_)));
        assert_eq!(sub[1].name, "skin_radius_factor");
        assert_eq!(sub[1].default, Some(ConfigValue::Float(1.0)));
        assert_eq!(fields[1].name, "support");
        assert_eq!(fields[1].default, None, "no .default tail: starts disabled");
        assert_eq!(fields[2].name, "weld");
        assert_eq!(fields[2].ty, ConfigFieldType::Bool);

        // Dotted lookup walks the nesting.
        assert_eq!(
            find_field(&fields, "surface.outside").unwrap().name,
            "outside"
        );
        assert_eq!(find_field(&fields, "weld").unwrap().name, "weld");
        assert!(find_field(&fields, "surface.nope").is_none());
        assert!(find_field(&fields, "weld.outside").is_none());

        // Literal dotted names (Lua parameter forms) win over descent.
        let lua = parse_schema("{ sphere.radius: float .default 1.0 }").unwrap();
        assert_eq!(
            find_field(&lua, "sphere.radius").unwrap().name,
            "sphere.radius"
        );
    }

    #[test]
    fn group_defaults_seed_and_encode_as_nested_maps() {
        let fields = parse_schema(BLOCKS).unwrap();
        let values = default_values(&fields);
        // The default-enabled group seeds its marker and sub-fields; the
        // disabled one stays entirely unset.
        assert_eq!(values.get("surface"), Some(&ConfigValue::Bool(true)));
        assert_eq!(
            values.get("surface.outside"),
            Some(&ConfigValue::Text("project".into()))
        );
        assert!(!values.contains_key("support"));
        assert!(!values.contains_key("support.axis"));

        // Encode emits surface as a nested map, omits support, and decode
        // round-trips the flattened form.
        let decoded = decode(&encode(&fields, &values));
        assert_eq!(decoded.get("surface"), Some(&ConfigValue::Bool(true)));
        assert_eq!(
            decoded.get("surface.skin_radius_factor"),
            Some(&ConfigValue::Float(1.0))
        );
        assert!(!decoded.contains_key("support"));
        assert_eq!(decoded.get("weld"), Some(&ConfigValue::Bool(true)));

        // The raw CBOR really is a nested map (what the operator sees).
        let raw: CborValue = ciborium::de::from_reader(&encode(&fields, &values)[..]).unwrap();
        let CborValue::Map(entries) = raw else {
            panic!("map")
        };
        let surface = entries
            .iter()
            .find(|(k, _)| matches!(k, CborValue::Text(t) if t == "surface"))
            .map(|(_, v)| v)
            .unwrap();
        assert!(
            matches!(surface, CborValue::Map(_)),
            "nested map, not a string"
        );
    }

    #[test]
    fn toggling_a_group_marker_adds_and_removes_its_map() {
        let fields = parse_schema(BLOCKS).unwrap();
        let mut values = default_values(&fields);
        values.insert("support".to_string(), ConfigValue::Bool(true));
        values.insert("support.max_descent".to_string(), ConfigValue::Float(20.0));
        let decoded = decode(&encode(&fields, &values));
        assert_eq!(
            decoded.get("support.max_descent"),
            Some(&ConfigValue::Float(20.0))
        );
        // Missing required sub-fields fall back to their seeds.
        assert_eq!(
            decoded.get("support.axis"),
            Some(&ConfigValue::Text("auto".into()))
        );
        // A false marker (the switch off) omits the block again.
        values.insert("surface".to_string(), ConfigValue::Bool(false));
        let decoded = decode(&encode(&fields, &values));
        assert!(!decoded.contains_key("surface"));
        assert!(!decoded.contains_key("surface.outside"));
    }

    #[test]
    fn parses_numeric_bounds_in_any_order() {
        let fields = parse_schema(
            "{ a: float .ge 0 .le 0.5, b: float .default 0.3 .le 1 .ge 0, c: int .le 10 }",
        )
        .unwrap();
        assert_eq!((fields[0].min, fields[0].max), (Some(0.0), Some(0.5)));
        assert_eq!(fields[0].default, None);
        assert_eq!((fields[1].min, fields[1].max), (Some(0.0), Some(1.0)));
        assert_eq!(fields[1].default, Some(ConfigValue::Float(0.3)));
        assert_eq!((fields[2].min, fields[2].max), (None, Some(10.0)));
    }

    #[test]
    fn uint_implies_a_zero_lower_bound() {
        let fields = parse_schema("{ n: uint .default 16, m: uint .ge 2 }").unwrap();
        assert_eq!(fields[0].ty, ConfigFieldType::Int);
        assert_eq!(fields[0].min, Some(0.0));
        assert_eq!(fields[0].default, Some(ConfigValue::Int(16)));
        // An explicit `.ge` wins over the implied zero.
        assert_eq!(fields[1].min, Some(2.0));
    }

    #[test]
    fn field_parse_gates_on_bounds() {
        let fields = parse_schema("{ ratio: float .ge 0 .le 0.5 }").unwrap();
        let field = &fields[0];
        assert_eq!(field.parse("0.3"), Some(ConfigValue::Float(0.3)));
        assert_eq!(field.parse("0"), Some(ConfigValue::Float(0.0))); // inclusive
        assert_eq!(field.parse("0.5"), Some(ConfigValue::Float(0.5))); // inclusive
        assert_eq!(field.parse("0.6"), None); // above max
        assert_eq!(field.parse("-0.1"), None); // below min
        assert_eq!(field.parse("abc"), None); // not a number

        // uint rejects negatives via its implied bound.
        let uint = &parse_schema("{ n: uint }").unwrap()[0];
        assert_eq!(uint.parse("3"), Some(ConfigValue::Int(3)));
        assert_eq!(uint.parse("-1"), None);
    }
}
