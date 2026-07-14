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
    /// A homogeneous array of a scalar element type (CDDL `[* T]` / `[+ T]`).
    /// `min_len` is 1 for `[+ T]` (at least one element) and 0 for `[* T]`.
    /// Elements are one of `Bool`/`Int`/`Float`/`Text` — not nested lists or
    /// enums.
    List {
        element: Box<ConfigFieldType>,
        min_len: usize,
    },
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
