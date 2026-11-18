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
//! Arrays take any element type but a group: `[* T]`, `[+ T]`, `[? T]`,
//! bounded `[n*m T]` (either bound may be left out: `[2* T]`, `[*4 T]`),
//! and homogeneous tuples `[T, T, T]` (exactly that many; `[T]` is one).
//! An element may itself be an array (`[* [2*2 int]]`) or a record
//! (`[* { at: [3*3 float], ? why: tstr }]`), which is one value: a CBOR map
//! inside the array. An array default is written as JSON
//! (`up: [3*3 float] .default [0.0, 0.0, 1.0]`).
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
    /// A homogeneous array (CDDL `[* T]`, `[+ T]`, `[n*m T]`, or a tuple
    /// `[T, T, T]`), holding between `min_len` and `max_len` elements
    /// (`None`: no upper bound). The element is any type but a group.
    List {
        element: Box<ConfigFieldType>,
        min_len: usize,
        max_len: Option<usize>,
    },
    /// A record as a value (`{ ... }` as an array element, or a field of
    /// such a record), encoded as a CBOR map. Unlike a [`Group`] it is a
    /// single value: an array element holds the whole record.
    ///
    /// [`Group`]: ConfigFieldType::Group
    Record(Vec<ConfigField>),
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
    /// A record's entries, in order (only fields that are set).
    Record(Vec<(String, ConfigValue)>),
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
    #[error(
        "unsupported CDDL type `{0}` (supported: bool, int, float, tstr, string enums, arrays and records of these)"
    )]
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
    /// The neutral value for this type when no default is declared: an
    /// array holds `min_len` neutral elements, a record its required fields.
    pub fn zero(&self) -> ConfigValue {
        match self {
            ConfigFieldType::Bool => ConfigValue::Bool(false),
            ConfigFieldType::Int => ConfigValue::Int(0),
            ConfigFieldType::Float => ConfigValue::Float(0.0),
            ConfigFieldType::Text => ConfigValue::Text(String::new()),
            ConfigFieldType::Enum(options) => {
                ConfigValue::Text(options.first().cloned().unwrap_or_default())
            }
            ConfigFieldType::List {
                element, min_len, ..
            } => ConfigValue::List(vec![element.zero(); *min_len]),
            ConfigFieldType::Record(fields) => ConfigValue::Record(
                fields
                    .iter()
                    .filter(|f| !f.optional)
                    .map(|f| (f.name.clone(), f.seed_value()))
                    .collect(),
            ),
            // A group's own value is its enablement marker.
            ConfigFieldType::Group(_) => ConfigValue::Bool(false),
        }
    }

    /// A type holding one value that is not an array or a record.
    fn is_scalar(&self) -> bool {
        !matches!(
            self,
            ConfigFieldType::List { .. } | ConfigFieldType::Record(_) | ConfigFieldType::Group(_)
        )
    }

    /// Check an untyped value (one read from JSON) against this type, at
    /// every depth: integers become floats where a float is declared (an
    /// operator decoding `f64` rejects a CBOR integer), enums must name an
    /// option, arrays must hold an allowed number of elements, and records
    /// must name only declared fields, set every required one, and keep
    /// numbers within their declared bounds. A record keeps its entries in
    /// the order given.
    pub fn coerce(&self, value: ConfigValue) -> Result<ConfigValue, String> {
        let got = |v: &ConfigValue| v.to_json_text();
        match (self, value) {
            (ConfigFieldType::Bool | ConfigFieldType::Group(_), v @ ConfigValue::Bool(_)) => Ok(v),
            (ConfigFieldType::Int, v @ ConfigValue::Int(_)) => Ok(v),
            (ConfigFieldType::Float, v @ ConfigValue::Float(_)) => Ok(v),
            (ConfigFieldType::Float, ConfigValue::Int(i)) => Ok(ConfigValue::Float(i as f64)),
            (ConfigFieldType::Text, v @ ConfigValue::Text(_)) => Ok(v),
            (ConfigFieldType::Enum(options), ConfigValue::Text(t)) => {
                if options.contains(&t) {
                    Ok(ConfigValue::Text(t))
                } else {
                    Err(format!("{t:?} is not one of {}", options.join("/")))
                }
            }
            (
                ConfigFieldType::List {
                    element,
                    min_len,
                    max_len,
                },
                ConfigValue::List(items),
            ) => {
                let n = items.len();
                if n < *min_len || max_len.is_some_and(|max| n > max) {
                    return Err(match max_len {
                        Some(max) if max == min_len => format!("needs {max} element(s), got {n}"),
                        Some(max) => format!("needs {min_len} to {max} element(s), got {n}"),
                        None => format!("needs at least {min_len} element(s), got {n}"),
                    });
                }
                items
                    .into_iter()
                    .enumerate()
                    .map(|(i, item)| element.coerce(item).map_err(|e| format!("[{i}]: {e}")))
                    .collect::<Result<Vec<_>, _>>()
                    .map(ConfigValue::List)
            }
            (ConfigFieldType::Record(fields), ConfigValue::Record(entries)) => {
                let mut out = Vec::with_capacity(entries.len());
                for (name, value) in entries {
                    let field = fields.iter().find(|f| f.name == name).ok_or_else(|| {
                        let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
                        format!("no field {name:?}; fields: {}", names.join(", "))
                    })?;
                    if out.iter().any(|(n, _)| *n == name) {
                        return Err(format!("{name:?} is given twice"));
                    }
                    let value = field.ty.coerce(value).map_err(|e| format!("{name}: {e}"))?;
                    if !field.in_bounds(&value) {
                        return Err(format!("{name}: {} is outside its bounds", got(&value)));
                    }
                    out.push((name, value));
                }
                if let Some(missing) = fields
                    .iter()
                    .find(|f| !f.optional && !out.iter().any(|(n, _)| *n == f.name))
                {
                    return Err(format!("{} is required", missing.name));
                }
                Ok(ConfigValue::Record(out))
            }
            (ty, v) => Err(format!("expected {}, got {}", ty.describe(), got(&v))),
        }
    }

    /// What a value of this type is, for messages.
    fn describe(&self) -> &'static str {
        match self {
            ConfigFieldType::Bool | ConfigFieldType::Group(_) => "a bool",
            ConfigFieldType::Int => "an integer",
            ConfigFieldType::Float => "a number",
            ConfigFieldType::Text | ConfigFieldType::Enum(_) => "a string",
            ConfigFieldType::List { .. } => "an array",
            ConfigFieldType::Record(_) => "a record",
        }
    }
}

impl ConfigValue {
    /// Render the value as the text an editor field shows, which
    /// [`ConfigValue::parse`] reads back exactly. Arrays and records are
    /// written as JSON (so an empty inner array, or text holding commas,
    /// survives); v2's array editor shows each element on its own.
    pub fn to_display_string(&self) -> String {
        match self {
            ConfigValue::Bool(b) => b.to_string(),
            ConfigValue::Int(i) => i.to_string(),
            ConfigValue::Float(f) => format_float(*f),
            ConfigValue::Text(t) => t.clone(),
            ConfigValue::List(_) | ConfigValue::Record(_) => self.to_json_text(),
        }
    }

    /// The value as JSON (a non-finite float, which JSON cannot hold, is
    /// written as `null` and does not read back).
    pub fn to_json_text(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Parse editor text back into a value of the given type. Returns `None`
    /// when the text isn't a valid value (empty, non-numeric, or an
    /// out-of-vocabulary enum), so the caller keeps the raw edit buffer without
    /// committing it to the config. An array or record written as JSON is
    /// checked against the type ([`ConfigFieldType::coerce`]); an array of
    /// scalars may also be written bare, split on commas, each non-empty
    /// element parsed as the element type.
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
            ConfigFieldType::Record(_) => ty.coerce(serde_json::from_str(text).ok()?).ok(),
            ConfigFieldType::List { element, .. } => {
                if text.trim_start().starts_with('[')
                    && let Ok(value) = serde_json::from_str::<ConfigValue>(text)
                {
                    return ty.coerce(value).ok();
                }
                if !element.is_scalar() {
                    return None;
                }
                let mut items = Vec::new();
                for part in text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    items.push(ConfigValue::parse(element, part)?);
                }
                ty.coerce(ConfigValue::List(items)).ok()
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
            ConfigValue::Record(entries) => CborValue::Map(
                entries
                    .iter()
                    .map(|(k, v)| (CborValue::Text(k.clone()), v.to_cbor()))
                    .collect(),
            ),
        }
    }
}

/// JSON form, records keeping their entry order.
impl serde::Serialize for ConfigValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self {
            ConfigValue::Bool(b) => s.serialize_bool(*b),
            ConfigValue::Int(i) => s.serialize_i64(*i),
            ConfigValue::Float(f) => s.serialize_f64(*f),
            ConfigValue::Text(t) => s.serialize_str(t),
            ConfigValue::List(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            ConfigValue::Record(entries) => {
                let mut map = s.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

/// An untyped value from JSON (or any self-describing format), records
/// keeping their key order; check it against a schema type with
/// [`ConfigFieldType::coerce`]. `null` has no config value.
impl<'de> serde::Deserialize<'de> for ConfigValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = ConfigValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a bool, number, string, array or map")
            }
            fn visit_bool<E>(self, v: bool) -> Result<ConfigValue, E> {
                Ok(ConfigValue::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<ConfigValue, E> {
                Ok(ConfigValue::Int(v))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<ConfigValue, E> {
                i64::try_from(v)
                    .map(ConfigValue::Int)
                    .map_err(|_| E::custom(format!("{v} is too large for an integer")))
            }
            fn visit_f64<E>(self, v: f64) -> Result<ConfigValue, E> {
                Ok(ConfigValue::Float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<ConfigValue, E> {
                Ok(ConfigValue::Text(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<ConfigValue, E> {
                Ok(ConfigValue::Text(v))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<ConfigValue, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(ConfigValue::List(items))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<ConfigValue, A::Error> {
                let mut entries = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, ConfigValue>()? {
                    entries.push((k, v));
                }
                Ok(ConfigValue::Record(entries))
            }
        }
        d.deserialize_any(Visitor)
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
    parse_record_body(s, true)
}

/// Parse the fields of a record body. With `groups`, a braced right-hand
/// side declares a group (the config record and the groups in it); without,
/// it is a record value (inside an array element, where nothing is
/// flattened to dotted paths).
fn parse_record_body(s: &str, groups: bool) -> Result<Vec<ConfigField>, ConfigSchemaError> {
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
        if groups && rhs.starts_with('{') {
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
/// record, and commas inside a quoted default (`.default "M0,0 H1"`)
/// belong to the string.
fn split_record_parts(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut quoted = false;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '"' => quoted = !quoted,
            _ if quoted => {}
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

/// Whitespace tokens, except that a double-quoted string, or a bracketed or
/// braced literal (`[0.0, 0.0, 1.0]`), is one token however many spaces it
/// holds.
fn annotation_tokens(s: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = None;
    let mut depth = 0usize;
    let mut quoted = false;
    for (i, c) in s.char_indices() {
        if start.is_none() {
            if c.is_whitespace() {
                continue;
            }
            start = Some(i);
        }
        match c {
            '"' => quoted = !quoted,
            _ if quoted => {}
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            c if c.is_whitespace() && depth == 0 => {
                tokens.push(&s[start.take().unwrap_or(i)..i]);
            }
            _ => {}
        }
    }
    if let Some(start) = start {
        tokens.push(&s[start..]);
    }
    tokens
}

/// Where `pattern` first occurs outside quotes and outside any brackets or
/// braces: a control operator inside an array's record element
/// (`[* { spacing: float .default 0.02 }]`) belongs to that element.
fn find_top_level(s: &str, pattern: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut quoted = false;
    for (i, c) in s.char_indices() {
        if !quoted && depth == 0 && s[i..].starts_with(pattern) {
            return Some(i);
        }
        match c {
            '"' => quoted = !quoted,
            _ if quoted => {}
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    None
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
    let sub = parse_record_body(rhs[1..close].trim(), true)?;
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
    if let Some(start) = OPS.iter().filter_map(|op| find_top_level(rhs, op)).min() {
        let ty = rhs[..start].trim();
        let mut default = None;
        let mut min = None;
        let mut max = None;
        let tokens = annotation_tokens(&rhs[start..]);
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
    if let Some(paren_idx) = find_top_level(rhs, "(")
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
    if let Some(inner) = ty.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        return parse_array(inner.trim(), ty);
    }
    if let Some(inner) = ty.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        return Ok(ConfigFieldType::Record(parse_record_body(
            inner.trim(),
            false,
        )?));
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

/// An array type's inside: an occurrence and one element type (`* T`,
/// `+ T`, `? T`, `n*m T`, `n* T`, `*m T`), or a homogeneous tuple
/// (`T, T, T`; a lone `T` is a one-element tuple).
fn parse_array(inner: &str, whole: &str) -> Result<ConfigFieldType, ConfigSchemaError> {
    let unsupported = |why: &str| ConfigSchemaError::UnsupportedType(format!("{whole} ({why})"));
    let parts: Vec<&str> = split_record_parts(inner)
        .into_iter()
        .map(str::trim)
        .collect();
    if parts.len() > 1 {
        let types = parts
            .iter()
            .map(|p| parse_field_type(p))
            .collect::<Result<Vec<_>, _>>()?;
        if types.windows(2).any(|w| w[0] != w[1]) {
            return Err(unsupported("a tuple's elements must share one type"));
        }
        return Ok(ConfigFieldType::List {
            element: Box::new(types[0].clone()),
            min_len: types.len(),
            max_len: Some(types.len()),
        });
    }
    let (min_len, max_len, element) = occurrence(inner);
    if max_len.is_some_and(|max| max < min_len) {
        return Err(unsupported("fewer allowed than required"));
    }
    if element.is_empty() {
        return Err(unsupported("no element type"));
    }
    Ok(ConfigFieldType::List {
        element: Box::new(parse_field_type(element)?),
        min_len,
        max_len,
    })
}

/// Split a leading CDDL occurrence indicator off an array's inside:
/// `(min, max, rest)`. Without one, the element occurs exactly once.
fn occurrence(s: &str) -> (usize, Option<usize>, &str) {
    let digits = |s: &str| s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if let Some(rest) = s.strip_prefix('+') {
        return (1, None, rest.trim());
    }
    if let Some(rest) = s.strip_prefix('?') {
        return (0, Some(1), rest.trim());
    }
    let n = digits(s);
    if let Some(after) = s[n..].strip_prefix('*') {
        let m = digits(after);
        let min = s[..n].parse().unwrap_or(0);
        let max = after[..m].parse().ok();
        return (min, max, after[m..].trim());
    }
    (1, Some(1), s)
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
/// map's own path (groups); a map inside an array is a record value, kept
/// whole and in order. Values with no config form (null, bytes, tags) are
/// skipped, and so is an array or record that holds one.
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
        // An array or map holding anything without a config value is
        // skipped whole rather than shortened.
        CborValue::Array(items) => items
            .iter()
            .map(cbor_to_value)
            .collect::<Option<Vec<_>>>()
            .map(ConfigValue::List),
        CborValue::Map(entries) => entries
            .iter()
            .map(|(k, v)| match k {
                CborValue::Text(k) => Some((k.clone(), cbor_to_value(v)?)),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(ConfigValue::Record),
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
    fn quoted_defaults_may_contain_spaces_and_commas() {
        let fields =
            parse_schema(r#"{ path: tstr .default "M0,0 H1 V1 Z", round: float .default 0.0 }"#)
                .unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(
            fields[0].default,
            Some(ConfigValue::Text("M0,0 H1 V1 Z".to_string()))
        );
        assert_eq!(fields[1].name, "round");
        assert_eq!(fields[1].default, Some(ConfigValue::Float(0.0)));
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
        bounded(element, min_len, None)
    }

    fn bounded(
        element: ConfigFieldType,
        min_len: usize,
        max_len: Option<usize>,
    ) -> ConfigFieldType {
        ConfigFieldType::List {
            element: Box::new(element),
            min_len,
            max_len,
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
    fn arrays_nest_and_take_enums() {
        // Once unsupported; operators need points (`[3*3 float]`), outlines
        // (`[* [* float]]`) and lists of authored edits (records).
        let fields = parse_schema(r#"{ x: [* [* int]], y: [* "a" / "b"], z: [ tstr] }"#).unwrap();
        assert_eq!(fields[0].ty, list(list(ConfigFieldType::Int, 0), 0));
        assert_eq!(
            fields[1].ty,
            list(ConfigFieldType::Enum(vec!["a".into(), "b".into()]), 0)
        );
        // No occurrence marker: CDDL's one-element array.
        assert_eq!(fields[2].ty, bounded(ConfigFieldType::Text, 1, Some(1)));
    }

    #[test]
    fn occurrences_bound_arrays_and_tuples_fix_them() {
        let fields = parse_schema(
            "{ a: [2*2 int], b: [2* float], c: [*4 float], d: [? int], e: [float, float, float], f: [+ bool] }",
        )
        .unwrap();
        assert_eq!(fields[0].ty, bounded(ConfigFieldType::Int, 2, Some(2)));
        assert_eq!(fields[1].ty, bounded(ConfigFieldType::Float, 2, None));
        assert_eq!(fields[2].ty, bounded(ConfigFieldType::Float, 0, Some(4)));
        assert_eq!(fields[3].ty, bounded(ConfigFieldType::Int, 0, Some(1)));
        assert_eq!(fields[4].ty, bounded(ConfigFieldType::Float, 3, Some(3)));
        assert_eq!(fields[5].ty, bounded(ConfigFieldType::Bool, 1, None));
        // A required fixed-size array seeds that many zeros.
        assert_eq!(
            fields[4].seed_value(),
            ConfigValue::List(vec![ConfigValue::Float(0.0); 3])
        );
        // Lengths are checked both ways.
        assert_eq!(ConfigValue::parse(&fields[0].ty, "1, 2, 3"), None);
        assert_eq!(ConfigValue::parse(&fields[0].ty, "1"), None);
        assert_eq!(
            ConfigValue::parse(&fields[0].ty, "1, 2"),
            Some(ConfigValue::List(vec![
                ConfigValue::Int(1),
                ConfigValue::Int(2)
            ]))
        );

        assert!(parse_schema("{ x: [int, float] }").is_err(), "mixed tuple");
        assert!(parse_schema("{ x: [3*2 int] }").is_err(), "max below min");
        assert!(parse_schema("{ x: [*] }").is_err(), "no element type");
    }

    #[test]
    fn array_defaults_are_json_and_take_the_declared_type() {
        let fields = parse_schema(
            "{ up: [3*3 float] .default [0, 0, 1], step: float .default 0.002, \
             ? outline: [* [* float]] .default [[0.0, 1.5], [2, 3]] }",
        )
        .unwrap();
        // Integers in a float array's default encode as floats.
        assert_eq!(
            fields[0].default,
            Some(ConfigValue::List(vec![
                ConfigValue::Float(0.0),
                ConfigValue::Float(0.0),
                ConfigValue::Float(1.0)
            ]))
        );
        assert_eq!(fields[1].name, "step");
        assert_eq!(fields[1].default, Some(ConfigValue::Float(0.002)));
        assert_eq!(
            fields[2].default,
            Some(ConfigValue::List(vec![
                ConfigValue::List(vec![ConfigValue::Float(0.0), ConfigValue::Float(1.5)]),
                ConfigValue::List(vec![ConfigValue::Float(2.0), ConfigValue::Float(3.0)]),
            ]))
        );
    }

    /// The schemas of the Kalogon cushion operators (scan-to-cushion), as
    /// written: the shapes this parser grew to cover.
    const CUSHION_SCHEMAS: [&str; 8] = [
        r#"{ closed: bool, ? start: [2*2 int], ? end: [2*2 int], ? drop: [* [2*2 int]], ? sequences: [* [2* [2*2 int]]], sample_step: float .default 0.001 }"#,
        r#"{ ? outline: [* [* float]], inset: float .ge 0.0 .default 0.0, ? z_min: float, ? z_max: float, up: [3*3 float] .default [0.0, 0.0, 1.0], ? facing_min: float, direction: tstr .default "normal", reach: float .ge 0.0001 .default 0.03, step: float .ge 0.00001 .default 0.001, band: float .ge 0.00001 .default 0.003, field: tstr .default "distance" }"#,
        r#"{ ? camera: { width: int .ge 1, height: int .ge 1, ? pos: [* float], ? target: [* float], ? up: [* float], ? fov: float, ? projection: tstr, ? ortho_scale: float, ? intrinsics: [* float], ? pose: [* float], ? near: float, ? far: float }, ? pixels: [* [* float]], ? rays: [* [* float]], hit: tstr .default "nearest", back_faces: bool .default true }"#,
        r#"{ edits: [* { why: tstr, ? drop: tstr, ? keep: tstr, ? point: [* float], ? normal: [* float], ? crossing: tstr, ? outline: [* [* float]], ? z0: float, ? z1: float, ? center: [* float], ? radius: float, ? at: [* float], ? max_distance: float, ? within: float, ? stop_at: [* { point: [* float], normal: [* float] }], ? area_below: float }], weld: float .ge 0.0 .default 0.000001 }"#,
        r#"{ ? x0: float, ? x1: float, ? y0: float, ? y1: float, seed_step: float .ge 0.0 .default 0.04, ? add: [* { at: [2*3 float], why: tstr }], ? lines: [* { points: [2* [2*3 float]], spacing: float .default 0.02, why: tstr }], ? remove: [* { at: [2*2 float], radius: float, why: tstr }], ? accept: [* { at: [2*2 float], radius: float, why: tstr }], line_spacing: float .default 0.01, out_step: float .default 0.002, merge: float .ge 0.0 .default 0.004, close_pair: float .ge 0.0 .default 0.015, departure_limit: float .default 0.002, departure_min_area_cm2: float .ge 0.0 .default 1.0, neighbour_radius: float .default 0.06 }"#,
        r#"{ ? outline: [* [* float]], inset: float .ge 0.0 .default 0.0, step: float .default 0.002, base_point: [3*3 float] .default [0.0, 0.0, 0.0], base_normal: [3*3 float] .default [0.0, 0.0, 1.0], ? flag_below: float, seen_within: float .default 0.003, min_normal_z: float .default 0.1, field: tstr .default "thickness" }"#,
        r#"{ edits: [* { kind: "ignore" / "offset" / "target" / "pin", ? at: [float, float, float], ? radius: float, ? amount: float, ? blend: float, ? key: tstr, ? value: float, why: tstr }] }"#,
        r#"{ x0: float, x1: float, ? y0: float, ? y1: float, ? depth: float, z_floor: float .default -0.08, cell: float .default 0.0025, floor_spacing: float .default 0.05, wall_station: float .default 0.03, conform_spacing: float .default 0.04, conform_stiffness: float .default 0.05, fit_band: float .default 0.025, robust_scale: float .default 0.0015, lump_min: float .default 0.001, front: [4*4 float], round_top: float .default 0.01, round_line: float .default 0.01, round_trough: float .default 0.012, round_corner: float .default 0.01, round_ridge: float .default 0.01, cane_cutout: float .default 0.0 }"#,
    ];

    #[test]
    fn the_cushion_operator_schemas_parse() {
        for schema in CUSHION_SCHEMAS {
            let fields = parse_schema(schema).unwrap_or_else(|e| panic!("{schema}: {e}"));
            // Their defaults encode and read back unchanged.
            let values = default_values(&fields);
            let decoded = decode(&encode(&fields, &values));
            assert_eq!(decoded, values, "{schema}");
        }
        // seat_points: a record inside an array, its own default kept to
        // itself rather than read as the array field's default.
        let fields = parse_schema(CUSHION_SCHEMAS[4]).unwrap();
        let lines = find_field(&fields, "lines").unwrap();
        assert_eq!(lines.default, None);
        let ConfigFieldType::List { element, .. } = &lines.ty else {
            panic!("lines is an array")
        };
        let ConfigFieldType::Record(sub) = element.as_ref() else {
            panic!("of records")
        };
        assert_eq!(
            sub[0].ty,
            list(bounded(ConfigFieldType::Float, 2, Some(3)), 2)
        );
        assert_eq!(sub[1].default, Some(ConfigValue::Float(0.02)));
        // A fresh record element holds its required fields.
        assert_eq!(
            element.zero(),
            ConfigValue::Record(vec![
                (
                    "points".into(),
                    ConfigValue::List(vec![ConfigValue::List(vec![ConfigValue::Float(0.0); 2]); 2])
                ),
                ("spacing".into(), ConfigValue::Float(0.02)),
                ("why".into(), ConfigValue::Text(String::new())),
            ])
        );
    }

    #[test]
    fn records_in_arrays_round_trip_whole_and_in_order() {
        let fields = parse_schema(CUSHION_SCHEMAS[3]).unwrap();
        // Written by hand, keys out of schema order, one optional set.
        let raw = CborValue::Map(vec![
            (
                CborValue::Text("edits".into()),
                CborValue::Array(vec![
                    CborValue::Map(vec![
                        (
                            CborValue::Text("drop".into()),
                            CborValue::Text("below".into()),
                        ),
                        (
                            CborValue::Text("why".into()),
                            CborValue::Text("the floor, \"under\" the seat".into()),
                        ),
                        (
                            CborValue::Text("outline".into()),
                            CborValue::Array(vec![
                                CborValue::Array(vec![
                                    CborValue::Float(0.1),
                                    CborValue::Float(-0.25),
                                ]),
                                CborValue::Array(vec![
                                    CborValue::Float(1e-7),
                                    CborValue::Float(3.0),
                                ]),
                            ]),
                        ),
                    ]),
                    CborValue::Map(vec![(
                        CborValue::Text("why".into()),
                        CborValue::Text("second".into()),
                    )]),
                ]),
            ),
            (CborValue::Text("weld".into()), CborValue::Float(1e-6)),
        ]);
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&raw, &mut bytes).unwrap();
        let values = decode(&bytes);
        assert_eq!(encode(&fields, &values), bytes, "byte for byte");

        // An editor's text for each element reads back as the same value.
        let edits = find_field(&fields, "edits").unwrap();
        let ConfigFieldType::List { element, .. } = &edits.ty else {
            panic!()
        };
        let Some(ConfigValue::List(items)) = values.get("edits") else {
            panic!()
        };
        for item in items {
            assert_eq!(
                ConfigValue::parse(element, &item.to_display_string()).as_ref(),
                Some(item)
            );
        }
        assert_eq!(
            ConfigValue::parse(&edits.ty, &values["edits"].to_display_string()).as_ref(),
            values.get("edits")
        );
    }

    #[test]
    fn seat_shape_front_seeds_four_zeros() {
        let fields = parse_schema(CUSHION_SCHEMAS[7]).unwrap();
        let values = default_values(&fields);
        assert_eq!(
            values.get("front"),
            Some(&ConfigValue::List(vec![ConfigValue::Float(0.0); 4]))
        );
        assert_eq!(values.get("z_floor"), Some(&ConfigValue::Float(-0.08)));
    }

    #[test]
    fn stl_imports_tuple_default_is_three_floats() {
        let fields = parse_schema(
            "{ scale: float .default 1.0, translate: [float, float, float] .default [0,0,0], center: bool .default false }",
        )
        .unwrap();
        assert_eq!(
            fields[1].default,
            Some(ConfigValue::List(vec![ConfigValue::Float(0.0); 3]))
        );
    }

    #[test]
    fn element_text_keeps_empty_arrays_and_commas_in_text() {
        let ty = list(list(ConfigFieldType::Text, 0), 0);
        let ConfigFieldType::List { element, .. } = &ty else {
            panic!()
        };
        for item in [
            ConfigValue::List(Vec::new()),
            ConfigValue::List(vec![
                ConfigValue::Text("a, b".into()),
                ConfigValue::Text(String::new()),
                ConfigValue::Text("say \"hi\"".into()),
            ]),
        ] {
            let text = item.to_display_string();
            assert!(
                !text.trim().is_empty(),
                "an empty array is not a blank buffer"
            );
            assert_eq!(ConfigValue::parse(element, &text), Some(item));
        }
    }

    #[test]
    fn floats_read_back_bit_for_bit() {
        // A point from a real outline; JSON parsing without
        // `float_roundtrip` read these two back one ulp off.
        let ty = list(list(ConfigFieldType::Float, 0), 0);
        let value = ConfigValue::List(vec![ConfigValue::List(vec![
            ConfigValue::Float(0.21883928571428574),
            ConfigValue::Float(-0.025982142857142898),
            ConfigValue::Float(5e-324),
            ConfigValue::Float(f64::MAX),
        ])]);
        assert_eq!(
            ConfigValue::parse(&ty, &value.to_display_string()),
            Some(value)
        );
    }

    #[test]
    fn records_are_checked_against_their_fields() {
        let fields = parse_schema(CUSHION_SCHEMAS[6]).unwrap();
        let edits = &fields[0].ty;
        let ok = r#"[{"kind": "pin", "at": [0, 0.1, 0.2], "why": "the ridge"}]"#;
        let Some(ConfigValue::List(items)) = ConfigValue::parse(edits, ok) else {
            panic!("parses")
        };
        assert_eq!(
            items[0],
            ConfigValue::Record(vec![
                ("kind".into(), ConfigValue::Text("pin".into())),
                (
                    "at".into(),
                    ConfigValue::List(vec![
                        ConfigValue::Float(0.0),
                        ConfigValue::Float(0.1),
                        ConfigValue::Float(0.2)
                    ])
                ),
                ("why".into(), ConfigValue::Text("the ridge".into())),
            ])
        );
        for bad in [
            r#"[{"kind": "pin"}]"#,                              // why is required
            r#"[{"kind": "push", "why": ""}]"#,                  // not an option
            r#"[{"kind": "pin", "why": "", "at": [0, 1]}]"#,     // two of three
            r#"[{"kind": "pin", "why": "", "colour": "red"}]"#,  // no such field
            r#"[{"kind": "pin", "why": "", "radius": "wide"}]"#, // not a number
        ] {
            assert_eq!(ConfigValue::parse(edits, bad), None, "{bad}");
        }
        let ConfigFieldType::List { element, .. } = edits else {
            panic!()
        };
        assert!(
            element
                .coerce(serde_json::from_str(r#"{"kind": "pin"}"#).unwrap())
                .unwrap_err()
                .contains("why is required")
        );
    }

    #[test]
    fn an_array_holding_a_value_with_no_config_form_is_skipped_whole() {
        let raw = CborValue::Map(vec![(
            CborValue::Text("xs".into()),
            CborValue::Array(vec![CborValue::Float(1.0), CborValue::Null]),
        )]);
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&raw, &mut bytes).unwrap();
        assert!(!decode(&bytes).contains_key("xs"), "not shortened to [1.0]");
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
