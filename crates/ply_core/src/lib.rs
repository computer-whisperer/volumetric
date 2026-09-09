//! Reader and writer for the PLY (Polygon File Format) container.
//!
//! PLY is the lingua franca of scanned and reconstructed geometry: point
//! clouds with colour and normals, Gaussian-splat parameter sets, and
//! polygon meshes from Blender or MeshLab all travel as PLY. The container
//! is a text header naming elements (`vertex`, `face`, ...) and their typed
//! properties, followed by the rows in ASCII or binary of either byte order.
//!
//! [`read_ply`] loads the whole container into typed columns without
//! interpreting them; [`read_header`] stops after the header, for sniffing.
//! [`write_ply`] writes a container back, always binary little-endian.
//! [`convert`] turns the conventional `vertex` and `face` elements into
//! positions, named per-vertex fields and triangle indices — the part the
//! volumetric importers use. Pure Rust with no dependencies beyond the ABI
//! crate, so the same code serves the wasm operators and the native hosts.

use volumetric_abi::fea::{COLOR_FIELD_NAME, NORMAL_FIELD_NAME};

/// The scalar types PLY defines, with both spellings accepted on read
/// (`char`/`int8`, `float`/`float32`, ...) and the short spelling written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::I8 => "char",
            Self::U8 => "uchar",
            Self::I16 => "short",
            Self::U16 => "ushort",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub fn is_integer(self) -> bool {
        !matches!(self, Self::F32 | Self::F64)
    }

    /// The largest value an integer type holds, the divisor that
    /// normalises an integer colour channel to `[0, 1]`; `None` for the
    /// float types.
    pub fn max_value(self) -> Option<f64> {
        Some(match self {
            Self::I8 => f64::from(i8::MAX),
            Self::U8 => f64::from(u8::MAX),
            Self::I16 => f64::from(i16::MAX),
            Self::U16 => f64::from(u16::MAX),
            Self::I32 => f64::from(i32::MAX),
            Self::U32 => f64::from(u32::MAX),
            Self::F32 | Self::F64 => return None,
        })
    }

    /// The representable range of an integer type, for checked writes.
    fn integer_range(self) -> Option<(f64, f64)> {
        Some(match self {
            Self::I8 => (f64::from(i8::MIN), f64::from(i8::MAX)),
            Self::U8 => (0.0, f64::from(u8::MAX)),
            Self::I16 => (f64::from(i16::MIN), f64::from(i16::MAX)),
            Self::U16 => (0.0, f64::from(u16::MAX)),
            Self::I32 => (f64::from(i32::MIN), f64::from(i32::MAX)),
            Self::U32 => (0.0, f64::from(u32::MAX)),
            Self::F32 | Self::F64 => return None,
        })
    }
}

/// A property is one scalar per row, or a variable-length list per row
/// (a face's vertex indices) with the count stored in `count`'s type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyKind {
    Scalar(ScalarType),
    List { count: ScalarType, item: ScalarType },
}

/// A property's column, decoded to `f64` whatever the declared type (every
/// PLY scalar type is exactly representable, u32 included).
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyData {
    /// One value per row.
    Scalar(Vec<f64>),
    /// Row `r` holds `items[offsets[r]..offsets[r + 1]]`
    /// (`offsets.len() == rows + 1`).
    List {
        offsets: Vec<usize>,
        items: Vec<f64>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    pub name: String,
    pub kind: PropertyKind,
    pub data: PropertyData,
}

/// One element table: `count` rows of the declared properties.
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    pub name: String,
    pub count: usize,
    pub properties: Vec<Property>,
}

impl Element {
    pub fn property(&self, name: &str) -> Option<&Property> {
        self.properties.iter().find(|p| p.name == name)
    }

    /// A scalar property's column, `None` when absent or a list.
    pub fn scalar(&self, name: &str) -> Option<&[f64]> {
        match &self.property(name)?.data {
            PropertyData::Scalar(values) => Some(values),
            PropertyData::List { .. } => None,
        }
    }

    /// A scalar property's declared type, `None` when absent or a list.
    pub fn scalar_type(&self, name: &str) -> Option<ScalarType> {
        match self.property(name)?.kind {
            PropertyKind::Scalar(ty) => Some(ty),
            PropertyKind::List { .. } => None,
        }
    }

    /// A list property's `(offsets, items)`, `None` when absent or scalar.
    pub fn list(&self, name: &str) -> Option<(&[usize], &[f64])> {
        match &self.property(name)?.data {
            PropertyData::List { offsets, items } => Some((offsets, items)),
            PropertyData::Scalar(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

impl Format {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "ascii" => Self::Ascii,
            "binary_little_endian" => Self::BinaryLittleEndian,
            "binary_big_endian" => Self::BinaryBigEndian,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Ascii => "ascii",
            Self::BinaryLittleEndian => "binary_little_endian",
            Self::BinaryBigEndian => "binary_big_endian",
        }
    }
}

/// An element as declared in the header, before any rows are read.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementDecl {
    pub name: String,
    pub count: usize,
    pub properties: Vec<(String, PropertyKind)>,
}

/// The parsed header: everything known about a file before its body.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub format: Format,
    pub comments: Vec<String>,
    pub obj_info: Vec<String>,
    pub elements: Vec<ElementDecl>,
}

impl Header {
    pub fn element(&self, name: &str) -> Option<&ElementDecl> {
        self.elements.iter().find(|e| e.name == name)
    }
}

/// A whole container: header fields plus every element's rows.
#[derive(Clone, Debug, PartialEq)]
pub struct PlyFile {
    pub format: Format,
    pub comments: Vec<String>,
    pub obj_info: Vec<String>,
    pub elements: Vec<Element>,
}

impl PlyFile {
    pub fn element(&self, name: &str) -> Option<&Element> {
        self.elements.iter().find(|e| e.name == name)
    }
}

/// True when the bytes start like a PLY file (the `ply` magic line).
pub fn is_ply(bytes: &[u8]) -> bool {
    bytes.starts_with(b"ply\n") || bytes.starts_with(b"ply\r\n")
}

/// Parses the header, returning it with the byte offset where the body
/// starts (the byte after the newline that ends `end_header`).
pub fn read_header(bytes: &[u8]) -> Result<(Header, usize), String> {
    if !is_ply(bytes) {
        return Err("not a PLY file (missing the `ply` magic line)".to_string());
    }
    let mut format = None;
    let mut comments = Vec::new();
    let mut obj_info = Vec::new();
    let mut elements: Vec<ElementDecl> = Vec::new();
    let mut pos = 0usize;
    let mut line_no = 0usize;
    loop {
        let rest = bytes.get(pos..).unwrap_or(&[]);
        let Some(nl) = rest.iter().position(|&b| b == b'\n') else {
            return Err("header has no `end_header` line".to_string());
        };
        let raw = &rest[..nl];
        pos += nl + 1;
        line_no += 1;
        let line = std::str::from_utf8(raw)
            .map_err(|_| format!("header line {line_no} is not UTF-8"))?
            .trim_end_matches('\r');
        let mut words = line.split_whitespace();
        let Some(keyword) = words.next() else {
            continue; // blank line: tolerated
        };
        match keyword {
            "ply" if line_no == 1 => {}
            "format" => {
                let name = words
                    .next()
                    .ok_or_else(|| format!("header line {line_no}: `format` without a name"))?;
                let version = words.next().unwrap_or("1.0");
                if version != "1.0" {
                    return Err(format!("unsupported PLY version {version}"));
                }
                format = Some(
                    Format::from_name(name)
                        .ok_or_else(|| format!("unsupported PLY format {name:?}"))?,
                );
            }
            "comment" => comments.push(after_keyword(line, keyword).to_string()),
            "obj_info" => obj_info.push(after_keyword(line, keyword).to_string()),
            "element" => {
                let name = words
                    .next()
                    .ok_or_else(|| format!("header line {line_no}: `element` without a name"))?;
                let count = words
                    .next()
                    .and_then(|c| c.parse::<usize>().ok())
                    .ok_or_else(|| {
                        format!("header line {line_no}: element {name:?} has no count")
                    })?;
                elements.push(ElementDecl {
                    name: name.to_string(),
                    count,
                    properties: Vec::new(),
                });
            }
            "property" => {
                let element = elements.last_mut().ok_or_else(|| {
                    format!("header line {line_no}: property declared before any element")
                })?;
                let first = words
                    .next()
                    .ok_or_else(|| format!("header line {line_no}: `property` without a type"))?;
                let (kind, name) = if first == "list" {
                    let count = scalar_type(words.next(), line_no)?;
                    let item = scalar_type(words.next(), line_no)?;
                    (PropertyKind::List { count, item }, words.next())
                } else {
                    (
                        PropertyKind::Scalar(scalar_type(Some(first), line_no)?),
                        words.next(),
                    )
                };
                let name =
                    name.ok_or_else(|| format!("header line {line_no}: property without a name"))?;
                if element.properties.iter().any(|(n, _)| n == name) {
                    return Err(format!(
                        "element {:?} declares property {name:?} twice",
                        element.name
                    ));
                }
                element.properties.push((name.to_string(), kind));
            }
            "end_header" => break,
            other => {
                return Err(format!(
                    "header line {line_no}: unexpected keyword {other:?}"
                ));
            }
        }
    }
    let format = format.ok_or_else(|| "header has no `format` line".to_string())?;
    Ok((
        Header {
            format,
            comments,
            obj_info,
            elements,
        },
        pos,
    ))
}

fn after_keyword<'a>(line: &'a str, keyword: &str) -> &'a str {
    line[keyword.len()..].trim_start()
}

fn scalar_type(word: Option<&str>, line_no: usize) -> Result<ScalarType, String> {
    let word = word.ok_or_else(|| format!("header line {line_no}: missing a property type"))?;
    ScalarType::from_name(word)
        .ok_or_else(|| format!("header line {line_no}: unknown property type {word:?}"))
}

/// Reads a whole PLY file into typed columns.
pub fn read_ply(bytes: &[u8]) -> Result<PlyFile, String> {
    let (header, body_start) = read_header(bytes)?;
    let body = &bytes[body_start..];
    let mut reader: Box<dyn ValueReader> = match header.format {
        Format::Ascii => Box::new(AsciiReader::new(body)),
        Format::BinaryLittleEndian => Box::new(BinaryReader {
            bytes: body,
            pos: 0,
            big_endian: false,
        }),
        Format::BinaryBigEndian => Box::new(BinaryReader {
            bytes: body,
            pos: 0,
            big_endian: true,
        }),
    };

    let mut elements = Vec::with_capacity(header.elements.len());
    for decl in &header.elements {
        let mut columns: Vec<PropertyData> = decl
            .properties
            .iter()
            .map(|(_, kind)| match kind {
                PropertyKind::Scalar(_) => PropertyData::Scalar(Vec::with_capacity(decl.count)),
                PropertyKind::List { .. } => PropertyData::List {
                    offsets: {
                        let mut offsets = Vec::with_capacity(decl.count + 1);
                        offsets.push(0);
                        offsets
                    },
                    items: Vec::new(),
                },
            })
            .collect();
        for row in 0..decl.count {
            for (slot, (name, kind)) in decl.properties.iter().enumerate() {
                let context = || format!("element {:?} row {row} property {name:?}", decl.name);
                match (kind, &mut columns[slot]) {
                    (PropertyKind::Scalar(ty), PropertyData::Scalar(values)) => {
                        values.push(
                            reader
                                .read(*ty)
                                .map_err(|e| format!("{}: {e}", context()))?,
                        );
                    }
                    (PropertyKind::List { count, item }, PropertyData::List { offsets, items }) => {
                        let n = reader
                            .read(*count)
                            .map_err(|e| format!("{}: {e}", context()))?;
                        if !(n.is_finite() && n >= 0.0 && n.fract() == 0.0) {
                            return Err(format!("{}: invalid list count {n}", context()));
                        }
                        for _ in 0..n as usize {
                            items.push(
                                reader
                                    .read(*item)
                                    .map_err(|e| format!("{}: {e}", context()))?,
                            );
                        }
                        offsets.push(items.len());
                    }
                    _ => unreachable!("column kind mirrors the declaration"),
                }
            }
        }
        elements.push(Element {
            name: decl.name.clone(),
            count: decl.count,
            properties: decl
                .properties
                .iter()
                .zip(columns)
                .map(|((name, kind), data)| Property {
                    name: name.clone(),
                    kind: *kind,
                    data,
                })
                .collect(),
        });
    }
    Ok(PlyFile {
        format: header.format,
        comments: header.comments,
        obj_info: header.obj_info,
        elements,
    })
}

trait ValueReader {
    fn read(&mut self, ty: ScalarType) -> Result<f64, String>;
}

struct AsciiReader<'a> {
    body: &'a [u8],
    pos: usize,
}

impl<'a> AsciiReader<'a> {
    fn new(body: &'a [u8]) -> Self {
        Self { body, pos: 0 }
    }
}

impl ValueReader for AsciiReader<'_> {
    fn read(&mut self, _ty: ScalarType) -> Result<f64, String> {
        while self.pos < self.body.len() && self.body[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        let start = self.pos;
        while self.pos < self.body.len() && !self.body[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err("body ends before the declared rows".to_string());
        }
        let token = std::str::from_utf8(&self.body[start..self.pos])
            .map_err(|_| "non-UTF-8 token in ASCII body".to_string())?;
        token
            .parse::<f64>()
            .map_err(|_| format!("malformed number {token:?}"))
    }
}

struct BinaryReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    big_endian: bool,
}

impl ValueReader for BinaryReader<'_> {
    fn read(&mut self, ty: ScalarType) -> Result<f64, String> {
        let n = ty.size();
        let slice = self
            .bytes
            .get(self.pos..self.pos + n)
            .ok_or_else(|| "body ends before the declared rows".to_string())?;
        self.pos += n;
        let mut buf = [0u8; 8];
        buf[..n].copy_from_slice(slice);
        macro_rules! decode {
            ($t:ty, $len:expr) => {{
                let mut b = [0u8; $len];
                b.copy_from_slice(&buf[..$len]);
                if self.big_endian {
                    <$t>::from_be_bytes(b) as f64
                } else {
                    <$t>::from_le_bytes(b) as f64
                }
            }};
        }
        Ok(match ty {
            ScalarType::I8 => decode!(i8, 1),
            ScalarType::U8 => decode!(u8, 1),
            ScalarType::I16 => decode!(i16, 2),
            ScalarType::U16 => decode!(u16, 2),
            ScalarType::I32 => decode!(i32, 4),
            ScalarType::U32 => decode!(u32, 4),
            ScalarType::F32 => decode!(f32, 4),
            ScalarType::F64 => decode!(f64, 8),
        })
    }
}

/// Serialises a container as binary little-endian PLY. Column lengths must
/// match their element's row count, and integer-typed values must be whole
/// numbers within the type's range.
pub fn write_ply(file: &PlyFile) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(b"ply\n");
    out.extend_from_slice(format!("format {} 1.0\n", Format::BinaryLittleEndian.name()).as_bytes());
    for comment in &file.comments {
        out.extend_from_slice(format!("comment {comment}\n").as_bytes());
    }
    for info in &file.obj_info {
        out.extend_from_slice(format!("obj_info {info}\n").as_bytes());
    }
    for element in &file.elements {
        out.extend_from_slice(format!("element {} {}\n", element.name, element.count).as_bytes());
        for property in &element.properties {
            let line = match property.kind {
                PropertyKind::Scalar(ty) => format!("property {} {}\n", ty.name(), property.name),
                PropertyKind::List { count, item } => format!(
                    "property list {} {} {}\n",
                    count.name(),
                    item.name(),
                    property.name
                ),
            };
            out.extend_from_slice(line.as_bytes());
            let rows = match &property.data {
                PropertyData::Scalar(values) => values.len(),
                PropertyData::List { offsets, .. } => offsets.len().saturating_sub(1),
            };
            if rows != element.count {
                return Err(format!(
                    "element {:?} property {:?} has {rows} rows, expected {}",
                    element.name, property.name, element.count
                ));
            }
        }
    }
    out.extend_from_slice(b"end_header\n");

    for element in &file.elements {
        for row in 0..element.count {
            for property in &element.properties {
                match (&property.kind, &property.data) {
                    (PropertyKind::Scalar(ty), PropertyData::Scalar(values)) => {
                        encode(&mut out, *ty, values[row])?;
                    }
                    (PropertyKind::List { count, item }, PropertyData::List { offsets, items }) => {
                        let range = offsets[row]..offsets[row + 1];
                        encode(&mut out, *count, range.len() as f64)?;
                        for &value in &items[range] {
                            encode(&mut out, *item, value)?;
                        }
                    }
                    _ => {
                        return Err(format!(
                            "element {:?} property {:?}: data shape does not match its kind",
                            element.name, property.name
                        ));
                    }
                }
            }
        }
    }
    Ok(out)
}

fn encode(out: &mut Vec<u8>, ty: ScalarType, value: f64) -> Result<(), String> {
    if let Some((lo, hi)) = ty.integer_range() {
        if !(value.is_finite() && value.fract() == 0.0 && value >= lo && value <= hi) {
            return Err(format!("value {value} does not fit PLY type {}", ty.name()));
        }
    }
    match ty {
        ScalarType::I8 => out.extend_from_slice(&(value as i8).to_le_bytes()),
        ScalarType::U8 => out.extend_from_slice(&(value as u8).to_le_bytes()),
        ScalarType::I16 => out.extend_from_slice(&(value as i16).to_le_bytes()),
        ScalarType::U16 => out.extend_from_slice(&(value as u16).to_le_bytes()),
        ScalarType::I32 => out.extend_from_slice(&(value as i32).to_le_bytes()),
        ScalarType::U32 => out.extend_from_slice(&(value as u32).to_le_bytes()),
        ScalarType::F32 => out.extend_from_slice(&(value as f32).to_le_bytes()),
        ScalarType::F64 => out.extend_from_slice(&value.to_le_bytes()),
    }
    Ok(())
}

/// Interpretation of the conventional `vertex` and `face` elements.
pub mod convert {
    use super::{Element, PlyFile, ScalarType};
    use volumetric_abi::fea::FeaField;

    /// The `vertex` element as positions plus named per-vertex fields.
    #[derive(Clone, Debug, PartialEq)]
    pub struct VertexData {
        /// xyz interleaved, one triple per vertex.
        pub positions: Vec<f64>,
        /// `normal` and `color` when the file carries them (see
        /// [`super::NORMAL_FIELD_NAME`], [`super::COLOR_FIELD_NAME`]), then
        /// one scalar field per requested extra property, in request order.
        pub fields: Vec<FeaField>,
    }

    /// The property-name triples read as a colour, in order of preference.
    const COLOR_TRIPLES: [[&str; 3]; 3] = [
        ["red", "green", "blue"],
        ["r", "g", "b"],
        ["diffuse_red", "diffuse_green", "diffuse_blue"],
    ];

    /// Reads the `vertex` element. `x`, `y`, `z` are required; `nx`/`ny`/
    /// `nz` become the normal field and a colour triple (integer-typed
    /// channels normalised by their type's maximum, float channels as
    /// stored) the colour field. `extra` names scalar properties to carry
    /// through as one-component fields under their own names; an absent
    /// or list-typed name is an error.
    pub fn vertices(file: &PlyFile, extra: &[String]) -> Result<VertexData, String> {
        let vertex = file
            .element("vertex")
            .ok_or_else(|| "the file has no `vertex` element".to_string())?;
        let axis = |name: &str| -> Result<&[f64], String> {
            vertex
                .scalar(name)
                .ok_or_else(|| format!("the vertex element has no scalar property {name:?}"))
        };
        let (x, y, z) = (axis("x")?, axis("y")?, axis("z")?);
        let mut positions = Vec::with_capacity(vertex.count * 3);
        for i in 0..vertex.count {
            positions.extend([x[i], y[i], z[i]]);
        }

        let mut fields = Vec::new();
        if let (Some(nx), Some(ny), Some(nz)) = (
            vertex.scalar("nx"),
            vertex.scalar("ny"),
            vertex.scalar("nz"),
        ) {
            fields.push(FeaField {
                name: super::NORMAL_FIELD_NAME.to_string(),
                components: 3,
                data: interleave(nx, ny, nz),
            });
        }
        if let Some(triple) = COLOR_TRIPLES
            .iter()
            .find(|names| names.iter().all(|n| vertex.scalar(n).is_some()))
        {
            let channel = |name: &str| -> Vec<f64> {
                let values = vertex.scalar(name).expect("triple was checked");
                match vertex.scalar_type(name).and_then(ScalarType::max_value) {
                    Some(max) => values.iter().map(|v| v / max).collect(),
                    None => values.to_vec(),
                }
            };
            let (r, g, b) = (channel(triple[0]), channel(triple[1]), channel(triple[2]));
            fields.push(FeaField {
                name: super::COLOR_FIELD_NAME.to_string(),
                components: 3,
                data: interleave(&r, &g, &b),
            });
        }
        for name in extra {
            if fields.iter().any(|f| &f.name == name) {
                return Err(format!(
                    "extra field {name:?} collides with a field the importer already emits"
                ));
            }
            let values = vertex.scalar(name).ok_or_else(|| {
                format!("the vertex element has no scalar property {name:?} to carry as a field")
            })?;
            fields.push(FeaField {
                name: name.clone(),
                components: 1,
                data: values.to_vec(),
            });
        }
        Ok(VertexData { positions, fields })
    }

    fn interleave(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(a.len() * 3);
        for i in 0..a.len() {
            out.extend([a[i], b[i], c[i]]);
        }
        out
    }

    /// The `face` element as triangles.
    #[derive(Clone, Debug, PartialEq)]
    pub struct FaceData {
        /// Vertex indices, 3 per triangle; polygons are fan-triangulated
        /// from their first vertex, keeping the file's winding.
        pub indices: Vec<u32>,
        /// Faces with fewer than three vertices, dropped.
        pub skipped: usize,
    }

    /// Reads the `face` element, `None` when the file has none (a point
    /// cloud). The index list is `vertex_indices` or, as some exporters
    /// spell it, `vertex_index`.
    pub fn faces(file: &PlyFile) -> Result<Option<FaceData>, String> {
        let Some(face) = file.element("face") else {
            return Ok(None);
        };
        let (offsets, items) = ["vertex_indices", "vertex_index"]
            .iter()
            .find_map(|name| face.list(name))
            .ok_or_else(|| "the face element has no `vertex_indices` list property".to_string())?;
        let vertex_count = file.element("vertex").map_or(0, |v: &Element| v.count);
        let mut indices = Vec::with_capacity(face.count * 3);
        let mut skipped = 0;
        for row in 0..face.count {
            let polygon = &items[offsets[row]..offsets[row + 1]];
            if polygon.len() < 3 {
                skipped += 1;
                continue;
            }
            let index = |value: f64| -> Result<u32, String> {
                if !(value.is_finite() && value.fract() == 0.0 && value >= 0.0) {
                    return Err(format!("face {row}: invalid vertex index {value}"));
                }
                if value >= vertex_count as f64 {
                    return Err(format!(
                        "face {row}: vertex index {value} but the file has {vertex_count} vertices"
                    ));
                }
                Ok(value as u32)
            };
            let first = index(polygon[0])?;
            for k in 1..polygon.len() - 1 {
                indices.extend([first, index(polygon[k])?, index(polygon[k + 1])?]);
            }
        }
        Ok(Some(FaceData { indices, skipped }))
    }

    /// True when the file declares a `face` element with at least one row.
    pub fn has_faces(file: &PlyFile) -> bool {
        file.element("face").is_some_and(|f| f.count > 0)
    }

    /// The importers' shared placement: optionally recentre the cloud's
    /// bounding box on the origin, then scale about it (centre → scale, as
    /// STL and 3MF Import do). `scale` must be finite and nonzero.
    pub fn place(positions: &mut [f64], center: bool, scale: f64) -> Result<(), String> {
        if !(scale.is_finite() && scale != 0.0) {
            return Err(format!("scale must be finite and nonzero, got {scale}"));
        }
        let mut offset = [0.0; 3];
        if center && !positions.is_empty() {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for p in positions.chunks_exact(3) {
                for a in 0..3 {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
            for a in 0..3 {
                offset[a] = -(lo[a] + hi[a]) / 2.0;
            }
        }
        for p in positions.chunks_exact_mut(3) {
            for a in 0..3 {
                p[a] = (p[a] + offset[a]) * scale;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::convert::{faces, has_faces, vertices};
    use super::*;

    /// Two coloured vertices and one triangle, as a container.
    fn sample() -> PlyFile {
        PlyFile {
            format: Format::BinaryLittleEndian,
            comments: vec!["made by tests".to_string()],
            obj_info: vec![],
            elements: vec![
                Element {
                    name: "vertex".to_string(),
                    count: 3,
                    properties: vec![
                        Property {
                            name: "x".to_string(),
                            kind: PropertyKind::Scalar(ScalarType::F32),
                            data: PropertyData::Scalar(vec![0.0, 1.0, 0.0]),
                        },
                        Property {
                            name: "y".to_string(),
                            kind: PropertyKind::Scalar(ScalarType::F32),
                            data: PropertyData::Scalar(vec![0.0, 0.0, 1.0]),
                        },
                        Property {
                            name: "z".to_string(),
                            kind: PropertyKind::Scalar(ScalarType::F64),
                            data: PropertyData::Scalar(vec![0.5, 0.5, 0.5]),
                        },
                        Property {
                            name: "red".to_string(),
                            kind: PropertyKind::Scalar(ScalarType::U8),
                            data: PropertyData::Scalar(vec![255.0, 0.0, 51.0]),
                        },
                        Property {
                            name: "green".to_string(),
                            kind: PropertyKind::Scalar(ScalarType::U8),
                            data: PropertyData::Scalar(vec![0.0, 255.0, 51.0]),
                        },
                        Property {
                            name: "blue".to_string(),
                            kind: PropertyKind::Scalar(ScalarType::U8),
                            data: PropertyData::Scalar(vec![0.0, 0.0, 51.0]),
                        },
                        Property {
                            name: "confidence".to_string(),
                            kind: PropertyKind::Scalar(ScalarType::I16),
                            data: PropertyData::Scalar(vec![-3.0, 7.0, 1000.0]),
                        },
                    ],
                },
                Element {
                    name: "face".to_string(),
                    count: 1,
                    properties: vec![Property {
                        name: "vertex_indices".to_string(),
                        kind: PropertyKind::List {
                            count: ScalarType::U8,
                            item: ScalarType::I32,
                        },
                        data: PropertyData::List {
                            offsets: vec![0, 3],
                            items: vec![0.0, 1.0, 2.0],
                        },
                    }],
                },
            ],
        }
    }

    #[test]
    fn binary_round_trip_preserves_everything() {
        let file = sample();
        let bytes = write_ply(&file).unwrap();
        assert!(is_ply(&bytes));
        let (header, body) = read_header(&bytes).unwrap();
        assert_eq!(header.format, Format::BinaryLittleEndian);
        assert_eq!(header.comments, vec!["made by tests".to_string()]);
        assert_eq!(header.element("vertex").unwrap().count, 3);
        assert_eq!(bytes[body - 1], b'\n');
        // 3 rows x (4 + 4 + 8 + 1 + 1 + 1 + 2) + 1 face x (1 + 3 x 4)
        assert_eq!(bytes.len() - body, 3 * 21 + 13);
        assert_eq!(read_ply(&bytes).unwrap(), file);
    }

    #[test]
    fn ascii_body_parses_lists_and_whitespace() {
        let text = "ply\r\n\
                    format ascii 1.0\r\n\
                    comment  two   spaces\r\n\
                    obj_info scanner v1\r\n\
                    element vertex 2\r\n\
                    property float x\r\n\
                    property float y\r\n\
                    property float z\r\n\
                    element face 2\r\n\
                    property list uchar int vertex_indices\r\n\
                    end_header\r\n\
                    1 2 3\n  4.5 -6 7e0\n\
                    3 0 1 1\n 2   1 0\n";
        let file = read_ply(text.as_bytes()).unwrap();
        assert_eq!(file.format, Format::Ascii);
        assert_eq!(file.comments, vec!["two   spaces".to_string()]);
        assert_eq!(file.obj_info, vec!["scanner v1".to_string()]);
        let vertex = file.element("vertex").unwrap();
        assert_eq!(vertex.scalar("x"), Some(&[1.0, 4.5][..]));
        assert_eq!(vertex.scalar("y"), Some(&[2.0, -6.0][..]));
        assert_eq!(vertex.scalar("z"), Some(&[3.0, 7.0][..]));
        let (offsets, items) = file
            .element("face")
            .unwrap()
            .list("vertex_indices")
            .unwrap();
        assert_eq!(offsets, &[0, 3, 5]);
        assert_eq!(items, &[0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn big_endian_bodies_decode() {
        let mut bytes = b"ply\nformat binary_big_endian 1.0\nelement vertex 1\n\
                          property short x\nproperty uint y\nproperty double z\nend_header\n"
            .to_vec();
        bytes.extend((-2i16).to_be_bytes());
        bytes.extend(70000u32.to_be_bytes());
        bytes.extend(2.5f64.to_be_bytes());
        let file = read_ply(&bytes).unwrap();
        let vertex = file.element("vertex").unwrap();
        assert_eq!(vertex.scalar("x"), Some(&[-2.0][..]));
        assert_eq!(vertex.scalar("y"), Some(&[70000.0][..]));
        assert_eq!(vertex.scalar("z"), Some(&[2.5][..]));
    }

    #[test]
    fn malformed_files_are_rejected_with_reasons() {
        let err = read_ply(b"solid nope").unwrap_err();
        assert!(err.contains("magic"), "{err}");

        let err =
            read_ply(b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n").unwrap_err();
        assert!(err.contains("end_header"), "{err}");

        let err = read_ply(
            b"ply\nformat ascii 1.0\nelement vertex 1\nproperty quaternion x\nend_header\n1\n",
        )
        .unwrap_err();
        assert!(err.contains("unknown property type"), "{err}");

        let err = read_ply(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nend_header\n\0\0\0\0")
            .unwrap_err();
        assert!(
            err.contains("row 1") && err.contains("ends before"),
            "{err}"
        );

        let err = read_ply(
            b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\nabc\n",
        )
        .unwrap_err();
        assert!(err.contains("malformed number"), "{err}");

        let err = read_ply(b"ply\nformat ascii 1.0\nproperty float x\nend_header\n").unwrap_err();
        assert!(err.contains("before any element"), "{err}");

        let mut file = sample();
        file.elements[0].properties[0].data = PropertyData::Scalar(vec![0.0]);
        let err = write_ply(&file).unwrap_err();
        assert!(err.contains("1 rows, expected 3"), "{err}");

        let mut file = sample();
        file.elements[0].properties[3].data = PropertyData::Scalar(vec![256.0, 0.0, 0.0]);
        let err = write_ply(&file).unwrap_err();
        assert!(err.contains("does not fit PLY type uchar"), "{err}");
    }

    #[test]
    fn vertices_convert_positions_colours_and_extras() {
        let file = sample();
        let data = vertices(&file, &["confidence".to_string()]).unwrap();
        assert_eq!(
            data.positions,
            vec![0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 0.5]
        );
        assert_eq!(data.fields.len(), 2);
        let color = &data.fields[0];
        assert_eq!(color.name, COLOR_FIELD_NAME);
        assert_eq!(color.components, 3);
        assert_eq!(color.data[0..3], [1.0, 0.0, 0.0]);
        assert!((color.data[6] - 0.2).abs() < 1e-12);
        let confidence = &data.fields[1];
        assert_eq!(confidence.name, "confidence");
        assert_eq!(confidence.components, 1);
        assert_eq!(confidence.data, vec![-3.0, 7.0, 1000.0]);

        let err = vertices(&file, &["nope".to_string()]).unwrap_err();
        assert!(err.contains("nope"), "{err}");
        let err = vertices(&file, &[COLOR_FIELD_NAME.to_string()]).unwrap_err();
        assert!(err.contains("collides"), "{err}");
    }

    #[test]
    fn float_colours_and_normals_pass_through() {
        let text = "ply\nformat ascii 1.0\nelement vertex 1\n\
                    property float x\nproperty float y\nproperty float z\n\
                    property float nx\nproperty float ny\nproperty float nz\n\
                    property float r\nproperty float g\nproperty float b\n\
                    end_header\n1 2 3 0 0 1 0.25 0.5 0.75\n";
        let file = read_ply(text.as_bytes()).unwrap();
        let data = vertices(&file, &[]).unwrap();
        assert_eq!(data.fields[0].name, NORMAL_FIELD_NAME);
        assert_eq!(data.fields[0].data, vec![0.0, 0.0, 1.0]);
        assert_eq!(data.fields[1].name, COLOR_FIELD_NAME);
        assert_eq!(data.fields[1].data, vec![0.25, 0.5, 0.75]);
        assert!(!has_faces(&file));
        assert_eq!(faces(&file).unwrap(), None);
    }

    #[test]
    fn faces_fan_triangulate_and_validate_indices() {
        let text = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
                    element face 3\nproperty list uchar int vertex_indices\nend_header\n\
                    0 0 0\n1 0 0\n1 1 0\n0 1 0\n\
                    4 0 1 2 3\n2 0 1\n3 3 2 1\n";
        let file = read_ply(text.as_bytes()).unwrap();
        assert!(has_faces(&file));
        let data = faces(&file).unwrap().unwrap();
        assert_eq!(data.indices, vec![0, 1, 2, 0, 2, 3, 3, 2, 1]);
        assert_eq!(data.skipped, 1);

        let text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n\
                    element face 1\nproperty list uchar int vertex_index\nend_header\n0 0 0\n3 0 0 9\n";
        let err = faces(&read_ply(text.as_bytes()).unwrap()).unwrap_err();
        assert!(
            err.contains("index 9") && err.contains("1 vertices"),
            "{err}"
        );
    }

    #[test]
    fn placement_centres_then_scales() {
        let mut positions = vec![1.0, 2.0, 3.0, 3.0, 6.0, 9.0];
        convert::place(&mut positions, true, 2.0).unwrap();
        assert_eq!(positions, vec![-2.0, -4.0, -6.0, 2.0, 4.0, 6.0]);
        let mut positions = vec![1.0, 2.0, 3.0];
        convert::place(&mut positions, false, 0.5).unwrap();
        assert_eq!(positions, vec![0.5, 1.0, 1.5]);
        assert!(convert::place(&mut positions, false, 0.0).is_err());
    }

    #[test]
    fn scalar_type_tables_agree() {
        for ty in [
            ScalarType::I8,
            ScalarType::U8,
            ScalarType::I16,
            ScalarType::U16,
            ScalarType::I32,
            ScalarType::U32,
            ScalarType::F32,
            ScalarType::F64,
        ] {
            assert_eq!(ScalarType::from_name(ty.name()), Some(ty));
            assert_eq!(ty.is_integer(), ty.max_value().is_some());
        }
        assert_eq!(ScalarType::from_name("float32"), Some(ScalarType::F32));
        assert_eq!(ScalarType::from_name("uint8"), Some(ScalarType::U8));
        assert_eq!(ScalarType::from_name("half"), None);
    }
}
