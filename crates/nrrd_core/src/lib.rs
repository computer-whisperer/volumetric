//! Reader and writer for NRRD ("nearly raw raster data") volumes.
//!
//! NRRD is the plain, tool-neutral container for sampled scalar fields: a
//! text header (sample type, sizes, spacing, origin, encoding) followed by
//! the samples themselves, raw or gzip-compressed, in one file. 3D Slicer,
//! ITK, teem and pynrrd all read and write it, which makes it the natural
//! way to hand a CT stack, a fused TSDF grid or any other regular grid to
//! the engine without minting a format.
//!
//! [`read_nrrd`] loads a whole file into a [`Nrrd`]: the samples as `f32`
//! with axis 0 varying fastest, plus the spacing and origin that place them
//! in the file's spatial frame. [`is_nrrd`] sniffs the magic. [`write_nrrd`]
//! writes a `float` volume back, raw, gzip or ASCII encoded, in the form
//! the same tools read (`space directions` + `space origin`).
//!
//! The reader accepts attached, all-spatial, axis-aligned volumes of any
//! scalar type in `raw`, `gzip` or `ascii` encoding and any dimension count.
//! It rejects, with a pointed error, detached data files, channel axes
//! (`kinds: list`/`vector`/...), rotated, sheared or flipped `space
//! directions`, and the `bzip2`/`hex` encodings. NaN samples pass through
//! untouched: they are how a volume marks what it never observed.

use std::io::Read;

/// A regular grid of scalar samples with its placement.
#[derive(Clone, Debug, PartialEq)]
pub struct Nrrd {
    /// Samples per axis; axis 0 varies fastest in `values`.
    pub sizes: Vec<usize>,
    /// Distance between neighbouring samples along each axis, in the
    /// file's spatial unit (NRRD carries none; callers decide).
    pub spacing: Vec<f64>,
    /// Position of sample `(0, ..., 0)`; sample `i` on an axis sits at
    /// `origin + i * spacing`.
    pub origin: Vec<f64>,
    /// The samples, axis 0 fastest. NaN marks an unobserved sample.
    pub values: Vec<f32>,
}

impl Nrrd {
    pub fn dimensions(&self) -> usize {
        self.sizes.len()
    }

    pub fn value_count(&self) -> usize {
        self.sizes.iter().product()
    }

    /// World coordinate of sample `index` along `axis`.
    pub fn position(&self, axis: usize, index: usize) -> f64 {
        self.origin[axis] + index as f64 * self.spacing[axis]
    }

    /// Checks that the geometry describes the samples: at least one axis,
    /// one sample per axis at least, finite positive spacing, finite origin,
    /// and exactly `sizes` product values.
    pub fn validate(&self) -> Result<(), String> {
        let d = self.sizes.len();
        if d == 0 {
            return Err("a volume needs at least one axis".to_string());
        }
        if self.spacing.len() != d || self.origin.len() != d {
            return Err(format!(
                "geometry mismatch: {d} axes but {} spacings and {} origin coordinates",
                self.spacing.len(),
                self.origin.len()
            ));
        }
        let mut count = 1usize;
        for axis in 0..d {
            if self.sizes[axis] == 0 {
                return Err(format!("axis {axis} has no samples"));
            }
            count = count
                .checked_mul(self.sizes[axis])
                .ok_or_else(|| "sample count overflows".to_string())?;
            if !(self.spacing[axis].is_finite() && self.spacing[axis] > 0.0) {
                return Err(format!(
                    "axis {axis} spacing must be finite and positive, got {}",
                    self.spacing[axis]
                ));
            }
            if !self.origin[axis].is_finite() {
                return Err(format!(
                    "axis {axis} origin must be finite, got {}",
                    self.origin[axis]
                ));
            }
        }
        if self.values.len() != count {
            return Err(format!(
                "sizes {:?} call for {count} samples but the volume holds {}",
                self.sizes,
                self.values.len()
            ));
        }
        Ok(())
    }
}

/// How the samples are stored after the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Raw,
    Gzip,
    Ascii,
}

/// True when the bytes start with the NRRD magic line.
pub fn is_nrrd(bytes: &[u8]) -> bool {
    bytes.starts_with(b"NRRD000")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Scalar {
    fn from_name(name: &str) -> Result<Self, String> {
        let normalised: String = name
            .to_ascii_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        Ok(match normalised.as_str() {
            "signed char" | "int8" | "int8_t" => Self::I8,
            "uchar" | "unsigned char" | "uint8" | "uint8_t" => Self::U8,
            "short" | "short int" | "signed short" | "signed short int" | "int16" | "int16_t" => {
                Self::I16
            }
            "ushort" | "unsigned short" | "unsigned short int" | "uint16" | "uint16_t" => Self::U16,
            "int" | "signed int" | "int32" | "int32_t" => Self::I32,
            "uint" | "unsigned int" | "uint32" | "uint32_t" => Self::U32,
            "longlong"
            | "long long"
            | "long long int"
            | "signed long long"
            | "signed long long int"
            | "int64"
            | "int64_t" => Self::I64,
            "ulonglong"
            | "unsigned long long"
            | "unsigned long long int"
            | "uint64"
            | "uint64_t" => Self::U64,
            "float" => Self::F32,
            "double" => Self::F64,
            "block" => return Err("`block` samples have no numeric value".to_string()),
            _ => return Err(format!("unknown sample type `{name}`")),
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    fn decode(self, bytes: &[u8], big_endian: bool) -> f32 {
        macro_rules! read {
            ($t:ty) => {{
                let array = bytes.try_into().unwrap();
                if big_endian {
                    <$t>::from_be_bytes(array)
                } else {
                    <$t>::from_le_bytes(array)
                }
            }};
        }
        match self {
            Self::I8 => read!(i8) as f32,
            Self::U8 => read!(u8) as f32,
            Self::I16 => read!(i16) as f32,
            Self::U16 => read!(u16) as f32,
            Self::I32 => read!(i32) as f32,
            Self::U32 => read!(u32) as f32,
            Self::I64 => read!(i64) as f32,
            Self::U64 => read!(u64) as f32,
            Self::F32 => read!(f32),
            Self::F64 => read!(f64) as f32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Centering {
    Node,
    Cell,
}

/// The header fields the reader acts on, as written.
#[derive(Default)]
struct Header {
    scalar: Option<Scalar>,
    dimension: Option<usize>,
    sizes: Option<Vec<usize>>,
    encoding: Option<Encoding>,
    big_endian: bool,
    spacings: Option<Vec<f64>>,
    directions: Option<Vec<Option<Vec<f64>>>>,
    space_origin: Option<Vec<f64>>,
    space_dimension: Option<usize>,
    axis_mins: Option<Vec<f64>>,
    axis_maxs: Option<Vec<f64>>,
    centers: Option<Vec<Option<Centering>>>,
    kinds: Option<Vec<String>>,
    line_skip: usize,
    byte_skip: i64,
}

fn parse_list<T: std::str::FromStr>(value: &str, what: &str) -> Result<Vec<T>, String> {
    value
        .split_whitespace()
        .map(|token| {
            token
                .parse::<T>()
                .map_err(|_| format!("{what}: cannot parse `{token}`"))
        })
        .collect()
}

/// Parses `(a,b,c) (d,e,f) none ...`: one entry per axis, `none` for a
/// non-spatial axis.
fn parse_vectors(value: &str) -> Result<Vec<Option<Vec<f64>>>, String> {
    let mut out = Vec::new();
    let mut rest = value.trim();
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('(') {
            let close = after
                .find(')')
                .ok_or_else(|| format!("unterminated vector in `{value}`"))?;
            let inner = &after[..close];
            let vector = inner
                .split(',')
                .map(|token| {
                    token
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| format!("cannot parse `{}` in `{value}`", token.trim()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            out.push(Some(vector));
            rest = after[close + 1..].trim_start();
        } else {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            let token = &rest[..end];
            if token.eq_ignore_ascii_case("none") {
                out.push(None);
            } else {
                return Err(format!(
                    "expected a `(...)` vector or `none`, got `{token}`"
                ));
            }
            rest = rest[end..].trim_start();
        }
    }
    Ok(out)
}

/// Parses the header lines, returning the fields and the byte offset of
/// the first data byte (just after the blank line that ends the header).
fn read_header(bytes: &[u8]) -> Result<(Header, usize), String> {
    if !is_nrrd(bytes) {
        return Err("not an NRRD file (missing the `NRRD000x` magic line)".to_string());
    }
    let mut header = Header::default();
    let mut offset = 0usize;
    let mut first = true;
    loop {
        let rest = &bytes[offset..];
        let newline = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| "header never ends (no blank line before the data)".to_string())?;
        let line = std::str::from_utf8(&rest[..newline])
            .map_err(|_| "header is not UTF-8".to_string())?
            .trim_end_matches('\r');
        offset += newline + 1;
        if first {
            first = false;
            let version = line.strip_prefix("NRRD000").unwrap_or("");
            if !matches!(version, "1" | "2" | "3" | "4" | "5") {
                return Err(format!("unsupported NRRD magic `{line}`"));
            }
            continue;
        }
        if line.is_empty() {
            return Ok((header, offset));
        }
        if line.starts_with('#') || line.contains(":=") {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(format!("malformed header line `{line}`"));
        };
        let key: String = name
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '_')
            .collect::<String>()
            .to_ascii_lowercase();
        let value = value.trim();
        match key.as_str() {
            "type" => header.scalar = Some(Scalar::from_name(value)?),
            "dimension" => {
                header.dimension = Some(
                    value
                        .parse()
                        .map_err(|_| format!("dimension: cannot parse `{value}`"))?,
                )
            }
            "sizes" => header.sizes = Some(parse_list(value, "sizes")?),
            "encoding" => {
                header.encoding = Some(match value.to_ascii_lowercase().as_str() {
                    "raw" => Encoding::Raw,
                    "gzip" | "gz" => Encoding::Gzip,
                    "ascii" | "txt" | "text" => Encoding::Ascii,
                    other => {
                        return Err(format!(
                            "encoding `{other}` is not supported (raw, gzip and ascii are)"
                        ));
                    }
                })
            }
            "endian" => {
                header.big_endian = match value.to_ascii_lowercase().as_str() {
                    "little" => false,
                    "big" => true,
                    other => return Err(format!("unknown endian `{other}`")),
                }
            }
            "spacings" => header.spacings = Some(parse_list(value, "spacings")?),
            "spacedirections" => header.directions = Some(parse_vectors(value)?),
            "spaceorigin" => {
                let mut vectors = parse_vectors(value)?;
                match (vectors.len(), vectors.pop().flatten()) {
                    (1, Some(origin)) => header.space_origin = Some(origin),
                    _ => return Err(format!("space origin: expected one vector, got `{value}`")),
                }
            }
            "spacedimension" => {
                header.space_dimension = Some(
                    value
                        .parse()
                        .map_err(|_| format!("space dimension: cannot parse `{value}`"))?,
                )
            }
            "axismins" => header.axis_mins = Some(parse_list(value, "axis mins")?),
            "axismaxs" => header.axis_maxs = Some(parse_list(value, "axis maxs")?),
            "centers" | "centerings" => {
                header.centers = Some(
                    value
                        .split_whitespace()
                        .map(|token| match token.to_ascii_lowercase().as_str() {
                            "node" => Ok(Some(Centering::Node)),
                            "cell" => Ok(Some(Centering::Cell)),
                            "none" | "???" => Ok(None),
                            other => Err(format!("unknown centering `{other}`")),
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            "kinds" => header.kinds = Some(value.split_whitespace().map(str::to_string).collect()),
            "datafile" => {
                return Err(
                    "the samples live in a separate data file; only attached data is supported"
                        .to_string(),
                );
            }
            "lineskip" => {
                header.line_skip = value
                    .parse()
                    .map_err(|_| format!("line skip: cannot parse `{value}`"))?
            }
            "byteskip" => {
                header.byte_skip = value
                    .parse()
                    .map_err(|_| format!("byte skip: cannot parse `{value}`"))?
            }
            _ => {}
        }
    }
}

/// Resolves spacing and origin per axis from whichever placement fields
/// the header carries: `space directions` (+ `space origin`), else
/// `spacings`, else `axis mins`/`axis maxs` (+ centering), else unit
/// spacing at the origin.
fn geometry(header: &Header, d: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if let Some(kinds) = &header.kinds {
        for (axis, kind) in kinds.iter().enumerate() {
            let lower = kind.to_ascii_lowercase();
            if !matches!(lower.as_str(), "domain" | "space" | "time" | "???") {
                return Err(format!(
                    "axis {axis} is a `{kind}` axis; only spatial axes are supported \
                     (write one scalar volume per channel)"
                ));
            }
        }
    }
    if let Some(space_dimension) = header.space_dimension {
        if space_dimension != d {
            return Err(format!(
                "the space has {space_dimension} dimensions but the grid has {d} axes; \
                 every axis must be spatial"
            ));
        }
    }

    let mut spacing = vec![1.0; d];
    let mut origin = vec![0.0; d];
    if let Some(directions) = &header.directions {
        if directions.len() != d {
            return Err(format!(
                "space directions lists {} axes, the grid has {d}",
                directions.len()
            ));
        }
        for (axis, direction) in directions.iter().enumerate() {
            let Some(direction) = direction else {
                return Err(format!(
                    "axis {axis} has no space direction, so it is not spatial; \
                     only spatial axes are supported"
                ));
            };
            if direction.len() != d {
                return Err(format!(
                    "axis {axis} direction has {} components in a {d}-axis grid; \
                     every axis must be spatial",
                    direction.len()
                ));
            }
            for (component, &value) in direction.iter().enumerate() {
                if component != axis && value != 0.0 {
                    return Err(format!(
                        "axis {axis} direction {direction:?} is not along the axis; rotated \
                         and sheared volumes are not supported (resample axis-aligned first)"
                    ));
                }
            }
            let along = direction[axis];
            if !(along.is_finite() && along > 0.0) {
                return Err(format!(
                    "axis {axis} direction {direction:?} must point along +axis with a \
                     positive, finite spacing (flipped axes are not supported)"
                ));
            }
            spacing[axis] = along;
        }
        if let Some(space_origin) = &header.space_origin {
            if space_origin.len() != d {
                return Err(format!(
                    "space origin has {} coordinates for a {d}-axis grid",
                    space_origin.len()
                ));
            }
            origin.copy_from_slice(space_origin);
        }
        return Ok((spacing, origin));
    }
    if let Some(spacings) = &header.spacings {
        if spacings.len() != d {
            return Err(format!(
                "spacings lists {} axes, the grid has {d}",
                spacings.len()
            ));
        }
        for (axis, &value) in spacings.iter().enumerate() {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!(
                    "axis {axis} spacing must be finite and positive, got {value}"
                ));
            }
        }
        spacing.copy_from_slice(spacings);
        if let Some(space_origin) = &header.space_origin {
            if space_origin.len() != d {
                return Err(format!(
                    "space origin has {} coordinates for a {d}-axis grid",
                    space_origin.len()
                ));
            }
            origin.copy_from_slice(space_origin);
        } else if let Some(mins) = &header.axis_mins {
            if mins.len() != d {
                return Err(format!(
                    "axis mins lists {} axes, the grid has {d}",
                    mins.len()
                ));
            }
            origin.copy_from_slice(mins);
        }
        return Ok((spacing, origin));
    }
    if let (Some(mins), Some(maxs)) = (&header.axis_mins, &header.axis_maxs) {
        if mins.len() != d || maxs.len() != d {
            return Err(format!(
                "axis mins/maxs list {}/{} axes, the grid has {d}",
                mins.len(),
                maxs.len()
            ));
        }
        let sizes = header
            .sizes
            .as_ref()
            .expect("sizes checked before geometry");
        for axis in 0..d {
            let centering = header
                .centers
                .as_ref()
                .and_then(|centers| centers.get(axis).copied().flatten())
                .unwrap_or(Centering::Node);
            let extent = maxs[axis] - mins[axis];
            if !(extent.is_finite() && extent > 0.0) {
                return Err(format!(
                    "axis {axis} mins/maxs [{}, {}] do not span a positive extent",
                    mins[axis], maxs[axis]
                ));
            }
            match centering {
                Centering::Node => {
                    if sizes[axis] < 2 {
                        return Err(format!(
                            "axis {axis} is node-centred with a single sample, so its \
                             spacing is undefined"
                        ));
                    }
                    spacing[axis] = extent / (sizes[axis] - 1) as f64;
                    origin[axis] = mins[axis];
                }
                Centering::Cell => {
                    spacing[axis] = extent / sizes[axis] as f64;
                    origin[axis] = mins[axis] + spacing[axis] / 2.0;
                }
            }
        }
        return Ok((spacing, origin));
    }
    if let Some(space_origin) = &header.space_origin {
        if space_origin.len() != d {
            return Err(format!(
                "space origin has {} coordinates for a {d}-axis grid",
                space_origin.len()
            ));
        }
        origin.copy_from_slice(space_origin);
    }
    Ok((spacing, origin))
}

/// Reads a whole NRRD file. See the crate docs for what is accepted.
pub fn read_nrrd(bytes: &[u8]) -> Result<Nrrd, String> {
    let (header, data_offset) = read_header(bytes)?;
    let d = header
        .dimension
        .ok_or_else(|| "header lacks `dimension`".to_string())?;
    if d == 0 {
        return Err("dimension must be at least 1".to_string());
    }
    let sizes = header
        .sizes
        .clone()
        .ok_or_else(|| "header lacks `sizes`".to_string())?;
    if sizes.len() != d {
        return Err(format!(
            "sizes lists {} axes but dimension is {d}",
            sizes.len()
        ));
    }
    let mut count = 1usize;
    for (axis, &size) in sizes.iter().enumerate() {
        if size == 0 {
            return Err(format!("axis {axis} has no samples"));
        }
        count = count
            .checked_mul(size)
            .ok_or_else(|| "sample count overflows".to_string())?;
    }
    let scalar = header
        .scalar
        .ok_or_else(|| "header lacks `type`".to_string())?;
    let encoding = header
        .encoding
        .ok_or_else(|| "header lacks `encoding`".to_string())?;
    let (spacing, origin) = geometry(&header, d)?;

    // Skip lines, then bytes, to reach the samples.
    let mut body = &bytes[data_offset..];
    for _ in 0..header.line_skip {
        let newline = body
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| "line skip runs past the end of the file".to_string())?;
        body = &body[newline + 1..];
    }
    let expected = count
        .checked_mul(scalar.size())
        .ok_or_else(|| "sample byte count overflows".to_string())?;
    if header.byte_skip < 0 {
        if header.byte_skip != -1 || encoding != Encoding::Raw {
            return Err(
                "byte skip -1 (samples at the end of the file) is only meaningful for raw \
                 encoding"
                    .to_string(),
            );
        }
        if body.len() < expected {
            return Err(format!(
                "file holds {} data bytes but {expected} are needed for {count} samples",
                body.len()
            ));
        }
        body = &body[body.len() - expected..];
    } else {
        let skip = header.byte_skip as usize;
        if body.len() < skip {
            return Err("byte skip runs past the end of the file".to_string());
        }
        body = &body[skip..];
    }

    let values = match encoding {
        Encoding::Ascii => {
            let text =
                std::str::from_utf8(body).map_err(|_| "ascii samples are not UTF-8".to_string())?;
            let mut values = Vec::with_capacity(count);
            for token in text.split_ascii_whitespace() {
                if values.len() == count {
                    break;
                }
                let value: f64 = token
                    .parse()
                    .map_err(|_| format!("ascii sample `{token}` is not a number"))?;
                values.push(value as f32);
            }
            if values.len() != count {
                return Err(format!(
                    "ascii data holds {} samples, sizes call for {count}",
                    values.len()
                ));
            }
            values
        }
        Encoding::Raw | Encoding::Gzip => {
            let inflated;
            let raw: &[u8] = if encoding == Encoding::Gzip {
                let mut out = Vec::with_capacity(expected);
                flate2::read::MultiGzDecoder::new(body)
                    .take(expected as u64 + 1)
                    .read_to_end(&mut out)
                    .map_err(|error| format!("gzip data is corrupt: {error}"))?;
                inflated = out;
                &inflated
            } else {
                body
            };
            if raw.len() < expected {
                return Err(format!(
                    "data holds {} bytes but {expected} are needed for {count} samples of {} \
                     bytes",
                    raw.len(),
                    scalar.size()
                ));
            }
            raw[..expected]
                .chunks_exact(scalar.size())
                .map(|chunk| scalar.decode(chunk, header.big_endian))
                .collect()
        }
    };
    let nrrd = Nrrd {
        sizes,
        spacing,
        origin,
        values,
    };
    nrrd.validate()?;
    Ok(nrrd)
}

/// Writes a `float` volume with `space directions`/`space origin`
/// placement, little-endian, in the requested encoding.
pub fn write_nrrd(nrrd: &Nrrd, encoding: Encoding) -> Result<Vec<u8>, String> {
    nrrd.validate()?;
    let d = nrrd.dimensions();
    let mut out = String::from("NRRD0004\n# written by volumetric nrrd_core\ntype: float\n");
    out.push_str(&format!("dimension: {d}\n"));
    out.push_str(&format!(
        "sizes: {}\n",
        nrrd.sizes
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    ));
    out.push_str(&format!("space dimension: {d}\n"));
    let directions: Vec<String> = (0..d)
        .map(|axis| {
            let components: Vec<String> = (0..d)
                .map(|component| {
                    if component == axis {
                        nrrd.spacing[axis].to_string()
                    } else {
                        "0".to_string()
                    }
                })
                .collect();
            format!("({})", components.join(","))
        })
        .collect();
    out.push_str(&format!("space directions: {}\n", directions.join(" ")));
    out.push_str(&format!(
        "space origin: ({})\n",
        nrrd.origin
            .iter()
            .map(f64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    ));
    out.push_str(&format!("kinds: {}\n", vec!["domain"; d].join(" ")));
    out.push_str("endian: little\n");
    out.push_str(match encoding {
        Encoding::Raw => "encoding: raw\n",
        Encoding::Gzip => "encoding: gzip\n",
        Encoding::Ascii => "encoding: ascii\n",
    });
    out.push('\n');
    let mut bytes = out.into_bytes();
    match encoding {
        Encoding::Ascii => {
            let row = nrrd.sizes[0];
            for (index, value) in nrrd.values.iter().enumerate() {
                if index > 0 {
                    bytes.push(if index % row == 0 { b'\n' } else { b' ' });
                }
                bytes.extend(value.to_string().as_bytes());
            }
            bytes.push(b'\n');
        }
        Encoding::Raw => {
            bytes.reserve(nrrd.values.len() * 4);
            for value in &nrrd.values {
                bytes.extend(value.to_le_bytes());
            }
        }
        Encoding::Gzip => {
            let mut raw = Vec::with_capacity(nrrd.values.len() * 4);
            for value in &nrrd.values {
                raw.extend(value.to_le_bytes());
            }
            let mut encoder =
                flate2::write::GzEncoder::new(&mut bytes, flate2::Compression::default());
            std::io::Write::write_all(&mut encoder, &raw)
                .and_then(|_| encoder.finish().map(|_| ()))
                .map_err(|error| format!("gzip encoding failed: {error}"))?;
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3 x 2 x 2 volume with a NaN hole, value = x + 10y + 100z.
    fn sample() -> Nrrd {
        let mut values = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..3 {
                    values.push((x + 10 * y + 100 * z) as f32);
                }
            }
        }
        values[4] = f32::NAN;
        Nrrd {
            sizes: vec![3, 2, 2],
            spacing: vec![0.5, 1.0, 2.0],
            origin: vec![-1.0, 0.0, 3.0],
            values,
        }
    }

    fn assert_same(a: &Nrrd, b: &Nrrd) {
        assert_eq!(a.sizes, b.sizes);
        assert_eq!(a.spacing, b.spacing);
        assert_eq!(a.origin, b.origin);
        assert_eq!(a.values.len(), b.values.len());
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!(x == y || (x.is_nan() && y.is_nan()), "{x} vs {y}");
        }
    }

    #[test]
    fn round_trips_in_every_encoding() {
        let volume = sample();
        for encoding in [Encoding::Raw, Encoding::Gzip, Encoding::Ascii] {
            let bytes = write_nrrd(&volume, encoding).unwrap();
            assert!(is_nrrd(&bytes));
            let back = read_nrrd(&bytes).unwrap_or_else(|e| panic!("{encoding:?}: {e}"));
            assert_same(&volume, &back);
        }
        let gz = write_nrrd(&volume, Encoding::Gzip).unwrap();
        let text = String::from_utf8_lossy(&gz[..gz.iter().position(|&b| b == 0x1f).unwrap()]);
        assert!(text.contains("space directions: (0.5,0,0) (0,1,0) (0,0,2)"));
        assert!(text.contains("space origin: (-1,0,3)"));
        assert_eq!(volume.position(2, 1), 5.0);
    }

    #[test]
    fn reads_big_endian_shorts_with_spacings_and_axis_mins() {
        let mut bytes = b"NRRD0001\ntype: short\ndimension: 2\nsizes: 2 2\n\
            spacings: 0.25 0.5\naxis mins: 10 20\nendian: big\nencoding: raw\n\n"
            .to_vec();
        for value in [-1i16, 2, 300, -400] {
            bytes.extend(value.to_be_bytes());
        }
        let volume = read_nrrd(&bytes).unwrap();
        assert_eq!(volume.sizes, vec![2, 2]);
        assert_eq!(volume.spacing, vec![0.25, 0.5]);
        assert_eq!(volume.origin, vec![10.0, 20.0]);
        assert_eq!(volume.values, vec![-1.0, 2.0, 300.0, -400.0]);
    }

    #[test]
    fn axis_mins_maxs_place_node_and_cell_centred_axes() {
        let bytes = b"NRRD0002\ntype: uchar\ndimension: 2\nsizes: 3 4\n\
            axis mins: 0 0\naxis maxs: 1 2\ncenters: node cell\nencoding: ascii\n\n\
            1 2 3\n4 5 6\n7 8 9\n10 11 12\n";
        let volume = read_nrrd(bytes).unwrap();
        assert_eq!(volume.spacing, vec![0.5, 0.5]);
        assert_eq!(volume.origin, vec![0.0, 0.25]);
        assert_eq!(volume.values[11], 12.0);
    }

    #[test]
    fn header_is_lenient_about_case_spacing_comments_and_key_values() {
        let bytes = b"NRRD0005\r\n# a comment\r\nTYPE: Float\r\nDimension: 1\r\nSizes: 3\r\n\
            Space Directions: (2, 0)\r\nspace_origin: (1,0)\r\nauthor:=someone\r\n\
            Encoding: TXT\r\n\r\nnan 1 -inf\r\n";
        // A 1-axis grid in a 2-D space is not all-spatial: rejected.
        let err = read_nrrd(bytes).unwrap_err();
        assert!(err.contains("every axis must be spatial"), "{err}");
        let fixed = String::from_utf8_lossy(bytes)
            .replace("(2, 0)", "(2)")
            .replace("(1,0)", "(1)");
        let volume = read_nrrd(fixed.as_bytes()).unwrap();
        assert_eq!(volume.spacing, vec![2.0]);
        assert_eq!(volume.origin, vec![1.0]);
        assert!(volume.values[0].is_nan());
        assert_eq!(volume.values[1], 1.0);
        assert_eq!(volume.values[2], f32::NEG_INFINITY);
    }

    #[test]
    fn line_and_byte_skips_and_trailing_data_are_honoured() {
        let mut bytes = b"NRRD0004\ntype: uint8\ndimension: 1\nsizes: 2\nline skip: 1\n\
            byte skip: 2\nencoding: raw\n\nskipped line\nXX"
            .to_vec();
        bytes.extend([7u8, 9]);
        assert_eq!(read_nrrd(&bytes).unwrap().values, vec![7.0, 9.0]);

        let mut tail = b"NRRD0004\ntype: uint8\ndimension: 1\nsizes: 2\nbyte skip: -1\n\
            encoding: raw\n\njunkjunk"
            .to_vec();
        tail.extend([1u8, 2]);
        assert_eq!(read_nrrd(&tail).unwrap().values, vec![1.0, 2.0]);
    }

    #[test]
    fn unsupported_layouts_are_named() {
        let err = read_nrrd(b"ply\n").unwrap_err();
        assert!(err.contains("magic"), "{err}");
        let cases: [(&str, &str); 6] = [
            ("data file: other.raw\n", "separate data file"),
            ("kinds: list domain domain\n", "`list` axis"),
            (
                "space directions: (0,1,0) (1,0,0) (0,0,1)\n",
                "not along the axis",
            ),
            (
                "space directions: (-1,0,0) (0,1,0) (0,0,1)\n",
                "flipped axes",
            ),
            ("encoding: bzip2\n", "not supported"),
            ("sizes: 3 2\n", "sizes lists 2 axes"),
        ];
        for (line, expected) in cases {
            let mut text = format!(
                "NRRD0004\ntype: uchar\ndimension: 3\nsizes: 3 2 2\nencoding: raw\n{line}\n"
            );
            if line.starts_with("encoding") || line.starts_with("sizes") {
                text = text.replacen(
                    if line.starts_with("encoding") {
                        "encoding: raw\n"
                    } else {
                        "sizes: 3 2 2\n"
                    },
                    "",
                    1,
                );
            }
            let mut bytes = text.into_bytes();
            bytes.extend([0u8; 12]);
            let err = read_nrrd(&bytes).unwrap_err();
            assert!(err.contains(expected), "{line:?}: {err}");
        }
        let short = b"NRRD0004\ntype: float\ndimension: 1\nsizes: 4\nencoding: raw\n\n\0\0\0\0";
        let err = read_nrrd(short).unwrap_err();
        assert!(err.contains("16 are needed"), "{err}");
    }

    #[test]
    fn validate_catches_inconsistent_geometry() {
        let mut volume = sample();
        volume.values.pop();
        assert!(volume.validate().unwrap_err().contains("12 samples"));
        let mut volume = sample();
        volume.spacing[1] = 0.0;
        assert!(volume.validate().is_err());
        assert!(write_nrrd(&volume, Encoding::Raw).is_err());
    }
}
