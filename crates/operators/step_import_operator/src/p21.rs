//! ISO 10303-21 (STEP Part 21) syntax parser: the DATA section as a flat
//! table of entity instances with structured arguments. Purely
//! syntactic — entity semantics live in `entities.rs`.
//!
//! Grammar subset (covers OCCT/KiCad/SolidWorks exports):
//! - `#id = TYPE(arg, ...);` — simple instance
//! - `#id = (TYPE1(...) TYPE2(...) ...);` — complex instance
//! - args: `#ref`, numbers, `'strings'` (with `''` escape), `.ENUM.`,
//!   nested `(lists)`, `TYPE(...)` typed values, `*` (derived), `$`
//!   (unset)
//! - `/* comments */` and arbitrary whitespace between tokens

use std::collections::HashMap;

/// One parsed argument.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Ref(u64),
    Number(f64),
    Str(String),
    /// `.IDENT.` — enums and the booleans `.T.` / `.F.`
    Enum(String),
    List(Vec<Arg>),
    /// `TYPE(args)` appearing in argument position (measures etc.).
    Typed(String, Vec<Arg>),
    /// `*`
    Derived,
    /// `$`
    Unset,
}

impl Arg {
    pub fn as_ref_id(&self) -> Option<u64> {
        match self {
            Arg::Ref(id) => Some(*id),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Arg::Number(v) => Some(*v),
            Arg::Typed(_, args) if args.len() == 1 => args[0].as_f64(),
            _ => None,
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Arg::Number(v) if *v >= 0.0 && v.fract() == 0.0 => Some(*v as usize),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Arg::Number(v) if v.fract() == 0.0 => Some(*v as i64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Arg::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Arg::Enum(e) if e == "T" => Some(true),
            Arg::Enum(e) if e == "F" => Some(false),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&[Arg]> {
        match self {
            Arg::List(items) => Some(items),
            _ => None,
        }
    }
}

/// One `TYPE(args)` record; simple instances have exactly one, complex
/// instances several.
#[derive(Clone, Debug)]
pub struct Record {
    pub name: String,
    pub args: Vec<Arg>,
}

/// A parsed instance: `#id = records`.
#[derive(Clone, Debug)]
pub struct Entity {
    pub records: Vec<Record>,
}

impl Entity {
    /// The record for a simple instance, or the named record of a
    /// complex instance.
    pub fn record(&self, name: &str) -> Option<&Record> {
        self.records.iter().find(|r| r.name == name)
    }
    pub fn simple(&self) -> Option<&Record> {
        match self.records.as_slice() {
            [r] => Some(r),
            _ => None,
        }
    }
    pub fn is(&self, name: &str) -> bool {
        self.records.iter().any(|r| r.name == name)
    }
}

/// The DATA section: id → entity.
pub struct DataSection {
    pub entities: HashMap<u64, Entity>,
}

impl DataSection {
    pub fn get(&self, id: u64) -> Result<&Entity, String> {
        self.entities
            .get(&id)
            .ok_or_else(|| format!("dangling entity reference #{id}"))
    }

    /// Follow a `Ref` argument.
    pub fn deref(&self, arg: &Arg) -> Result<&Entity, String> {
        match arg {
            Arg::Ref(id) => self.get(*id),
            other => Err(format!("expected an entity reference, got {other:?}")),
        }
    }
}

pub fn parse(input: &str) -> Result<DataSection, String> {
    let bytes = input.as_bytes();
    let mut lex = Lexer { bytes, pos: 0 };

    // Verify the file magic, then skip to DATA;
    lex.skip_ws();
    // Tolerate a UTF-8 BOM — they do occur in the wild.
    lex.eat_keyword("\u{FEFF}");
    lex.skip_ws();
    if !lex.eat_keyword("ISO-10303-21;") {
        return Err("not a STEP file: missing ISO-10303-21 header".into());
    }
    let data_start = find_keyword(bytes, lex.pos, b"DATA;").ok_or("no DATA section")?;
    lex.pos = data_start + 5;

    let mut entities = HashMap::new();
    loop {
        lex.skip_ws();
        if lex.eat_keyword("ENDSEC;") || lex.pos >= bytes.len() {
            break;
        }
        let id = lex.instance_id()?;
        lex.skip_ws();
        if !lex.eat(b'=') {
            return Err(format!("#{id}: expected '='"));
        }
        lex.skip_ws();
        let records = if lex.peek() == Some(b'(') {
            // Complex instance: a parenthesized run of records.
            lex.pos += 1;
            let mut records = Vec::new();
            loop {
                lex.skip_ws();
                if lex.eat(b')') {
                    break;
                }
                records.push(lex.record()?);
            }
            records
        } else {
            vec![lex.record()?]
        };
        lex.skip_ws();
        if !lex.eat(b';') {
            return Err(format!("#{id}: expected ';'"));
        }
        entities.insert(id, Entity { records });
    }
    Ok(DataSection { entities })
}

fn find_keyword(bytes: &[u8], from: usize, keyword: &[u8]) -> Option<usize> {
    // Scan outside strings/comments so 'DATA;' in a description can't
    // fool us.
    let mut lex = Lexer { bytes, pos: from };
    loop {
        lex.skip_ws();
        if lex.pos >= bytes.len() {
            return None;
        }
        if bytes[lex.pos..].starts_with(keyword) {
            return Some(lex.pos);
        }
        if bytes[lex.pos] == b'\'' {
            lex.string().ok()?;
        } else {
            lex.pos += 1;
        }
    }
}

struct Lexer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Lexer<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_keyword(&mut self, kw: &str) -> bool {
        if self.bytes[self.pos..].starts_with(kw.as_bytes()) {
            self.pos += kw.len();
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r' | b'\n') => self.pos += 1,
                Some(b'/') if self.bytes.get(self.pos + 1) == Some(&b'*') => {
                    self.pos += 2;
                    while self.pos < self.bytes.len()
                        && !(self.bytes[self.pos] == b'*'
                            && self.bytes.get(self.pos + 1) == Some(&b'/'))
                    {
                        self.pos += 1;
                    }
                    self.pos = (self.pos + 2).min(self.bytes.len());
                }
                _ => return,
            }
        }
    }

    fn instance_id(&mut self) -> Result<u64, String> {
        if !self.eat(b'#') {
            return Err(format!(
                "expected '#' at byte {} ({:?}...)",
                self.pos,
                String::from_utf8_lossy(
                    &self.bytes[self.pos..(self.pos + 24).min(self.bytes.len())]
                )
            ));
        }
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(format!("expected digits after '#' at byte {start}"));
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .unwrap()
            .parse()
            .map_err(|e| format!("bad instance id: {e}"))
    }

    /// `TYPE(args)`.
    fn record(&mut self) -> Result<Record, String> {
        let name = self.keyword()?;
        self.skip_ws();
        if !self.eat(b'(') {
            return Err(format!("{name}: expected '('"));
        }
        let args = self.args()?;
        Ok(Record { name, args })
    }

    /// Uppercase identifier (letters, digits, underscore).
    fn keyword(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(format!(
                "expected a keyword at byte {} ({:?}...)",
                start,
                String::from_utf8_lossy(&self.bytes[start..(start + 24).min(self.bytes.len())])
            ));
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }

    /// Arguments up to and including the closing ')'.
    fn args(&mut self) -> Result<Vec<Arg>, String> {
        let mut args = Vec::new();
        loop {
            self.skip_ws();
            if self.eat(b')') {
                return Ok(args);
            }
            if !args.is_empty() {
                if !self.eat(b',') {
                    return Err(format!("expected ',' or ')' at byte {}", self.pos));
                }
                self.skip_ws();
            }
            args.push(self.arg()?);
        }
    }

    fn arg(&mut self) -> Result<Arg, String> {
        match self.peek() {
            Some(b'#') => Ok(Arg::Ref(self.instance_id()?)),
            Some(b'$') => {
                self.pos += 1;
                Ok(Arg::Unset)
            }
            Some(b'*') => {
                self.pos += 1;
                Ok(Arg::Derived)
            }
            Some(b'\'') => Ok(Arg::Str(self.string()?)),
            Some(b'.') => {
                self.pos += 1;
                let name = self.keyword()?;
                if !self.eat(b'.') {
                    return Err(format!("unterminated enum .{name}"));
                }
                Ok(Arg::Enum(name))
            }
            Some(b'(') => {
                self.pos += 1;
                Ok(Arg::List(self.args()?))
            }
            Some(b) if b == b'-' || b == b'+' || b.is_ascii_digit() => self.number(),
            Some(b) if b.is_ascii_uppercase() => {
                let r = self.record()?;
                Ok(Arg::Typed(r.name, r.args))
            }
            other => Err(format!(
                "unexpected argument start {other:?} at byte {}",
                self.pos
            )),
        }
    }

    fn number(&mut self) -> Result<Arg, String> {
        let start = self.pos;
        if matches!(self.peek(), Some(b'-' | b'+')) {
            self.pos += 1;
        }
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_digit() || b == b'.' || b == b'E' || b == b'e')
        {
            self.pos += 1;
            // Exponent sign.
            if matches!(self.bytes.get(self.pos - 1), Some(b'E' | b'e'))
                && matches!(self.peek(), Some(b'-' | b'+'))
            {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        text.parse::<f64>()
            .map(Arg::Number)
            .map_err(|e| format!("bad number {text:?}: {e}"))
    }

    /// A `'...'` string; `''` is an escaped quote.
    fn string(&mut self) -> Result<String, String> {
        if !self.eat(b'\'') {
            return Err("expected a string".into());
        }
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".into()),
                Some(b'\'') => {
                    self.pos += 1;
                    if self.peek() == Some(b'\'') {
                        out.push(b'\'');
                        self.pos += 1;
                    } else {
                        return Ok(String::from_utf8_lossy(&out).into_owned());
                    }
                }
                Some(b) => {
                    out.push(b);
                    self.pos += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('has DATA; in a string'),'2;1');
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('',(0.,-1.5E-3,2.));
#2 = DIRECTION('name''with quote',(0.,0.,1.));
#3 = ADVANCED_FACE('',(#1),#2,.T.);
#4 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3)
GLOBAL_UNIT_ASSIGNED_CONTEXT((#1,#2)) REPRESENTATION_CONTEXT('x','y') );
#5 = MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#4); /* comment, with ) */
ENDSEC;
END-ISO-10303-21;
"#;

    #[test]
    fn parses_the_zoo() {
        let data = parse(MINI).unwrap();
        assert_eq!(data.entities.len(), 5);

        let p = data.get(1).unwrap().simple().unwrap();
        assert_eq!(p.name, "CARTESIAN_POINT");
        let coords = p.args[1].as_list().unwrap();
        assert_eq!(coords[0].as_f64(), Some(0.0));
        assert_eq!(coords[1].as_f64(), Some(-1.5e-3));
        assert_eq!(coords[2].as_f64(), Some(2.0));

        let d = data.get(2).unwrap().simple().unwrap();
        assert_eq!(d.args[0].as_str(), Some("name'with quote"));

        let f = data.get(3).unwrap().simple().unwrap();
        assert_eq!(f.args[1].as_list().unwrap()[0].as_ref_id(), Some(1));
        assert_eq!(f.args[2].as_ref_id(), Some(2));
        assert_eq!(f.args[3].as_bool(), Some(true));

        let ctx = data.get(4).unwrap();
        assert_eq!(ctx.records.len(), 3);
        assert!(ctx.is("GLOBAL_UNIT_ASSIGNED_CONTEXT"));
        assert!(ctx.record("REPRESENTATION_CONTEXT").is_some());

        let m = data.get(5).unwrap().simple().unwrap();
        match &m.args[0] {
            Arg::Typed(name, args) => {
                assert_eq!(name, "LENGTH_MEASURE");
                assert_eq!(args[0].as_f64(), Some(25.4));
            }
            other => panic!("expected typed measure, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_step() {
        assert!(parse("not a step file").is_err());
    }

    #[test]
    fn dangling_reference_reported() {
        let data = parse(MINI).unwrap();
        assert!(data.get(99).is_err());
    }
}
