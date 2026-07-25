//! The HTML subset parser: a deliberately tiny, hand-written reader for
//! the fragment language the card operator accepts — lowercase tags from
//! [`SUPPORTED_TAGS`], a `class` attribute, text with the standard named
//! entities, comments, and `<br>`/`<hr>` voids. Anything outside the
//! subset is a hard error naming the offender, never a silent skip: the
//! operator's contract is that markup either renders as specified or
//! fails loudly enough for the author to converge in one round.

use std::fmt::Write as _;

/// Tags the card language accepts. `div` lays out; `p`/`h1`–`h4` hold
/// text (headings carry no default styling, exactly like Tailwind's
/// preflight — size and weight come from classes); `span`/`b`/`strong`
/// style inline runs; `br`/`hr` are voids.
pub const SUPPORTED_TAGS: &[&str] = &[
    "div", "p", "h1", "h2", "h3", "h4", "span", "b", "strong", "br", "hr",
];

const VOID_TAGS: &[&str] = &["br", "hr"];

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Element(Element),
    /// Raw text with entities decoded; whitespace not yet collapsed
    /// (collapse happens during paragraph shaping, where runs merge).
    Text(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    pub tag: String,
    pub classes: Vec<String>,
    pub children: Vec<Node>,
}

/// Parse an HTML fragment into its single root element.
///
/// The fragment must contain exactly one top-level element (the card);
/// whitespace and comments around it are fine.
pub fn parse_fragment(html: &str) -> Result<Element, String> {
    let mut parser = Parser {
        bytes: html.as_bytes(),
        pos: 0,
        depth: 0,
        errors: Vec::new(),
    };
    let mut roots = Vec::new();
    parser.parse_nodes(None, &mut roots);
    if !parser.errors.is_empty() {
        return Err(parser.errors.join("\n"));
    }

    let mut elements = roots.into_iter().filter_map(|node| match node {
        Node::Element(el) => Some(el),
        Node::Text(t) if t.trim().is_empty() => None,
        Node::Text(t) => {
            let preview: String = t.trim().chars().take(30).collect();
            Some(Element {
                tag: format!("!text {preview:?}"),
                classes: Vec::new(),
                children: Vec::new(),
            })
        }
    });
    match (elements.next(), elements.next()) {
        (Some(root), None) if !root.tag.starts_with('!') => Ok(root),
        (Some(bad), None) => Err(format!(
            "the fragment's top level holds bare text ({}) — wrap the card in a single root element",
            &bad.tag[1..]
        )),
        (None, _) => Err("the fragment contains no elements".to_string()),
        (Some(_), Some(_)) => {
            Err("the fragment has multiple top-level elements — wrap the card in a single root element"
                .to_string())
        }
    }
}

/// Nesting deeper than any real card markup; a cap keeps the recursive
/// parser (and every recursive consumer of the tree) off the stack limit.
const MAX_DEPTH: usize = 128;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
    errors: Vec<String>,
}

impl Parser<'_> {
    fn rest(&self) -> &[u8] {
        &self.bytes[self.pos..]
    }

    fn starts_with(&self, s: &str) -> bool {
        self.rest().starts_with(s.as_bytes())
    }

    fn line(&self) -> usize {
        1 + self.bytes[..self.pos].iter().filter(|&&b| b == b'\n').count()
    }

    fn error(&mut self, msg: String) {
        // A pathological input (a large non-HTML paste) can generate one
        // error per byte; past a generous cap, stop parsing outright so
        // error collection stays linear.
        if self.errors.len() >= 64 {
            if self.errors.len() == 64 {
                self.errors
                    .push("too many errors — stopping here".to_string());
            }
            self.pos = self.bytes.len();
            return;
        }
        let line = self.line();
        self.errors.push(format!("line {line}: {msg}"));
    }

    /// Parse child nodes until `</closing>` (or end of input for `None`).
    fn parse_nodes(&mut self, closing: Option<&str>, out: &mut Vec<Node>) {
        let mut text = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                if let Some(tag) = closing {
                    self.error(format!("unclosed <{tag}> (reached end of input)"));
                }
                break;
            }
            if self.starts_with("<!--") {
                Self::flush_text(&mut text, out);
                match self.rest().windows(3).position(|w| w == b"-->") {
                    Some(end) => self.pos += end + 3,
                    None => {
                        self.error("unterminated comment".to_string());
                        self.pos = self.bytes.len();
                    }
                }
                continue;
            }
            if self.starts_with("</") {
                Self::flush_text(&mut text, out);
                let start = self.pos;
                self.pos += 2;
                let name = self.read_name();
                self.skip_whitespace();
                if !self.starts_with(">") {
                    self.error(format!("malformed closing tag </{name}"));
                    self.pos = self.bytes.len();
                    break;
                }
                self.pos += 1;
                if closing == Some(name.as_str()) {
                    return;
                }
                self.error(match closing {
                    Some(open) => format!("</{name}> closes nothing (inside <{open}>)"),
                    None => format!("</{name}> closes nothing (at the top level)"),
                });
                let _ = start;
                continue;
            }
            if self.starts_with("<") {
                Self::flush_text(&mut text, out);
                if let Some(node) = self.parse_element() {
                    out.push(node);
                }
                continue;
            }
            // Text run up to the next markup character.
            let ch = self.decode_text_char();
            if let Some(c) = ch {
                text.push(c);
            }
        }
        Self::flush_text(&mut text, out);
    }

    fn flush_text(text: &mut String, out: &mut Vec<Node>) {
        if !text.is_empty() {
            out.push(Node::Text(std::mem::take(text)));
        }
    }

    /// One character of text content, decoding entities.
    fn decode_text_char(&mut self) -> Option<char> {
        let rest = self.rest();
        if rest[0] == b'&' {
            let end = rest[..rest.len().min(14)].iter().position(|&b| b == b';');
            let entity = end.map(|e| &rest[1..e]);
            let decoded = match entity {
                Some(b"amp") => Some('&'),
                Some(b"lt") => Some('<'),
                Some(b"gt") => Some('>'),
                Some(b"quot") => Some('"'),
                Some(b"apos") => Some('\''),
                Some(b"nbsp") => Some('\u{a0}'),
                Some(num) if num.first() == Some(&b'#') => {
                    let digits = &num[1..];
                    let code = if digits.first() == Some(&b'x') || digits.first() == Some(&b'X') {
                        u32::from_str_radix(std::str::from_utf8(&digits[1..]).ok()?, 16).ok()
                    } else {
                        std::str::from_utf8(digits).ok()?.parse().ok()
                    };
                    code.and_then(char::from_u32)
                }
                _ => None,
            };
            match decoded {
                Some(c) if end.is_some() && end.unwrap() <= 12 => {
                    self.pos += end.unwrap() + 1;
                    return Some(c);
                }
                _ => {
                    let preview: String =
                        String::from_utf8_lossy(&rest[..rest.len().min(8)]).into_owned();
                    self.error(format!("unknown entity starting at {preview:?}"));
                    self.pos += 1;
                    return None;
                }
            }
        }
        // Plain UTF-8 character.
        let s = std::str::from_utf8(rest).ok()?;
        let c = s.chars().next()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn read_name(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'-')
        {
            self.pos += 1;
        }
        String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned()
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn parse_element(&mut self) -> Option<Node> {
        if self.depth >= MAX_DEPTH {
            self.error(format!("markup nested deeper than {MAX_DEPTH} levels"));
            self.pos = self.bytes.len();
            return None;
        }
        self.pos += 1; // consume '<'
        let tag = self.read_name();
        if tag.is_empty() {
            self.error("stray '<' (write it as &lt;)".to_string());
            return None;
        }
        let lowered = tag.to_ascii_lowercase();
        if lowered != tag {
            self.error(format!("tag <{tag}> must be lowercase"));
        }
        if !SUPPORTED_TAGS.contains(&lowered.as_str()) {
            let mut msg = format!("unsupported tag <{lowered}>");
            let _ = write!(msg, " (supported: {})", SUPPORTED_TAGS.join(", "));
            self.error(msg);
        }

        let mut classes = Vec::new();
        loop {
            self.skip_whitespace();
            if self.pos >= self.bytes.len() {
                self.error(format!("unterminated <{lowered}> tag"));
                return None;
            }
            if self.starts_with("/>") {
                self.pos += 2;
                return Some(Node::Element(Element {
                    tag: lowered,
                    classes,
                    children: Vec::new(),
                }));
            }
            if self.starts_with(">") {
                self.pos += 1;
                break;
            }
            let attr = self.read_name();
            if attr.is_empty() {
                self.error(format!(
                    "malformed <{lowered}> tag near {:?}",
                    String::from_utf8_lossy(&self.rest()[..self.rest().len().min(8)])
                ));
                self.pos = self.bytes.len();
                return None;
            }
            let value = if self.starts_with("=") {
                self.pos += 1;
                self.read_attr_value(&lowered)?
            } else {
                String::new()
            };
            if attr == "class" {
                classes.extend(value.split_ascii_whitespace().map(str::to_string));
            } else {
                self.error(format!(
                    "unsupported attribute {attr:?} on <{lowered}> (only class is read)"
                ));
            }
        }

        if VOID_TAGS.contains(&lowered.as_str()) {
            return Some(Node::Element(Element {
                tag: lowered,
                classes,
                children: Vec::new(),
            }));
        }
        let mut children = Vec::new();
        self.depth += 1;
        self.parse_nodes(Some(&lowered), &mut children);
        self.depth -= 1;
        Some(Node::Element(Element {
            tag: lowered,
            classes,
            children,
        }))
    }

    fn read_attr_value(&mut self, tag: &str) -> Option<String> {
        let quote = *self.rest().first()?;
        if quote != b'"' && quote != b'\'' {
            self.error(format!("unquoted attribute value on <{tag}>"));
            return None;
        }
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != quote {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            self.error(format!("unterminated attribute value on <{tag}>"));
            return None;
        }
        let value = String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned();
        self.pos += 1;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parses(html: &str) -> Element {
        parse_fragment(html).expect("fragment parses")
    }

    #[test]
    fn nested_structure_with_classes_and_text() {
        let root = parses(
            r#"<div class="flex gap-2">
                 <p class="font-bold">Hi &amp; bye</p>
                 <span>x</span>
               </div>"#,
        );
        assert_eq!(root.tag, "div");
        assert_eq!(root.classes, ["flex", "gap-2"]);
        let elements: Vec<&Element> = root
            .children
            .iter()
            .filter_map(|n| match n {
                Node::Element(e) => Some(e),
                Node::Text(_) => None,
            })
            .collect();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].tag, "p");
        assert_eq!(elements[0].children, [Node::Text("Hi & bye".to_string())]);
    }

    #[test]
    fn voids_comments_and_self_closing() {
        let root = parses("<div><!-- note --><br><hr class=\"border-2\"/><span/></div>");
        let tags: Vec<String> = root
            .children
            .iter()
            .filter_map(|n| match n {
                Node::Element(e) => Some(e.tag.clone()),
                Node::Text(_) => None,
            })
            .collect();
        assert_eq!(tags, ["br", "hr", "span"]);
    }

    #[test]
    fn entities_decode() {
        let root = parses("<p>&lt;3 &quot;a&quot; &#65;&#x42; &nbsp;</p>");
        assert_eq!(
            root.children,
            [Node::Text("<3 \"a\" AB \u{a0}".to_string())]
        );
    }

    #[test]
    fn errors_are_specific_and_collected() {
        let err = parse_fragment("<div CLASS=\"x\"><img src=\"a\"><ul></ul></div>").unwrap_err();
        assert!(err.contains("unsupported tag <img>"), "{err}");
        assert!(err.contains("unsupported tag <ul>"), "{err}");
        assert!(err.contains("unsupported attribute"), "{err}");

        let err = parse_fragment("<div>").unwrap_err();
        assert!(err.contains("unclosed <div>"), "{err}");

        let err = parse_fragment("<div></span></div>").unwrap_err();
        assert!(err.contains("</span> closes nothing"), "{err}");

        let err = parse_fragment("hello").unwrap_err();
        assert!(err.contains("wrap the card"), "{err}");

        let err = parse_fragment("<div></div><div></div>").unwrap_err();
        assert!(err.contains("multiple top-level"), "{err}");

        let err = parse_fragment("<p>a &unknown; b</p>").unwrap_err();
        assert!(err.contains("unknown entity"), "{err}");

        let err = parse_fragment("<p>3 < 4</p>").unwrap_err();
        assert!(err.contains("stray '<'"), "{err}");
    }

    #[test]
    fn line_numbers_locate_errors() {
        let err = parse_fragment("<div>\n\n<video></video>\n</div>").unwrap_err();
        assert!(err.contains("line 3"), "{err}");
    }

    #[test]
    fn deep_nesting_errors_instead_of_overflowing() {
        let html = format!("{}x{}", "<div>".repeat(20_000), "</div>".repeat(20_000));
        let err = parse_fragment(&html).unwrap_err();
        assert!(err.contains("nested deeper"), "{err}");
    }

    #[test]
    fn pathological_error_floods_terminate_quickly() {
        // 200KB of bare ampersands: capped error collection keeps this
        // linear instead of quadratic-in-input.
        let html = format!("<p>{}</p>", "& ".repeat(100_000));
        let start = std::time::Instant::now();
        let err = parse_fragment(&html).unwrap_err();
        assert!(start.elapsed().as_secs_f64() < 2.0, "took {:?}", start.elapsed());
        assert!(err.contains("too many errors"), "{err}");
        assert!(err.lines().count() <= 65, "{} lines", err.lines().count());
    }
}
