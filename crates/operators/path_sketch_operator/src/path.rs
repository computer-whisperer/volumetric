//! SVG path data parser: the `d` attribute grammar — `M L H V C S Q T A Z`
//! in absolute and relative forms, implicit command repetition, compact
//! number runs (`10-5`, `1.5.5`) and adjacent arc flags (`0 01`) —
//! resolved to absolute-coordinate pieces per subpath. Geometry (arc
//! conversion, flattening, rounding) lives in the crate root; this module
//! only decodes.

pub type Point = [f64; 2];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Piece {
    Line(Point),
    Quad {
        ctrl: Point,
        end: Point,
    },
    Cubic {
        c1: Point,
        c2: Point,
        end: Point,
    },
    /// SVG endpoint-parameterised arc: radii, x-axis rotation in degrees,
    /// the large-arc and sweep flags, and the end point.
    Arc {
        rx: f64,
        ry: f64,
        rotation_deg: f64,
        large: bool,
        sweep: bool,
        end: Point,
    },
}

impl Piece {
    pub fn end(&self) -> Point {
        match *self {
            Piece::Line(end)
            | Piece::Quad { end, .. }
            | Piece::Cubic { end, .. }
            | Piece::Arc { end, .. } => end,
        }
    }

    pub fn is_line(&self) -> bool {
        matches!(self, Piece::Line(_))
    }
}

/// One subpath: a start point and the pieces drawn from it. Every subpath
/// is filled closed by the consumer, so `Z` carries no state here beyond
/// resetting the current point.
#[derive(Clone, Debug, PartialEq)]
pub struct Subpath {
    pub start: Point,
    pub pieces: Vec<Piece>,
}

struct Scanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Scanner<'_> {
    fn skip_separators(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos)
            && (b.is_ascii_whitespace() || b == b',')
        {
            self.pos += 1;
        }
    }

    fn at_end(&mut self) -> bool {
        self.skip_separators();
        self.pos >= self.bytes.len()
    }

    /// The next command letter, if the next token is one.
    fn command(&mut self) -> Option<u8> {
        self.skip_separators();
        let c = *self.bytes.get(self.pos)?;
        if c.is_ascii_alphabetic() {
            self.pos += 1;
            Some(c)
        } else {
            None
        }
    }

    /// What the scanner is looking at, for error messages.
    fn found(&self, at: usize) -> String {
        if at >= self.bytes.len() {
            return "end of data".to_string();
        }
        let end = (at + 8).min(self.bytes.len());
        format!("`{}`", String::from_utf8_lossy(&self.bytes[at..end]))
    }

    fn number(&mut self, cmd: u8, what: &str) -> Result<f64, String> {
        self.skip_separators();
        let b = self.bytes;
        let start = self.pos;
        let mut p = start;
        if matches!(b.get(p), Some(b'+' | b'-')) {
            p += 1;
        }
        let int_start = p;
        while matches!(b.get(p), Some(c) if c.is_ascii_digit()) {
            p += 1;
        }
        let mut digits = p - int_start;
        if b.get(p) == Some(&b'.') {
            p += 1;
            let frac_start = p;
            while matches!(b.get(p), Some(c) if c.is_ascii_digit()) {
                p += 1;
            }
            digits += p - frac_start;
        }
        if digits == 0 {
            return Err(format!(
                "`{}` expects {what} at byte {start}, found {}",
                cmd as char,
                self.found(start)
            ));
        }
        if matches!(b.get(p), Some(b'e' | b'E')) {
            let mut q = p + 1;
            if matches!(b.get(q), Some(b'+' | b'-')) {
                q += 1;
            }
            let exp_start = q;
            while matches!(b.get(q), Some(c) if c.is_ascii_digit()) {
                q += 1;
            }
            if q > exp_start {
                p = q;
            }
        }
        let text = std::str::from_utf8(&b[start..p]).map_err(|e| e.to_string())?;
        let value: f64 = text
            .parse()
            .map_err(|_| format!("`{}`: `{text}` is not a number", cmd as char))?;
        if !value.is_finite() {
            return Err(format!("`{}`: `{text}` is not finite", cmd as char));
        }
        self.pos = p;
        Ok(value)
    }

    fn flag(&mut self, cmd: u8, what: &str) -> Result<bool, String> {
        self.skip_separators();
        let at = self.pos;
        match self.bytes.get(at) {
            Some(b'0') => {
                self.pos += 1;
                Ok(false)
            }
            Some(b'1') => {
                self.pos += 1;
                Ok(true)
            }
            _ => Err(format!(
                "`{}` expects {what} (0 or 1) at byte {at}, found {}",
                cmd as char,
                self.found(at)
            )),
        }
    }

    fn point(&mut self, cmd: u8, base: Point, what: &str) -> Result<Point, String> {
        let x = self.number(cmd, what)?;
        let y = self.number(cmd, what)?;
        Ok([base[0] + x, base[1] + y])
    }
}

fn reflect(ctrl: Point, about: Point) -> Point {
    [2.0 * about[0] - ctrl[0], 2.0 * about[1] - ctrl[1]]
}

/// Decode path data into subpaths. Subpaths without pieces (a move
/// followed by another move) are dropped.
pub fn parse(d: &str) -> Result<Vec<Subpath>, String> {
    let mut sc = Scanner {
        bytes: d.as_bytes(),
        pos: 0,
    };
    let mut subpaths: Vec<Subpath> = Vec::new();
    let mut current: Option<Subpath> = None;
    let mut pos: Point = [0.0, 0.0];
    let mut start: Point = [0.0, 0.0];
    let mut last_cmd: Option<u8> = None;
    let mut last_cubic_ctrl: Option<Point> = None;
    let mut last_quad_ctrl: Option<Point> = None;

    let flush = |current: &mut Option<Subpath>, subpaths: &mut Vec<Subpath>| {
        if let Some(sp) = current.take()
            && !sp.pieces.is_empty()
        {
            subpaths.push(sp);
        }
    };

    while !sc.at_end() {
        let cmd = match sc.command() {
            Some(c) => c,
            None => match last_cmd {
                None => return Err("path data must start with a move (M or m)".to_string()),
                Some(b'M') => b'L',
                Some(b'm') => b'l',
                Some(b'Z' | b'z') => {
                    return Err(format!(
                        "numbers after Z need a command letter at byte {}",
                        sc.pos
                    ));
                }
                Some(c) => c,
            },
        };
        let upper = cmd.to_ascii_uppercase();
        if last_cmd.is_none() && upper != b'M' {
            return Err(format!(
                "path data must start with a move (M or m), not `{}`",
                cmd as char
            ));
        }
        let base = if cmd.is_ascii_lowercase() {
            pos
        } else {
            [0.0, 0.0]
        };
        let mut cubic_ctrl = None;
        let mut quad_ctrl = None;

        match upper {
            b'M' => {
                let p = sc.point(cmd, base, "a point")?;
                flush(&mut current, &mut subpaths);
                current = Some(Subpath {
                    start: p,
                    pieces: Vec::new(),
                });
                pos = p;
                start = p;
            }
            b'Z' => {
                flush(&mut current, &mut subpaths);
                pos = start;
            }
            _ => {
                let piece = match upper {
                    b'L' => Piece::Line(sc.point(cmd, base, "a point")?),
                    b'H' => {
                        let x = sc.number(cmd, "an x coordinate")?;
                        Piece::Line([base[0] + x, pos[1]])
                    }
                    b'V' => {
                        let y = sc.number(cmd, "a y coordinate")?;
                        Piece::Line([pos[0], base[1] + y])
                    }
                    b'C' => {
                        let c1 = sc.point(cmd, base, "a control point")?;
                        let c2 = sc.point(cmd, base, "a control point")?;
                        let end = sc.point(cmd, base, "an end point")?;
                        cubic_ctrl = Some(c2);
                        Piece::Cubic { c1, c2, end }
                    }
                    b'S' => {
                        let c2 = sc.point(cmd, base, "a control point")?;
                        let end = sc.point(cmd, base, "an end point")?;
                        let c1 = last_cubic_ctrl.map_or(pos, |c| reflect(c, pos));
                        cubic_ctrl = Some(c2);
                        Piece::Cubic { c1, c2, end }
                    }
                    b'Q' => {
                        let ctrl = sc.point(cmd, base, "a control point")?;
                        let end = sc.point(cmd, base, "an end point")?;
                        quad_ctrl = Some(ctrl);
                        Piece::Quad { ctrl, end }
                    }
                    b'T' => {
                        let end = sc.point(cmd, base, "an end point")?;
                        let ctrl = last_quad_ctrl.map_or(pos, |c| reflect(c, pos));
                        quad_ctrl = Some(ctrl);
                        Piece::Quad { ctrl, end }
                    }
                    b'A' => {
                        let rx = sc.number(cmd, "an x radius")?;
                        let ry = sc.number(cmd, "a y radius")?;
                        let rotation_deg = sc.number(cmd, "a rotation")?;
                        let large = sc.flag(cmd, "the large-arc flag")?;
                        let sweep = sc.flag(cmd, "the sweep flag")?;
                        let end = sc.point(cmd, base, "an end point")?;
                        Piece::Arc {
                            rx,
                            ry,
                            rotation_deg,
                            large,
                            sweep,
                            end,
                        }
                    }
                    other => {
                        return Err(format!("unknown path command `{}`", other as char));
                    }
                };
                pos = piece.end();
                // A command after Z continues from the subpath start.
                current
                    .get_or_insert_with(|| Subpath {
                        start,
                        pieces: Vec::new(),
                    })
                    .pieces
                    .push(piece);
            }
        }

        last_cmd = Some(cmd);
        last_cubic_ctrl = cubic_ctrl;
        last_quad_ctrl = quad_ctrl;
    }
    flush(&mut current, &mut subpaths);

    if subpaths.is_empty() {
        return Err("path data draws nothing".to_string());
    }
    Ok(subpaths)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(d: &str) -> Subpath {
        let mut subpaths = parse(d).unwrap();
        assert_eq!(subpaths.len(), 1, "{d}");
        subpaths.remove(0)
    }

    #[test]
    fn absolute_and_relative_lines() {
        let sp = one("M 1 2 l 3 4 L 0 0 z");
        assert_eq!(sp.start, [1.0, 2.0]);
        assert_eq!(
            sp.pieces,
            [Piece::Line([4.0, 6.0]), Piece::Line([0.0, 0.0])]
        );
    }

    #[test]
    fn implicit_repetition_after_a_move_draws_lines() {
        let sp = one("M0 0 1 1 2 2");
        assert_eq!(
            sp.pieces,
            [Piece::Line([1.0, 1.0]), Piece::Line([2.0, 2.0])]
        );
        let sp = one("m1 1 1 1 1 1");
        assert_eq!(
            sp.pieces,
            [Piece::Line([2.0, 2.0]), Piece::Line([3.0, 3.0])]
        );
    }

    #[test]
    fn horizontal_and_vertical_keep_the_other_coordinate() {
        let sp = one("M1 1 H 3 v 2 h -1 V 0");
        assert_eq!(
            sp.pieces,
            [
                Piece::Line([3.0, 1.0]),
                Piece::Line([3.0, 3.0]),
                Piece::Line([2.0, 3.0]),
                Piece::Line([2.0, 0.0]),
            ]
        );
    }

    #[test]
    fn compact_numbers_and_adjacent_arc_flags() {
        let sp = one("M10-5L1.5.5A1 1 0 01 5 5e0");
        assert_eq!(sp.start, [10.0, -5.0]);
        assert_eq!(sp.pieces[0], Piece::Line([1.5, 0.5]));
        assert_eq!(
            sp.pieces[1],
            Piece::Arc {
                rx: 1.0,
                ry: 1.0,
                rotation_deg: 0.0,
                large: false,
                sweep: true,
                end: [5.0, 5.0],
            }
        );
    }

    #[test]
    fn smooth_curves_reflect_the_previous_control_point() {
        let sp = one("M0 0 C 1 1 2 1 3 0 S 5 -1 6 0 Q 7 1 8 0 T 10 0 T 12 0");
        assert_eq!(
            sp.pieces[1],
            Piece::Cubic {
                c1: [4.0, -1.0],
                c2: [5.0, -1.0],
                end: [6.0, 0.0],
            }
        );
        assert_eq!(
            sp.pieces[3],
            Piece::Quad {
                ctrl: [9.0, -1.0],
                end: [10.0, 0.0],
            }
        );
        assert_eq!(
            sp.pieces[4],
            Piece::Quad {
                ctrl: [11.0, 1.0],
                end: [12.0, 0.0],
            }
        );
        // No preceding curve of the right kind: the control point is the
        // current point.
        let sp = one("M0 0 L 1 0 S 2 1 3 0 L 4 0 T 5 0");
        assert_eq!(
            sp.pieces[1],
            Piece::Cubic {
                c1: [1.0, 0.0],
                c2: [2.0, 1.0],
                end: [3.0, 0.0],
            }
        );
        assert_eq!(
            sp.pieces[3],
            Piece::Quad {
                ctrl: [4.0, 0.0],
                end: [5.0, 0.0],
            }
        );
    }

    #[test]
    fn close_resets_the_current_point_and_splits_subpaths() {
        let subpaths = parse("M1 1 h1 v1 z l 5 0 v1 z M9 9 M8 8 l1 0 l0 1").unwrap();
        assert_eq!(subpaths.len(), 3);
        assert_eq!(subpaths[1].start, [1.0, 1.0]);
        assert_eq!(subpaths[1].pieces[0], Piece::Line([6.0, 1.0]));
        // The empty `M9 9` subpath is dropped.
        assert_eq!(subpaths[2].start, [8.0, 8.0]);
    }

    #[test]
    fn errors_name_the_problem() {
        assert!(parse("L 1 2").unwrap_err().contains("start with a move"));
        assert!(parse("M 1").unwrap_err().contains("expects a point"));
        assert!(
            parse("M 0 0 X 1")
                .unwrap_err()
                .contains("unknown path command `X`")
        );
        assert!(
            parse("M 0 0 A 1 1 0 2 0 3 3")
                .unwrap_err()
                .contains("large-arc flag")
        );
        assert!(parse("M 0 0 Z 1 1").unwrap_err().contains("after Z"));
        assert!(parse("").unwrap_err().contains("draws nothing"));
        assert!(parse("M 1 2").unwrap_err().contains("draws nothing"));
    }
}
