//! Marker dictionaries: bit patterns and their identification with
//! rotation and error tolerance. ArUco's 5x5 and 4x4 sets and the
//! AprilTag 36h11 family are the same thing to the detector: a square of
//! `n x n` bits inside a dark border.
//!
//! A marker's `n x n` inner bits are packed row-major from the top-left
//! cell, most significant bit first, as OpenCV's rotation 0. Rotating the
//! picture rotates the bits; identification tries all four so the corners
//! can be put into the marker's canonical order.

use crate::dict_tables::{APRILTAG_36H11_CODES, ARUCO_4X4_50_CODES, ARUCO_5X5_100_CODES};

#[derive(Clone, Debug)]
pub struct Dictionary {
    pub name: &'static str,
    /// Bits per side of the inner pattern.
    pub size: u32,
    /// Each marker's bits at rotations 0..4 (clockwise quarter turns).
    codes: Vec<[u64; 4]>,
    /// OpenCV's `maxCorrectionBits` for the dictionary.
    pub max_correction: u32,
}

/// A marker identified in a dictionary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identification {
    pub id: u32,
    /// Clockwise quarter turns the observed pattern is from the
    /// dictionary's rotation 0: the observation is the code turned this
    /// many times clockwise, so the code's top-left corner sits at the
    /// observation's corner `rotation` (counting clockwise from its
    /// top-left).
    pub rotation: u32,
    /// Bits that differed.
    pub distance: u32,
}

/// The bits of an `n x n` pattern turned a quarter turn clockwise: the
/// cell at (row r, column c) moves to (c, n - 1 - r).
pub fn rotate_cw(bits: u64, n: u32) -> u64 {
    let mut out = 0u64;
    for r in 0..n {
        for c in 0..n {
            let bit = (bits >> (n * n - 1 - (r * n + c))) & 1;
            let (nr, nc) = (c, n - 1 - r);
            out |= bit << (n * n - 1 - (nr * n + nc));
        }
    }
    out
}

impl Dictionary {
    fn new(name: &'static str, size: u32, codes: &[u64], max_correction: u32) -> Self {
        let codes = codes
            .iter()
            .map(|&c0| {
                let c1 = rotate_cw(c0, size);
                let c2 = rotate_cw(c1, size);
                let c3 = rotate_cw(c2, size);
                [c0, c1, c2, c3]
            })
            .collect();
        Self {
            name,
            size,
            codes,
            max_correction,
        }
    }

    /// OpenCV `DICT_5X5_100`: the scanner's swatch markers.
    pub fn aruco_5x5_100() -> Self {
        Self::new("5x5_100", 5, &ARUCO_5X5_100_CODES, 3)
    }

    /// OpenCV `DICT_4X4_50`: the scanner's calibration board.
    pub fn aruco_4x4_50() -> Self {
        Self::new("4x4_50", 4, &ARUCO_4X4_50_CODES, 1)
    }

    /// OpenCV `DICT_APRILTAG_36h11`: the survey card's tags (ids 100–165
    /// of the 587 in the family).
    pub fn apriltag_36h11() -> Self {
        Self::new("36h11", 6, &APRILTAG_36H11_CODES, 5)
    }

    /// The families by name.
    pub const NAMES: [&'static str; 3] = ["5x5_100", "4x4_50", "36h11"];

    /// `5x5_100`, `4x4_50` or `36h11`, also accepting OpenCV's
    /// `DICT_5X5_100` and `DICT_APRILTAG_36h11` forms.
    pub fn by_name(name: &str) -> Option<Self> {
        match name
            .trim_start_matches("DICT_")
            .trim_start_matches("APRILTAG_")
            .to_ascii_lowercase()
            .as_str()
        {
            "5x5_100" => Some(Self::aruco_5x5_100()),
            "4x4_50" => Some(Self::aruco_4x4_50()),
            "36h11" => Some(Self::apriltag_36h11()),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// The marker's bits at rotation 0.
    pub fn code(&self, id: u32) -> Option<u64> {
        self.codes.get(id as usize).map(|c| c[0])
    }

    /// Bits that may differ and still identify, as OpenCV's default
    /// `errorCorrectionRate` (0.6) of the maximum correction.
    pub fn tolerance(&self) -> u32 {
        (0.6 * f64::from(self.max_correction)).floor() as u32
    }

    /// The marker whose pattern, at some rotation, is nearest to `bits`
    /// within the tolerance. `bits` are the observed inner pattern packed
    /// as the tables are.
    pub fn identify(&self, bits: u64) -> Option<Identification> {
        let tolerance = self.tolerance();
        let mut best: Option<Identification> = None;
        for (id, rotations) in self.codes.iter().enumerate() {
            for (rotation, code) in rotations.iter().enumerate() {
                let distance = (code ^ bits).count_ones();
                if distance <= tolerance && best.is_none_or(|b| distance < b.distance) {
                    best = Some(Identification {
                        id: id as u32,
                        rotation: rotation as u32,
                        distance,
                    });
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_a_quarter_turn() {
        // A single set bit at the top-right of a 3x3 goes to the
        // bottom-right, then bottom-left, then top-left, then home.
        let n = 3;
        let top_right = 1 << (8 - 2);
        let r1 = rotate_cw(top_right, n);
        assert_eq!(r1, 1 << (8 - 8), "top-right -> bottom-right");
        let r2 = rotate_cw(r1, n);
        assert_eq!(r2, 1 << (8 - 6), "-> bottom-left");
        let r3 = rotate_cw(r2, n);
        assert_eq!(r3, 1 << 8, "-> top-left");
        assert_eq!(rotate_cw(r3, n), top_right);
        for code in [0b101_011_110u64, 0b111_000_010] {
            let mut b = code;
            for _ in 0..4 {
                b = rotate_cw(b, n);
            }
            assert_eq!(b, code);
        }
    }

    #[test]
    fn markers_identify_at_every_rotation_within_tolerance() {
        let dict = Dictionary::aruco_5x5_100();
        assert_eq!(dict.len(), 100);
        assert_eq!(dict.tolerance(), 1);
        for id in [0u32, 1, 5, 49, 53, 99] {
            let code = dict.code(id).unwrap();
            let mut bits = code;
            for rotation in 0..4 {
                let found = dict.identify(bits).unwrap();
                assert_eq!(
                    (found.id, found.distance),
                    (id, 0),
                    "id {id} rot {rotation}"
                );
                // The observed bits are the code turned counter-clockwise
                // `rotation` times, so `rotation` clockwise turns undo it.
                assert_eq!((4 - found.rotation) % 4, rotation % 4, "id {id}");
                bits = rotate_cw(rotate_cw(rotate_cw(bits, 5), 5), 5);
            }
            // One wrong bit still identifies; two do not.
            let one = dict.identify(code ^ 1).unwrap();
            assert_eq!((one.id, one.distance), (id, 1));
            assert_eq!(dict.identify(code ^ 0b11), None);
        }
        let board = Dictionary::aruco_4x4_50();
        assert_eq!((board.len(), board.size, board.tolerance()), (50, 4, 0));
        assert_eq!(board.identify(board.code(7).unwrap()).unwrap().id, 7);
        assert_eq!(board.identify(board.code(7).unwrap() ^ 1), None);
        assert_eq!(Dictionary::by_name("DICT_4X4_50").unwrap().name, "4x4_50");
        assert!(Dictionary::by_name("6x6").is_none());
    }

    #[test]
    fn apriltags_identify_with_three_bits_corrected() {
        let tags = Dictionary::apriltag_36h11();
        assert_eq!((tags.len(), tags.size, tags.tolerance()), (587, 6, 3));
        for id in [0u32, 100, 133, 165, 586] {
            let code = tags.code(id).unwrap();
            assert!(code < 1 << 36);
            let mut bits = code;
            for rotation in 0..4 {
                let found = tags.identify(bits).unwrap();
                assert_eq!(
                    (found.id, found.distance),
                    (id, 0),
                    "id {id} rot {rotation}"
                );
                assert_eq!((4 - found.rotation) % 4, rotation % 4);
                bits = rotate_cw(rotate_cw(rotate_cw(bits, 6), 6), 6);
            }
            let three = tags.identify(code ^ 0b1000_0000_0101).unwrap();
            assert_eq!((three.id, three.distance), (id, 3));
            assert_eq!(tags.identify(code ^ 0b1000_0000_1111), None);
        }
        // The families never read each other's markers, even with the
        // tolerance: a 5x5 code padded into a 6x6 is not a tag.
        let swatches = Dictionary::aruco_5x5_100();
        for id in 0..swatches.len() as u32 {
            let bits = swatches.code(id).unwrap();
            assert!(tags.identify(bits).is_none(), "swatch {id} reads as a tag");
            assert!(
                tags.identify(bits << 11).is_none(),
                "swatch {id} reads as a tag"
            );
        }
        assert_eq!(
            Dictionary::by_name("DICT_APRILTAG_36h11").unwrap().name,
            "36h11"
        );
        assert_eq!(Dictionary::by_name("36H11").unwrap().name, "36h11");
    }
}
