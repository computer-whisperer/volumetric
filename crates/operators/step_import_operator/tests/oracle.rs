//! Classification oracle: compare our ray-parity classifier against
//! OCCT's exact point classifier (`bclassify` in DRAWEXE) on the
//! representative board file.
//!
//! Runs only when both DRAWEXE and board.step are available (developer
//! machines); skips silently elsewhere. Two DRAWEXE passes: one to
//! collect per-solid bounding boxes (to prune the point x solid classify
//! matrix), one to classify. Every verdict is exact on both sides —
//! disagreements are bugs, except points OCCT calls ON (surface
//! contact), which are skipped.

use std::io::Write as _;
use std::process::{Command, Stdio};
use step_import_operator::{StepConfig, import};

const POINT_COUNT: usize = 2000;

fn drawexe_available() -> bool {
    // -b = batch mode: no viewer window, exits on stdin EOF.
    let Ok(mut child) = Command::new("DRAWEXE")
        .arg("-b")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    let _ = child.stdin.take().unwrap().write_all(b"exit\n");
    child.wait().is_ok()
}

fn board_path() -> String {
    let raw = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../board.step");
    std::fs::canonicalize(raw)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| raw.to_string())
}

/// Run a DRAW script (an `exit` is appended — without it, batch DRAWEXE
/// can drop buffered stdout at EOF) and return stdout.
fn drawexe(script: &str) -> String {
    let mut child = Command::new("DRAWEXE")
        .arg("-b")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn DRAWEXE");
    let mut stdin = child.stdin.take().unwrap();
    stdin
        .write_all(script.as_bytes())
        .and_then(|_| stdin.write_all(b"\nexit\n"))
        .expect("write script");
    drop(stdin);
    let out = child.wait_with_output().expect("DRAWEXE run");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Strip `Draw[n]> ` prompt prefixes — in batch mode prompts share
/// lines with command output.
fn strip_prompts(line: &str) -> &str {
    let mut l = line;
    while l.starts_with("Draw[") {
        match l.find("> ") {
            Some(i) => l = &l[i + 2..],
            None => break,
        }
    }
    l
}

/// Deterministic point sequence (LCG) inside `bbox` inflated by `pad`.
fn sample_points(bbox: [f64; 6], pad: f64, count: usize) -> Vec<[f64; 3]> {
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            core::array::from_fn(|axis| {
                let lo = bbox[axis * 2] - pad;
                let hi = bbox[axis * 2 + 1] + pad;
                lo + (hi - lo) * next()
            })
        })
        .collect()
}

#[test]
fn agrees_with_occt_point_classification() {
    let Ok(text) = std::fs::read_to_string(board_path()) else {
        eprintln!("board.step not present; skipping oracle");
        return;
    };
    if !drawexe_available() {
        eprintln!("DRAWEXE not available; skipping oracle");
        return;
    }

    // Our side.
    let model = import(&text, &StepConfig::default()).expect("import");
    let payload = brep_core::payload::build_payload(&model).expect("payload");
    let view = brep_core::payload::PayloadView::new(&payload).unwrap();
    let bounds = view.bounds();

    // Pass 1: OCCT solid list and bounding boxes.
    let script = format!(
        "pload ALL\nReadStep D {}\nXGetOneShape a D\nputs \"NSOLIDS [llength [explode a SO]]\"\n",
        board_path()
    );
    let out = drawexe(&script);
    let n_solids: usize = out
        .lines()
        .find_map(|l| strip_prompts(l).strip_prefix("NSOLIDS "))
        .unwrap_or_else(|| panic!("no NSOLIDS in DRAWEXE output: {:?}", &out[..out.len().min(800)]))
        .trim()
        .parse()
        .expect("parse solid count");
    assert!(n_solids > 100, "expected ~121 solids, got {n_solids}");

    let mut script = format!(
        "pload ALL\nReadStep D {}\nXGetOneShape a D\nexplode a SO\n",
        board_path()
    );
    for i in 1..=n_solids {
        script.push_str(&format!("puts \"BOX {i}\"\nbounding a_{i}\n"));
    }
    let out = drawexe(&script);
    let mut boxes: Vec<[f64; 6]> = Vec::with_capacity(n_solids);
    let mut lines = out.lines().map(strip_prompts);
    while let Some(line) = lines.next() {
        if line.starts_with("BOX ") {
            let nums: Vec<f64> = lines
                .next()
                .expect("bounding output")
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            assert_eq!(nums.len(), 6, "bounding format: {out:.0?}");
            // DRAW prints xmin ymin zmin xmax ymax zmax.
            boxes.push([nums[0], nums[3], nums[1], nums[4], nums[2], nums[5]]);
        }
    }
    assert_eq!(boxes.len(), n_solids);

    // Pass 2: classify each point against every solid whose (padded)
    // box contains it.
    let points = sample_points(bounds, 1.0, POINT_COUNT);
    let mut script = format!(
        "pload ALL\nReadStep D {}\nXGetOneShape a D\nexplode a SO\n",
        board_path()
    );
    let pad = 1e-6;
    let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(points.len());
    for (pi, p) in points.iter().enumerate() {
        script.push_str(&format!("puts \"PT {pi}\"\n"));
        let cands: Vec<usize> = (0..n_solids)
            .filter(|&i| {
                (0..3).all(|a| {
                    p[a] > boxes[i][a * 2] - pad && p[a] < boxes[i][a * 2 + 1] + pad
                })
            })
            .collect();
        for &i in &cands {
            script.push_str(&format!(
                "point pp {} {} {}\nputs \"CLS\"\nbclassify a_{} pp\n",
                p[0],
                p[1],
                p[2],
                i + 1
            ));
        }
        candidates.push(cands);
    }
    let out = drawexe(&script);

    // Parse the stream: per point, IN from any candidate wins; ON
    // anywhere skips the point.
    let mut verdicts: Vec<Option<bool>> = Vec::with_capacity(points.len());
    let mut current: Option<(usize, bool, bool)> = None; // (idx, inside, on)
    for line in out.lines().map(strip_prompts) {
        if let Some(rest) = line.strip_prefix("PT ") {
            if let Some((idx, inside, on)) = current.take() {
                assert_eq!(idx, verdicts.len());
                verdicts.push(if on { None } else { Some(inside) });
            }
            current = Some((rest.trim().parse().unwrap(), false, false));
        } else if line.contains("point is") {
            let (_, inside, on) = current.as_mut().expect("classification before PT");
            if line.contains("is IN") {
                *inside = true;
            } else if line.contains("is ON") {
                *on = true;
            }
        }
    }
    if let Some((idx, inside, on)) = current.take() {
        assert_eq!(idx, verdicts.len());
        verdicts.push(if on { None } else { Some(inside) });
    }
    assert_eq!(verdicts.len(), points.len(), "oracle output truncated");

    // Compare.
    let mut checked = 0usize;
    let mut disagreements = Vec::new();
    for ((p, verdict), cands) in points.iter().zip(&verdicts).zip(&candidates) {
        let Some(expect) = verdict else { continue };
        checked += 1;
        let got = view.is_inside(*p);
        if got != *expect && disagreements.len() < 10 {
            disagreements.push((*p, *expect, got, cands.len()));
        }
    }
    eprintln!("oracle: {checked}/{} points checked", points.len());
    assert!(
        disagreements.is_empty(),
        "{} disagreements with OCCT, first: {:?}",
        disagreements.len(),
        disagreements
    );

    // While everything is loaded: a rough classification throughput
    // number for the meshing-cost conversation.
    let t = std::time::Instant::now();
    let mut acc = 0usize;
    for p in points.iter().cycle().take(20_000) {
        acc += view.is_inside(*p) as usize;
    }
    let dt = t.elapsed();
    eprintln!(
        "throughput: {:.0} samples/s (acc {acc})",
        20_000.0 / dt.as_secs_f64()
    );
}
