use std::path::PathBuf;

use damascene_core::prelude::{Rect, write_bundle};
use volumetric_ui_v2::shell_bundle;

fn main() -> std::io::Result<()> {
    let viewport = Rect::new(0.0, 0.0, 1280.0, 800.0);
    let bundle = shell_bundle(viewport);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");
    let written = write_bundle(&bundle, &out_dir, "shell")?;
    for path in written {
        println!("wrote {}", path.display());
    }
    if !bundle.lint.findings.is_empty() {
        eprint!("{}", bundle.lint.text());
    }
    Ok(())
}
