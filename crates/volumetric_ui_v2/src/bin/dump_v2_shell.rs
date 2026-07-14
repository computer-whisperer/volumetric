use std::path::PathBuf;

use damascene_core::prelude::{Rect, write_bundle};
use volumetric_ui_v2::{catalog_sheet_bundle, export_modal_bundle, shell_bundle};

fn main() -> std::io::Result<()> {
    let viewport = Rect::new(0.0, 0.0, 1280.0, 800.0);
    // Tall enough that no browse row of the warmed catalog is clipped.
    let sheet_viewport = Rect::new(0.0, 0.0, 720.0, 2000.0);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");
    for (name, bundle) in [
        ("shell", shell_bundle(viewport)),
        ("export_modal", export_modal_bundle(viewport)),
        ("catalog_sheet", catalog_sheet_bundle(sheet_viewport)),
    ] {
        let written = write_bundle(&bundle, &out_dir, name)?;
        for path in written {
            println!("wrote {}", path.display());
        }
        if !bundle.lint.findings.is_empty() {
            eprint!("{}", bundle.lint.text());
        }
    }
    Ok(())
}
