use damascene_core::prelude::Rect;
use volumetric_ui_v2::VolumetricUiV2;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let viewport = Rect::new(0.0, 0.0, 1280.0, 800.0);
    volumetric_ui_v2::host::run("Volumetric UI v2", viewport, VolumetricUiV2::default())
}

// Trunk still links a `main`, but the browser entry is the
// `#[wasm_bindgen(start)]` function below.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    // The Rect is the pre-layout fallback; the shell sizes the surface from
    // the canvas's CSS box once it measures it.
    let viewport = Rect::new(0.0, 0.0, 1280.0, 800.0);
    volumetric_ui_v2::web_host::run("Volumetric UI v2", viewport, VolumetricUiV2::default());
}
