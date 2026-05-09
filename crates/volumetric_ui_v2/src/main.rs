use aetna_core::prelude::Rect;
use volumetric_ui_v2::VolumetricUiV2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let viewport = Rect::new(0.0, 0.0, 1280.0, 800.0);
    aetna_winit_wgpu::run("Volumetric UI v2", viewport, VolumetricUiV2::default())
}
