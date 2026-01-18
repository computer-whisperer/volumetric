//! Core types for the rendering engine.
//!
//! Defines render primitives (mesh, line, point data) and their associated styles.

use bytemuck::{Pod, Zeroable};

// ============================================================================
// Mesh Types
// ============================================================================

/// A batch of triangle mesh data.
#[derive(Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Option<Vec<u32>>,
}

/// A single mesh vertex with position and normal.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Material identifier for meshes.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

// ============================================================================
// Line Types
// ============================================================================

/// A batch of line segments.
#[derive(Clone, Default)]
pub struct LineData {
    pub segments: Vec<LineSegment>,
}

/// A single line segment with start/end points and color.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineSegment {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4], // RGBA with alpha
}

/// GPU instance data for line rendering (includes width).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineInstance {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4],
    pub width: f32,
    pub _pad: [f32; 3],
}

impl LineInstance {
    /// Create a LineInstance from a LineSegment with the given width.
    pub fn from_segment(segment: &LineSegment, width: f32) -> Self {
        Self {
            start: segment.start,
            end: segment.end,
            color: segment.color,
            width,
            _pad: [0.0; 3],
        }
    }
}

/// Line rendering style.
#[derive(Clone, Debug)]
pub struct LineStyle {
    /// Line width (interpretation depends on width_mode)
    pub width: f32,
    /// How width is interpreted
    pub width_mode: WidthMode,
    /// Line pattern (solid, dashed, etc.)
    pub pattern: LinePattern,
    /// Depth testing mode
    pub depth_mode: DepthMode,
}

impl Default for LineStyle {
    fn default() -> Self {
        Self {
            width: 2.0,
            width_mode: WidthMode::ScreenSpace,
            pattern: LinePattern::Solid,
            depth_mode: DepthMode::Normal,
        }
    }
}

/// How line/point width is interpreted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum WidthMode {
    /// Width in screen pixels (constant regardless of distance)
    #[default]
    ScreenSpace,
    /// Width in world units (appears smaller at distance)
    WorldSpace,
}

/// Line pattern for dashed/dotted lines.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum LinePattern {
    #[default]
    Solid,
    Dashed {
        dash_length: f32,
        gap_length: f32,
    },
    Dotted {
        spacing: f32,
    },
}

/// Depth testing mode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DepthMode {
    /// Normal depth testing against scene
    #[default]
    Normal,
    /// Render on top of everything (for annotations/overlays)
    Overlay,
}

// ============================================================================
// Point Types
// ============================================================================

/// A batch of points.
#[derive(Clone, Default)]
pub struct PointData {
    pub points: Vec<PointInstance>,
}

/// A single point instance with position and color.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub color: [f32; 4], // RGBA with alpha
}

/// Point rendering style.
#[derive(Clone, Debug)]
pub struct PointStyle {
    /// Point size (interpretation depends on size_mode)
    pub size: f32,
    /// How size is interpreted
    pub size_mode: WidthMode,
    /// Point shape
    pub shape: PointShape,
    /// Depth testing mode
    pub depth_mode: DepthMode,
}

impl Default for PointStyle {
    fn default() -> Self {
        Self {
            size: 4.0,
            size_mode: WidthMode::ScreenSpace,
            shape: PointShape::Circle,
            depth_mode: DepthMode::Normal,
        }
    }
}

/// Point shape for rendering.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum PointShape {
    #[default]
    Circle,
    Square,
    Diamond,
}

impl PointShape {
    /// Convert to shader uniform value.
    pub fn to_shader_value(self) -> u32 {
        match self {
            PointShape::Circle => 0,
            PointShape::Square => 1,
            PointShape::Diamond => 2,
        }
    }
}

// ============================================================================
// Grid Types
// ============================================================================

/// Which grid planes to display.
#[derive(Copy, Clone, Debug, Default)]
pub struct GridPlanes {
    /// XY plane (z = 0)
    pub xy: bool,
    /// XZ plane (y = 0, ground plane)
    pub xz: bool,
    /// YZ plane (x = 0)
    pub yz: bool,
}

impl GridPlanes {
    pub const NONE: Self = Self {
        xy: false,
        xz: false,
        yz: false,
    };
    pub const XY: Self = Self {
        xy: true,
        xz: false,
        yz: false,
    };
    pub const XZ: Self = Self {
        xy: false,
        xz: true,
        yz: false,
    };
    pub const YZ: Self = Self {
        xy: false,
        xz: false,
        yz: true,
    };
    pub const ALL: Self = Self {
        xy: true,
        xz: true,
        yz: true,
    };
}

/// Grid rendering settings.
#[derive(Clone, Debug)]
pub struct GridSettings {
    /// Which planes to display
    pub planes: GridPlanes,
    /// Spacing between grid lines (world units)
    pub spacing: f32,
    /// Extent of grid from origin (world units)
    pub extent: f32,
    /// Primary line color (every N lines)
    pub major_color: [f32; 4],
    /// Secondary line color
    pub minor_color: [f32; 4],
    /// Lines between major lines
    pub subdivisions: u32,
    /// Color for X axis line
    pub x_axis_color: [f32; 4],
    /// Color for Z axis line (on XZ plane)
    pub z_axis_color: [f32; 4],
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            planes: GridPlanes::XZ, // Ground plane only
            spacing: 1.0,
            extent: 10.0,
            major_color: [0.5, 0.5, 0.5, 0.8],
            minor_color: [0.3, 0.3, 0.3, 0.4],
            subdivisions: 5,
            x_axis_color: [0.8, 0.2, 0.2, 1.0], // Red
            z_axis_color: [0.2, 0.2, 0.8, 1.0], // Blue
        }
    }
}

impl GridSettings {
    /// Generate line segments for the grid.
    pub fn generate_lines(&self) -> Vec<LineSegment> {
        let mut lines = Vec::new();
        let n = (self.extent / self.spacing) as i32;

        if self.planes.xz {
            // Ground plane (y = 0)
            for i in -n..=n {
                let pos = i as f32 * self.spacing;
                let is_major = i % self.subdivisions as i32 == 0;
                let is_axis = i == 0;

                // Line parallel to X axis (varying Z)
                let color_z = if is_axis {
                    self.x_axis_color
                } else if is_major {
                    self.major_color
                } else {
                    self.minor_color
                };
                lines.push(LineSegment {
                    start: [-self.extent, 0.0, pos],
                    end: [self.extent, 0.0, pos],
                    color: color_z,
                });

                // Line parallel to Z axis (varying X)
                let color_x = if is_axis {
                    self.z_axis_color
                } else if is_major {
                    self.major_color
                } else {
                    self.minor_color
                };
                lines.push(LineSegment {
                    start: [pos, 0.0, -self.extent],
                    end: [pos, 0.0, self.extent],
                    color: color_x,
                });
            }
        }

        if self.planes.xy {
            // XY plane (z = 0)
            for i in -n..=n {
                let pos = i as f32 * self.spacing;
                let is_major = i % self.subdivisions as i32 == 0;
                let color = if is_major {
                    self.major_color
                } else {
                    self.minor_color
                };

                // Line parallel to X
                lines.push(LineSegment {
                    start: [-self.extent, pos, 0.0],
                    end: [self.extent, pos, 0.0],
                    color,
                });
                // Line parallel to Y
                lines.push(LineSegment {
                    start: [pos, -self.extent, 0.0],
                    end: [pos, self.extent, 0.0],
                    color,
                });
            }
        }

        if self.planes.yz {
            // YZ plane (x = 0)
            for i in -n..=n {
                let pos = i as f32 * self.spacing;
                let is_major = i % self.subdivisions as i32 == 0;
                let color = if is_major {
                    self.major_color
                } else {
                    self.minor_color
                };

                // Line parallel to Y
                lines.push(LineSegment {
                    start: [0.0, -self.extent, pos],
                    end: [0.0, self.extent, pos],
                    color,
                });
                // Line parallel to Z
                lines.push(LineSegment {
                    start: [0.0, pos, -self.extent],
                    end: [0.0, pos, self.extent],
                    color,
                });
            }
        }

        lines
    }
}

// ============================================================================
// Axis Indicator
// ============================================================================

/// Configuration for the axis indicator widget.
#[derive(Clone, Debug)]
pub struct AxisIndicator {
    /// Screen position (normalized, 0-1, from bottom-left)
    pub position: [f32; 2],
    /// Size in pixels
    pub size: f32,
    /// X axis color (red)
    pub x_color: [f32; 4],
    /// Y axis color (green)
    pub y_color: [f32; 4],
    /// Z axis color (blue)
    pub z_color: [f32; 4],
    /// Whether to show axis labels
    pub show_labels: bool,
}

impl Default for AxisIndicator {
    fn default() -> Self {
        Self {
            position: [0.92, 0.08], // Bottom-right corner
            size: 60.0,
            x_color: [1.0, 0.2, 0.2, 1.0], // Red
            y_color: [0.2, 1.0, 0.2, 1.0], // Green
            z_color: [0.2, 0.2, 1.0, 1.0], // Blue
            show_labels: true,
        }
    }
}

// ============================================================================
// Render Settings
// ============================================================================

/// Global render settings.
#[derive(Clone, Debug)]
pub struct RenderSettings {
    /// Enable SSAO (screen-space ambient occlusion)
    pub ssao_enabled: bool,
    /// SSAO sample count (8 for web, 16 for native)
    pub ssao_samples: u32,
    /// SSAO radius in world units
    pub ssao_radius: f32,
    /// SSAO bias
    pub ssao_bias: f32,
    /// SSAO strength
    pub ssao_strength: f32,
    /// Grid settings
    pub grid: GridSettings,
    /// Axis indicator settings
    pub axis_indicator: AxisIndicator,
    /// Show axis indicator
    pub show_axis_indicator: bool,
    /// Background color
    pub background_color: [f32; 4],
}

impl Default for RenderSettings {
    fn default() -> Self {
        let is_web = cfg!(target_arch = "wasm32");
        Self {
            ssao_enabled: true,
            ssao_samples: if is_web { 8 } else { 16 },
            ssao_radius: 0.5,
            ssao_bias: 0.025,
            ssao_strength: 1.0,
            grid: GridSettings::default(),
            axis_indicator: AxisIndicator::default(),
            show_axis_indicator: true,
            background_color: [0.1, 0.1, 0.1, 1.0],
        }
    }
}
