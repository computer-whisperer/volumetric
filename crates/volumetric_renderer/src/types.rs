//! Core types for the rendering engine.
//!
//! Defines render primitives (mesh, line, point data) and their associated styles.

use bytemuck::{Pod, Zeroable};

use crate::ViewGizmo;

// ============================================================================
// Mesh Types
// ============================================================================

/// A batch of triangle mesh data.
#[derive(Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Option<Vec<u32>>,
}

/// A single mesh vertex with position, normal and color.
/// Padded for GPU alignment (48 bytes total).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
    /// Linear RGBA multiplied into the material base color; white for the
    /// plain untinted look.
    pub color: [f32; 4],
}

/// A surface point drawn as a disc: the geometry a direct cast of a model
/// produces (see `DIRECT_CASTING_PLAN.md`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SurfelVertex {
    pub position: [f32; 3],
    /// Radius of the disc, in the surfels' own units.
    pub radius: f32,
    /// Unit, pointing out of the model.
    pub normal: [f32; 3],
    pub _pad: f32,
    /// Linear RGBA multiplied into the draw's colour and the material
    /// base colour, as a mesh vertex's is; white for the plain look.
    pub color: [f32; 4],
}

impl SurfelVertex {
    /// A white surfel.
    pub fn new(position: [f32; 3], normal: [f32; 3], radius: f32) -> Self {
        Self::colored(position, normal, radius, [1.0; 4])
    }

    pub fn colored(position: [f32; 3], normal: [f32; 3], radius: f32, color: [f32; 4]) -> Self {
        Self {
            position,
            radius,
            normal,
            _pad: 0.0,
            color,
        }
    }
}

/// Surfels to draw, in their own coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfelData {
    pub surfels: Vec<SurfelVertex>,
}

impl MeshVertex {
    /// Create a new mesh vertex with the given position and normal, untinted.
    pub fn new(position: [f32; 3], normal: [f32; 3]) -> Self {
        Self::colored(position, normal, [1.0; 4])
    }

    /// Create a new mesh vertex with an explicit vertex color.
    pub fn colored(position: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> Self {
        Self {
            position,
            _pad0: 0.0,
            normal,
            _pad1: 0.0,
            color,
        }
    }
}

/// Material identifier for meshes: an index into the frame's material
/// table (0..=255), carried through the G-buffer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

/// What a mesh draw is, to whoever asks what is under a pixel: written to
/// the G-buffer and handed back by a pick. The host chooses the
/// numbering; [`ObjectId::NONE`] is the background and any draw the host
/// does not care to tell apart.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ObjectId(pub u32);

impl ObjectId {
    pub const NONE: Self = Self(0);
}

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

/// A Gaussian splat as the renderer draws it: world-space centres and
/// covariances, activated opacities, and spherical-harmonic colour. Built
/// once from a splat value; the renderer sorts and colours it per view.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SplatData {
    /// Flat surfels (2DGS: each pixel's ray is intersected with the
    /// surfel's plane) rather than 3D Gaussians (EWA projection).
    pub surfels: bool,
    /// World centres, one per primitive.
    pub positions: Vec<[f32; 3]>,
    /// The three scaled local axes of each primitive in world coordinates
    /// (`R · diag(s)`, column by column, xyz each): the covariance is their
    /// outer-product sum. A surfel's third axis is zero.
    pub axes: Vec<[f32; 9]>,
    /// Opacity per primitive in `[0, 1]`.
    pub opacities: Vec<f32>,
    /// Spherical-harmonic degree of the colour, 0 to 3.
    pub sh_degree: u32,
    /// Colour coefficients per primitive: the three degree-0 values, then
    /// the `(d + 1)² − 1` higher coefficients of red, then green, then
    /// blue.
    pub sh: Vec<f32>,
}

impl SplatData {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The world covariance of primitive `i` as `xx, xy, xz, yy, yz, zz`.
    pub fn covariance(&self, i: usize) -> [f32; 6] {
        let a = &self.axes[i];
        let dot = |r: usize, c: usize| (0..3).map(|k| a[3 * k + r] * a[3 * k + c]).sum::<f32>();
        [
            dot(0, 0),
            dot(0, 1),
            dot(0, 2),
            dot(1, 1),
            dot(1, 2),
            dot(2, 2),
        ]
    }

    /// Coefficients per primitive in `sh`.
    pub fn sh_per_point(&self) -> usize {
        3 * ((self.sh_degree + 1) * (self.sh_degree + 1)) as usize
    }

    /// The largest axis of the positions' bounding box, for deciding when a
    /// camera has moved enough to re-sort.
    pub fn extent(&self) -> f32 {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        if self.positions.is_empty() {
            0.0
        } else {
            (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f32::max)
        }
    }
}

/// How a splat is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct SplatStyle {
    /// Standard deviations the footprint reaches (3 covers 99 %).
    pub kernel_radius: f32,
    /// Multiplier on every primitive's opacity, for fading a splat against
    /// a photograph.
    pub opacity_scale: f32,
}

impl Default for SplatStyle {
    fn default() -> Self {
        Self {
            kernel_radius: 3.0,
            opacity_scale: 1.0,
        }
    }
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

/// The world plane the grid lies in, through the origin.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum GridPlane {
    /// z = 0: the ground of a Z-up world.
    #[default]
    XY,
    /// y = 0.
    XZ,
    /// x = 0.
    YZ,
}

impl GridPlane {
    /// The world axes (0 = X, 1 = Y, 2 = Z) the plane spans, then its
    /// normal.
    pub fn axes(self) -> [usize; 3] {
        match self {
            Self::XY => [0, 1, 2],
            Self::XZ => [0, 2, 1],
            Self::YZ => [1, 2, 0],
        }
    }
}

/// How far apart the grid's lines are.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GridSpacing {
    /// A power of ten chosen each frame so the finest cells stay a
    /// readable size on screen where the view is looking: `focus_depth`
    /// in front of the eye (the orbit camera's distance). Finer levels
    /// fade in and out as the view zooms.
    ///
    /// `min_cell_px` is the smallest a minor cell is drawn, in target
    /// pixels; a decade finer would be smaller, so the grid steps up.
    /// [`GridSpacing::MIN_CELL_PX`] per logical pixel is the usual value.
    Auto { focus_depth: f32, min_cell_px: f32 },
    /// Minor lines this far apart (world units), a major line every ten.
    Fixed(f32),
}

impl GridSpacing {
    /// The usual smallest minor cell, in logical pixels: minor cells are
    /// then this to ten times this across.
    pub const MIN_CELL_PX: f32 = 16.0;
}

/// The ground grid and world axis lines: drawn analytically per pixel, so
/// they have no extent and no geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct GridSettings {
    pub visible: bool,
    pub plane: GridPlane,
    pub spacing: GridSpacing,
    /// Draw the world axes: the two in the plane as coloured grid lines,
    /// the third as a line through the origin.
    pub axes: bool,
    /// Multiplies the opacity of every line.
    pub opacity: f32,
    /// Linear RGB of the grid's lines.
    pub line_color: [f32; 3],
    /// Linear RGB of the world X, Y and Z axis lines.
    pub axis_colors: [[f32; 3]; 3],
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            visible: true,
            plane: GridPlane::XY,
            spacing: GridSpacing::Fixed(1.0),
            axes: true,
            opacity: 1.0,
            line_color: [0.5, 0.5, 0.5],
            axis_colors: AXIS_COLORS,
        }
    }
}

/// Linear RGB of the world X (red), Y (green) and Z (blue) axes, shared
/// by the grid's axis lines and the view gizmo.
pub const AXIS_COLORS: [[f32; 3]; 3] = [[0.85, 0.2, 0.22], [0.3, 0.65, 0.15], [0.2, 0.4, 0.9]];

// ============================================================================
// Shading
// ============================================================================

/// One directional light of the rig.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Light {
    /// The direction the light shines from, in the camera's frame: x to
    /// the right of the frame, y up it, z toward the viewer. The lights
    /// turn with the view, so no face goes dark as it turns.
    pub direction: [f32; 3],
    /// Linear RGB, intensity included.
    pub color: [f32; 3],
}

/// The lights a frame is lit by: the resolve pass's whole lighting model
/// is driven by this value, so a different look is different data.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LightingRig {
    /// Key, fill and rim.
    pub lights: [Light; 3],
    /// Ambient light on a surface facing straight up the world Z axis.
    pub sky: [f32; 3],
    /// Ambient light on a surface facing straight down it.
    pub ground: [f32; 3],
}

impl Default for LightingRig {
    /// A neutral studio: a key from the upper left, a weaker fill from the
    /// right, a rim from behind, and a soft sky.
    fn default() -> Self {
        Self {
            lights: [
                Light {
                    direction: [-0.5, 0.6, 0.65],
                    color: [0.62; 3],
                },
                Light {
                    direction: [0.7, -0.1, 0.5],
                    color: [0.22; 3],
                },
                Light {
                    direction: [0.2, 0.6, -0.75],
                    color: [0.25; 3],
                },
            ],
            sky: [0.30; 3],
            ground: [0.16; 3],
        }
    }
}

/// The lighting rigs offered by name.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum LightingPreset {
    /// Key, fill and rim under a soft sky: the default.
    #[default]
    Studio,
    /// Mostly ambient, with little difference between faces: for reading
    /// colours and edge lines rather than form.
    Flat,
    /// One light at the camera: every face is lit by how squarely it
    /// faces the viewer.
    Headlight,
}

impl LightingPreset {
    pub const ALL: [Self; 3] = [Self::Studio, Self::Flat, Self::Headlight];

    pub fn name(self) -> &'static str {
        match self {
            Self::Studio => "studio",
            Self::Flat => "flat",
            Self::Headlight => "headlight",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.name() == name)
    }

    pub fn rig(self) -> LightingRig {
        let light = |direction, intensity: f32| Light {
            direction,
            color: [intensity; 3],
        };
        let off = light([0.0, 0.0, 1.0], 0.0);
        match self {
            Self::Studio => LightingRig::default(),
            Self::Flat => LightingRig {
                lights: [light([-0.3, 0.4, 0.85], 0.22), off, off],
                sky: [0.66; 3],
                ground: [0.52; 3],
            },
            Self::Headlight => LightingRig {
                lights: [light([0.0, 0.0, 1.0], 0.72), off, off],
                sky: [0.2; 3],
                ground: [0.2; 3],
            },
        }
    }
}

/// How a surface answers the light. [`MaterialId`] indexes the frame's
/// table of these.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Material {
    /// Linear RGB multiplied into the vertex colour.
    pub base_tint: [f32; 3],
    /// 0 is a mirror-tight highlight, 1 a broad dull one.
    pub roughness: f32,
    /// The strength of the highlight; 0 is matte.
    pub specular: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_tint: [0.85, 0.9, 1.0],
            roughness: 0.45,
            specular: 0.18,
        }
    }
}

/// The most materials a frame's table holds; ids past the table's end
/// take its first entry.
pub const MAX_MATERIALS: usize = 16;

/// Ambient occlusion: the darkening of creases and contact from nearby
/// geometry, from the G-buffer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AoSettings {
    pub enabled: bool,
    /// How far occluders are looked for, as a fraction of the diagonal of
    /// the frame's meshes: the same look at any part size.
    pub radius: f32,
    /// Exponent on the result; above 1 darkens.
    pub strength: f32,
}

impl Default for AoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 0.06,
            strength: 1.0,
        }
    }
}

/// Edge lines from discontinuities in the G-buffer: silhouettes, creases
/// and the boundaries between objects, one pixel wide.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EdgeSettings {
    pub enabled: bool,
    /// Faces meeting at more than this angle (radians) get a line.
    pub crease_angle: f32,
    /// Linear RGB of the lines.
    pub color: [f32; 3],
    /// How opaque the lines are.
    pub opacity: f32,
}

impl Default for EdgeSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            crease_angle: 0.6,
            color: [0.02; 3],
            opacity: 0.55,
        }
    }
}

// ============================================================================
// Render Settings
// ============================================================================

/// Global render settings.
#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub lighting: LightingRig,
    /// The material table; at most [`MAX_MATERIALS`] entries are used.
    pub materials: Vec<Material>,
    pub ao: AoSettings,
    pub edges: EdgeSettings,
    /// Grid settings
    pub grid: GridSettings,
    /// The view gizmo, where the host wants one drawn.
    pub gizmo: Option<ViewGizmo>,
    /// Background color
    pub background_color: [f32; 4],
    /// Smooth the lit scene's stair-stepped edges (FXAA). Lines, the
    /// grid and the gizmo are anti-aliased either way.
    pub antialiasing: bool,
    /// How many target pixels across one pixel of the delivered picture
    /// is: 1 unless the frame is drawn larger to be scaled down
    /// (supersampling). Every width given in pixels (lines, points, the
    /// grid's lines, edge lines) is multiplied by it, so the picture
    /// looks the same and only gets smoother.
    pub pixel_scale: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            lighting: LightingRig::default(),
            materials: vec![Material::default()],
            ao: AoSettings::default(),
            edges: EdgeSettings::default(),
            grid: GridSettings::default(),
            gizmo: None,
            background_color: [0.1, 0.1, 0.1, 1.0],
            antialiasing: true,
            pixel_scale: 1.0,
        }
    }
}
