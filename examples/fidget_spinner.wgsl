// Parameterized three-lobed fidget spinner with four bearing seats.
// WGSL model-dialect port of examples/fidget_spinner.lua. All dimensions
// are metres. Changing the overrides in the first block updates the solid,
// its clearances, and its declared bounds together.

alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;

override bearing_outer_diameter: float = 0.022; // @param key="spinner.bearing_outer_diameter" min=0.005 max=0.05
override radial_clearance: float = 0.00015; // @param key="spinner.radial_clearance" min=0.0 max=0.002
override bearing_pitch: float = 0.035; // @param key="spinner.bearing_pitch" min=0.02 max=0.1
override body_thickness: float = 0.0072; // @param key="spinner.body_thickness" min=0.001 max=0.03

override center_outer_radius: float = 0.0165; // @param key="spinner.center_outer_radius" min=0.005 max=0.05
override lobe_outer_radius: float = 0.0155; // @param key="spinner.lobe_outer_radius" min=0.005 max=0.05
override web_radius: float = 0.0105; // @param key="spinner.web_radius" min=0.002 max=0.03
const bounds_margin: float = 0.001;

// Derived values may reference the overrides above (WGSL `const` cannot).
override bearing_radius: float = bearing_outer_diameter / 2.0 + radial_clearance;
override half_thickness: float = body_thickness / 2.0;
override half_pitch: float = bearing_pitch / 2.0;
override lobe_y: float = bearing_pitch * sqrt(3.0) / 2.0;

fn disk(p: vec2d, center: vec2d, radius: float) -> bool {
    return length(p - center) <= radius;
}

fn capsule(p: vec2d, start: vec2d, end: vec2d, radius: float) -> bool {
    let segment = end - start;
    let along = clamp(dot(p - start, segment) / dot(segment, segment), 0.0, 1.0);
    return length(p - (start + segment * along)) <= radius;
}

fn scene(p: vec3d) -> bool {
    let right = vec2d(bearing_pitch, 0.0);
    let upper_left = vec2d(-half_pitch, lobe_y);
    let lower_left = vec2d(-half_pitch, -lobe_y);
    let origin = vec2d(0.0, 0.0);
    let q = p.xy;

    let body = disk(q, origin, center_outer_radius)
        || disk(q, right, lobe_outer_radius)
        || disk(q, upper_left, lobe_outer_radius)
        || disk(q, lower_left, lobe_outer_radius)
        || capsule(q, origin, right, web_radius)
        || capsule(q, origin, upper_left, web_radius)
        || capsule(q, origin, lower_left, web_radius);

    let bearing_hole = disk(q, origin, bearing_radius)
        || disk(q, right, bearing_radius)
        || disk(q, upper_left, bearing_radius)
        || disk(q, lower_left, bearing_radius);

    return abs(p.z) <= half_thickness && body && !bearing_hole;
}

fn bounds_min() -> vec3d {
    return vec3d(
        -(bearing_pitch + lobe_outer_radius + bounds_margin),
        -(lobe_y + lobe_outer_radius + bounds_margin),
        -half_thickness,
    );
}

fn bounds_max() -> vec3d {
    return vec3d(
        bearing_pitch + lobe_outer_radius + bounds_margin,
        lobe_y + lobe_outer_radius + bounds_margin,
        half_thickness,
    );
}
