// Parametric, open-top Raspberry Pi 4 Model B tray.
// WGSL model-dialect port of examples/raspberry_pi_4_tray.lua.
//
// All dimensions are metres. The board envelope, 3 mm corner radius, and
// 58 x 49 mm mounting-hole pattern follow Raspberry Pi's official mechanical
// drawing. Connector apertures include printable clearance and are
// intentionally generous: verify them against the exact board revision and
// printer before use.
//
// Official reference:
// https://pip-assets.raspberrypi.com/categories/545-raspberry-pi-4-model-b/documents/RP-008343-DS-1-raspberry-pi-4-mechanical-drawing.pdf

alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;

override board_length: float = 0.085; // @param key="rpi4.board_length" min=0.08 max=0.09
override board_width: float = 0.056; // @param key="rpi4.board_width" min=0.05 max=0.06
override board_clearance: float = 0.0005; // @param key="case.board_clearance" min=0.0001 max=0.002
override wall_thickness: float = 0.002; // @param key="case.wall_thickness" min=0.0012 max=0.005
override floor_thickness: float = 0.002; // @param key="case.floor_thickness" min=0.0012 max=0.005
override case_height: float = 0.023; // @param key="case.height" min=0.012 max=0.04
override case_corner_radius: float = 0.005; // @param key="case.corner_radius" min=0.003 max=0.01

override standoff_height: float = 0.003; // @param key="case.standoff_height" min=0.001 max=0.008
override post_outer_diameter: float = 0.006; // @param key="case.post_outer_diameter" min=0.004 max=0.01
override screw_hole_diameter: float = 0.0029; // @param key="case.screw_hole_diameter" min=0.0024 max=0.004

override port_clearance: float = 0.0006; // @param key="case.port_clearance" min=0.0002 max=0.002
override vent_width: float = 0.0025; // @param key="case.vent_width" min=0.0012 max=0.005
override vent_length: float = 0.036; // @param key="case.vent_length" min=0.015 max=0.055
override vent_pitch: float = 0.008; // @param key="case.vent_pitch" min=0.005 max=0.012

override inner_min_x: float = -board_clearance;
override inner_max_x: float = board_length + board_clearance;
override inner_min_y: float = -board_clearance;
override inner_max_y: float = board_width + board_clearance;
override outer_min_x: float = inner_min_x - wall_thickness;
override outer_max_x: float = inner_max_x + wall_thickness;
override outer_min_y: float = inner_min_y - wall_thickness;
override outer_max_y: float = inner_max_y + wall_thickness;
override inner_corner_radius: float = max(0.0005, case_corner_radius - wall_thickness);

override board_z: float = floor_thickness + standoff_height;
override post_radius: float = post_outer_diameter / 2.0;
override screw_radius: float = screw_hole_diameter / 2.0;

// Official mounting-hole centers relative to the board's lower-left corner:
// a 58 x 49 mm rectangle anchored at (3.5 mm, 3.5 mm).
const mount_anchor = vec2d(0.0035, 0.0035);
const mount_offsets = array<vec2d, 4>(
    vec2d(0.0, 0.0),
    vec2d(0.0, 0.049),
    vec2d(0.058, 0.0),
    vec2d(0.058, 0.049),
);

fn disk(p: vec2d, center: vec2d, radius: float) -> bool {
    return length(p - center) <= radius;
}

fn capsule(p: vec2d, start: vec2d, end: vec2d, radius: float) -> bool {
    let segment = end - start;
    let along = clamp(dot(p - start, segment) / dot(segment, segment), 0.0, 1.0);
    return length(p - (start + segment * along)) <= radius;
}

fn rounded_rectangle(p: vec2d, min_corner: vec2d, max_corner: vec2d, radius: float) -> bool {
    let nearest = clamp(p, min_corner + vec2d(radius), max_corner - vec2d(radius));
    return disk(p, nearest, radius);
}

fn box3(p: vec3d, min_corner: vec3d, max_corner: vec3d) -> bool {
    return all(p >= min_corner) && all(p <= max_corner);
}

fn at_mount_hole(q: vec2d, radius: float) -> bool {
    var hit = false;
    for (var i: i32 = 0; i < 4; i++) {
        hit = hit || disk(q, mount_anchor + mount_offsets[i], radius);
    }
    return hit;
}

fn mounting_post(p: vec3d) -> bool {
    let inside_height = p.z >= floor_thickness && p.z <= board_z;
    return inside_height && at_mount_hole(p.xy, post_radius);
}

fn screw_hole(p: vec3d) -> bool {
    let inside_height = p.z >= 0.0 && p.z <= board_z;
    return inside_height && at_mount_hole(p.xy, screw_radius);
}

fn floor_vent(p: vec3d) -> bool {
    let vent_start_x = (board_length - vent_length) / 2.0;
    let vent_end_x = vent_start_x + vent_length;
    let first_vent_y: float = 0.012;
    var vent = false;
    for (var row: i32 = 0; row < 5; row++) {
        let vent_y = first_vent_y + f64(row) * vent_pitch;
        vent = vent || capsule(
            p.xy,
            vec2d(vent_start_x, vent_y),
            vec2d(vent_end_x, vent_y),
            vent_width / 2.0,
        );
    }
    return p.z >= 0.0 && p.z <= floor_thickness && vent;
}

fn connector_cutout(p: vec3d) -> bool {
    let side_min_x = inner_max_x - port_clearance;
    let side_max_x = outer_max_x + port_clearance;
    let edge_min_y = outer_min_y - port_clearance;
    let edge_max_y = inner_min_y + port_clearance;

    // Right edge: lower/upper USB stacks and Ethernet. These retain narrow
    // wall ribs between apertures instead of removing the whole side.
    let lower_usb = box3(
        p,
        vec3d(side_min_x, 0.001 - port_clearance, board_z - 0.001),
        vec3d(side_max_x, 0.017 + port_clearance, board_z + 0.019),
    );
    let upper_usb = box3(
        p,
        vec3d(side_min_x, 0.018 - port_clearance, board_z - 0.001),
        vec3d(side_max_x, 0.034 + port_clearance, board_z + 0.019),
    );
    let ethernet = box3(
        p,
        vec3d(side_min_x, 0.036 - port_clearance, board_z - 0.001),
        vec3d(side_max_x, board_width + port_clearance, board_z + 0.017),
    );

    // Lower edge: USB-C power, two micro-HDMI ports, and the A/V jack.
    let usb_c = box3(
        p,
        vec3d(0.005 - port_clearance, edge_min_y, board_z - 0.001),
        vec3d(0.016 + port_clearance, edge_max_y, board_z + 0.0045),
    );
    let hdmi_zero = box3(
        p,
        vec3d(0.022 - port_clearance, edge_min_y, board_z - 0.001),
        vec3d(0.030 + port_clearance, edge_max_y, board_z + 0.006),
    );
    let hdmi_one = box3(
        p,
        vec3d(0.035 - port_clearance, edge_min_y, board_z - 0.001),
        vec3d(0.044 + port_clearance, edge_max_y, board_z + 0.006),
    );
    let av_jack = box3(
        p,
        vec3d(0.050 - port_clearance, edge_min_y, board_z - 0.001),
        vec3d(0.058 + port_clearance, edge_max_y, board_z + 0.0095),
    );

    return lower_usb || upper_usb || ethernet || usb_c || hdmi_zero || hdmi_one || av_jack;
}

fn scene(p: vec3d) -> bool {
    let outer = rounded_rectangle(
        p.xy,
        vec2d(outer_min_x, outer_min_y),
        vec2d(outer_max_x, outer_max_y),
        case_corner_radius,
    ) && p.z >= 0.0 && p.z <= case_height;

    let cavity = rounded_rectangle(
        p.xy,
        vec2d(inner_min_x, inner_min_y),
        vec2d(inner_max_x, inner_max_y),
        inner_corner_radius,
    ) && p.z >= floor_thickness && p.z <= case_height;

    let structure = (outer && !cavity) || mounting_post(p);
    let opening = screw_hole(p) || floor_vent(p) || connector_cutout(p);
    return structure && !opening;
}

fn bounds_min() -> vec3d {
    return vec3d(outer_min_x, outer_min_y, 0.0);
}

fn bounds_max() -> vec3d {
    return vec3d(outer_max_x, outer_max_y, case_height);
}
