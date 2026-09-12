// Simplified upper mounting parts, in metres in the measured local frame.
// Routed measurements and explicitly assumed thickness/bend locations.
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;

override part: float = 0; // @param key="part"
override thickness: float = 0.0025; // @param key="assumed_thickness"
override xmin: float = -0.123; // @param key="cross_xmin"
override xmax: float = 0.119; // @param key="cross_xmax"
override ymin: float = -0.011; // @param key="cross_ymin"
override ymax: float = 0.016; // @param key="cross_ymax"
override bend_start: float = 0.083; // @param key="assumed_bend_start"
override bend_end: float = 0.094; // @param key="assumed_bend_end"
override ax: float = 0.107; // @param key="cross_a_x"
override ay: float = 0.0; // @param key="cross_a_y"
override az: float = 0.0035; // @param key="cross_a_z"
override bx: float = -0.107; // @param key="cross_b_x"
override by: float = 0.0; // @param key="cross_b_y"
override bz: float = 0.0035; // @param key="cross_b_z"
override aix: float = 0.055; // @param key="cross_a_inner_x"
override aiy: float = 0.0; // @param key="cross_a_inner_y"
override bix: float = -0.055; // @param key="cross_b_inner_x"
override biy: float = 0.0; // @param key="cross_b_inner_y"
override cross_slot_length: float = 0.015; // @param key="cross_slot_length"
override cross_slot_width: float = 0.009; // @param key="cross_slot_width"
override cross_hole_diameter: float = 0.008; // @param key="cross_hole_diameter"
override rail_start: float = 0.014; // @param key="assumed_rail_start"
override rail_end: float = 0.243; // @param key="rail_end"
override rail_front_halfwidth: float = 0.026; // @param key="rail_front_halfwidth"
override rail_back_halfwidth: float = 0.0225; // @param key="rail_back_halfwidth"
override rax: float = 0.015; // @param key="rail_a_x"
override ray: float = 0.175; // @param key="rail_a_y"
override rbx: float = -0.015; // @param key="rail_b_x"
override rby: float = 0.175; // @param key="rail_b_y"
override rail_slot_length: float = 0.018; // @param key="rail_slot_length"
override rail_slot_width: float = 0.008; // @param key="rail_slot_width"

fn slot(p: vec2d, total_length: float, width: float) -> bool {
    let q = vec2d(max(abs(p.x) - (total_length - width) / 2.0, 0.0), p.y);
    return dot(q, q) <= width * width / 4.0;
}

fn scene(p: vec3d) -> bool {
    if part < 0.5 {
        if p.x < xmin || p.x > xmax || p.y < ymin || p.y > ymax { return false; }
        let rise = select(bz, az, p.x > 0.0);
        let top = rise * clamp((abs(p.x) - bend_start) / (bend_end - bend_start), 0.0, 1.0);
        if p.z > top || p.z < top - thickness { return false; }
        return !slot(p.xy - vec2d(ax, ay), cross_slot_length, cross_slot_width)
            && !slot(p.xy - vec2d(bx, by), cross_slot_length, cross_slot_width)
            && !slot(p.xy - vec2d(aix, aiy), cross_hole_diameter, cross_hole_diameter)
            && !slot(p.xy - vec2d(bix, biy), cross_hole_diameter, cross_hole_diameter);
    }
    if p.y < rail_start || p.y > rail_end || p.z > 0.0 || p.z < -thickness { return false; }
    let fraction = (p.y - rail_start) / (rail_end - rail_start);
    let halfwidth = mix(rail_front_halfwidth, rail_back_halfwidth, fraction);
    return abs(p.x) <= halfwidth
        && !slot(p.yx - vec2d(ray, rax), rail_slot_length, rail_slot_width)
        && !slot(p.yx - vec2d(rby, rbx), rail_slot_length, rail_slot_width);
}

fn bounds_min() -> vec3d {
    if part < 0.5 { return vec3d(xmin, ymin, -thickness); }
    return vec3d(-max(rail_front_halfwidth, rail_back_halfwidth), rail_start, -thickness);
}

fn bounds_max() -> vec3d {
    if part < 0.5 { return vec3d(xmax, ymax, max(az, bz)); }
    return vec3d(-bounds_min().x, rail_end, 0.0);
}
