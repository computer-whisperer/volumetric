// Keychain sleeve for the e-paper tag, in metres in the tag frame: origin on
// the display face at the centre of the outline, x along the length (+x is
// the plain end, -x the end with the hanger slot), z out of the back.
//
// part 0: the tag's measured envelope (for fit checks and photo overlays)
// part 1: the sleeve (rigid: PLA/PETG). The tag slides in from +x, display
//         first against the front lip; a tongue in the back plate carries a
//         detent that drops behind the tag's back face. Keyring lug on the
//         closed -x end.
// part 2: the bumper (TPU). A full surround hugging the tag's envelope: a
//         lip all round the display face and a frame over the back face;
//         the back opening stretches over the tag. Keyring lug at -x on the
//         display side.
//
// measured_* come from measure.py (work/measurements.json); assumed_* are
// design choices, not observations.
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;

override part: float = 1; // @param key="part"

// The tag.
override tag_length: float = 0.0704; // @param key="measured_length"
override tag_width: float = 0.0357; // @param key="measured_width"
override tag_height: float = 0.01366; // @param key="measured_height"
override face_length: float = 0.0695; // @param key="measured_face_length"
override face_width: float = 0.0344; // @param key="measured_face_width"
override back_length: float = 0.0574; // @param key="measured_back_length"
override back_width: float = 0.0310; // @param key="measured_back_width"
override taper_z: float = 0.0035; // @param key="measured_length_taper_z"
override narrow_z: float = 0.0103; // @param key="measured_width_taper_z"
override tag_radius: float = 0.003; // @param key="assumed_tag_corner_radius"

// The sleeve.
override clearance: float = 0.0003; // @param key="assumed_clearance"
override wall: float = 0.0018; // @param key="assumed_wall"
override back_plate: float = 0.0016; // @param key="assumed_back_plate"
override lip: float = 0.0010; // @param key="assumed_lip_overlap"
override lip_thickness: float = 0.0010; // @param key="assumed_lip_thickness"
override edge_radius: float = 0.0008; // @param key="assumed_edge_radius"
override tongue_length: float = 0.014; // @param key="assumed_tongue_length"
override tongue_width: float = 0.016; // @param key="assumed_tongue_width"
override slit: float = 0.0008; // @param key="assumed_slit"
override detent_depth: float = 0.0014; // @param key="assumed_detent_depth"
override detent_gap: float = 0.0010; // @param key="assumed_detent_gap"
override detent_length: float = 0.0030; // @param key="assumed_detent_length"
override lug_reach: float = 0.0095; // @param key="assumed_lug_reach"
override lug_width: float = 0.014; // @param key="assumed_lug_width"
override lug_thickness: float = 0.004; // @param key="assumed_lug_thickness"
override hole_diameter: float = 0.005; // @param key="assumed_hole_diameter"

// The bumper.
override tpu_clearance: float = 0.0; // @param key="assumed_tpu_clearance"
override tpu_wall: float = 0.0015; // @param key="assumed_tpu_wall"
override tpu_front: float = 0.0010; // @param key="assumed_tpu_front_thickness"
override tpu_front_lip: float = 0.0010; // @param key="assumed_tpu_front_overlap"
override tpu_back: float = 0.0012; // @param key="assumed_tpu_back_thickness"
override tpu_back_lip: float = 0.0020; // @param key="assumed_tpu_back_overlap"
override tpu_chamfer: float = 0.0006; // @param key="assumed_tpu_chamfer"
override tpu_lug_thickness: float = 0.0035; // @param key="assumed_tpu_lug_thickness"

// Signed distance to a rectangle of half-size h with corner radius r.
fn rounded_rect(p: vec2d, h: vec2d, r: float) -> float {
    let q = abs(p) - h + vec2d(r, r);
    return length(max(q, vec2d(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Signed distance to a box of half-size h with every edge rounded by r.
fn round_box(p: vec3d, h: vec3d, r: float) -> float {
    let q = abs(p) - h + vec3d(r, r, r);
    return length(max(q, vec3d(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// The tag's half-size at height z: full outline to taper_z (length) and
// narrow_z (width), then straight to the back face's outline.
fn tag_half(z: float) -> vec2d {
    let tl = clamp((z - taper_z) / (tag_height - taper_z), 0.0, 1.0);
    let tw = clamp((z - narrow_z) / (tag_height - narrow_z), 0.0, 1.0);
    return 0.5 * vec2d(mix(tag_length, back_length, tl), mix(tag_width, back_width, tw));
}

fn tag(p: vec3d) -> bool {
    if p.z < 0.0 || p.z > tag_height { return false; }
    return rounded_rect(p.xy, tag_half(p.z), tag_radius) <= 0.0;
}

// The pocket's half-size: the tag's largest outline plus clearance, the
// same at every height (the tag slides in along x).
fn pocket_half() -> vec2d {
    return 0.5 * vec2d(tag_length, tag_width) + vec2d(clearance, clearance);
}

// Inside a rounded rectangle whose +x half runs on to infinity.
fn open_toward_x(p: vec2d, h: vec2d, r: float) -> bool {
    return rounded_rect(p, h, r) <= 0.0 || (p.x > 0.0 && abs(p.y) <= h.y);
}

fn pocket_top() -> float { return tag_height + clearance; }

fn closed_end() -> float { return -pocket_half().x - wall; }

fn open_end() -> float { return pocket_half().x; }

fn sleeve(p: vec3d) -> bool {
    let ph = pocket_half();
    let z0 = -lip_thickness;
    let z1 = pocket_top() + back_plate;
    let x0 = closed_end();
    let x1 = open_end();
    // Outer shell: a rounded box, plus the lug beyond the closed end.
    let centre = vec3d(0.5 * (x0 + x1), 0.0, 0.5 * (z0 + z1));
    let half = vec3d(0.5 * (x1 - x0), ph.y + wall, 0.5 * (z1 - z0));
    var solid = round_box(p - centre, half, edge_radius) <= 0.0;
    let hole_x = x0 - lug_reach + 0.5 * lug_width;
    let lug_z0 = z1 - lug_thickness;
    if !solid && p.x < x0 + edge_radius && p.z >= lug_z0 && p.z <= z1 {
        let q = p.xy - vec2d(hole_x, 0.0);
        let round_end = length(q) <= 0.5 * lug_width;
        let neck = p.x >= hole_x && abs(p.y) <= 0.5 * lug_width;
        solid = (round_end || neck) && length(q) > 0.5 * hole_diameter;
    }
    if !solid { return false; }
    // The pocket, open at +x (square there: the tag slides in through it).
    if p.z >= 0.0 && p.z <= pocket_top() && open_toward_x(p.xy, ph, tag_radius + clearance) {
        return false;
    }
    // The display window through the front lip, open at +x: the face's
    // outline less the lip's overlap onto it.
    let wh = 0.5 * vec2d(face_length, face_width) - vec2d(lip, lip);
    if p.z < 0.0 && open_toward_x(p.xy, wh, max(tag_radius - lip, 0.001)) {
        return false;
    }
    // Slits either side of the tongue in the back plate.
    let tongue_root = x1 - tongue_length;
    if p.z > pocket_top() && p.x > tongue_root
        && abs(abs(p.y) - 0.5 * tongue_width - 0.5 * slit) <= 0.5 * slit {
        return false;
    }
    return true;
}

// The detent under the tongue: a block hanging from the back plate behind
// the tag's back face, ramped on its +x side for the tag sliding in.
fn detent(p: vec3d) -> bool {
    let top = pocket_top();
    let bottom = top - detent_depth;
    let x0 = 0.5 * back_length + detent_gap;
    let x1 = x0 + detent_length;
    if p.z < bottom || p.z > top || abs(p.y) > 0.5 * tongue_width || p.x < x0 { return false; }
    // Ramp: full depth at x0, rising to the plate by x1.
    return p.x <= x1 && p.z >= bottom + (p.x - x0) / (x1 - x0) * detent_depth;
}

// The bumper's outer half-size at height z: the envelope plus clearance and
// wall, held at the face's and the back's outline across the lips, less a
// chamfer at the two faces.
fn bumper_outer(z: float) -> vec2d {
    let z0 = -tpu_front;
    let z1 = tag_height + tpu_clearance + tpu_back;
    let h = tag_half(clamp(z, 0.0, tag_height)) + vec2d(tpu_clearance + tpu_wall, tpu_clearance + tpu_wall);
    let cut = max(tpu_chamfer - min(z - z0, z1 - z), 0.0);
    return h - vec2d(cut, cut);
}

fn bumper(p: vec3d) -> bool {
    let z0 = -tpu_front;
    let z1 = tag_height + tpu_clearance + tpu_back;
    if p.z < z0 || p.z > z1 { return false; }
    let r = tag_radius + tpu_clearance + tpu_wall;
    let shell = rounded_rect(p.xy, bumper_outer(p.z), r) <= 0.0;
    // Lug off the -x end on the display side (the print bed side).
    let face_end = -bumper_outer(0.0).x;
    let hole_x = face_end + edge_radius - lug_reach + 0.5 * lug_width;
    var lug = false;
    if !shell && p.x < face_end + r && p.z <= z0 + tpu_lug_thickness {
        let q = p.xy - vec2d(hole_x, 0.0);
        lug = (length(q) <= 0.5 * lug_width || (p.x >= hole_x && abs(p.y) <= 0.5 * lug_width))
            && length(q) > 0.5 * hole_diameter;
    }
    if !shell && !lug { return false; }
    // The pocket: the envelope itself, plus clearance.
    if p.z >= 0.0 && p.z <= tag_height + tpu_clearance
        && rounded_rect(p.xy, tag_half(min(p.z, tag_height)) + vec2d(tpu_clearance, tpu_clearance),
                        tag_radius + tpu_clearance) <= 0.0 {
        return false;
    }
    // The display window: the face's outline less the front lip's overlap.
    if p.z < 0.0 && rounded_rect(p.xy, 0.5 * vec2d(face_length, face_width) - vec2d(tpu_front_lip, tpu_front_lip),
                                 max(tag_radius - tpu_front_lip, 0.001)) <= 0.0 {
        return false;
    }
    // The back opening: the back face's outline less the back lip's overlap.
    if p.z > tag_height && rounded_rect(p.xy, 0.5 * vec2d(back_length, back_width) - vec2d(tpu_back_lip, tpu_back_lip),
                                        max(tag_radius - tpu_back_lip, 0.001)) <= 0.0 {
        return false;
    }
    return true;
}

fn scene(p: vec3d) -> bool {
    if part < 0.5 { return tag(p); }
    if part > 1.5 { return bumper(p); }
    return sleeve(p) || detent(p);
}

fn bounds_min() -> vec3d {
    if part < 0.5 {
        return vec3d(-0.5 * tag_length, -0.5 * tag_width, 0.0) - vec3d(0.001, 0.001, 0.001);
    }
    if part > 1.5 {
        let h = bumper_outer(0.5 * tag_height);
        return vec3d(-bumper_outer(0.0).x - lug_reach, -h.y, -tpu_front) - vec3d(0.001, 0.001, 0.001);
    }
    return vec3d(closed_end() - lug_reach, -pocket_half().y - wall, -lip_thickness) - vec3d(0.001, 0.001, 0.001);
}

fn bounds_max() -> vec3d {
    if part < 0.5 {
        return vec3d(0.5 * tag_length, 0.5 * tag_width, tag_height) + vec3d(0.001, 0.001, 0.001);
    }
    if part > 1.5 {
        let h = bumper_outer(0.5 * tag_height);
        return vec3d(bumper_outer(0.0).x, h.y, tag_height + tpu_clearance + tpu_back) + vec3d(0.001, 0.001, 0.001);
    }
    return vec3d(open_end(), pocket_half().y + wall, pocket_top() + back_plate) + vec3d(0.001, 0.001, 0.001);
}
