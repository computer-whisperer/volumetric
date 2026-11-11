// Keychain bodies for the Hanshow Stellar-L3N e-paper tag, in metres in the
// tag frame: origin on the display face at the centre of its outline, x
// along the length toward the end whose back carries the model label (the
// back body is flush with the face there), y toward the side with the clip
// strip (flush too), z out of the back. The back body steps in at -x and -y
// onto ledges a few millimetres up.
//
// part 0: the tag's measured envelope (fit checks and photo overlays)
// part 1: the sleeve (rigid: PETG). Open at -x; the tag slides in +x end
//         first, display against the front lip. A tongue in the back plate
//         carries a detent that drops into the -x step behind the back body.
//         A groove in the +y wall lets the clip strip through. Keyring lug
//         on the closed +x end.
// part 2: the bumper (TPU). A full surround: a lip all round the display
//         face, walls on the face's outline, a back frame on the back face
//         around the logo panel. Stretch the back frame over the tag. Keyring
//         lug at +x on the back side; print back frame down.
//
// measured_* come from measure.py (work/parameters.json); assumed_* are
// design choices or readings not measured, not observations.
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;

override part: float = 1; // @param key="part"

// The tag.
override face_length: float = 0.09044; // @param key="measured_face_length"
override face_width: float = 0.04061; // @param key="measured_face_width"
override face_radius: float = 0.00266; // @param key="measured_face_corner_radius"
override back_z: float = 0.01238; // @param key="measured_back_z"
override panel_z: float = 0.01319; // @param key="measured_panel_z"
override body_x0: float = -0.03715; // @param key="measured_body_x0"
override body_x1: float = 0.04485; // @param key="measured_body_x1"
override body_y0: float = -0.01227; // @param key="measured_body_y0"
override body_y1: float = 0.01934; // @param key="measured_body_y1"
override panel_x1: float = 0.01711; // @param key="measured_panel_x1"
override panel_y0: float = -0.01003; // @param key="measured_panel_y0"
override panel_y1: float = 0.01724; // @param key="measured_panel_y1"
override step_x_z: float = 0.00375; // @param key="measured_step_x_z"
override step_y_z: float = 0.00515; // @param key="measured_step_y_z"
override clip_out: float = 0.00116; // @param key="measured_clip_protrusion"
// The panel's -x edge sits just inside the body's; not separated in the
// photographs.
override panel_x0_inset: float = 0.0004; // @param key="assumed_panel_x0_inset"
// The body's top edges are rounded: its walls stand this far outside the
// top face's measured edges.
override body_fillet: float = 0.0010; // @param key="assumed_body_fillet"
// The clip strip, from two photographs (DSC03274, DSC03238), not measured.
override clip_x0: float = -0.0115; // @param key="assumed_clip_x0"
override clip_x1: float = 0.0115; // @param key="assumed_clip_x1"
override clip_z0: float = 0.0085; // @param key="assumed_clip_z0"
override clip_z1: float = 0.0118; // @param key="assumed_clip_z1"

// The sleeve.
override clearance: float = 0.0003; // @param key="assumed_clearance"
override wall: float = 0.0022; // @param key="assumed_wall"
override back_plate: float = 0.0016; // @param key="assumed_back_plate"
override lip: float = 0.0015; // @param key="assumed_lip_overlap"
override lip_thickness: float = 0.0010; // @param key="assumed_lip_thickness"
override edge_radius: float = 0.0008; // @param key="assumed_edge_radius"
override groove_margin: float = 0.0010; // @param key="assumed_groove_margin"
override tongue_length: float = 0.025; // @param key="assumed_tongue_length"
override tongue_width: float = 0.016; // @param key="assumed_tongue_width"
override slit: float = 0.0008; // @param key="assumed_slit"
override detent_catch: float = 0.0018; // @param key="assumed_detent_catch"
override detent_gap: float = 0.0005; // @param key="assumed_detent_gap"
override detent_length: float = 0.0040; // @param key="assumed_detent_length"
override lug_reach: float = 0.0095; // @param key="assumed_lug_reach"
override lug_width: float = 0.014; // @param key="assumed_lug_width"
override lug_thickness: float = 0.004; // @param key="assumed_lug_thickness"
override hole_diameter: float = 0.005; // @param key="assumed_hole_diameter"

// The bumper.
override tpu_clearance: float = 0.0; // @param key="assumed_tpu_clearance"
override tpu_wall: float = 0.0015; // @param key="assumed_tpu_wall"
override tpu_front: float = 0.0010; // @param key="assumed_tpu_front_thickness"
override tpu_front_lip: float = 0.0015; // @param key="assumed_tpu_front_overlap"
override tpu_back: float = 0.0012; // @param key="assumed_tpu_back_thickness"
override tpu_back_lip: float = 0.0020; // @param key="assumed_tpu_back_overlap"
override tpu_panel_gap: float = 0.0003; // @param key="assumed_tpu_panel_gap"
override tpu_chamfer: float = 0.0006; // @param key="assumed_tpu_chamfer"
override tpu_lug_thickness: float = 0.0035; // @param key="assumed_tpu_lug_thickness"

// Signed distance to a rectangle of half-size h with corner radius r.
fn rounded_rect(p: vec2d, h: vec2d, r: float) -> float {
    let q = abs(p) - h + vec2d(r, r);
    return length(max(q, vec2d(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// The same for a rectangle from lo to hi.
fn rect_between(p: vec2d, lo: vec2d, hi: vec2d, r: float) -> float {
    return rounded_rect(p - 0.5 * (lo + hi), 0.5 * (hi - lo), r);
}

// Signed distance to a box of half-size h with every edge rounded by r.
fn round_box(p: vec3d, h: vec3d, r: float) -> float {
    let q = abs(p) - h + vec3d(r, r, r);
    return length(max(q, vec3d(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn face_half() -> vec2d {
    return 0.5 * vec2d(face_length, face_width);
}

// The tag: the face plate up to its ledges, the back body (flush with the
// face at +x and +y) up to the back face, the logo panel on top, the clip
// strip on the +y wall.
fn tag(p: vec3d) -> bool {
    if p.z < 0.0 || p.z > panel_z { return false; }
    let h = face_half();
    if rounded_rect(p.xy, h, face_radius) > 0.0 {
        // Outside the face's outline: only the clip strip.
        return p.x >= clip_x0 && p.x <= clip_x1 && p.y >= h.y - 0.0005 && p.y <= h.y + clip_out
            && p.z >= clip_z0 && p.z <= clip_z1;
    }
    let bx0 = body_x0 - body_fillet;
    let by0 = body_y0 - body_fillet;
    let in_body = p.x >= bx0 && p.y >= by0;
    if !in_body {
        // On a ledge: the face plate's height there.
        var ledge = step_y_z;
        if p.x < bx0 { ledge = step_x_z; }
        return p.z <= ledge;
    }
    if p.z <= back_z - body_fillet { return true; }
    if p.z <= back_z {
        // The rounded top edge, as a chamfer from the walls to the top face.
        let t = (p.z - (back_z - body_fillet)) / body_fillet;
        let lo = vec2d(bx0, by0) + t * vec2d(body_fillet, body_fillet);
        let hi = vec2d(h.x - t * (h.x - body_x1), h.y - t * (h.y - body_y1));
        return p.x >= lo.x && p.y >= lo.y && p.x <= hi.x && p.y <= hi.y;
    }
    return rect_between(p.xy, vec2d(body_x0 + panel_x0_inset, panel_y0), vec2d(panel_x1, panel_y1), 0.001) <= 0.0;
}

// --- The sleeve ---

fn pocket_half() -> vec2d {
    return face_half() + vec2d(clearance, clearance);
}

fn pocket_top() -> float { return panel_z + clearance; }

fn open_end() -> float { return -pocket_half().x; }

fn closed_end() -> float { return pocket_half().x + wall; }

// Inside a rounded rectangle whose -x half runs on to infinity.
fn open_toward_minus_x(p: vec2d, h: vec2d, r: float) -> bool {
    return rounded_rect(p, h, r) <= 0.0 || (p.x < 0.0 && abs(p.y) <= h.y);
}

fn tongue_y() -> float { return 0.5 * (body_y0 + body_y1); }

fn sleeve(p: vec3d) -> bool {
    let ph = pocket_half();
    let z0 = -lip_thickness;
    let z1 = pocket_top() + back_plate;
    let x0 = open_end();
    let x1 = closed_end();
    // Outer shell: a rounded box, plus the lug beyond the closed end.
    let centre = vec3d(0.5 * (x0 + x1), 0.0, 0.5 * (z0 + z1));
    let half = vec3d(0.5 * (x1 - x0), ph.y + wall, 0.5 * (z1 - z0));
    var solid = round_box(p - centre, half, edge_radius) <= 0.0;
    let hole_x = x1 + lug_reach - 0.5 * lug_width;
    if !solid && p.x > x1 - edge_radius && p.z >= z1 - lug_thickness && p.z <= z1 {
        let q = p.xy - vec2d(hole_x, 0.0);
        let round_end = length(q) <= 0.5 * lug_width;
        let neck = p.x <= hole_x && abs(p.y) <= 0.5 * lug_width;
        solid = (round_end || neck) && length(q) > 0.5 * hole_diameter;
    }
    if !solid { return false; }
    // The pocket, open at -x.
    if p.z >= 0.0 && p.z <= pocket_top() && open_toward_minus_x(p.xy, ph, face_radius + clearance) {
        return false;
    }
    // The groove the clip strip runs in, from the open end to past its seat.
    if p.y >= 0.0 && p.y <= face_half().y + clip_out + clearance && p.x <= clip_x1 + groove_margin
        && p.z >= clip_z0 - groove_margin && p.z <= min(clip_z1 + groove_margin, pocket_top()) {
        return false;
    }
    // The display window through the front lip, open at -x.
    let wh = face_half() - vec2d(lip, lip);
    if p.z < 0.0 && open_toward_minus_x(p.xy, wh, max(face_radius - lip, 0.0005)) {
        return false;
    }
    // Slits either side of the tongue in the back plate, from the open end.
    if p.z > pocket_top() && p.x < x0 + tongue_length
        && abs(abs(p.y - tongue_y()) - 0.5 * tongue_width - 0.5 * slit) <= 0.5 * slit {
        return false;
    }
    return true;
}

// The detent under the tongue: hanging from the back plate into the -x
// step, its catch face just beyond the back body's -x wall, ramped on its
// -x side for the tag sliding in.
fn detent(p: vec3d) -> bool {
    let top = pocket_top();
    let bottom = back_z - detent_catch;
    let x1 = body_x0 - body_fillet - detent_gap;
    let x0 = x1 - detent_length;
    if p.z < bottom || p.z > top || abs(p.y - tongue_y()) > 0.5 * tongue_width || p.x > x1 || p.x < x0 {
        return false;
    }
    // Ramp: full depth at the catch face, rising to the plate at x0.
    return p.z >= bottom + (x1 - p.x) / (x1 - x0) * (top - bottom);
}

// --- The bumper ---

// The pocket: the face's outline (plus clearance) at every height, with the
// clip strip's room on +y. Signed distance in the plane.
fn bumper_pocket(p: vec3d) -> float {
    let c = tpu_clearance;
    let body = rounded_rect(p.xy, face_half() + vec2d(c, c), face_radius + c);
    let h = face_half();
    let strip = rect_between(p.xy, vec2d(clip_x0 - groove_margin, h.y - 0.001),
                             vec2d(clip_x1 + groove_margin, h.y + clip_out + c), 0.0005);
    var d = body;
    if p.z >= clip_z0 - groove_margin && p.z <= clip_z1 + groove_margin { d = min(d, strip); }
    return d;
}

fn bumper_top() -> float { return back_z + tpu_clearance + tpu_back; }

fn bumper(p: vec3d) -> bool {
    let z0 = -tpu_front;
    let z1 = bumper_top();
    let h = face_half();
    let c = tpu_clearance;
    let pocket = bumper_pocket(p);
    // The shell: the pocket grown by the wall, less a chamfer at both faces.
    let cut = max(tpu_chamfer - min(p.z - z0, z1 - p.z), 0.0);
    let shell = p.z >= z0 && p.z <= z1 && pocket <= tpu_wall - cut;
    // Lug off the +x end on the back side (the print bed side).
    let end = h.x + c + tpu_wall;
    let hole_x = end + lug_reach - 0.5 * lug_width;
    var lug = false;
    if !shell && p.x > end - face_radius && p.z >= z1 - tpu_lug_thickness && p.z <= z1 {
        let q = p.xy - vec2d(hole_x, 0.0);
        lug = (length(q) <= 0.5 * lug_width || (p.x <= hole_x && abs(p.y) <= 0.5 * lug_width))
            && length(q) > 0.5 * hole_diameter;
    }
    if !shell && !lug { return false; }
    // The pocket itself, up to the back face.
    if p.z >= 0.0 && p.z <= back_z + c && pocket <= 0.0 { return false; }
    // The display window: the face's outline less the front lip's overlap.
    if p.z < 0.0 && rounded_rect(p.xy, h - vec2d(tpu_front_lip, tpu_front_lip),
                                 max(face_radius - tpu_front_lip, 0.0005)) <= 0.0 {
        return false;
    }
    // The back opening: the back body's top face less the back lip's
    // overlap, and room for the logo panel standing through it.
    if p.z > back_z + c {
        let top_face = rect_between(p.xy, vec2d(body_x0, body_y0) + vec2d(tpu_back_lip, tpu_back_lip),
                                    vec2d(body_x1, body_y1) - vec2d(tpu_back_lip, tpu_back_lip), 0.001);
        let g = tpu_panel_gap;
        let panel = rect_between(p.xy, vec2d(body_x0 + panel_x0_inset - g, panel_y0 - g),
                                 vec2d(panel_x1 + g, panel_y1 + g), 0.001 + g);
        if min(top_face, panel) <= 0.0 { return false; }
    }
    return true;
}

fn scene(p: vec3d) -> bool {
    if part < 0.5 { return tag(p); }
    if part > 1.5 { return bumper(p); }
    return sleeve(p) || detent(p);
}

fn bounds_min() -> vec3d {
    let m = vec3d(0.001, 0.001, 0.001);
    let h = face_half();
    if part < 0.5 {
        return vec3d(-h.x, -h.y, 0.0) - m;
    }
    if part > 1.5 {
        return vec3d(-h.x, -h.y, 0.0) - vec3d(tpu_clearance + tpu_wall, tpu_clearance + tpu_wall, tpu_front) - m;
    }
    return vec3d(open_end(), -pocket_half().y - wall, -lip_thickness) - m;
}

fn bounds_max() -> vec3d {
    let m = vec3d(0.001, 0.001, 0.001);
    let h = face_half();
    if part < 0.5 {
        return vec3d(h.x, h.y + clip_out, panel_z) + m;
    }
    if part > 1.5 {
        let end = h.x + tpu_clearance + tpu_wall;
        return vec3d(end + lug_reach, h.y + clip_out + tpu_clearance + tpu_wall, bumper_top()) + m;
    }
    return vec3d(closed_end() + lug_reach, pocket_half().y + wall, pocket_top() + back_plate) + m;
}
