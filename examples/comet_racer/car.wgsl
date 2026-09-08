// COMET: an original open-cockpit toy racer. Authored in millimetres;
// engine-facing coordinates and bounds are metres. +x is the nose, +y up.
alias float = f64;
alias vec3d = vec3<f64>;

override scale: float = 1.0; // @param key="comet.scale" min=0.5 max=2.0

fn rounded_box(x: float, y: float, z: float, hx: float, hy: float, hz: float, r: float) -> bool {
    let a = max(abs(x) - hx + r, 0.0);
    let b = max(abs(y) - hy + r, 0.0);
    let c = max(abs(z) - hz + r, 0.0);
    return a*a + b*b + c*c <= r*r;
}

fn wheel(x: float, y: float, z: float) -> bool {
    // Rounded slick tyre with two circumferential channels, recessed
    // outer hub, six substantial spokes, and a projecting centre cap.
    let radius = sqrt(x*x + y*y);
    let shoulder_r = max(radius - 10.5, 0.0);
    let shoulder_z = max(abs(z) - 3.5, 0.0);
    var tyre = shoulder_r*shoulder_r + shoulder_z*shoulder_z <= 4.0;
    if (abs(abs(z) - 2.6) < 0.55 && radius > 11.5) {
        tyre = false;
    }
    if (z > 3.5 && radius < 8.4) {
        tyre = false;
    }
    var spoke = false;
    // Three crossing bars give six spokes; no angular seam or atan needed.
    if (abs(y) < 1.0 || abs(0.8660254*x - 0.5*y) < 1.0 || abs(0.8660254*x + 0.5*y) < 1.0) {
        spoke = radius <= 8.7 && z >= 3.0 && z <= 4.8;
    }
    return tyre || spoke || (radius <= 3.0 && abs(z) <= 5.8);
}

fn scene(p: vec3d) -> bool {
    let x = p.x * 1000.0 / scale;
    let y = p.y * 1000.0 / scale;
    let z = p.z * 1000.0 / scale;
    let az = abs(z);

    // Low tapered monocoque, rounded analytically without a field bake.
    let taper = 1.0 + max(x - 8.0, 0.0) * 0.018;
    let hood_top = 25.0 - max(x - 8.0, 0.0) * 0.15;
    var body = rounded_box(x, y - 16.0, z*taper, 48.0, 9.0, 15.0, 4.0)
        && y <= hood_top;
    let engine = rounded_box(x + 27.0, y - 24.0, z, 16.0, 6.5, 12.0, 4.0);
    body = body || engine;

    // Open cockpit has a solid floor. A separate backrest remains inside.
    let cockpit = (x + 3.0)*(x + 3.0)/169.0 + z*z/81.0 < 1.0 && y > 15.5;
    body = body && !cockpit;
    // Engine cooling slots on both sides, and twin fine hood stripes.
    let vent = (abs(x + 34.0) < 0.7 || abs(x + 29.5) < 0.7 || abs(x + 25.0) < 0.7)
        && y > 23.0 && y < 28.0 && az > 10.0;
    let stripe = abs(az - 3.0) < 0.6 && x > 13.0 && x < 42.0 && y > hood_top - 0.8;
    body = body && !vent && !stripe;

    // Sidepods, splitter and diffuser make a deliberately broad stance.
    let pod = rounded_box(x + 1.0, y - 12.0, az - 16.5, 15.0, 4.0, 5.0, 2.0);
    let splitter = rounded_box(x - 43.5, y - 8.0, z, 9.5, 1.5, 23.0, 1.0);
    let diffuser = rounded_box(x + 43.0, y - 8.0, z, 6.0, 1.5, 22.0, 1.0);
    let seat = rounded_box(x + 11.0, y - 20.0, z, 2.0, 5.5, 6.5, 1.5);
    let steering_r2 = (y - 23.0)*(y - 23.0) + z*z;
    let steering = abs(x - 5.0) <= 0.9 && steering_r2 <= 20.25
        && (steering_r2 >= 9.0 || abs(z) <= 0.8 || abs(y - 23.0) <= 0.8);
    let column = steering_r2 <= 1.44 && x >= 5.0 && x <= 13.0;

    // Roll hoop: upper semicircle and two vertical feet behind the seat.
    let hoop_radius = sqrt((y - 27.0)*(y - 27.0) + z*z);
    let hoop = (x + 18.0)*(x + 18.0) + (hoop_radius - 9.5)*(hoop_radius - 9.5) <= 2.25 && y >= 27.0;
    let hoop_foot = (x + 18.0)*(x + 18.0) + (az - 9.5)*(az - 9.5) <= 2.25 && y >= 21.0 && y <= 27.0;

    // Rear wing and endplates; chunky supports connect into the engine cover.
    let wing = rounded_box(x + 41.0, y - 34.0, z, 8.0, 2.0, 29.0, 1.5);
    let endplate = rounded_box(x + 41.0, y - 34.0, az - 28.0, 8.0, 4.0, 1.3, 1.0);
    let support = rounded_box(x + 39.0, y - 27.0, az - 9.0, 2.0, 7.0, 1.8, 1.0);

    // Transverse axles connect all four wheels into a single static model.
    let axle_x = abs(x) - 30.0;
    let axle = axle_x*axle_x + (y - 12.5)*(y - 12.5) <= 4.0 && az <= 25.0;
    let wheels = wheel(axle_x, y - 12.5, az - 25.0);
    return body || pod || splitter || diffuser || seat || steering || column || hoop || hoop_foot || wing || endplate || support || axle || wheels;
}

fn bounds_min() -> vec3d {
    return vec3d(-0.050, -0.001, -0.032) * scale;
}

fn bounds_max() -> vec3d {
    return vec3d(0.054, 0.039, 0.032) * scale;
}
