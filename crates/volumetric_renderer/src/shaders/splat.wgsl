// Gaussian splat rendering. Every primitive is drawn as a screen-space
// quad enclosing its projected footprint (the 3D covariance through the
// view's Jacobian, EWA splatting), back to front with premultiplied
// alpha, depth-tested at its centre against the scene. Inside the quad a
// 3D Gaussian's weight is the projected 2D Gaussian; a surfel's is the
// 2DGS evaluation: the pixel's ray intersected with the surfel's plane,
// measured in the surfel's own frame, with a screen-space low-pass so a
// surfel never vanishes between pixels.

struct Uniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    screen_size_px: vec2<f32>,
    // How many standard deviations the quad reaches.
    kernel_radius: f32,
    opacity_scale: f32,
    // 1: surfels (ray–plane intersection); 0: Gaussians (EWA).
    surfels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VsIn {
    @location(0) corner: vec2<f32>,
    @location(1) uv: vec2<f32>,
    // Per instance: world centre and activated opacity.
    @location(2) position: vec4<f32>,
    // Scaled local axes in world coordinates (a surfel's third is zero).
    @location(3) axis_u: vec4<f32>,
    @location(4) axis_v: vec4<f32>,
    @location(5) axis_w: vec4<f32>,
    // View-dependent colour, evaluated on the CPU at the last sort.
    @location(6) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Offset from the centre in pixels.
    @location(1) offset_px: vec2<f32>,
    // The inverse 2D covariance: xx, xy, yy.
    @location(2) conic: vec3<f32>,
    // The surfel's camera-space axes and centre, for the intersection.
    @location(3) @interpolate(flat) cam_u: vec3<f32>,
    @location(4) @interpolate(flat) cam_v: vec3<f32>,
    @location(5) @interpolate(flat) cam_c: vec3<f32>,
    // The centre's pixel.
    @location(6) @interpolate(flat) centre_px: vec2<f32>,
};

fn dropped() -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.color = vec4<f32>(0.0);
    out.offset_px = vec2<f32>(0.0);
    out.conic = vec3<f32>(1.0, 0.0, 1.0);
    out.cam_u = vec3<f32>(0.0);
    out.cam_v = vec3<f32>(0.0);
    out.cam_c = vec3<f32>(0.0);
    out.centre_px = vec2<f32>(0.0);
    return out;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let t4 = uniforms.view * vec4<f32>(in.position.xyz, 1.0);
    let clip = uniforms.proj * t4;
    // Behind the camera, before the near plane, or well outside the
    // frame: not drawn.
    if clip.w <= 1e-6 || clip.z < 0.0 || abs(clip.x) > 1.3 * clip.w || abs(clip.y) > 1.3 * clip.w {
        return dropped();
    }
    let w = clip.w;
    let hw = 0.5 * uniforms.screen_size_px.x;
    let hh = 0.5 * uniforms.screen_size_px.y;
    let p = uniforms.proj;
    // Jacobian of the pixel position (x right, y down) with respect to the
    // view-space point: column k holds d(px, py)/d(t_k).
    var j: mat3x2<f32>;
    for (var k = 0; k < 3; k++) {
        let col = p[k];
        j[k] = vec2<f32>(
            hw * (col.x * w - clip.x * col.w) / (w * w),
            -hh * (col.y * w - clip.y * col.w) / (w * w),
        );
    }
    let wr = mat3x3<f32>(uniforms.view[0].xyz, uniforms.view[1].xyz, uniforms.view[2].xyz);
    let m = j * wr;
    // The 2D covariance is the outer-product sum of the projected axes.
    let pu = m * in.axis_u.xyz;
    let pv = m * in.axis_v.xyz;
    let pw = m * in.axis_w.xyz;
    // Dilate by a third of a pixel so nothing falls between samples.
    let a = pu.x * pu.x + pv.x * pv.x + pw.x * pw.x + 0.3;
    let b = pu.x * pu.y + pv.x * pv.y + pw.x * pw.y;
    let d = pu.y * pu.y + pv.y * pv.y + pw.y * pw.y + 0.3;
    let det = a * d - b * b;
    if det <= 0.0 {
        return dropped();
    }
    let mid = 0.5 * (a + d);
    let disc = sqrt(max(mid * mid - det, 1e-6));
    let l1 = mid + disc;
    let l2 = max(mid - disc, 1e-6);
    var v1: vec2<f32>;
    if abs(b) > 1e-6 {
        v1 = normalize(vec2<f32>(b, l1 - a));
    } else if a >= d {
        v1 = vec2<f32>(1.0, 0.0);
    } else {
        v1 = vec2<f32>(0.0, 1.0);
    }
    let v2 = vec2<f32>(-v1.y, v1.x);
    let r1 = uniforms.kernel_radius * sqrt(l1);
    let r2 = uniforms.kernel_radius * sqrt(l2);
    if r1 > 2.0 * max(uniforms.screen_size_px.x, uniforms.screen_size_px.y) {
        // A footprint wider than two frames is a stray primitive near the
        // eye, not geometry.
        return dropped();
    }
    let corner = in.uv * 2.0 - vec2<f32>(1.0, 1.0);
    let offset_px = corner.x * r1 * v1 + corner.y * r2 * v2;
    let offset_ndc = vec2<f32>(offset_px.x / hw, -offset_px.y / hh);
    out.position = vec4<f32>(clip.xy + offset_ndc * w, clip.z, w);
    out.offset_px = offset_px;
    out.conic = vec3<f32>(d / det, -b / det, a / det);
    out.color = vec4<f32>(in.color.rgb, in.position.w * uniforms.opacity_scale);
    out.cam_u = wr * in.axis_u.xyz;
    out.cam_v = wr * in.axis_v.xyz;
    out.cam_c = t4.xyz;
    out.centre_px = vec2<f32>((clip.x / w + 1.0) * hw, (1.0 - clip.y / w) * hh);
    return out;
}

// The weight of a surfel at the fragment: the pixel's ray meets the
// surfel's plane at local coordinates (u, v) in units of the surfel's
// scales (2DGS), or, when that lands farther out than a screen-space
// Gaussian of half a pixel's variance, the latter.
fn surfel_weight(in: VsOut) -> f32 {
    let p = uniforms.proj;
    let ndc_x = in.position.x / (0.5 * uniforms.screen_size_px.x) - 1.0;
    let ndc_y = 1.0 - in.position.y / (0.5 * uniforms.screen_size_px.y);
    // Rows of the projection: row r has components p[k][r].
    let row0 = vec4<f32>(p[0].x, p[1].x, p[2].x, p[3].x);
    let row1 = vec4<f32>(p[0].y, p[1].y, p[2].y, p[3].y);
    let row3 = vec4<f32>(p[0].w, p[1].w, p[2].w, p[3].w);
    // The planes through the pixel's ray, in camera space.
    let hx = row0 - ndc_x * row3;
    let hy = row1 - ndc_y * row3;
    let c4 = vec4<f32>(in.cam_c, 1.0);
    let hu = vec3<f32>(dot(hx.xyz, in.cam_u), dot(hx.xyz, in.cam_v), dot(hx, c4));
    let hv = vec3<f32>(dot(hy.xyz, in.cam_u), dot(hy.xyz, in.cam_v), dot(hy, c4));
    let cr = cross(hu, hv);
    let d = in.position.xy - in.centre_px;
    let rho2d = 2.0 * dot(d, d);
    var rho = rho2d;
    if abs(cr.z) > 1e-12 {
        let u = cr.x / cr.z;
        let v = cr.y / cr.z;
        rho = min(u * u + v * v, rho2d);
    }
    return exp(-0.5 * rho);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var weight: f32;
    var cap: f32;
    if uniforms.surfels == 1u {
        weight = surfel_weight(in);
        cap = 0.999;
    } else {
        let d = in.offset_px;
        let power = -0.5 * (in.conic.x * d.x * d.x + 2.0 * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);
        if power > 0.0 {
            discard;
        }
        weight = exp(power);
        cap = 0.99;
    }
    let alpha = min(cap, in.color.a * weight);
    if alpha < 1.0 / 255.0 {
        discard;
    }
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
