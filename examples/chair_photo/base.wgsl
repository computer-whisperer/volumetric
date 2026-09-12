// Representative lower base in a vertical frame at the column/floor datum.
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;
override part: float = 0; // @param key="part"
override radius: float = 0.336; // @param key="star_radius"
override joint_z: float = 0.052; // @param key="joint_height"
override phase: float = 0.539; // @param key="star_phase"
override outer_radius: float = 0.024; // @param key="column_outer_radius"
override rod_radius: float = 0.0135; // @param key="column_rod_radius"
override collar_z: float = 0.266; // @param key="column_collar_z"
override top_z: float = 0.419; // @param key="column_top_z"
override hub_z: float = 0.165; // @param key="assumed_hub_top_z"
override wheel_radius: float = 0.025; // @param key="assumed_wheel_radius"
override caster_yaw: float = 0; // @param key="caster_yaw"

fn cylinder(p: vec3d, r: float, lo: float, hi: float) -> bool {
    return dot(p.xy,p.xy) <= r*r && p.z >= lo && p.z <= hi;
}
fn scene(p: vec3d) -> bool {
    if part < 0.5 {
        return cylinder(p, outer_radius, 0.035, collar_z)
            || cylinder(p, outer_radius+0.001, collar_z-0.004, collar_z)
            || cylinder(p, rod_radius, collar_z, top_z);
    }
    if part < 1.5 {
        if cylinder(p, 0.047, hub_z-0.07, hub_z) { return !cylinder(p, outer_radius, 0.0, hub_z+0.01); }
        for (var i: i32=0; i<5; i=i+1) {
            let angle = phase + float(i)*1.2566370614359172;
            let q=vec3d(p.x*cos(angle)+p.y*sin(angle), -p.x*sin(angle)+p.y*cos(angle), p.z);
            if q.x >= 0.029 && q.x <= radius+0.016 {
                let t=clamp(q.x/radius,0.0,1.0);
                // Elliptical arms arch out from the hub, then turn down to the caster.
                let drop=smoothstep(0.80,1.0,t);
                let z=hub_z-0.045-0.004*sin(t*3.141592653589793) - (hub_z-0.045-joint_z-0.018)*drop;
                let halfwidth=mix(0.020,0.029,sin(t*3.141592653589793));
                let halfheight=mix(0.019,0.024,drop);
                let end= max(q.x-radius,0.0)/0.016;
                if q.y*q.y/(halfwidth*halfwidth)+(q.z-z)*(q.z-z)/(halfheight*halfheight)+end*end <= 1.0 { return true; }
            }
        }
        return false;
    }
    // One twin-wheel swivel caster per exported instance. Wheel orientation
    // and fork details are representative, not recovered articulation.
    let angle=phase+(part-2.0)*1.2566370614359172;
    let at=vec3d(radius*cos(angle),radius*sin(angle),0.0);
    let d=p-at;
    let yaw=angle+caster_yaw;
    let q=vec3d(d.x*cos(yaw)+d.y*sin(yaw),-d.x*sin(yaw)+d.y*cos(yaw),d.z);
    if cylinder(q,0.012,joint_z-0.014,joint_z+0.014) { return true; }
    let wheel=vec2d(q.x-0.018,q.z-wheel_radius);
    let rim=dot(wheel,wheel)<=wheel_radius*wheel_radius;
    if rim && abs(q.y)>=0.005 && abs(q.y)<=0.022 { return true; }
    return q.x>=-0.010 && q.x<=0.035 && abs(q.y)<=0.007 && q.z>=wheel_radius && q.z<=joint_z-0.005;
}
fn bounds_min() -> vec3d {
    if part < 0.5 { return vec3d(-outer_radius-0.002,-outer_radius-0.002,0.035); }
    if part < 1.5 { return vec3d(-radius-0.025,-radius-0.025,0.0); }
    let angle=phase+(part-2.0)*1.2566370614359172;
    return vec3d(radius*cos(angle)-0.055,radius*sin(angle)-0.055,0.0);
}
fn bounds_max() -> vec3d {
    if part < 0.5 { return vec3d(outer_radius+0.002,outer_radius+0.002,top_z); }
    if part < 1.5 { return vec3d(radius+0.025,radius+0.025,hub_z+0.02); }
    let angle=phase+(part-2.0)*1.2566370614359172;
    return vec3d(radius*cos(angle)+0.055,radius*sin(angle)+0.055,joint_z+0.02);
}
