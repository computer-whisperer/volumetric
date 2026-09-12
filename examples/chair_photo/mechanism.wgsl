// Representative folded housing, visible hardware and three control levers.
// Same fixed frame as the accepted seat mounting apertures.
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;
override part: float = 0; // @param key="part"
override pin_x: float = -0.034; // @param key="rear_pin_x"
override upper_x: float = -0.032; // @param key="rear_upper_fastener_x"
override lower_x: float = -0.030; // @param key="rear_lower_fastener_x"
override end_x: float = -0.195; // @param key="control_end_x"
override end_y: float = 0.100; // @param key="control_end_y"
override end_z: float = -0.080; // @param key="control_end_z"
override root_y: float = 0.070; // @param key="control_root_y"
override pin_y: float = 0.260; // @param key="rear_pin_y"
override pin_z: float = -0.035; // @param key="rear_pin_z"
override upper_y: float = 0.244; // @param key="rear_upper_fastener_y"
override upper_z: float = -0.012; // @param key="rear_upper_fastener_z"
override lower_y: float = 0.243; // @param key="rear_lower_fastener_y"
override lower_z: float = -0.055; // @param key="rear_lower_fastener_z"
override receiver_y: float = 0.275; // @param key="receiver_y"
override knob_z: float = -0.024; // @param key="assumed_knob_z"
fn rod(p: vec3d,a: vec3d,b: vec3d,r: float) -> bool {
    let d=b-a;
    let t=clamp(dot(p-a,d)/dot(d,d),0.0,1.0);
    let q=p-a-d*t;
    return dot(q,q)<=r*r;
}
fn x_cylinder(p: vec3d, y: float,z: float,r: float,x0: float,x1: float) -> bool {
    let q=p.yz-vec2d(y,z);
    return dot(q,q)<=r*r && p.x>=x0 && p.x<=x1;
}
fn scene(p: vec3d) -> bool {
    if part < 0.5 {
        // Side cheeks; the rail and its six apertures remain a separate part.
        let bottom: float=select(float(-0.050),float(-0.065),p.y>0.210);
        if p.y>=0.010 && p.y<=0.278 && p.z<=-0.003 && p.z>=bottom && abs(p.x)>=0.026 && abs(p.x)<=0.030 { return true; }
        if p.y>=0.028 && p.y<=0.218 && p.z>=-0.056 && p.z<=-0.050 && abs(p.x)<=0.030 { return true; }
        if rod(p,vec3d(-0.025,0.016,-0.002),vec3d(-0.025,0.242,-0.002),0.0025) { return true; }
        return rod(p,vec3d(0.025,0.016,-0.002),vec3d(0.025,0.242,-0.002),0.0025);
    }
    if part < 1.5 {
        // Observed cap/fastener stations. Opposite caps and transverse axes
        // are symmetry assumptions; internal linkage is not reconstructed.
        return x_cylinder(p,pin_y,pin_z,0.009,pin_x,-pin_x)
            || x_cylinder(p,upper_y,upper_z,0.005,upper_x,-upper_x)
            || x_cylinder(p,lower_y,lower_z,0.005,lower_x,-lower_x);
    }
    if part < 2.5 {
        if x_cylinder(p,receiver_y,knob_z,0.007,0.020,0.055) { return true; }
        let q=p.yz-vec2d(receiver_y,knob_z);
        let flutes=0.025+0.0015*cos(12.0*atan2(q.y,q.x));
        return dot(q,q)<=flutes*flutes && p.x>=0.044 && p.x<=0.073;
    }
    let side=select(float(-1.0),float(1.0),end_x>0.0);
    let start=vec3d(side*0.030,root_y,-0.030);
    let end=vec3d(end_x,end_y,end_z);
    let q=p-end;
    let along=normalize(end-start);
    let across=normalize(cross(vec3d(0.0,0.0,1.0),along));
    let normal=cross(along,across);
    let paddle=vec3d(dot(q,along)/0.031,dot(q,across)/0.019,dot(q,normal)/0.005);
    return rod(p,start,end,0.007) || dot(paddle,paddle)<=1.0;
}

fn bounds_min() -> vec3d {
    if part < 0.5 { return vec3d(-0.030,0.010,-0.065); }
    if part < 1.5 { return vec3d(min(pin_x,min(upper_x,lower_x)),min(pin_y-0.009,min(upper_y-0.005,lower_y-0.005)),min(pin_z-0.009,min(upper_z-0.005,lower_z-0.005))); }
    if part < 2.5 { return vec3d(0.020,receiver_y-0.027,knob_z-0.027); }
    return vec3d(min(end_x-0.04,-0.04),min(end_y-0.04,root_y-0.01),min(end_z-0.04,-0.04));
}
fn bounds_max() -> vec3d {
    if part < 0.5 { return vec3d(0.030,0.278,0.0005); }
    if part < 1.5 { return vec3d(-min(pin_x,min(upper_x,lower_x)),max(pin_y+0.009,max(upper_y+0.005,lower_y+0.005)),max(pin_z+0.009,max(upper_z+0.005,lower_z+0.005))); }
    if part < 2.5 { return vec3d(0.073,receiver_y+0.027,knob_z+0.027); }
    return vec3d(max(end_x+0.04,0.04),max(end_y+0.04,root_y+0.01),max(end_z+0.04,-0.02));
}
