// Receiver in the fitted entry-mouth frame. Z=0 is the mouth plane.
// Throat inset, insertion axis and interior depth are explicit assumptions.
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;
override width: float = 0.039; // @param key="mouth_width"
override depth: float = 0.018; // @param key="mouth_depth"
override corner: float = 0.009; // @param key="mouth_corner_radius"
override throat_inset: float = 0.001; // @param key="assumed_throat_inset"
override leadin: float = 0.003; // @param key="assumed_leadin_depth"
override socket_depth: float = 0.065; // @param key="assumed_socket_depth"
override wall: float = 0.003; // @param key="assumed_receiver_wall"
fn rounded(p: vec2d, halfsize: vec2d, radius: float) -> bool {
    let q=abs(p)-halfsize+radius;
    return length(max(q,vec2d(0.0))) + min(max(q.x,q.y),0.0) <= radius;
}
fn scene(p: vec3d) -> bool {
    if p.z > 0.0 || p.z < -socket_depth { return false; }
    let cover=p.z>=-0.004;
    let outer=select(rounded(p.xy-vec2d(0.0,-0.009),vec2d(0.028,0.024),0.005),
                     rounded(p.xy-vec2d(0.0,-0.009),vec2d(0.029,0.024),0.006),cover);
    if !outer { return false; }
    let inset=throat_inset*clamp(-p.z/leadin,0.0,1.0);
    let bore=rounded(p.xy,vec2d(width,depth)/2.0-inset,max(corner-inset,0.0001));
    // Open bore: its termination is unseen; do not invent a bottom stop.
    if bore { return false; }
    // Below the collar, retain a hollow outer shell and front web.
    if !cover && abs(p.x)<0.028-wall && p.y < -depth/2.0-wall && p.y > -0.033+wall { return false; }
    return true;
}
fn bounds_min() -> vec3d { return vec3d(-0.029,-0.033,-socket_depth); }
fn bounds_max() -> vec3d { return vec3d(0.029,0.015,0.0); }
