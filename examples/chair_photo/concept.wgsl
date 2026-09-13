// Concept A. Metres, seat-local frame. Dimensions are design inputs, not ratings.
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;
override part: float = 0; // @param key="part"
override half: float = 0.258; // @param key="half"
override front: float = -0.281; // @param key="front"
override rear: float = 0.235; // @param key="rear"
override side: float = 1; // @param key="side"
override station: float = 0; // @param key="station"
override hole_x: float = 0; // @param key="hole_x"
override hole_y: float = 0; // @param key="hole_y"
override hole2_x: float = 0; // @param key="hole2_x"
override hole2_y: float = 0; // @param key="hole2_y"
override tongue_clearance: float = 0; // @param key="tongue_clearance"
override contact_z: float = 0; // @param key="contact_z"
override tube_radius: float = 0.01105; // @param key="tube_radius"
override tube_half: float = 0.1257; // @param key="tube_half"
override bolt_width: float = 0.0763; // @param key="bolt_width"
override bolt_height: float = 0.1995; // @param key="bolt_height"

fn box(p:vec3d, c:vec3d, h:vec3d)->bool { return all(abs(p-c)<=h); }
fn rod(p:vec3d,a:vec3d,b:vec3d,r:float)->bool {
    let d=b-a; let t=dot(p-a,d)/dot(d,d);
    let q=p-a-clamp(t,0.0,1.0)*d;
    return t>=0.0 && t<=1.0 && dot(q,q)<=r*r;
}
fn tube(p:vec3d,a:vec3d,b:vec3d,r:float,wall:float)->bool {
    return rod(p,a,b,r) && !rod(p,a,b,r-wall);
}
fn rounded(p:vec2d,h:vec2d,r:float)->bool {
    let q=max(abs(p)-(h-vec2d(r)),vec2d(0.0)); return dot(q,q)<=r*r;
}
fn pan_hole(p:vec3d)->bool {
    let q=vec2d(abs(p.x)-0.11,min(abs(p.y-(front+0.055)),abs(p.y-0.19)));
    let r=max(0.0033,0.0066+p.z-0.043);
    return dot(q,q)<=r*r;
}
fn mount_hole(p:vec3d)->bool {
    // Centers passed directly from the accepted photo-derived fit.
    return dot(p.xy-vec2d(hole_x,hole_y),p.xy-vec2d(hole_x,hole_y))<=0.0033*0.0033;
}
fn scene(p:vec3d)->bool {
    let cy=(front+rear)/2.0; let hy=(rear-front)/2.0;
    let arm=side*(half+0.04);
    if part<0.5 {
        let outline=rounded(p.xy-vec2d(0.0,cy),vec2d(half,hy),0.008);
        let sheet=p.z>=0.040 && p.z<=0.043;
        let flange=p.z>=0.020 && p.z<=0.040 && (abs(p.x)>=half-0.003 || p.y>=rear-0.003)
            && !(abs(p.y-0.01)<0.017 && abs(p.x)>half-0.004);
        return outline && (sheet || flange) && !pan_hole(p);
    }
    if part<1.5 {
        let q=p-vec3d(side*0.11,(front+0.025+0.215)/2.0,0.025);
        return abs(q.y)<=(0.215-front-0.025)/2.0 && rounded(q.xz,vec2d(0.01,0.015),0.003)
            && !rounded(q.xz,vec2d(0.008,0.013),0.001) && !pan_hole(p);
    }
    if part<2.5 {
        // Both mounting stations come directly from the measured fit.
        let bore=mount_hole(p) || dot(p.xy-vec2d(hole2_x,hole2_y),p.xy-vec2d(hole2_x,hole2_y))<0.0033*0.0033;
        return box(p,vec3d(0.0,station,0.007),vec3d(0.125,0.016,0.003)) && !bore;
    }
    if part<3.5 {
        return abs(p.x-side*0.11)<=0.012 && min(abs(p.y-(front+0.055)),abs(p.y-0.19))<=0.012
            && p.z>=0.034 && p.z<=0.040 && !pan_hole(p);
    }
    if part<4.5 {
        let q=vec2d(abs(p.x)-0.11,min(abs(p.y-(front+0.055)),abs(p.y-0.19)));
        let r=select(float(0.003),p.z-0.0367,p.z>0.0397);
        return p.z>=0.027 && p.z<=0.0427 && dot(q,q)<=r*r;
    }
    if part<5.5 {
        let q=p.xy-vec2d(hole_x,hole_y);
        return p.z>=contact_z && p.z<=0.004 && dot(q,q)<=0.009*0.009 && !mount_hole(p);
    }
    if part<6.5 {
        return tube(p,vec3d(arm,0.06,0.024),vec3d(arm,0.06,0.25),0.015,0.002)
            || tube(p,vec3d(side*0.11,0.06,0.024),vec3d(arm,0.06,0.024),0.014,0.002);
    }
    if part<7.5 { return tube(p,vec3d(arm,0.06,0.12),vec3d(arm,0.06,0.29),0.0128,0.002); }
    if part<8.5 {
        let q=p-vec3d(arm-side*0.0475,0.06,0.296);
        return rounded(q.xy,vec2d(0.0725,0.026),0.006) && abs(q.z)<=0.013
            && !(abs(q.y)<0.0105 && abs(q.z)<0.0045)
            && !(abs(q.y)<0.0065 && q.z>=0.0)
            && !rod(p,vec3d(arm,0.06,0.282),vec3d(arm,0.06,0.291),0.013);
    }
    if part<9.5 {
        return box(p,vec3d(arm-side*0.035,0.06,0.296),vec3d(0.075,0.010,0.004))
            || box(p,vec3d(arm,0.06,0.307),vec3d(0.015,0.006,0.011));
    }
    if part<10.5 { return rounded(p.xy-vec2d(arm,0.02),vec2d(0.035,0.115),0.022) && abs(p.z-0.328)<=0.01; }
    if part<11.5 {
        // Separate tongue model is authored in the measured receiver frame.
        return rounded(p.xy,vec2d(0.016+tongue_clearance,0.005+tongue_clearance),0.004) && p.z>=-0.045 && p.z<=0.060;
    }
    if part<12.5 {
        // Metal lower spine plus shoe joining the tongue above the socket mouth.
        let spine=tube(p,vec3d(0.0,0.355,0.105),vec3d(0.0,0.355,0.345),0.017,0.002)
            || tube(p,vec3d(0.0,0.278,0.055),vec3d(0.0,0.355,0.105),0.017,0.002);
        let shoe=box(p,vec3d(0.0,0.278,0.055),vec3d(0.024,0.031,0.015));
        return spine || shoe;
    }
    if part<13.5 { return tube(p,vec3d(0.0,0.355,0.215),vec3d(0.0,0.355,0.385),0.0148,0.002); }
    if part<14.5 {
        let a=vec3d(0.0,0.355,0.375);
        let left=vec3d(-0.09,0.315,0.365);let right=vec3d(0.09,0.315,0.365);
        let arms=rod(p,a,left,0.014)||rod(p,a,right,0.014);
        let socket=rod(p,vec3d(0.0,0.355,0.353),vec3d(0.0,0.355,0.39),0.022);
        let bores=rod(p,vec3d(0.0,0.355,0.352),vec3d(0.0,0.355,0.386),0.015);
        let collars=rod(p,left-vec3d(0.012,0.0,0.0),left+vec3d(0.012,0.0,0.0),tube_radius+0.007)
            || rod(p,right-vec3d(0.012,0.0,0.0),right+vec3d(0.012,0.0,0.0),tube_radius+0.007);
        let tube_bore=rod(p,vec3d(-0.13,0.315,0.365),vec3d(0.13,0.315,0.365),tube_radius+0.0003);
        // Rear slit represents a split clamp; clamp screws remain to detail.
        let slit=abs(p.z-0.365)<0.0006 && p.y>0.315 && abs(p.x)>0.075;
        return (arms||socket||collars) && !bores && !tube_bore && !slit;
    }
    if part<15.5 { return tube(p,vec3d(-tube_half,0.315,0.365),vec3d(tube_half,0.315,0.365),tube_radius,0.002); }
    if part<16.5 {
        return box(p,vec3d(0.0,0.30225,0.365),vec3d(0.025,0.02275,0.018))
            && !rod(p,vec3d(-0.026,0.315,0.365),vec3d(0.026,0.315,0.365),tube_radius+0.0002);
    }
    if part<17.5 {
        let q=p-vec3d(0.0,0.278,0.385);
        let plate=rounded(q.xz,vec2d(bolt_width/2.0+0.014,bolt_height/2.0+0.014),0.014);
        let holes=length(vec2d(abs(q.x)-bolt_width/2.0,abs(q.z)-bolt_height/2.0))<0.0035;
        let slots=rounded(vec2d(abs(q.x)-0.018,q.z),vec2d(0.0045,0.079),0.0045);
        return plate && abs(q.y)<=0.0015 && !holes && !slots;
    }
    if part<18.5 {
        // Clearance shell only: label envelope, stylized forward-curving wings.
        let q=p-vec3d(0.0,0.261,0.405);
        let surface=-0.1524*pow(abs(q.x)/0.1905,3.0);
        let shell=rounded(q.xz,vec2d(0.1905,0.1905),0.045) && q.y>=surface-0.004 && q.y<=surface;
        let bosses=length(vec2d(abs(p.x)-bolt_width/2.0,abs(p.z-0.385)-bolt_height/2.0))<=0.013
            && p.y>=0.249 && p.y<=0.2765;
        return shell||bosses;
    }
    // Cushion clearance envelope, separate assembly part; 80 mm is unmeasured.
    return rounded(p.xy-vec2d(0.0,cy),vec2d(half-0.004,hy-0.004),0.012) && p.z>=0.043 && p.z<=0.123;
}
fn bounds_min()->vec3d {
    let arm=side*(half+0.04);
    if part<0.5 { return vec3d(-half,front,0.020); }
    if part<1.5 { return vec3d(side*0.11-0.01,front+0.025,0.010); }
    if part<2.5 { return vec3d(-0.125,station-0.016,0.004); }
    if part<3.5 { return vec3d(side*0.11-0.012,front+0.043,0.034); }
    if part<4.5 { return vec3d(-0.117,front+0.048,0.027); }
    if part<5.5 { return vec3d(hole_x-0.009,hole_y-0.009,contact_z); }
    if part<6.5 { return vec3d(min(arm,side*0.11)-0.015,0.045,0.010); }
    if part<7.5 { return vec3d(arm-0.0128,0.0472,0.12); }
    if part<8.5 { return vec3d(arm-side*0.0475-0.0725,0.034,0.283); }
    if part<9.5 { return vec3d(arm-side*0.035-0.075,0.05,0.292); }
    if part<10.5 { return vec3d(arm-0.035,-0.095,0.318); }
    if part<11.5 { return vec3d(-0.016-tongue_clearance,-0.005-tongue_clearance,-0.045); }
    if part<12.5 { return vec3d(-0.024,0.247,0.040); }
    if part<13.5 { return vec3d(-0.0148,0.3402,0.215); }
    if part<14.5 { return vec3d(-0.115,0.296,0.346); }
    if part<15.5 { return vec3d(-tube_half,0.315-tube_radius,0.365-tube_radius); }
    if part<16.5 { return vec3d(-0.025,0.2795,0.347); }
    if part<17.5 { return vec3d(-bolt_width/2.0-0.014,0.2765,0.385-bolt_height/2.0-0.014); }
    if part<18.5 { return vec3d(-0.1905,0.1046,0.2145); }
    return vec3d(-half+0.004,front+0.004,0.043);
}
fn bounds_max()->vec3d {
    let arm=side*(half+0.04);
    if part<0.5 { return vec3d(half,rear,0.043); }
    if part<1.5 { return vec3d(side*0.11+0.01,0.215,0.040); }
    if part<2.5 { return vec3d(0.125,station+0.016,0.010); }
    if part<3.5 { return vec3d(side*0.11+0.012,0.202,0.040); }
    if part<4.5 { return vec3d(0.117,0.197,0.0427); }
    if part<5.5 { return vec3d(hole_x+0.009,hole_y+0.009,0.004); }
    if part<6.5 { return vec3d(max(arm,side*0.11)+0.015,0.075,0.25); }
    if part<7.5 { return vec3d(arm+0.0128,0.0728,0.29); }
    if part<8.5 { return vec3d(arm-side*0.0475+0.0725,0.086,0.309); }
    if part<9.5 { return vec3d(arm-side*0.035+0.075,0.07,0.318); }
    if part<10.5 { return vec3d(arm+0.035,0.135,0.338); }
    if part<11.5 { return vec3d(0.016+tongue_clearance,0.005+tongue_clearance,0.060); }
    if part<12.5 { return vec3d(0.024,0.372,0.345); }
    if part<13.5 { return vec3d(0.0148,0.3698,0.385); }
    if part<14.5 { return vec3d(0.115,0.377,0.393); }
    if part<15.5 { return vec3d(tube_half,0.315+tube_radius,0.365+tube_radius); }
    if part<16.5 { return vec3d(0.025,0.325,0.383); }
    if part<17.5 { return vec3d(bolt_width/2.0+0.014,0.2795,0.385+bolt_height/2.0+0.014); }
    if part<18.5 { return vec3d(0.1905,0.2765,0.5955); }
    return vec3d(half-0.004,rear-0.004,0.123);
}
