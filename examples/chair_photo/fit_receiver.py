#!/usr/bin/env python3
"""Fit one planar rounded rectangular mouth to multi-view rim observations.

All camera projection and ray/plane intersections are performed by volumetric.
SciPy fits Euclidean profile dimensions and the plane, without requiring
corresponding contour samples in different pictures.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from measure import HERE, cli, vector


def profile_distance(points, halfsize, radius):
    q = np.abs(points) - halfsize + radius
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0) - radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=HERE/'work')
    parser.add_argument('--audit', action='store_true', help='Write original-pixel crops with observed and projected mouth samples')
    args = parser.parse_args()
    work = args.work.resolve()
    report = json.loads((work/'measurements.json').read_text())
    observations = json.loads((HERE/'receiver-observations.json').read_text())
    origin = np.array(report['frame']['origin'])
    basis = np.array(report['frame']['basis'])
    survey = work/'survey.vviews'

    def geometry(p):
        # Fit variables: center (mm), extrinsic XYZ rotations (degrees),
        # full mouth width/depth (mm), corner radius as a fraction of depth/2.
        center = origin + p[:3] @ basis / 1000
        axes = Rotation.from_euler('xyz', p[3:6], degrees=True).as_matrix().T @ basis
        return center, axes

    def samples(p, view):
        center, axes = geometry(p)
        call = ['view-pick','-i',survey,'--view',view,'--plane-point',vector(center),
                '--plane-normal',vector(axes[2]),'--json']
        for pixel in observations['contours'][view]:
            call += ['--pixel',vector(pixel)]
        points = np.array([q['world'] for q in json.loads(cli(*call))['picked']])
        return (points-center) @ axes[:2].T * 1000

    def residual(p, views):
        return np.concatenate([profile_distance(samples(p,v), p[6:8]/2, p[8]*p[7]/2) for v in views])

    def jacobian(p):
        steps = [.05]*8 + [.005]
        columns = []
        for i, step in enumerate(steps):
            d = np.zeros(9); d[i] = step
            columns.append((residual(p+d, observations['fit_views']) - residual(p-d, observations['fit_views'])) / (2*step))
        return np.array(columns).T

    initial = [0.1, 275, 5, 1, 1, 0.1, 38, 19, .85]
    result = least_squares(lambda p: residual(p, observations['fit_views']), initial,
        bounds=([-10,245,-15,-25,-25,-15,20,10,.1],[10,290,20,25,25,15,50,30,1]),
        jac=jacobian, x_scale='jac', ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=100)
    if not result.success:
        raise RuntimeError(result.message)
    p=result.x
    center,axes=geometry(p)
    stats={}
    for view in observations['fit_views']+observations['check_views']:
        r=residual(p,[view])
        stats[view]={'rms_mm':float(np.sqrt(np.mean(r*r))), 'max_abs_mm':float(np.max(np.abs(r)))}
        points=samples(p,view)
        stats[view]['rim_plane_coordinates_mm']=points.tolist()
        independent=least_squares(lambda q: profile_distance(points-q[:2],q[2:]/2,q[3]/2),
                                  [0.,0.,p[6],p[7]],bounds=([-10,-10,20,10],[10,10,50,30]))
        stats[view]['independent_capsule_on_shared_plane']={
            'center_offset_mm':independent.x[:2].tolist(),
            'width_mm':float(independent.x[2]),'depth_mm':float(independent.x[3]),
            'rms_mm':float(np.sqrt(np.mean(independent.fun**2)))}
    output={'description':observations['description'], 'fit_views':observations['fit_views'],
        'check_views':observations['check_views'], 'center_local_mm':p[:3].tolist(),
        'rotation_xyz_deg':p[3:6].tolist(), 'mouth_width_mm':float(p[6]),
        'mouth_depth_mm':float(p[7]), 'mouth_corner_radius_mm':float(p[8]*p[7]/2),
        'world_frame':{'origin':center.tolist(),'basis':axes.tolist()},'residuals':stats,
        'limitations':'Residuals are distances on the fitted rim plane, not accuracy bounds. Manual traces share camera calibration. Opening below lead-in and insertion depth are not measured by this fit.',
        'optimizer':{'evaluations':result.nfev,'message':result.message,'radius_at_capsule_bound':bool(p[8]>.999)},
        'profile_note':'Rounded rectangle fit reaches capsule limit when radius_at_capsule_bound is true; radius is then constrained rather than independently resolved. The mouth normal does not prove insertion direction.'}
    (work/'receiver-fit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['residuals','world_frame']},indent=2))
    print({v:{k:r[k] for k in ['rms_mm','max_abs_mm']} for v,r in stats.items()})
    if args.audit:
        xy=[]
        half=p[6:8]/2; radius=p[8]*p[7]/2
        for sx,sy,start in [(1,1,0),(-1,1,90),(-1,-1,180),(1,-1,270)]:
            for theta in np.deg2rad(np.linspace(start,start+90,9)):
                xy.append(np.array([sx,sy])*(half-radius)+radius*np.array([np.cos(theta),np.sin(theta)]))
        world=center+np.array(xy)@axes[:2]/1000
        for view in observations['fit_views']+observations['check_views']:
            pixels=observations['contours'][view]
            extent=np.ptp(pixels,axis=0)+140
            command=['view-crop','-i',survey,'--view',view,'--center',vector(np.mean(pixels,axis=0)),
                     '--size',f'{int(extent[0])}x{int(extent[1])}','--scale','2','--grid','0',
                     '-o',work/f'rim-audit-{view}.png']
            for pixel in pixels: command+=['--mark',vector(pixel)]
            for point in world: command+=['--mark-world',vector(point)]
            cli(*command)


if __name__ == '__main__':
    main()
