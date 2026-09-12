#!/usr/bin/env python3
"""Recover approximate base datums using volumetric's existing survey."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from measure import HERE, cli, triangulate, vector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=HERE/'work')
    args=parser.parse_args(); work=args.work.resolve(); survey=work/'survey.vviews'
    obs=json.loads((HERE/'assembly-observations.json').read_text())
    mount=json.loads((work/'measurements.json').read_text()); origin=np.array(mount['frame']['origin']); basis=np.array(mount['frame']['basis'])
    features={}
    for name, pixels in obs['features'].items():
        result=triangulate(survey,pixels)
        result['local_mm']=((np.array(result['world'])-origin)@basis.T*1000).tolist()
        features[name]=result
    collar=np.array(features['column_collar']['world'])

    def pick(view,pixels,point,normal):
        args=['view-pick','-i',survey,'--view',view,'--plane-point',vector(point),'--plane-normal',vector(normal),'--json']
        for pixel in pixels:args+=['--pixel',vector(pixel)]
        return np.array([p['world'] for p in json.loads(cli(*args))['picked']])

    ends=obs['star_endpoints']
    def star(z):
        return pick(ends['view'],ends['pixels'],[0,0,z],[0,0,1])
    def cost(z):
        radii=np.linalg.norm(star(z)[:,:2]-collar[:2],axis=1)
        return np.var(radii)
    fit=minimize_scalar(cost,bounds=(.04,.12),method='bounded',options={'xatol':1e-5})
    points=star(fit.x); offsets=points[:,:2]-collar[:2]
    radius=np.linalg.norm(offsets,axis=1).mean()
    heading=np.arctan2(basis[0,1],basis[0,0])
    phases=np.arctan2(offsets[:,1],offsets[:,0])-heading
    phase=np.arctan2(np.sin(5*phases).mean(),np.cos(5*phases).mean())/5
    cameras=json.loads((work/'cameras.json').read_text())
    view=obs['column_widths']['view'];cam=next(v for v in cameras['views'] if v['id']==view)
    normal=np.array(cam['position'])-collar;normal[2]=0;normal/=np.linalg.norm(normal)
    widths={}
    for key in ['outer','rod']:
        p=pick(view,obs['column_widths'][key],collar,normal)
        widths[key+'_diameter_mm']=float(np.linalg.norm(p[1]-p[0])*1000)
    wheels=obs['wheel_centers']; caster_yaws={}
    for index, pixel in wheels['pixels'].items():
        point=pick(wheels['view'],[pixel],[0,0,wheels['assumed_height_m']],[0,0,1])[0]
        angle=heading+phase+int(index)*2*np.pi/5
        joint=collar[:2]+radius*np.array([np.cos(angle),np.sin(angle)])
        d=point[:2]-joint
        caster_yaws[index]=float(np.arctan2(d[1],d[0])-angle)
    knob=obs['knob_center']
    knob_point=pick(knob['view'],[knob['pixel']],origin+basis[0]*knob['assumed_face_x_m'],basis[0])[0]
    knob_local=(knob_point-origin)@basis.T
    result={'features':features,'column':{'ground_axis_xy':collar[:2].tolist(),'collar_height_mm':float(collar[2]*1000),**widths},
        'caster_yaws_rad':caster_yaws,'knob_center_local_m':knob_local.tolist(),
        'star':{'joint_height_mm':float(fit.x*1000),'radius_mm':float(radius*1000), 'phase_deg':float(np.rad2deg(phase)), 'heading_world_deg':float(np.rad2deg(heading)),
        'picked_world':points.tolist(),'radial_rms_mm':float(np.sqrt(cost(fit.x))*1000)},
        'limitations':'Approximate collar ellipse centers and caster joint centers, horizontal floor/card datum, circular fivefold star, and vertical gas lift. Radial fit residual is not an accuracy bound.'}
    (work/'assembly-measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
