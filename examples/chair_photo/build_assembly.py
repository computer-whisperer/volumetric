#!/usr/bin/env python3
"""Build the representative complete base around the accepted WGSL seat interface."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from measure import HERE, cli
from build import add_operator


def pose(frame):
    angles=Rotation.from_matrix(np.array(frame['basis']).T).as_euler('xyz',degrees=True)
    return {'rotate':dict(zip(['rx_deg','ry_deg','rz_deg'],angles.tolist())),
            'translate':dict(zip(['dx','dy','dz'],frame['origin']))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=HERE/'work')
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args();w=args.work.resolve()
    mount=json.loads((w/'measurements.json').read_text())
    receiver=json.loads((w/'receiver-fit.json').read_text())
    assembly=json.loads((w/'assembly-measurements.json').read_text())
    subprocess.run([sys.executable,str(HERE/'build.py'),'--work',str(w)],check=True,stdout=subprocess.DEVNULL)
    project=w/'base.vproj';shutil.copyfile(w/'mount.vproj',project)
    select=['view-select','-i',w/'survey.vviews','-p',project,'--embed','preview','--preview-px','1600','--asset-id','assembly_photos']
    for view in ['DSC00753','DSC00760','DSC00762']:select+=['--id',view]
    cli(*select)
    for name in ['base','mechanism','receiver']:
        cli('project-add-asset','-p',project,'-i',HERE/f'{name}.wgsl','--type','wgsl','--asset-id',name+'_source')

    def op(operator,inputs,output,exported=False):add_operator(project,operator,inputs,output,exported)
    params={}
    def part(name,source,parameters,frame):
        params[name]=parameters
        op('wgsl_script_operator',['asset:'+source+'_source',parameters],name+'_local')
        op('pose_operator',['asset:'+name+'_local',pose(frame)],name,True)

    column=assembly['column'];star=assembly['star'];angle=np.deg2rad(star['heading_world_deg'])
    base_frame={'origin':[*column['ground_axis_xy'],0.0], 'basis':[[float(np.cos(angle)),float(np.sin(angle)),0.0],[float(-np.sin(angle)),float(np.cos(angle)),0.0],[0.,0.,1.]]}
    op('subspace_operator',[{'kind':'frame'},base_frame['origin'],*base_frame['basis'][:2]],'base_frame')
    p={'star_radius':star['radius_mm']/1000,'joint_height':star['joint_height_mm']/1000,'star_phase':float(np.deg2rad(star['phase_deg'])),
       'column_outer_radius':column['outer_diameter_mm']/2000,'column_rod_radius':column['rod_diameter_mm']/2000,
       'column_collar_z':column['collar_height_mm']/1000,'column_top_z':mount['frame']['origin'][2]-.052,
       'assumed_hub_top_z':.182,'assumed_wheel_radius':.025,'caster_yaw':0.0}
    for index,name in enumerate(['gas_lift','star_base',*[f'caster_{i+1}' for i in range(5)]]):
        part(name,'base',{**p,'part':float(index),'caster_yaw':assembly['caster_yaws_rad'].get(str(index-2),0.0)},base_frame)
    p={}
    for key in ['rear_pin','rear_upper_fastener','rear_lower_fastener']:
        local=assembly['features'][key]['local_mm']
        p[key+'_x']=local[0]/1000;p[key+'_y']=local[1]/1000;p[key+'_z']=local[2]/1000
    p.update(receiver_y=assembly['knob_center_local_m'][1],assumed_knob_z=assembly['knob_center_local_m'][2])
    for index,name in enumerate(['mechanism_housing','rear_pins','backrest_clamp_knob','rear_control','lift_control','tilt_control']):
        control=[-.195,.100,-.080]
        root=.070
        if index>=3:
            control=(np.array(assembly['features'][name]['local_mm'])/1000).tolist()
            root=[.205,.070,.180][index-3]
        part(name,'mechanism',{**p,'part':float(index),'control_end_x':control[0],
             'control_end_y':control[1],'control_end_z':control[2],'control_root_y':root},mount['frame'])
    p={'mouth_width':receiver['mouth_width_mm']/1000,'mouth_depth':receiver['mouth_depth_mm']/1000,
       'mouth_corner_radius':receiver['mouth_corner_radius_mm']/1000,'assumed_throat_inset':.001,
       'assumed_leadin_depth':.003,'assumed_socket_depth':.065,'assumed_receiver_wall':.003}
    part('backrest_receiver','receiver',p,receiver['world_frame'])
    frame=receiver['world_frame']
    op('subspace_operator',[{'kind':'frame'},frame['origin'],*frame['basis'][:2]],'backrest_mouth_frame')
    # The measured cap center locates a visible transverse pin. Its role as
    # the sole tilt axis and the actual motion range are not established.
    pin=assembly['features']['rear_pin']['local_mm'];basis=np.array(mount['frame']['basis'])
    pin_origin=np.array(mount['frame']['origin'])+np.array([0,pin[1],pin[2]])@basis/1000
    op('subspace_operator',[{'kind':'frame'},pin_origin.tolist(),*mount['frame']['basis'][:2]],'rear_pin_reference_frame')
    (w/'assembly-parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    (w/'base-run.json').write_text(cli('project-run','-p',project,'--json'))
    if args.render:
        common=['render','-i',project,'--up','0,0,1','--resolution','192','--no-sharp','--no-simplify','--grid','0','--no-ssao']
        target=np.array([*column['ground_axis_xy'],.245])
        camera=target+np.array([-.85,.95,.60])
        from measure import vector
        print(cli(*common,'--camera-pos',vector(camera),'--camera-target',vector(target),'--width','1600','--height','1600','--fov','38','-o',w/'base_iso.png'))
        for view in ['DSC00755','DSC00758','DSC00753','DSC00762']:
            print(cli(*common,'--through',('views:' if view in ['DSC00755','DSC00758'] else 'assembly_photos:')+view,'--overlay','edge','--width','2400','--height','1600','-o',w/f'base-overlay-{view}.png'))
        detail=['render','-i',project,'--asset','backrest_receiver','--asset','rear_pins','--asset','backrest_clamp_knob','--up','0,0,1','--resolution','192','--no-sharp','--no-simplify','--grid','0','--no-ssao']
        target=np.array(receiver['world_frame']['origin'])+np.array([0,0,-.022])
        camera=target+np.array([-.13,.17,.13])
        print(cli(*detail,'--camera-pos',vector(camera),'--camera-target',vector(target),'--fov','40','-o',w/'backrest-detail_iso.png'))
        print(cli(*detail,'--views','top','--projection','ortho','--ortho-scale','.16','-o',w/'backrest-detail_top.png'))
    print(f'Built {project}')


if __name__=='__main__':main()
