#!/usr/bin/env python3
"""Build the representative complete base around the accepted WGSL seat interface."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from measure import HERE
from build import mount_parameters
import volumetric as v


def pose(frame):
    angles=Rotation.from_matrix(np.array(frame['basis']).T).as_euler('xyz',degrees=True)
    return {'rotate':dict(zip(['rx_deg','ry_deg','rz_deg'],angles.tolist())),
            'translate':dict(zip(['dx','dy','dz'],frame['origin']))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=HERE/'work')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--state',type=Path,help='Joint state JSON; writes base-posed.* and preserves the canonical rest build')
    args=parser.parse_args();w=args.work.resolve()
    output_stem = 'base-posed' if args.state else 'base'
    mount=json.loads((w/'measurements.json').read_text())
    receiver=json.loads((w/'receiver-fit.json').read_text())
    assembly=json.loads((w/'assembly-measurements.json').read_text())
    project = v.Project()
    observations = json.loads((HERE/'observations.json').read_text())
    rim = json.loads((HERE/'receiver-observations.json').read_text())
    views = v.ViewSet.load(str(w/'survey.vviews'))
    picks = {name: {view: px for view, px in pixels.items() if view in observations['fit_views']}
             for name, pixels in observations['features'].items()}
    checks = {name: {view: px for view, px in pixels.items() if view in observations['check_views']}
              for name, pixels in observations['features'].items()}
    views = views.with_picks(picks, check=checks).with_contours({'receiver_mouth': rim['contours']})
    project.add_views(views.select(ids=['DSC00753','DSC00755','DSC00756','DSC00758','DSC00760','DSC00762'],
                                  embed='preview', preview_px=1600), id='views')
    project.add_asset(json.dumps({'mount': observations, 'receiver': rim,
        'kinematic_assumptions': (HERE/'ARTICULATION.md').read_text()}).encode(), 'blob', id='evidence_provenance')
    for name in ['mount','base','mechanism','receiver']:
        project.add_asset((HERE/f'{name}.wgsl').read_bytes(), 'wgsl', id=name+'_source')

    def op(operator, inputs, output, exported=False):
        return project.add_op(operator, inputs, output=output, export=exported)
    params={}
    def part(name,source,parameters,frame):
        params[name]=parameters
        op('wgsl_script_operator',[source+'_source',parameters],name+'_local')
        op('pose_operator',[name+'_local',pose(frame)],name)

    frame = mount['frame']
    op('subspace_operator', [{'kind':'frame'},frame['origin'],*frame['basis'][:2]], 'mount_frame')
    mounting = mount_parameters(mount)
    (w/'parameters.json').write_text(json.dumps(mounting,indent=2)+'\n')
    for index, name in enumerate(['crossbar','rail']):
        part(name,'mount',{**mounting,'part':float(index)},frame)

    column=assembly['column'];star=assembly['star'];angle=np.deg2rad(star['heading_world_deg'])
    base_frame={'origin':[*column['ground_axis_xy'],0.0], 'basis':[[float(np.cos(angle)),float(np.sin(angle)),0.0],[float(-np.sin(angle)),float(np.cos(angle)),0.0],[0.,0.,1.]]}
    op('subspace_operator',[{'kind':'frame'},base_frame['origin'],*base_frame['basis'][:2]],'base_frame')
    p={'star_radius':star['radius_mm']/1000,'joint_height':star['joint_height_mm']/1000,'star_phase':float(np.deg2rad(star['phase_deg'])),
       'column_outer_radius':column['outer_diameter_mm']/2000,'column_rod_radius':column['rod_diameter_mm']/2000,
       'column_collar_z':column['collar_height_mm']/1000,'column_top_z':mount['frame']['origin'][2]-.052,
       'assumed_hub_top_z':.182,'assumed_wheel_radius':.025,'caster_yaw':0.0}
    for index, name in enumerate(['gas_lift_body','gas_lift_rod','star_base']):
        part(name,'base',{**p,'part':float(index)},base_frame)
    caster_axes = []
    for i in range(5):
        yaw = assembly['caster_yaws_rad'].get(str(i),0.0)
        part(f'caster_{i+1}_fork','base',{**p,'part':float(3+i),'caster_yaw':yaw},base_frame)
        part(f'caster_{i+1}_wheels','base',{**p,'part':float(8+i),'caster_yaw':yaw},base_frame)
        a = p['star_phase'] + i*2*np.pi/5
        at = np.array([p['star_radius']*np.cos(a),p['star_radius']*np.sin(a),p['joint_height']])
        wheel = at + np.array([.018*np.cos(a+yaw),.018*np.sin(a+yaw),p['assumed_wheel_radius']-p['joint_height']])
        basis = np.array(base_frame['basis'])
        caster_axes.append((np.array(base_frame['origin'])+at@basis,
                            np.array(base_frame['origin'])+wheel@basis,
                            np.array([-np.sin(a+yaw),np.cos(a+yaw),0.0])@basis))
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
    op('subspace_operator',[{'kind':'line'},base_frame['origin'],[0.,0.,1.],None],'lift_axis')
    joints = [
        {'name':'ground','child':'star_base'},
        {'name':'column_mount','parent':'star_base','child':'gas_lift_body'},
        {'name':'lift','kind':'prismatic','parent':'gas_lift_body','child':'gas_lift_rod',
         'axis_input':0,'min':-.020,'max':.080},
        {'name':'swivel','kind':'revolute','continuous':True,'parent':'gas_lift_rod',
         'child':'mechanism_housing','axis_input':0},
    ]
    for name in ['crossbar','rail','rear_pins','backrest_clamp_knob','rear_control','lift_control','tilt_control','backrest_receiver']:
        joints.append({'name':name+'_mount','parent':'mechanism_housing','child':name})
    for i,(stem,axle,direction) in enumerate(caster_axes,1):
        joints += [
            {'name':f'caster_{i}_swivel','kind':'revolute','continuous':True,'parent':'star_base',
             'child':f'caster_{i}_fork','axis':{'origin':stem.tolist(),'direction':[0.,0.,1.]}},
            {'name':f'caster_{i}_roll','kind':'revolute','continuous':True,'parent':f'caster_{i}_fork',
             'child':f'caster_{i}_wheels','axis':{'origin':axle.tolist(),'direction':direction.tolist()}},
        ]
    mechanism = {'parts':list(params),'joints':joints}
    op('mechanism_operator',[mechanism,'lift_axis'],'base_mechanism')
    state = json.loads(args.state.read_text()) if args.state else {}
    op('assemble_operator',['base_mechanism',*params,state],'chair_base',True)
    assert project.validate() == []
    interfaces = {
        'seat_mount': {'part':'crossbar','rest_frame':mount['frame']},
        'backrest_mouth': {'part':'backrest_receiver','rest_frame':receiver['world_frame']},
        'rear_pin_reference': {'part':'rear_pins','rest_frame':{'origin':pin_origin.tolist(),'basis':mount['frame']['basis']}},
    }
    project.add_asset(json.dumps(interfaces).encode(),'blob',id='interface_attachments')
    project.save(str(w/f'{output_stem}.vproj'))
    exports = project.run()
    articulated = exports['chair_base'].assembly()
    articulated.save(str(w/f'{output_stem}.vasm'))
    articulated.mechanism.save(str(w/f'{output_stem}.vmech'))
    transforms = dict(zip(articulated.parts,articulated.poses()))
    for interface in interfaces.values():
        transform = transforms[interface['part']]
        rest = interface['rest_frame']
        interface['posed_frame'] = {
            'origin': (transform[:,:3]@rest['origin']+transform[:,3]).tolist(),
            'basis': (np.array(rest['basis'])@transform[:,:3].T).tolist(),
        }
    (w/f'{output_stem}-interfaces.json').write_text(json.dumps({'state':articulated.state,'interfaces':interfaces},indent=2)+'\n')
    (w/'base-mechanism.json').write_text(json.dumps(mechanism,indent=2)+'\n')
    (w/'assembly-parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    if args.render:
        from PIL import Image
        from audit_assembly import posed_parts
        common = dict(assets=['chair_base'],up=[0,0,1],resolution=96,sharp=False,simplify=False,grid=0,ssao=False)
        target=np.array([*column['ground_axis_xy'],.245])
        camera=target+np.array([-.85,.95,.60])
        frame=v.render(project,camera=(camera,target),width=1200,height=1200,fov=38,**common)
        Image.fromarray(frame.image).save(w/f'{output_stem}_iso.png')
        for view in ['DSC00755','DSC00756','DSC00760']:
            frame=v.render(project,through='views:'+view,overlay='edge',marks=True,width=1548,height=1032,**common)
            Image.fromarray(frame.image).save(w/f'{output_stem}-overlay-{view}.png')
        detail = posed_parts(exports['chair_base'])
        target = np.array(interfaces['backrest_mouth']['posed_frame']['origin'])+[0,0,-.022]
        frame = v.render([detail[n] for n in ['backrest_receiver','rear_pins','backrest_clamp_knob']],
                         camera=(target+[-.13,.17,.13],target),fov=40,width=1000,height=1000,
                         up=[0,0,1],resolution=192,sharp=False,simplify=False,grid=0,ssao=False)
        Image.fromarray(frame.image).save(w/f'{output_stem}-backrest-detail.png')
    print(f"Built {w/f'{output_stem}.vproj'}: {len(params)} parts, {len(exports['chair_base'].assembly().mechanism.state_keys)} states")


if __name__=='__main__':main()
