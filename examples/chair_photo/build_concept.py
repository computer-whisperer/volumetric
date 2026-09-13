#!/usr/bin/env python3
"""Build concept A on the measured base with native WGSL and assembly operators."""
import argparse
import json
from pathlib import Path

import numpy as np
import volumetric as v
from build_assembly import pose
from measure import HERE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, default=HERE/'work')
    ap.add_argument('--inches', type=float, default=20)
    ap.add_argument('--state', type=Path)
    ap.add_argument('--render', action='store_true')
    args = ap.parse_args()
    if not 14 <= args.inches <= 22:
        ap.error('Concept tray family is 14 through 22 inches')
    w = args.work.resolve(); out = w/'concept'; out.mkdir(exist_ok=True)
    stem = f'chair-{args.inches:g}'+('-posed' if args.state else '')
    base = v.Assembly.load(str(w/'base.vasm'))
    mount = json.loads((w/'measurements.json').read_text())
    receiver = json.loads((w/'receiver-fit.json').read_text())
    evidence = json.loads((HERE/'backrest-report.json').read_text())
    dims = evidence['dimensions_mm']
    frame = mount['frame']; basis = np.array(frame['basis']); origin = np.array(frame['origin'])
    def world(p): return (origin+np.array(p)@basis).tolist()
    half = (args.inches*.0254+.008)/2; rear = .235; front = rear-2*half
    params = {'half':half, 'front':front, 'rear':rear,
              'tube_radius':dims['tube_apparent_diameter']/2000,
              'tube_half':dims['tube_length']/2000,
              'bolt_width':dims['bolt_width']/1000, 'bolt_height':dims['bolt_height']/1000}
    project = v.Project()
    project.add_asset((HERE/'concept.wgsl').read_bytes(), 'wgsl', id='concept_source')
    project.add_asset(json.dumps({'design':(HERE/'CONCEPT_A.md').read_text(),
        'hardware':evidence, 'observations':json.loads((HERE/'backrest-observations.json').read_text())}).encode(), 'blob', id='concept_provenance')
    # Flatten the existing tree, retaining all original joints and rest-world part bytes.
    # Decoded joints resolve axis_input references to self-contained inline axes.
    parts = list(base.parts); joints = list(base.mechanism.joints)
    for name in parts:
        project.add_model(base.part_model(name), id=name)
    part_params = {}; printed = []
    def geometry(name, kind, local_frame=frame, **kw):
        p = {**params,'part':float(kind), **kw}; part_params[name] = p
        project.add_op('wgsl_script_operator', ['concept_source',p], output=name+'_local', export=False)
        project.add_op('pose_operator', [name+'_local',pose(local_frame)], output=name, export=False)
    def add(name, kind, parent='crossbar', joint=None, printed_part=False, local_frame=frame, **kw):
        geometry(name,kind,local_frame,**kw)
        parts.append(name)
        joints.append({'name':name+'_mount','parent':parent,'child':name,**(joint or {})})
        if printed_part: printed.append(name)
    def slide(name, direction, maximum, drive=None):
        j = {'name':name,'kind':'prismatic','axis':{'origin':world([0,0,0]),
            'direction':(np.array(direction)@basis).tolist()},'min':0.,'max':maximum}
        if drive: j['drive'] = {'joint':drive,'ratio':1.,'offset':0.}
        return j
    add('seat_pan',0)
    arm_frame = {'origin':world([0,-.05,0]),'basis':frame['basis']}
    for side,label in [(-1.,'left'),(1.,'right')]:
        add(label+'_seat_rail',1,side=side)
        add(label+'_pan_backing',3,parent=label+'_seat_rail',side=side)
        add(label+'_arm_post',6,local_frame=arm_frame,parent=label+'_seat_rail',side=side)
        add(label+'_arm_inner',7,local_frame=arm_frame,parent=label+'_arm_post',side=side,
            joint=slide(label+'_arm_height',[0,0,1],.08))
        add(label+'_arm_guide',8,local_frame=arm_frame,parent=label+'_arm_inner',side=side,printed_part=True)
        # Symmetric setup coupling, not a claim of a mechanical linkage.
        add(label+'_arm_carriage',9,local_frame=arm_frame,parent=label+'_arm_guide',side=side,
            joint=slide('arm_inset' if side<0 else 'right_arm_inset',[-side,0,0],.075,
                        drive='arm_inset' if side>0 else None))
        add(label+'_arm_pad',10,local_frame=arm_frame,parent=label+'_arm_carriage',side=side,printed_part=True)
    for label,a,b,y in [('front','cross_a','cross_b',0.),('rear','rail_a','rail_b',.17465)]:
        pa=mount['features'][a]['local_mm']; pb=mount['features'][b]['local_mm']
        add(label+'_adapter',2,station=y,hole_x=pa[0]/1000,hole_y=pa[1]/1000,
            hole2_x=pb[0]/1000,hole2_y=pb[1]/1000)
        for n in [a,b]:
            x,y,z=np.array(mount['features'][n]['local_mm'])/1000
            add(n+'_shim',5,parent=label+'_adapter',hole_x=x,hole_y=y,contact_z=z)
    add('pan_screw_envelopes',4,parent='seat_pan')
    add('receiver_tongue',11,parent='backrest_receiver',local_frame=receiver['world_frame'])
    geometry('tongue_clearance',11,receiver['world_frame'],tongue_clearance=.0003)
    geometry('lower_spine_blank',12)
    project.add_op('boolean_operator',['lower_spine_blank','tongue_clearance',{'op':'subtract'}],output='lower_spine',export=False)
    parts.append('lower_spine');joints.append({'name':'spine_mount','parent':'receiver_tongue','child':'lower_spine'})
    add('upper_spine',13,parent='lower_spine',joint=slide('backrest_height',[0,0,1],.08))
    add('backrest_yoke',14,parent='upper_spine',printed_part=True)
    add('oem_backrest_tube',15,parent='backrest_yoke')
    add('oem_tube_clamp',16,parent='oem_backrest_tube')
    add('oem_slotted_plate',17,parent='oem_tube_clamp')
    add('backrest_shell_proxy',18,parent='oem_slotted_plate')
    add('cushion_envelope',19,parent='seat_pan')
    project.add_op('mechanism_operator',[{'parts':parts,'joints':joints},None],output='chair_mechanism',export=False)
    state = json.loads(args.state.read_text()) if args.state else {}
    project.add_op('assemble_operator',['chair_mechanism',*parts,state],output='chair')
    interfaces = {
        'cushion_support':{'part':'seat_pan','rest_frame':{'origin':world([0,(front+rear)/2,.043]),'basis':frame['basis']}},
        'backrest_tube':{'part':'oem_backrest_tube','rest_frame':{'origin':world([0,.315,.365]),'basis':frame['basis']}},
        'receiver_mouth':{'part':'backrest_receiver','rest_frame':receiver['world_frame']}}
    project.add_asset(json.dumps(interfaces).encode(),'blob',id='interface_attachments')
    assert project.validate() == []
    project.save(str(out/f'{stem}.vproj'))
    exports = project.run(); assembly = exports['chair'].assembly()
    assembly.save(str(out/f'{stem}.vasm'))
    transforms = dict(zip(parts,assembly.poses()))
    for d in interfaces.values():
        t = transforms[d['part']]; f = d['rest_frame']
        d['posed_frame'] = {'origin':(t[:,:3]@f['origin']+t[:,3]).tolist(),
                            'basis':(np.array(f['basis'])@t[:,:3].T).tolist()}
    # Compile local part bounds separately: the print envelope must not use a
    # world-rotated AABB, which would penalize the arbitrary survey orientation.
    pp = v.Project(); pp.add_asset((HERE/'concept.wgsl').read_bytes(),'wgsl',id='source')
    for n in printed:
        pp.add_op('wgsl_script_operator',['source',part_params[n]],output=n)
    boxes = {}
    for n,a in pp.run().items():
        lo,hi = a.bounds(); size = (np.array(hi)-lo)*1000
        boxes[n] = {'local_size_mm':size.tolist(), 'fits_carbon_axis_aligned':bool(np.all(size <= [410,256,460]))}
    report = {'concept':'A','cushion_inches':args.inches,'parameters':params,'parts':parts,
              'states':assembly.mechanism.state_keys,'state':assembly.state,'interfaces':interfaces,
              'printed_parts':boxes,'part_parameters':part_params,
              'limitations':'Layout and kinematic model; see CONCEPT_A.md. Shell and cushion are proxies. Section sizes, fastener envelopes and joint travel are provisional; locking, positive retention, bend allowances and structural qualification remain.'}
    (out/f'{stem}.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.render:
        from PIL import Image
        from audit_assembly import posed_parts
        posed = posed_parts(exports['chair'])
        common = dict(width=1200,height=1200,fov=38,up=basis[2].tolist(),resolution=256,
                      sharp=False,simplify=False,grid=0,ssao=False)
        target=world([0,.02,.05])
        for label,offset in [('front',[-1.05,-1.5,.85]),('rear',[1.05,1.5,.85])]:
            cam=(np.array(target)+np.array(offset)@basis).tolist()
            result=v.render([a for n,a in posed.items() if n != 'cushion_envelope'],camera=(cam,target),**common)
            Image.fromarray(result.image).save(out/f'{stem}-{label}.png')
        underside = [a for n,a in posed.items() if n not in ['backrest_shell_proxy','cushion_envelope']]
        target=world([0,.025,.025]); cam=(np.array(target)+np.array([-.75,-.7,-.65])@basis).tolist()
        result=v.render(underside,camera=(cam,target),**common)
        Image.fromarray(result.image).save(out/f'{stem}-underside.png')
    print(f'{out/stem}: {len(parts)} parts, {len(assembly.mechanism.state_keys)} states; Carbon envelopes {boxes}')


if __name__ == '__main__':
    main()
