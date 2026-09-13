#!/usr/bin/env python3
"""Check concept occupancy, interfaces and motion; no structural rating implied."""
import argparse
import json
from pathlib import Path

import numpy as np
import volumetric as v
from audit_assembly import posed_parts
from measure import HERE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,default=HERE/'work')
    ap.add_argument('--inches',type=float,default=20)
    args = ap.parse_args(); w = args.work.resolve(); out=w/'concept'; stem=f'chair-{args.inches:g}'
    info = json.loads((out/f'{stem}.json').read_text())
    mount = json.loads((w/'measurements.json').read_text())
    frame = mount['frame']; basis = np.array(frame['basis']); origin=np.array(frame['origin'])
    def world(points): return origin+np.asarray(points)@basis
    project=v.Project.open(str(out/f'{stem}.vproj')); exports=project.run(); asset=exports['chair']
    assembly=asset.assembly(); parts=posed_parts(asset)
    base=v.Assembly.load(str(w/'base.vasm'))
    assert all(base.part_model(n)==assembly.part_model(n) for n in base.parts)
    assert len(parts)==53 and len(assembly.mechanism.state_keys)==16
    assert v.Assembly.decode(assembly.encode()).parts==assembly.parts
    half=info['parameters']['half']; front=info['parameters']['front']; rear=.235
    # Dense area sample inside the intended cushion rectangle, just below the
    # support top. Report small countersink openings rather than claiming 100%.
    xy=np.array([(x,y) for x in np.arange(-half+.004,half-.004,.002)
                       for y in np.arange(front+.004,rear-.004,.002)])
    support=parts['seat_pan'].occupied(world(np.column_stack([xy,np.full(len(xy),.0429)])))
    fraction=float(support.mean()); assert fraction>.995, fraction
    body=world(np.column_stack([xy,np.full(len(xy),.0431)]))
    assert not parts['pan_screw_envelopes'].occupied(body).any()
    assert not parts['seat_pan'].occupied(body).any()
    # Real positive controls at the screw head tops and plate surfaces.
    heads=np.array([[x,y,.04269] for x in [-.11,.11] for y in [front+.055,.19]])
    assert parts['pan_screw_envelopes'].occupied(world(heads)).all()
    assert not parts['seat_pan'].occupied(world(heads)).any()
    # Measured station bores remain open in both adapters and shims.
    for label,names in [('front',['cross_a','cross_b']),('rear',['rail_a','rail_b'])]:
        for n in names:
            x,y,z=np.array(mount['features'][n]['local_mm'])/1000
            assert not parts[label+'_adapter'].occupied(world([[x,y,.007]])).any()
            assert not parts[n+'_shim'].occupied(world([[x,y,(z+.004)/2]])).any()
            assert parts[n+'_shim'].occupied(world([[x+.006,y,(z+.004)/2]])).all()
    # Sample pairs in world AABB intersections. This is a collision smoke test,
    # not a tolerance proof; deterministic occupied witness points caused fixes.
    rng=np.random.default_rng(20260912)
    checks=[]
    def clear(a,b,models=parts,count=25000):
        alo,ahi=models[a].bounds(); blo,bhi=models[b].bounds()
        lo=np.maximum(alo,blo); hi=np.minimum(ahi,bhi)
        if np.any(hi<=lo):
            checks.append([a,b,0]);return
        points=rng.uniform(lo,hi,(count,3))
        collision=models[a].occupied(points)&models[b].occupied(points)
        assert not collision.any(), (a,b,((points[collision][0]-origin)@basis.T).tolist())
        checks.append([a,b,count])
    for new in ['receiver_tongue','lower_spine','upper_spine','backrest_yoke']:
        for retained in ['oem_tube_clamp','oem_slotted_plate','backrest_shell_proxy']:
            clear(new,retained)
    for side in ['left','right']:
        clear(side+'_arm_post','seat_pan')
        clear(side+'_arm_inner',side+'_arm_post')
        clear(side+'_arm_carriage',side+'_arm_guide')
        clear(side+'_arm_pad','backrest_shell_proxy')
    clear('receiver_tongue','backrest_receiver')
    clear('receiver_tongue','lower_spine')
    clear('upper_spine','lower_spine')
    clear('backrest_yoke','oem_backrest_tube')
    # Independent translation oracle, including CAD-only symmetric setup coupling.
    state={'left_arm_height':.08,'right_arm_height':.04,'arm_inset':.075,'backrest_height':.08}
    poses=dict(zip(assembly.parts,assembly.mechanism.pose(state)))
    for n,delta in [('left_arm_pad',[.075,0,.08]),('right_arm_pad',[-.075,0,.04]),
                    ('oem_backrest_tube',[0,0,.08]),('backrest_shell_proxy',[0,0,.08]),
                    ('seat_pan',[0,0,0]),('cushion_envelope',[0,0,0])]:
        np.testing.assert_allclose(poses[n][:,:3],np.eye(3),atol=1e-12)
        np.testing.assert_allclose(poses[n][:,3],np.array(delta)@basis,atol=1e-12)
    moved=posed_parts(asset,state)
    for side in ['left','right']:
        clear(side+'_arm_inner',side+'_arm_post',moved)
        clear(side+'_arm_carriage',side+'_arm_guide',moved)
        clear(side+'_arm_pad','backrest_shell_proxy',moved)
    clear('upper_spine','lower_spine',moved)
    # Ensure the inserted inner walls remain present over a 50 mm interval
    # at the worst designed extension (not a strength adequacy criterion).
    top=posed_parts(asset,{'left_arm_height':.08,'right_arm_height':.08})
    for side,sign in [('left',-1),('right',1)]:
        q=[[sign*(half+.04)+.012, .01,z] for z in [.201,.225,.249]]
        assert top[side+'_arm_inner'].occupied(world(q)).all()
    assert moved['upper_spine'].occupied(world([[.014,.355,z] for z in [.296,.32,.344]])).all()
    # Cushion must follow its owning seat through base swivel and lift.
    poses2=dict(zip(assembly.parts,assembly.mechanism.pose({'lift':.06,'swivel':37})))
    np.testing.assert_allclose(poses2['cushion_envelope'],poses2['crossbar'],atol=1e-12)
    assert all(p['fits_carbon_axis_aligned'] for p in info['printed_parts'].values())
    report={'parts':len(parts),'states':len(assembly.mechanism.state_keys),'base_parts_byte_identical':len(base.parts),
            'support_samples':len(xy),'support_fraction_at_0_1mm_below_top':fraction,
            'fastener_head_clearance_mm':.3,'minimum_designed_telescoping_overlap_mm':50,
            'collision_checks':checks,'motion_checks':'independent translations, coupling, cushion owner pass',
            'limitations':'Deterministic samples can miss small intersections. Openings are sampled, not integrated. No strength, clamp retention, bend/weld process, ergonomic fit or occupied stability validation.'}
    (out/f'{stem}-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='collision_checks'},indent=2))


if __name__=='__main__':main()
