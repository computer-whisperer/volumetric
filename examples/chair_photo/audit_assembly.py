#!/usr/bin/env python3
"""Audit the articulated base's rest geometry and independent joint motions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import volumetric as v
from measure import HERE


def posed_parts(asset, state=None):
    """Extract typed, posed Model assets through the public Assembly Model operator."""
    p = v.Project()
    assembly_id = p.add_asset(asset.bytes, 'assembly', id='assembly')
    for name in asset.assembly().parts:
        p.add_op('assembly_model_operator', [assembly_id, {'part':name}, state], output=name)
    return p.run()


def transformed(matrix, points):
    return np.asarray(points) @ matrix[:, :3].T + matrix[:, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=HERE/'work')
    parser.add_argument('--reference', type=Path, help='Pre-migration project with 16 flat model exports')
    parser.add_argument('--posed', type=Path, help='Posed .vasm to compare part bytes and attached interface frames')
    args = parser.parse_args()
    w = args.work.resolve()
    mount = json.loads((w/'measurements.json').read_text())
    receiver = json.loads((w/'receiver-fit.json').read_text())
    measurements = json.loads((w/'assembly-measurements.json').read_text())
    exports = v.Project.open(str(w/'base.vproj')).run()
    asset = exports['chair_base']
    assembly = asset.assembly()
    mech = assembly.mechanism
    np.testing.assert_allclose(assembly.poses(), np.tile(np.eye(3,4),(22,1,1)), atol=1e-12,
                               err_msg='Run the rest-state build before this audit')
    parts = posed_parts(asset)
    assert len(parts) == 22 and len(mech.state_keys) == 12
    assert v.Assembly.decode(assembly.encode()).state == assembly.state
    points = []; labels = []
    z = np.array(mount['frame']['basis'][2])
    for name, feature in mount['features'].items():
        points.append(np.array(feature['world'])-z*.001); labels.append(name)
    o = np.array(receiver['world_frame']['origin']); b = np.array(receiver['world_frame']['basis'])
    for depth in [.001,.010,.030,.060]:
        points.append(o-b[2]*depth); labels.append(f'receiver_axis_{depth*1000:.0f}mm')
    for name, model in {**parts,'union':exports['chair_base_model']}.items():
        blocked = np.array(labels)[model.occupied(np.array(points))].tolist()
        assert not blocked, (name,blocked)
    assert parts['backrest_receiver'].occupied(o+np.array([[.025,0,-.002],[.026,0,-.030]])@b).all()
    axis = np.array([*measurements['column']['ground_axis_xy'],0.0])
    assert parts['gas_lift_rod'].occupied(np.array([axis+[0,0,.30]])).all()
    assert parts['gas_lift_body'].occupied(np.array([axis+[0,0,.20]])).all()
    report = {'parts':len(parts), 'states':mech.state_keys, 'critical_void_count':len(points),
              'void_audit':'All parts and the rest union clear all probes; receiver/body/rod positive controls pass',
              'limitations':'Geometry checks only; stroke, wheel coupling and ground contact are not measured or validated.'}
    if args.reference:
        old = v.Project.open(str(args.reference)).run()
        rng = np.random.default_rng(20260912)
        comparisons = {}
        for name, model in old.items():
            if model.kind != 'Model':
                continue
            if name == 'gas_lift':
                new_names = ['gas_lift_body','gas_lift_rod']
            elif name.startswith('caster_'):
                new_names = [name+'_fork',name+'_wheels']
            else:
                new_names = [name]
            bounds = [model.bounds(),*[parts[n].bounds() for n in new_names]]
            lo = np.min([bound[0] for bound in bounds],axis=0)
            hi = np.max([bound[1] for bound in bounds],axis=0)
            samples = rng.uniform(lo-.001,hi+.001,(12000,3))
            before = model.occupied(samples)
            after = np.logical_or.reduce([parts[n].occupied(samples) for n in new_names])
            assert np.array_equal(before,after), (name,int(np.count_nonzero(before!=after)))
            comparisons[name] = {'samples':len(samples),'occupied':int(before.sum()),'mismatches':0}
            assert before.any(), name
        assert len(comparisons) == 16
        report['rest_comparison'] = comparisons

    # Independent rigid-motion oracle, not a second call to Mechanism.pose.
    state = {'lift':.060,'swivel':37.,'caster_1_swivel':43.,'caster_1_roll':725.}
    poses = dict(zip(mech.parts,mech.pose(state)))
    r = Rotation.from_rotvec(np.array([0.,0.,np.deg2rad(state['swivel'])])).as_matrix()
    point = np.array(mount['features']['cross_a']['world'])
    expected = axis+r@(point-axis)+[0,0,state['lift']]
    np.testing.assert_allclose(transformed(poses['crossbar'],point),expected,atol=1e-12)
    for name in ['star_base','gas_lift_body','caster_2_fork','caster_2_wheels']:
        np.testing.assert_allclose(poses[name],np.eye(3,4),atol=1e-12)
    np.testing.assert_allclose(poses['gas_lift_rod'][:,3],[0,0,.060],atol=1e-12)
    joints = {j['name']:j for j in mech.joints}
    stem = np.array(joints['caster_1_swivel']['axis']['origin'])
    axle = np.array(joints['caster_1_roll']['axis']['origin'])
    direction = np.array(joints['caster_1_roll']['axis']['direction'])
    yaw = Rotation.from_rotvec(np.array([0.,0.,np.deg2rad(43.)])).as_matrix()
    roll = Rotation.from_rotvec(direction*np.deg2rad(725.)).as_matrix()
    wheel_point = axle+np.array([0,0,.01])
    expected_wheel = stem+yaw@(axle+roll@(wheel_point-axle)-stem)
    np.testing.assert_allclose(transformed(poses['caster_1_wheels'],wheel_point),expected_wheel,atol=1e-12)
    # The unchanged wheel WASM can still be sampled through a posed extraction.
    moved = posed_parts(asset,state)
    wheel_material = axle+direction*.012
    moved_material = stem+yaw@(wheel_material-stem)
    assert moved['caster_1_wheels'].occupied(np.array([moved_material]))[0]
    moved_voids = transformed(poses['crossbar'],np.array(points))
    for model in moved.values():
        assert not model.occupied(moved_voids).any()
    pulled = mech.pull('crossbar',point.tolist(),expected.tolist(),iterations=40)
    reached = transformed(mech.pose(pulled)[mech.parts.index('crossbar')],point)
    assert np.linalg.norm(reached-expected) < 1e-7
    velocity = mech.velocity('crossbar',expected.tolist(),state)
    for index,key in enumerate(mech.state_keys):
        eps = 1e-6 if key == 'lift' else 1e-4
        plus={**state,key:state.get(key,0)+eps};minus={**state,key:state.get(key,0)-eps}
        pi = mech.parts.index('crossbar')
        finite=(transformed(mech.pose(plus)[pi],point)-transformed(mech.pose(minus)[pi],point))/(2*eps)
        np.testing.assert_allclose(velocity[index],finite,atol=1e-8)
    report['motion_checks'] = {'state':state,'lift_swivel_caster_oracles':'pass',
        'moved_voids':'clear in all 22 parts','pull_error_m':float(np.linalg.norm(reached-expected)),
        'velocity_finite_difference':'all 12 state derivatives agree'}
    if args.posed:
        posed = v.Assembly.load(str(args.posed))
        assert posed.parts == assembly.parts
        assert all(posed.part_model(name) == assembly.part_model(name) for name in assembly.parts)
        np.testing.assert_allclose(posed.poses(),mech.pose(posed.state),atol=1e-12)
        path = args.posed.with_name(args.posed.stem+'-interfaces.json')
        definitions = json.loads(path.read_text())
        canonical = json.loads((w/'base-interfaces.json').read_text())['interfaces']
        assert definitions['state'] == posed.state
        assert definitions['interfaces'].keys() == canonical.keys()
        for name, interface in definitions['interfaces'].items():
            assert interface['part'] == canonical[name]['part']
            assert interface['rest_frame'] == canonical[name]['rest_frame']
            matrix = posed.poses()[posed.parts.index(interface['part'])]
            rest = interface['rest_frame']; actual = interface['posed_frame']
            np.testing.assert_allclose(actual['origin'],transformed(matrix,rest['origin']),atol=1e-12)
            np.testing.assert_allclose(actual['basis'],np.array(rest['basis'])@matrix[:,:3].T,atol=1e-12)
        report['posed_artifact'] = 'Identical part WASM, expected poses and owned interface frames; canonical assembly remains at rest'
    (w/'assembly-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (w/'motion-state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
