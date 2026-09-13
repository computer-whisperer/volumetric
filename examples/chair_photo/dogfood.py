#!/usr/bin/env python3
"""Exercise persisted photo evidence and the Python workbench on the accepted base.

Run after build_assembly.py, with the volumetric Python package installed.
Writes separate artifacts under work/dogfood; does not replace accepted geometry.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import volumetric as v
from audit_assembly import posed_parts

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=HERE/'work')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--benchmark-cli', action='store_true', help='Repeat the slower CLI receiver fit too')
    args = parser.parse_args()
    work = args.work.resolve()
    out = work/'dogfood'
    out.mkdir(exist_ok=True)
    observations = read(HERE/'observations.json')
    rim = read(HERE/'receiver-observations.json')
    baseline = read(work/'measurements.json')
    views = v.ViewSet.load(str(work/'survey.vviews'))
    picks = {name: {view: pixel for view, pixel in pixels.items() if view in observations['fit_views']}
             for name, pixels in observations['features'].items()}
    checks = {name: {view: pixel for view, pixel in pixels.items() if view in observations['check_views']}
              for name, pixels in observations['features'].items()}
    views = views.with_picks(picks, check=checks).with_contours({'receiver_mouth': rim['contours']})
    ids = sorted(set(observations['fit_views'] + observations['check_views'] + rim['fit_views'] + rim['check_views']))
    evidence = views.select(ids=ids, embed='full')
    evidence.save(str(out/'evidence.vviews'))
    restored = v.ViewSet.load(str(out/'evidence.vviews'))
    assert restored.picks() == evidence.picks()
    assert restored.contours() == evidence.contours()
    start = perf_counter()
    fitted = restored.fit_picks()
    fit_seconds = perf_counter() - start
    errors = {name: float(np.linalg.norm(value['world'] - baseline['features'][name]['world']) * 1000)
              for name, value in fitted.items()}
    assert max(errors.values()) < 1e-6, errors
    report = {'feature_delta_mm': errors, 'feature_fit_seconds': fit_seconds,
              'check_errors_px': {name: value['max_check_px'] for name, value in fitted.items()},
              'check_note': baseline['check_note'],
              'contour_roles': 'Fit/check membership remains in receiver-observations.json; ViewSet contours do not encode roles.'}
    for backend in ['python'] + (['cli'] if args.benchmark_cli else []):
        command = [sys.executable, str(HERE/'fit_receiver.py'), '--work', str(work), '--backend', backend,
                   '--output', str(out/f'receiver-{backend}.json')]
        if backend == 'python':
            command += ['--views', str(out/'evidence.vviews')]
        start = perf_counter()
        with (out/f'receiver-{backend}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        report[f'receiver_{backend}_seconds'] = perf_counter() - start
    reference = read(out/'receiver-cli.json') if args.benchmark_cli else read(work/'receiver-fit.json')
    native = read(out/'receiver-python.json')
    report['receiver_reference'] = 'fresh CLI fit' if args.benchmark_cli else 'saved CLI fit'
    report['receiver_delta_mm'] = {key: native[key]-reference[key]
        for key in ['mouth_width_mm', 'mouth_depth_mm', 'mouth_corner_radius_mm']}
    report['receiver_center_delta_mm'] = float(np.linalg.norm(np.array(native['center_local_mm'])-reference['center_local_mm']))
    report['receiver_world_origin_delta_mm'] = float(np.linalg.norm(np.array(native['world_frame']['origin'])-reference['world_frame']['origin'])*1000)
    report['receiver_basis_max_delta'] = float(np.max(np.abs(np.array(native['world_frame']['basis'])-reference['world_frame']['basis'])))
    assert max(abs(x) for x in report['receiver_delta_mm'].values()) < .001
    assert report['receiver_center_delta_mm'] < .001
    assert report['receiver_world_origin_delta_mm'] < .001
    assert report['receiver_basis_max_delta'] < 1e-6
    project = v.Project.open(str(work/'base.vproj'))
    evidence_id = project.add_views(restored, id='chair_evidence')
    # Keep interpretation alongside the pixel evidence in the durable project.
    project.add_asset(json.dumps({'mount': observations, 'receiver': rim}).encode(), 'blob', id='evidence_provenance')
    project_bytes = project.to_bytes()
    project.save(str(out/'base-evidence.vproj'))
    project = v.Project.open(str(out/'base-evidence.vproj'))
    assert project.to_bytes() == project_bytes
    assert project.validate() == []
    # Exercise config editing on a copy: WGSL parameters are F64Map, not config.
    receiver_step = next(i for i, step in enumerate(project.steps()) if step['outputs'] == ['backrest_receiver_local'])
    try:
        project.set_config(receiver_step, {'mouth_width': .039})
    except ValueError as error:
        report['wgsl_parameter_edit_limitation'] = str(error)
    else:
        raise AssertionError('Revisit this probe: WGSL parameter editing is now supported')
    start = perf_counter()
    exports = project.run()
    report['project_run_seconds'] = perf_counter()-start
    assembly_value = exports['chair_base'].assembly()
    np.testing.assert_allclose(assembly_value.poses(), np.tile(np.eye(3,4),(len(assembly_value),1,1)),
                               atol=1e-12, err_msg='Photo dogfood requires the rest-state base')
    solids = posed_parts(exports['chair_base'])
    assert len(solids) == 22
    z = np.array(baseline['frame']['basis'][2])
    points = [np.array(feature['world']) - z*.001 for feature in baseline['features'].values()]
    origin = np.array(reference['world_frame']['origin'])
    basis = np.array(reference['world_frame']['basis'])
    points += [origin - basis[2]*depth for depth in [.001, .010, .030, .060]]
    for name, asset in solids.items():
        assert not asset.occupied(np.array(points)).any(), name
    assert solids['backrest_receiver'].occupied(origin + np.array([[.025, 0, -.002], [.026, 0, -.030]]) @ basis).all()
    assembly = read(work/'assembly-measurements.json')
    assert solids['gas_lift_rod'].occupied(np.array([[*assembly['column']['ground_axis_xy'], .30]])).all()
    report['void_audit'] = '22 parts x 10 void probes clear; receiver walls and column positive controls pass'
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.render:
        from PIL import Image
        report['renders'] = {}
        for view in ['DSC00755', 'DSC00756', 'DSC00760']:
            start = perf_counter()
            frame = v.render(project, assets=['chair_base'], through=f'{evidence_id}:{view}', overlay='edge', marks=True,
                             width=1548, height=1032, resolution=96, sharp=False, simplify=False, grid=0, ssao=False)
            Image.fromarray(frame.image).save(out/f'overlay-{view}.png')
            report['renders'][view] = {'seconds': perf_counter()-start, 'report': frame.report}
        (out/'render-report.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
