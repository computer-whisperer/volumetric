#!/usr/bin/env python3
"""Check critical voids against every exported solid using native occupancy."""
import argparse
import json
from pathlib import Path
import numpy as np
from measure import HERE, cli, vector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=HERE/'work')
    args=parser.parse_args();w=args.work.resolve()
    mount=json.loads((w/'measurements.json').read_text())
    receiver=json.loads((w/'receiver-fit.json').read_text())
    project=w/'base.vproj';target=w/'audit-models'
    cli('project-export','-p',project,'-o',target,'--json')
    points=[]; labels=[]
    # Test points just below the accepted top surfaces, inside each aperture.
    z=np.array(mount['frame']['basis'][2])
    for name,feature in mount['features'].items():
        points.append(np.array(feature['world'])-z*.001);labels.append(name)
    o=np.array(receiver['world_frame']['origin']);b=np.array(receiver['world_frame']['basis'])
    for depth in [.001,.010,.030,.060]:
        points.append(o-b[2]*depth);labels.append(f'receiver_axis_{depth*1000:.0f}mm')
    results={}
    exports=json.loads(cli('project-list','-p',project,'--json'))['exports']
    if not exports:
        raise AssertionError('Assembly has no exported solids')
    for name in sorted(exports):
        model=target/(name+'.wasm')
        command=['sample','-i',model,'--json']
        for point in points:command+=['--point',vector(point)]
        samples=json.loads(cli(*command))['samples']
        occupied=[label for label,s in zip(labels,samples) if s['occupied']]
        if occupied:raise AssertionError(f'{model.stem} blocks critical voids: {occupied}')
        results[model.stem]={'clear_probe_count':len(samples)}
    # Positive controls: collar and side wall must be present, so an empty
    # export or incorrect pose cannot make the void checks pass trivially.
    command=['sample','-i',target/'backrest_receiver.wasm','--json']
    for q in [[.025,0,-.002],[.026,0,-.030]]:
        command+=['--point',vector(o+np.array(q)@b)]
    assert all(p['occupied'] for p in json.loads(cli(*command))['samples'])
    assembly=json.loads((w/'assembly-measurements.json').read_text())
    point=[*assembly['column']['ground_axis_xy'],.30]
    assert json.loads(cli('sample','-i',target/'gas_lift.wasm','--point',vector(point),'--json'))['samples'][0]['occupied']
    output={'probes':dict(zip(labels,[p.tolist() for p in points])),'exports':results,
            'positive_controls':'Receiver collar/side wall and column axis are occupied',
            'limitation':'Checks this model geometry, not physical insertion clearance or hidden depth.'}
    (w/'assembly-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(f'{len(results)} exports leave all {len(points)} critical void probes clear; positive controls pass.')


if __name__=='__main__':main()
