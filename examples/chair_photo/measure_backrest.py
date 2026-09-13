#!/usr/bin/env python3
"""Replay the backrest survey and saved hardware observations with native kernels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import volumetric as v
from measure import HERE
from intake_backrest import jpeg_codestream


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, default=HERE/'work'/'backrest')
    ap.add_argument('--survey', action='store_true', help='Import work/photos and solve again')
    args = ap.parse_args()
    w = args.work.resolve()
    inventory = HERE.parent/'calibration-targets.json'
    targets = json.loads(inventory.read_text())['targets']
    # Recognize allocated physical identities before selecting metric calibration.
    detected_ids = set()
    for photo in sorted((w/'photos').glob('*.JPG')):
        for d in v.detect(v.Gray.open(str(photo)), ['36h11']):
            detected_ids.add((d['family'], d['id']))
    candidates = [t for t in targets if sum(
        (t['family'], i) in detected_ids
        for i in range(t['first_id'], t['first_id']+t['marker_count'])) >= 4]
    if len(candidates) != 1:
        raise ValueError(f'Expected one known card, recognized {[t["key"] for t in candidates]}')
    target = candidates[0]
    card = inventory.parent/target['card_spec']
    manifest = json.loads((HERE/'backrest-photo-manifest.json').read_text())
    if {p.name for p in (w/'photos').glob('*.JPG')} != {i['jpeg'] for i in manifest['photos']}:
        raise ValueError('Photo directory does not match the session manifest')
    for item in manifest['photos']:
        actual = hashlib.sha256(jpeg_codestream((w/'photos'/item['jpeg']).read_bytes())).hexdigest()
        if actual != item['jpeg_codestream_sha256']:
            raise ValueError(f'JPEG coding stream hash mismatch: {item["jpeg"]}')
    identity = {'key':target['key'], 'card_spec_sha256':hashlib.sha256(card.read_bytes()).hexdigest()}
    if args.survey:
        views, _ = v.import_stills(str(w/'photos'), embed='preview', labels={'session':'printed-backrest-20260912'})
        views, detection, _ = views.detect(swatches=None, card=str(card))
        (w/'detection.json').write_text(json.dumps(detection, indent=2)+'\n')
        views, survey = v.survey(views, rounds=6)
        (w/'survey.json').write_text(json.dumps(survey, indent=2)+'\n')
        views.save(str(w/'survey.vviews'))
        (w/'target.json').write_text(json.dumps(identity,indent=2)+'\n')
    if not (w/'target.json').exists() or json.loads((w/'target.json').read_text()) != identity:
        raise ValueError('Survey calibration identity is absent or changed; run with --survey')
    views = v.ViewSet.load(str(w/'survey.vviews'))
    obs = json.loads((HERE/'backrest-observations.json').read_text())
    fit = {n:{k:px for k,px in picks.items() if k in obs['fit_views']} for n,picks in obs['features'].items()}
    check = {n:{k:px for k,px in picks.items() if k in obs['check_views']} for n,picks in obs['features'].items()}
    views = views.with_picks(fit, check=check)
    views.save(str(w/'hardware.vviews'))
    features = {}
    for n,picks in fit.items():
        point, gaps = views.triangulate(picks)
        checks = {k:float(np.linalg.norm(views.view(k).project(np.array([point]))[0]-px)) for k,px in check[n].items()}
        features[n] = {'world':np.asarray(point).tolist(), 'gaps_m':list(gaps), 'check_error_px':checks}
    pts = {n:np.array(f['world']) for n,f in features.items()}
    origin = (pts['tube_low']+pts['tube_high'])/2
    x = pts['tube_low']-pts['tube_high']; x /= np.linalg.norm(x)
    z = (pts['plain_low']+pts['plain_high']-pts['label_low']-pts['label_high'])/2
    z -= x*np.dot(z,x); z /= np.linalg.norm(z)
    basis = np.array([x,np.cross(z,x),z])
    for n,f in features.items():
        f['local_mm'] = ((pts[n]-origin)@basis.T*1000).tolist()
    width = obs['tube_width']
    sides, _ = views.view(width['view']).cast(np.array(width['pixels'],dtype=float), z=float(origin[2]))
    report = {
        'target': {**target, 'card_spec_sha256':hashlib.sha256(card.read_bytes()).hexdigest()},
        'frame': {'origin':origin.tolist(), 'basis':basis.tolist()}, 'features':features,
        'dimensions_mm': {
            'tube_length':float(np.linalg.norm(pts['tube_low']-pts['tube_high'])*1000),
            'tube_apparent_diameter':float(np.linalg.norm(sides[1]-sides[0])*1000),
            'bolt_width':float(np.mean([np.linalg.norm(pts[s+'_low']-pts[s+'_high']) for s in ['label','plain']])*1000),
            'bolt_height':float(np.mean([np.linalg.norm(pts['label_'+s]-pts['plain_'+s]) for s in ['low','high']])*1000)},
        'label':{'width_in':15,'length_in':15,'lateral_depths_in':[6,6],'serial':'BDR-00044'},
        'limitations':'Approximate manual socket-center picks. Additional-view bolt disagreement is reported, not suppressed; target RMS and ray misses are not hardware accuracy bounds. Tube diameter is a silhouette width projected onto a horizontal plane through the tube center. Shell contour, plate thickness, tube wall, threads and clamp travel are unmeasured.'}
    (w/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'backrest-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'target':target['key'], 'dimensions_mm':report['dimensions_mm'],
                      'check_error_px':{n:f['check_error_px'] for n,f in features.items()}},indent=2))


if __name__ == '__main__':
    main()
