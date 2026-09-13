#!/usr/bin/env python3
"""Plot the measured mounting stations and lift-axis intersection, in millimetres."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from measure import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=HERE/'work')
    w = parser.parse_args().work
    mount = json.loads((w/'measurements.json').read_text())
    base = json.loads((w/'assembly-measurements.json').read_text())
    receiver = json.loads((w/'receiver-fit.json').read_text())
    origin = np.array(mount['frame']['origin']); basis = np.array(mount['frame']['basis'])
    axis = np.array([*base['column']['ground_axis_xy'],0.])
    # Intersect the vertical lift axis with the mounting datum plane.
    axis[2] = np.dot(basis[2],origin-axis)/basis[2,2]
    local = (axis-origin)@basis.T*1000
    fig, ax = plt.subplots(figsize=(8,8),layout='constrained')
    for name, feature in mount['features'].items():
        x,y,z = feature['local_mm']
        ax.scatter(x,y,s=65,color='#245c9e')
        if abs(z)<.05: z=0.
        inner = name.endswith('_inner')
        offset = ((7 if x>=0 else -7),(-14 if inner else 12))
        ax.annotate(f'{name}\nZ {z:+.1f}',(x,y),xytext=offset,textcoords='offset points',
                    ha='left' if x>=0 else 'right',va='top' if inner else 'bottom',fontsize=9)
    polygon = [mount['features'][n]['local_mm'][:2] for n in ['cross_b','cross_a','rail_a','rail_b','cross_b']]
    ax.plot(*np.array(polygon).T,'--',color='#9ca3af',lw=1,label='Outer attachment-station polygon')
    ax.scatter(*local[:2],marker='+',s=180,color='#d25826',linewidths=2,label='Vertical lift axis at seat datum')
    ax.annotate(f'Lift axis ({local[0]:.1f}, {local[1]:.1f})',local[:2],xytext=(10,5),
                textcoords='offset points',fontsize=10,color='#a14119')
    center = np.array(receiver['center_local_mm'])
    ax.scatter(*center[:2],marker='s',s=70,color='#32845b')
    label = f"Receiver mouth center\nEntry rim ≈ {receiver['mouth_width_mm']:.0f} × {receiver['mouth_depth_mm']:.0f} mm; throat unknown"
    ax.annotate(label,center[:2],xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
    ax.set(xlim=(-180,180),ylim=(-40,320),xlabel='Seat-local X (mm)',
           ylabel='Seat-local Y toward receiver (mm)',
           title='Measured base interfaces — top view\nZ labels are relative to the inner front-hole datum')
    ax.set_aspect('equal'); ax.grid(alpha=.18)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.10),frameon=False,fontsize=9)
    fig.savefig(w/'interface-map.png',dpi=160)
    fig.savefig(w/'interface-map.svg')


if __name__ == '__main__':
    main()
