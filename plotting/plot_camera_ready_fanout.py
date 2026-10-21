#!/usr/bin/env python3
"""Combine saved paper fanout heatmaps and confidence-band curves for publication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from common import (
    ROOT, MISSING, MUTED, PROBABILITY_CMAP, cell_edges, configure,
    interval, series_style, style_axis,
)
from plot_capacity import _matrix_for_heatmap


def label_panel(ax, title, letter, **kwargs):
    style_axis(ax, title, **kwargs)
    ax.text(-.14, 1.075, r'$\mathsf{' + letter + '}$',
            transform=ax.transAxes, math_fontfamily='cm', fontsize=10,
            color='black', va='bottom', clip_on=False)


def render(frame, activation, output_dir):
    phase = frame[(frame.experiment == 'fanout_phase') & (frame.activation == activation)]
    curves = frame[(frame.experiment == 'fanout') & (frame.activation == activation)]
    ns = sorted(phase.N.unique())
    degrees = sorted(curves.outdegree.unique())
    if len(ns) != 3 or len(degrees) != 5:
        raise ValueError('This layout requires three network sizes and five out-degrees.')
    if sorted(curves.N.unique()) != ns:
        raise ValueError('Heatmaps and curves must contain the same network sizes.')
    configure()
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.1),
                             gridspec_kw={'height_ratios': [1.2, 1, 1]})
    fig.suptitle(('Tanh' if activation == 'tanh' else 'SmoothF') + ' · fanout experiments',
                 fontsize=11, fontweight='bold')
    missing = False
    bounds = []
    for j, (ax, n) in enumerate(zip(axes[0], ns)):
        loads, outdegrees, matrix = _matrix_for_heatmap(phase[phase.N == n])
        xe, ye = cell_edges(loads, .05), cell_edges(outdegrees)
        bounds.append((xe[0], xe[-1]))
        image = ax.pcolormesh(xe, ye, np.ma.masked_invalid(matrix),
                              cmap=PROBABILITY_CMAP, vmin=0, vmax=1,
                              shading='flat', rasterized=True)
        for iy, ix in np.argwhere(~np.isfinite(matrix)):
            missing = True
            ax.add_patch(Rectangle(
                (xe[ix], ye[iy]), xe[ix+1]-xe[ix], ye[iy+1]-ye[iy],
                facecolor=MISSING, edgecolor='#A1ABA9', hatch='///', lw=0))
        label_panel(ax, f'N = {int(n)}', 'ABC'[j], grid=None)
        ax.set_xlabel(r'Realized load $P/N$')
        ax.set_yticks([2, 6, 10, 14, 18])
        if j == 0:
            ax.set_ylabel('Out-degree')
        else:
            ax.tick_params(labelleft=False)
    cb = fig.colorbar(image, ax=list(axes[0]), location='right',
                      fraction=.018, pad=.018, aspect=25)
    cb.set_label('Transition probability', rotation=270, labelpad=14)
    cb.set_ticks([0, .25, .5, .75, 1])
    cb.outline.set_visible(False)

    lower = list(axes[1:].flat)
    handles = []
    for j, (ax, degree) in enumerate(zip(lower, degrees)):
        for i, n in enumerate(ns):
            curve = curves[(curves.outdegree == degree) & (curves.N == n)].sort_values('alpha_actual')
            if curve.empty:
                raise ValueError(f'Missing fanout curve for {activation}, N={n}, degree={degree}')
            style = series_style(i, True)
            interval(ax, curve.alpha_actual, curve.ci_low, curve.ci_high, style['color'])
            line, = ax.plot(curve.alpha_actual, curve.success_probability,
                            **style, label=f'N = {int(n)}')
            if j == 0:
                handles.append(line)
        label_panel(ax, f'Out-degree {int(degree)}', 'DEFGH'[j], probability=True)
        ax.set_xlabel(r'Realized load $P/N$')
        if j % 3 == 0:
            ax.set_ylabel('Transition probability')
        else:
            ax.tick_params(labelleft=False)
    # Common load limits allow comparisons across both experiment grids.
    xmin = min(b[0] for b in bounds)
    xmax = max(b[1] for b in bounds)
    for ax in list(axes[0]) + lower[:5]:
        ax.set_xlim(xmin, xmax)
        ax.set_xticks([.05, .10, .15, .20, .25, .30])
    key = lower[5]
    key.set_axis_off()
    key.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, .95),
               title='Network size', fontsize=8, title_fontsize=9,
               handlelength=3, labelspacing=.8)
    key.text(.03, .43, 'Shading: reported 95% confidence bands',
             transform=key.transAxes, fontsize=8, color=MUTED, va='top')
    if missing:
        legend = key.get_legend()
        key.add_artist(legend)
        key.legend(handles=[Patch(facecolor=MISSING, edgecolor='#A1ABA9', hatch='///',
                                  label='Not sampled / infeasible graph')],
                   loc='lower left', bbox_to_anchor=(0, .15), fontsize=7.5)
    fig.patch.set_facecolor('none')
    for ax in fig.axes:
        ax.patch.set_facecolor('none')
    stem = 'Fanout_' + activation
    outputs = []
    for extension in ('pdf', 'svg', 'png'):
        path = output_dir / f'{stem}.{extension}'
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=.06,
                    transparent=True, facecolor='none', edgecolor='none')
        outputs.append(path.name)
    plt.close(fig)
    print(f'Created {stem}: {len(phase)} heatmap samples, {len(curves)} curve samples', flush=True)
    return {
        'activation': activation,
        'panels': {**{letter: f'fanout_phase: N={int(n)}' for letter, n in zip('ABC', ns)},
                   **{letter: f'fanout: outdegree={int(d)}' for letter, d in zip('DEFGH', degrees)}},
        'layout': [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'legend']],
        'phase_rows': len(phase), 'curve_rows': len(curves),
        'transparent_background': True,
        'panel_label_style': r'Black Computer Modern sans-serif, $\mathsf{A}$ through $\mathsf{H}$',
        'raster_dpi': 600, 'outputs': outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT/'data/raw/paper/capacity/capacity_probability.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/camera_ready')
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for activation in ('tanh', 'smoothf'):
        manifest = render(frame, activation, args.output_dir)
        manifest['source'] = str(args.input.resolve())
        manifest['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        (args.output_dir/f'Fanout_{activation}_manifest.json').write_text(
            json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
