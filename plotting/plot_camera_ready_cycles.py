#!/usr/bin/env python3
"""Combine the saved paper scaled- and fixed-cycle capacity experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from common import ROOT, configure, interval, series_style
from plot_camera_ready_fanout import label_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/raw/paper/capacity/capacity_probability.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/camera_ready')
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    frame = frame[frame.experiment.isin(['scaled_cycle', 'fixed_cycle'])]
    ns = sorted(frame.N.unique())
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.3), sharey=True)
    handles = {}
    counts = {}
    for row, experiment in enumerate(('scaled_cycle', 'fixed_cycle')):
        subset = frame[frame.experiment == experiment]
        lengths = sorted(subset.cycle_length.unique())
        if experiment == 'fixed_cycle' and len(lengths) != 1:
            raise ValueError('Expected one fixed cycle length.')
        for col, activation in enumerate(('smoothf', 'tanh')):
            ax = axes[row, col]
            panel = subset[subset.activation == activation]
            if panel.empty:
                raise ValueError(f'Missing {experiment} data for {activation}')
            for n, curve in panel.groupby('N'):
                curve = curve.sort_values('alpha_actual')
                sty = series_style(ns.index(n), True)
                interval(ax, curve.alpha_actual, curve.ci_low, curve.ci_high, sty['color'])
                line, = ax.plot(curve.alpha_actual, curve.success_probability,
                                **sty, label=f'N = {int(n)}')
                handles[n] = line
            name = 'SmoothF' if activation == 'smoothf' else 'Tanh'
            regime = r'scaled $L\simeq P/3$' if row == 0 else f'fixed $L={int(lengths[0])}$'
            label_panel(ax, name+' · '+regime, 'ABCD'[row*2+col], probability=True)
            ax.set_xlabel(r'Realized load $P/N$')
            if col == 0:
                ax.set_ylabel('Full-cycle probability')
            # Every panel displays its own sampled load range and ticks.
            ax.tick_params(axis='x', labelbottom=True)
            counts['ABCD'[row*2+col]] = len(panel)
    fig.align_ylabels(axes[:, 0])
    fig.legend(handles=[handles[n] for n in ns], loc='outside upper center',
               ncol=len(ns), title='Cycle capacity · reported 95% confidence bands')
    fig.patch.set_facecolor('none')
    for ax in fig.axes:
        ax.patch.set_facecolor('none')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'capacity_scaled_fixed_cycle'
    outputs = []
    for extension in ('pdf', 'svg', 'png'):
        path = args.output_dir/f'{stem}.{extension}'
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=.06,
                    transparent=True, facecolor='none', edgecolor='none')
        outputs.append(path.name)
    plt.close(fig)
    manifest = {'source': str(args.input.resolve()),
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'layout': [['A: SmoothF scaled cycle', 'B: Tanh scaled cycle'],
                           ['C: SmoothF fixed cycle', 'D: Tanh fixed cycle']],
                'panel_rows': counts, 'transparent_background': True,
                'raster_dpi': 600, 'outputs': outputs}
    (args.output_dir/f'{stem}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Created {stem}: PDF, SVG, PNG', flush=True)


if __name__ == '__main__':
    main()
