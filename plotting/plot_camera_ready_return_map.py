#!/usr/bin/env python3
"""Combine paper return maps with the saved analytical reduced source-branch flow."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common import ROOT, INK, PAPER, MUTED, CORAL, MAGNITUDE_CMAP, configure, series_style, style_axis
from plot_reduced_phase_plane import arrow_along


def panel_label(ax, title, letter):
    style_axis(ax, title, grid=None)
    ax.text(-.14, 1.075, r'$\mathsf{' + letter + '}$', transform=ax.transAxes,
            math_fontfamily='cm', color='black', fontsize=10, va='bottom', clip_on=False)


def draw_return(ax, maps, points, activation, kappa, letter):
    panel = maps[(maps.activation == activation) & (maps.kappa == kappa)]
    if panel.empty:
        raise ValueError(f'Missing return map for {activation}, kappa={kappa}')
    ax.plot(panel.Z, panel.Z_plus, **series_style(0), label=rf'$\kappa={kappa:g}$')
    fixed = points[(points.activation == activation) & (points.kappa == kappa)]
    for stable, group in fixed.groupby('stable'):
        ax.scatter(group.fixed_point, group.fixed_point, s=25,
                   marker='o' if stable else 'D', facecolor=INK if stable else PAPER,
                   edgecolor=INK, lw=1, zorder=6)
    high = max(panel.Z.max(), panel.Z_plus.max())*1.02
    ax.plot([0, high], [0, high], ls='--', color=MUTED, lw=.85, label='Identity')
    ax.set_xlim(0, high)
    ax.set_ylim(0, high)
    ax.set_aspect('equal', adjustable='box')
    panel_label(ax, ('SmoothF' if activation == 'smoothf' else 'Tanh')+' · return map', letter)
    ax.set_xlabel(r'Entry saliency $Z$')
    ax.set_ylabel(r'Next entry saliency $Z^+$')
    ax.legend(loc='lower right')


def draw_flow(ax, field, boundaries, paths, activation, kappa, norm, letter):
    f = field[field.activation == activation]
    x, y = np.sort(f.source.unique()), np.sort(f.successor.unique())
    def grid(key):
        return f.pivot(index='successor', columns='source', values=key).reindex(index=y, columns=x).to_numpy()
    mask = ~grid('valid').astype(bool)
    speed = np.ma.masked_where(mask, grid('speed'))
    image = ax.pcolormesh(x, y, speed, cmap=MAGNITUDE_CMAP, norm=norm,
                          shading='nearest', rasterized=True)
    b = boundaries[(boundaries.activation == activation) & (boundaries.kappa == kappa)]
    ax.fill_between(b.source, np.minimum(b.boundary, y.max()), y.max(), color='#ECEEEA', zorder=2)
    u = np.ma.masked_where(mask, grid('d_source'))
    v = np.ma.masked_where(mask, grid('d_successor'))
    ax.streamplot(x, y, u, v, density=(.65, .8), color=(1, 1, 1, .28),
                  linewidth=.45+.35*speed.filled(0)/norm.vmax, arrowsize=.55,
                  minlength=.18, maxlength=2.5, zorder=3)
    visible = b.boundary <= y.max()
    ax.plot(b.loc[visible, 'source'], b.loc[visible, 'boundary'], color=PAPER,
            lw=1.4, ls=(0, (4, 2)), zorder=5,
            path_effects=[pe.Stroke(linewidth=2.6, foreground=INK), pe.Normal()])
    pths = paths[(paths.activation == activation) & (paths.kappa == kappa)]
    for pid, path in pths.groupby('path_id'):
        selected = bool(path.selected.iloc[0])
        ax.plot(path.source, path.successor, color=PAPER, lw=2 if selected else .85,
                alpha=1 if selected else .52, zorder=7 if selected else 4,
                path_effects=[pe.Stroke(linewidth=3.2, foreground=INK), pe.Normal()] if selected else None)
        arrow_along(ax, path.source, path.successor, color=PAPER, lw=1 if selected else .65)
        if selected:
            ax.scatter(path.source.iloc[0], path.successor.iloc[0], s=24,
                       facecolor=INK, edgecolor=PAPER, lw=1.1, zorder=9, clip_on=False)
            if path.escaped.iloc[-1]:
                ax.scatter(path.source.iloc[-1], path.successor.iloc[-1], s=26,
                           marker='s', facecolor=CORAL, edgecolor=PAPER, lw=.85, zorder=9)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, y.max())
    ax.set_aspect('equal', adjustable='box')
    panel_label(ax, ('SmoothF' if activation == 'smoothf' else 'Tanh')+' · source-branch flow', letter)
    ax.set_xlabel(r'Source saliency $s=|z_{\rm source}|$')
    ax.set_ylabel(r'Successor saliency $r=|z_{\rm successor}|$')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/raw/paper/return_map')
    parser.add_argument('--reduced-input', type=Path, default=ROOT/'figures/data/reduced_phase_plane')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/camera_ready')
    args = parser.parse_args()
    sources = {'maps': args.input/'return_map.csv', 'points': args.input/'return_map_fixed_points.csv',
               'fields': args.reduced_input/'reduced_phase_field.csv',
               'boundaries': args.reduced_input/'reduced_phase_boundary.csv',
               'paths': args.reduced_input/'reduced_phase_paths.csv'}
    data = {name: pd.read_csv(path) for name, path in sources.items()}
    configure()
    for kappa, field in data['fields'].groupby('kappa'):
        fig, axes = plt.subplots(2, 2, figsize=(7.8, 8))
        fig.suptitle(rf'Return maps and reduced source-branch flow · $\kappa={kappa:g}$',
                     fontsize=11, fontweight='bold')
        norm = Normalize(0, field.loc[field.valid.astype(bool), 'speed'].max())
        for j, activation in enumerate(('smoothf', 'tanh')):
            draw_return(axes[0, j], data['maps'], data['points'], activation, kappa, 'AB'[j])
            image = draw_flow(axes[1, j], field, data['boundaries'], data['paths'],
                               activation, kappa, norm, 'CD'[j])
        cb = fig.colorbar(image, ax=list(axes[1]), location='right', fraction=.025, pad=.025, aspect=25)
        cb.set_label(r'Reduced saliency speed $\sqrt{\dot{s}^{\,2}+\dot{r}^{\,2}}$',
                     rotation=270, labelpad=17)
        cb.outline.set_visible(False)
        handles = [
            Line2D([], [], marker='o', color=INK, ls='none', ms=4, label='Stable fixed point'),
            Line2D([], [], marker='D', color=INK, mfc=PAPER, ls='none', ms=4, label='Unstable fixed point'),
            Line2D([], [], color=INK, ls='--', lw=1, label='Stability boundary'),
            Line2D([], [], color=INK, marker='o', mfc=PAPER, ls='none', ms=4, label='Entry'),
            Line2D([], [], color=CORAL, marker='s', ls='none', ms=4, label='Stability loss'),
            Patch(facecolor='#ECEEEA', edgecolor=MUTED, lw=.5, label='Outside stable branch'),
        ]
        legend = fig.legend(
            handles=handles, loc='outside lower center', ncol=3, fontsize=6.7,
            title='Orthogonal memories · unit gains · singular limit · paths stop at branch loss\n'
                  'Analytical reduced model · not a full-network phase portrait',
            title_fontsize=6.5)
        legend.get_title().set_color(MUTED)
        legend.get_title().set_multialignment('center')
        fig.patch.set_facecolor('none')
        for ax in fig.axes:
            ax.patch.set_facecolor('none')
        # Resolve equal-aspect axes before placing labels on common column anchors.
        fig.canvas.draw()
        fig.set_layout_engine('none')
        # Keep each column's y-axis on one shared left edge, including after
        # Matplotlib applies the different equal-aspect data ranges.
        for column in range(2):
            boxes = [ax.get_position().frozen() for ax in axes[:, column]]
            left = min(box.x0 for box in boxes)
            width = min(box.width for box in boxes)
            for ax, box in zip(axes[:, column], boxes):
                ax.set_anchor('W')
                ax.set_position([left, box.y0, width, box.height])
            fig.align_ylabels(axes[:, column])
        fig.canvas.draw()
        axes_positions = {letter: list(ax.get_position().bounds)
                          for letter, ax in zip('ABCD', axes.flat)}
        assert np.isclose(axes_positions['A'][0], axes_positions['C'][0])
        assert np.isclose(axes_positions['B'][0], axes_positions['D'][0])
        label_positions = {}
        for column in range(2):
            labels = [next(text for text in ax.texts
                           if text.get_text().startswith(r'$\mathsf{'))
                      for ax in axes[:, column]]
            positions = [fig.transFigure.inverted().transform(
                text.get_transform().transform(text.get_position())) for text in labels]
            anchor = min(position[0] for position in positions)
            for row, (text, position) in enumerate(zip(labels, positions)):
                text.set_transform(fig.transFigure)
                text.set_position((anchor, position[1]))
                label_positions['ABCD'[row*2+column]] = [float(anchor), float(position[1])]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f'return_map_reduced_phase_plane_kappa_{kappa:g}'
        outputs = []
        for ext in ('pdf', 'svg', 'png'):
            path = args.output_dir/f'{stem}.{ext}'
            fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=.06,
                        transparent=True, facecolor='none', edgecolor='none')
            outputs.append(path.name)
        plt.close(fig)
        manifest = {'kappa': float(kappa), 'layout': [['A: SmoothF return map', 'B: Tanh return map'],
                     ['C: SmoothF reduced flow', 'D: Tanh reduced flow']],
                    'input_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sources.values()},
                    'axes_positions': axes_positions,
                    'panel_label_positions': label_positions,
                    'transparent_background': True, 'raster_dpi': 600, 'outputs': outputs}
        (args.output_dir/f'{stem}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'Created {stem}: PDF, SVG, PNG', flush=True)


if __name__ == '__main__':
    main()
