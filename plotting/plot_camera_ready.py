#!/usr/bin/env python3
"""Build correlation_noise_test directly from saved paper-run data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from common import (
    ROOT, INK, TEAL, CORAL, MUTED, PAPER, MISSING, MAGNITUDE_CMAP,
    PROBABILITY_CMAP, cell_edges, configure, interval, series_style, style_axis,
)


def draw_escape(ax, frame):
    for key, index, label in [
        ('T_esc_theory', 0, 'Reduced theory'),
        ('T_esc_empirical', 1, 'Stability loss'),
        ('T_switch_empirical', 2, 'Visible switch'),
    ]:
        # Keep the full event grid so undefined events remain gaps.
        ax.plot(frame.event, frame[key], **series_style(index, True), label=label)
    ax.set_xlabel('Transition event')
    ax.set_ylabel(r'Time since branch entry $T$')
    style_axis(ax, 'Escape and visible switching', 'a')
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc='best')


def draw_kappa(ax, summary, theory, critical):
    ax.plot(theory.kappa, theory.T_esc_theory_infty, color=INK, lw=1.8,
            label='Stable fixed-point theory')
    settled = summary[summary.plateau_ok.astype(bool)]
    unsettled = summary[~summary.plateau_ok.astype(bool)]
    if len(settled):
        ax.errorbar(settled.kappa, settled.T_esc_empirical_infty,
                    yerr=settled.T_esc_empirical_window_std, fmt='o',
                    color=TEAL, mfc=PAPER, ms=4, capsize=2, elinewidth=.85,
                    label='Settled simulation ± window SD')
    if len(unsettled):
        ax.scatter(unsettled.kappa, unsettled.T_esc_empirical_infty,
                   facecolors=PAPER, edgecolors=CORAL, marker='D', s=22,
                   label='Unsettled window', zorder=4)
    ax.axvline(critical.kappa_critical, color=MUTED, ls='--', lw=.8,
               label=rf'$\kappa_{{\rm crit}}={critical.kappa_critical:.3f}$')
    ax.set_xlabel(r'Slow feedback gain $\kappa$')
    ax.set_ylabel(r'Asymptotic $T_{\rm esc}$')
    style_axis(ax, 'Gain controls residence time', 'b')
    ax.legend(loc='best')


def draw_phase_plane(fig, axes, frame):
    eps = np.sort(frame.epsilon.unique())
    kappas = np.sort(frame.kappa.unique())

    def matrix(column):
        return frame.pivot(index='epsilon', columns='kappa', values=column).reindex(
            index=eps, columns=kappas).to_numpy(float)

    times = matrix('mean_T_esc')
    probability = matrix('transition_probability')
    no_success = np.isfinite(probability) & (probability <= 0)
    missing = ~np.isfinite(probability)
    undefined = ~np.isfinite(times) & ~no_success
    times = np.ma.masked_where(no_success | undefined, times)
    xe, ye = cell_edges(kappas), cell_edges(eps, singleton_width=.1)
    finite = times.compressed()
    if finite.size:
        low, high = float(finite.min()), float(finite.max())
        pad = max(abs(low)*.02, .01) if low == high else 0
        norm = Normalize(low-pad, high+pad)
    else:
        norm = Normalize(0, 1)
    for i, (ax, data, title, label) in enumerate(zip(
        axes, [times, np.ma.masked_invalid(probability)],
        ['Escape timing', 'Retrieval reliability'],
        [r'$T_{\rm esc}$ | correct transition', 'Transition probability'],
    )):
        image = ax.pcolormesh(
            xe, ye, data, cmap=MAGNITUDE_CMAP if i == 0 else PROBABILITY_CMAP,
            norm=norm if i == 0 else Normalize(0, 1), shading='flat', rasterized=True)
        mask = (no_success | undefined) if i == 0 else (no_success | missing)
        for iy, ix in np.argwhere(mask):
            ax.add_patch(Rectangle(
                (xe[ix], ye[iy]), xe[ix+1]-xe[ix], ye[iy+1]-ye[iy],
                facecolor=MISSING, edgecolor='#A1ABA9', lw=0,
                hatch='///' if no_success[iy, ix] else '...'))
        if data.size <= 64:
            for iy, y in enumerate(eps):
                for ix, x in enumerate(kappas):
                    value = data[iy, ix]
                    if not np.ma.is_masked(value) and np.isfinite(value):
                        level = norm(value) if i == 0 else value
                        ax.text(x, y, f'{value:.3g}', ha='center', va='center',
                                fontsize=7, color='white' if level < .52 else INK)
        ax.set_xlabel(r'Slow feedback gain $\kappa$')
        style_axis(ax, title, 'cf'[i], grid=None)
        ax.set_xticks(kappas if len(kappas) <= 6 else kappas[::max(1, len(kappas)//5)])
        ax.set_yticks(eps if len(eps) <= 6 else eps[::max(1, len(eps)//5)])
        cb = fig.colorbar(image, ax=ax, orientation='vertical', location='right',
                          fraction=.045, pad=.04, aspect=25)
        cb.set_label(label, rotation=270, labelpad=14)
        cb.outline.set_visible(False)
        if i == 0 and not finite.size:
            ax.text(.5, .5, 'No finite escape times', transform=ax.transAxes,
                    ha='center', color=INK)
    for ax in axes:
        ax.set_ylabel(r'Correlation parameter $\epsilon$')



def draw_noise(axes, frame):
    frame = frame.sort_values('noise_amplitude')
    x = frame.noise_amplitude
    axes[0].plot(x, frame.mean_gap, **series_style(0, True), label='Mean gap')
    interval(axes[0], x, frame.mean_gap_ci_low, frame.mean_gap_ci_high)
    axes[0].plot(x, frame.median_gap, **series_style(1, True), label='Median gap')
    axes[0].axhline(0, color=MUTED, ls=':', lw=.8)
    axes[0].set_ylabel(r'Switching delay $T_{\rm switch}-T_{\rm esc}$')
    axes[0].text(.02, .98, 'Shading: 95% CI of the mean gap',
                 transform=axes[0].transAxes, va='top', fontsize=6.5, color=MUTED)
    axes[1].plot(x, frame.post_escape_switch_probability, **series_style(0, True),
                 label='After stability loss')
    axes[1].plot(x, frame.premature_switch_probability, **series_style(2, True),
                 label='Premature switch')
    axes[1].set_ylabel('Outcome probability')
    for i, ax in enumerate(axes):
        if np.all(x > 0):
            ax.set_xscale('log')
        elif np.any(x > 0):
            ax.set_xscale('symlog', linthresh=float(x[x > 0].min()))
        ax.set_xlabel('Fast-state noise amplitude')
        style_axis(ax, ['Ejection delay', 'Switching outcomes'][i], 'de'[i],
                   probability=i == 1)
        ax.legend(loc='best')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT/'data/raw/paper')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/camera_ready')
    args = parser.parse_args()
    sources = {
        'escape': args.input_root/'escape_time/escape_time_events.csv',
        'kappa': args.input_root/'kappa_sweep/kappa_sweep_summary.csv',
        'theory': args.input_root/'kappa_sweep/kappa_sweep_theory.csv',
        'critical': args.input_root/'kappa_sweep/kappa_critical_point.csv',
        'noise': args.input_root/'noise_sweep/noise_sweep_summary.csv',
        'phase': args.input_root/'correlation_phase_plane/correlation_phase_plane_summary.csv',
    }
    tables = {name: pd.read_csv(path) for name, path in sources.items()}
    configure()
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.3))
    draw_escape(axes[0, 0], tables['escape'])
    draw_kappa(axes[0, 1], tables['kappa'], tables['theory'], tables['critical'].iloc[0])
    draw_noise(axes[1, :2], tables['noise'])
    draw_phase_plane(fig, axes[:, 2], tables['phase'])
    # Computer Modern sans-serif matches TikZ's LaTeX \sf panel lettering.
    for ax in axes.flat:
        for text in ax.texts:
            if text.get_text() in ('a', 'b', 'c', 'd', 'e', 'f'):
                text.set_text(r'$\mathsf{' + text.get_text().upper() + '}$')
                text.set_math_fontfamily('cm')
                text.set_fontweight('normal')
                text.set_color('black')
    fig.patch.set_facecolor('none')
    for ax in fig.axes:
        ax.patch.set_facecolor('none')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'correlation_noise_test'
    outputs = []
    for extension in ('pdf', 'svg', 'png'):
        path = args.output_dir/f'{stem}.{extension}'
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=.06,
                    transparent=True, facecolor='none', edgecolor='none')
        outputs.append(path.name)
    plt.close(fig)
    manifest = {
        'source_root': str(args.input_root.resolve()),
        'panels': {'A': 'escape_time_validation: panel a', 'B': 'kappa_sweep',
                   'C': 'correlation_phase_plane: escape timing',
                   'D': 'noise_sweep: panel a (ejection delay)',
                   'E': 'noise_sweep: panel b (switching outcomes)',
                   'F': 'correlation_phase_plane: retrieval reliability'},
        'layout': [['A', 'B', 'C'], ['D', 'E', 'F']],
        'panel_label_style': r'Computer Modern sans-serif, $\mathsf{A}$ through $\mathsf{F}$',
        'input_sha256': {str(path.relative_to(args.input_root)): hashlib.sha256(
            path.read_bytes()).hexdigest() for path in sources.values()},
        'outputs': outputs, 'raster_dpi': 600,
        'transparent_background': True, 'missing_transition_legend': False,
    }
    (args.output_dir/f'{stem}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Created {", ".join(outputs)} in {args.output_dir}')


if __name__ == '__main__':
    main()
