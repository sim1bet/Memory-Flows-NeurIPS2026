#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common import configure,export,save,memory_lines,style_axis

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();configure()
    frame=pd.read_csv(Path(a.input)/'limit_cycles_timeseries.csv');cases=list(frame.case.drop_duplicates())
    rows=(len(cases)+1)//2
    emphasis=frame.groupby('memory')['saliency'].max().nlargest(5).index
    fig,axes=plt.subplots(rows,2,figsize=(7.2,2.35*rows),squeeze=False)
    ymax=frame.saliency.max()*1.12
    for i,(ax,case) in enumerate(zip(axes.flat,cases)):
        panel=frame[frame.case==case];first=panel.iloc[0]
        memory_lines(ax,panel,'time_slow','saliency',emphasis=emphasis)
        # Config case IDs are not used to assert sub/supercritical behavior.
        activation='SmoothF' if first['activation']=='smoothf' else 'Tanh'
        title=rf"{activation} · {first['patterns']} · $\kappa={first['kappa']:g}$"
        style_axis(ax,title,chr(97+i));ax.set_ylim(0,ymax)
        ax.set_xlabel(r'Slow time $t/\tau_z$');ax.set_ylabel(r'Saliency $|z_\mu|$')
    for ax in axes.flat[len(cases):]:ax.set_visible(False)
    legend={}
    for ax in axes.flat:
        for handle,label in zip(*ax.get_legend_handles_labels()):legend.setdefault(label,handle)
    fig.legend(list(legend.values()),list(legend),loc='outside upper center',ncol=min(7,len(legend)),fontsize=6.8)
    save(fig,a.output);export(frame,a.data_dir,'limit_cycles.csv')
    for case,panel in frame.groupby('case'):export(panel,a.data_dir,f'limit_cycle_{case}.csv')
if __name__=='__main__':main()
