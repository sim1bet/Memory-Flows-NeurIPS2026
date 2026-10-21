#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import configure,export,save,style_axis,series_style,memory_lines,INK,TEAL,CORAL,MUTED,note

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();configure()
    source=Path(a.input);frame=pd.read_csv(source/'escape_time_events.csv');valid=frame.dropna(subset=['T_esc_empirical','T_esc_theory'])
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.05),gridspec_kw={'width_ratios':[1.35,1]})
    for key,index,label in [('T_esc_theory',0,'Reduced theory'),('T_esc_empirical',1,'Stability loss'),('T_switch_empirical',2,'Visible switch')]:
        # Full event grid preserves gaps; undefined events are not bridged.
        axes[0].plot(frame.event,frame[key],**series_style(index,True),label=label)
    axes[0].set_xlabel('Transition event');axes[0].set_ylabel(r'Time since branch entry $T$')
    style_axis(axes[0],'Escape and visible switching','a');axes[0].legend()
    axes[0].xaxis.get_major_locator().set_params(integer=True)
    if len(valid):
        error=100*(valid.T_esc_empirical-valid.T_esc_theory)/valid.T_esc_theory
        axes[1].axhline(0,color=MUTED,ls=':',lw=.8)
        axes[1].plot(valid.event,error,**series_style(1,True))
    else:note(axes[1],'No paired escape estimates')
    axes[1].set_xlabel('Transition event');axes[1].set_ylabel('Relative timing error (%)');style_axis(axes[1],'Theory discrepancy','b')
    axes[1].xaxis.get_major_locator().set_params(integer=True)
    save(fig,a.output);export(valid,a.data_dir,'escape_time_validation.csv')
    series=pd.read_csv(source/'escape_time_timeseries.csv')
    fig,ax=plt.subplots(figsize=(7.2,2.85));memory_lines(ax,series,'time_slow','saliency')
    for t in frame.absolute_escape_time_slow.dropna():ax.axvline(t,color=MUTED,ls=':',lw=.7,zorder=0)
    ax.plot([],[],color=MUTED,ls=':',lw=.7,label='Stability loss')
    ax.set_xlabel(r'Slow time $t/\tau_z$');ax.set_ylabel(r'Saliency $|z_\mu|$');style_axis(ax,'Sequential saliency transfer')
    fig.legend(*ax.get_legend_handles_labels(),loc='outside upper center',ncol=7,fontsize=6.8)
    save(fig,Path(a.output).with_name('tanh_saliency_trajectory'))
    active=series.groupby('memory').saliency.max().sort_values(ascending=False).head(5).index
    stride=max(1,int(np.ceil(series.time_slow.nunique()/1200)))
    compact=pd.concat([series[series.memory==memory].iloc[::stride] for memory in active],ignore_index=True)
    export(compact,a.data_dir,'tanh_saliency_trajectory.csv')
    for memory,panel in compact.groupby('memory'):export(panel,a.data_dir,f'tanh_saliency_memory_{int(memory)}.csv')
if __name__=='__main__':main()
