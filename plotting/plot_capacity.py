#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle
import numpy as np
import pandas as pd
from common import configure,export,save,series_style,style_axis,interval,cell_edges,PROBABILITY_CMAP,MISSING

def _curves(frame,ylabel,output):
    activations=list(frame.activation.drop_duplicates());ns=sorted(frame.N.unique())
    fig,axes=plt.subplots(1,len(activations),figsize=(3.55*len(activations),3.05),squeeze=False,sharey=True)
    for j,(ax,activation) in enumerate(zip(axes.flat,activations)):
        for n,curve in frame[frame.activation==activation].groupby('N'):
            curve=curve.sort_values('alpha_actual');sty=series_style(ns.index(n),True)
            interval(ax,curve.alpha_actual,curve.ci_low,curve.ci_high,sty['color'])
            ax.plot(curve.alpha_actual,curve.success_probability,**sty,label=f'N = {int(n)}')
        style_axis(ax,'SmoothF' if activation=='smoothf' else 'Tanh',chr(97+j),probability=True)
        ax.set_xlabel(r'Realized load $P/N$')
    axes[0,0].set_ylabel(ylabel)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=max(1,len(ns)),title='Reported 95% confidence bands')
    save(fig,output)

def plot_scaled_cycle(frame,output):_curves(frame,r'Probability of completing $L\simeq P/3$',output)
def plot_fixed_cycle(frame,output):
    lengths=sorted(frame.cycle_length.unique());label=f'Probability of completing L = {int(lengths[0])}' if len(lengths)==1 else 'Full-cycle probability'
    _curves(frame,label,output)

def plot_fanout(frame,activation,output):
    panel=frame[frame.activation==activation];degrees=sorted(panel.outdegree.unique());ns=sorted(panel.N.unique())
    columns=min(3,max(1,len(degrees)));rows=int(np.ceil(len(degrees)/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(7.2,2.15*rows),squeeze=False,sharex=True,sharey=True)
    handles=[]
    for j,(ax,degree) in enumerate(zip(axes.flat,degrees)):
        for n,curve in panel[panel.outdegree==degree].groupby('N'):
            curve=curve.sort_values('alpha_actual');sty=series_style(ns.index(n),True)
            interval(ax,curve.alpha_actual,curve.ci_low,curve.ci_high,sty['color'])
            line,=ax.plot(curve.alpha_actual,curve.success_probability,**sty,label=f'N = {int(n)}')
            if j==0:handles.append(line)
        style_axis(ax,f'Out-degree {int(degree)}',chr(97+j),probability=True)
        ax.set_xlabel(r'Realized load $P/N$')
        ax.tick_params(axis='x',which='both',bottom=True,labelbottom=True)
        if j%columns==0:ax.set_ylabel('Transition probability')
    for ax in axes.flat[len(degrees):]:ax.set_visible(False)
    fig.legend(handles=handles,loc='outside upper center',ncol=len(handles),title=('SmoothF' if activation=='smoothf' else 'Tanh')+' · reported 95% confidence bands')
    save(fig,output)

def _matrix_for_heatmap(panel):
    degrees=np.sort(panel.outdegree.unique()).astype(float);loads=np.sort(panel.alpha_actual.unique()).astype(float)
    matrix=panel.pivot(index='outdegree',columns='alpha_actual',values='success_probability').reindex(index=degrees,columns=loads).to_numpy(float)
    return loads,degrees,matrix

def plot_fanout_phase(frame,activation,output):
    panel=frame[frame.activation==activation];ns=sorted(panel.N.unique())
    fig,axes=plt.subplots(1,len(ns),figsize=(max(4.4,2.5*len(ns)),3.6),squeeze=False,sharey=True)
    absent=False
    for j,(ax,n) in enumerate(zip(axes.flat,ns)):
        loads,degrees,matrix=_matrix_for_heatmap(panel[panel.N==n]);xe=cell_edges(loads,.05);ye=cell_edges(degrees)
        image=ax.pcolormesh(xe,ye,np.ma.masked_invalid(matrix),cmap=PROBABILITY_CMAP,vmin=0,vmax=1,shading='flat',rasterized=True)
        for iy,ix in np.argwhere(~np.isfinite(matrix)):
            absent=True;ax.add_patch(Rectangle((xe[ix],ye[iy]),xe[ix+1]-xe[ix],ye[iy+1]-ye[iy],facecolor=MISSING,edgecolor='#A1ABA9',hatch='///',lw=0))
        if matrix.size<=40:
            for iy,y in enumerate(degrees):
                for ix,x in enumerate(loads):
                    v=matrix[iy,ix]
                    if np.isfinite(v):ax.text(x,y,f'{v:.2g}',ha='center',va='center',fontsize=7,color='white' if v<.52 else '#16324F')
        style_axis(ax,f'N = {int(n)}',chr(97+j),grid=None);ax.set_xlabel(r'Realized load $P/N$')
        ax.set_yticks(degrees[::max(1,len(degrees)//6)])
    axes[0,0].set_ylabel('Out-degree')
    cb=fig.colorbar(image,ax=axes.ravel().tolist(),location='bottom',fraction=.10,pad=.14,aspect=40,shrink=.85)
    cb.set_label('Transition probability');cb.set_ticks([0,.25,.5,.75,1]);cb.outline.set_visible(False)
    fig.suptitle(('SmoothF' if activation=='smoothf' else 'Tanh')+' · branching reliability',fontsize=10,fontweight='bold')
    if absent:fig.legend(handles=[Patch(facecolor=MISSING,edgecolor='#A1ABA9',hatch='///',label='Not sampled / infeasible graph')],loc='outside upper right',fontsize=6)
    save(fig,output)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();configure()
    frame=pd.read_csv(Path(a.input)/'capacity_probability.csv');output=Path(a.output_dir)
    for experiment,name,fn in [('scaled_cycle','capacity_scaled_cycle',plot_scaled_cycle),('fixed_cycle','capacity_fixed_cycle_L5',plot_fixed_cycle)]:
        subset=frame[frame.experiment==experiment].copy()
        if len(subset):
            fn(subset,str(output/name));export(subset,a.data_dir,name+'.csv')
            for activation,panel in subset.groupby('activation'):export(panel,a.data_dir,f'{name}_{activation}.csv')
    for experiment,name,fn in [('fanout','capacity_fanout',plot_fanout),('fanout_phase','capacity_fanout_phase_plane',plot_fanout_phase)]:
        subset=frame[frame.experiment==experiment].copy()
        if len(subset):
            export(subset,a.data_dir,name+'.csv')
            for activation,panel in subset.groupby('activation'):
                export(panel,a.data_dir,f'{name}_{activation}.csv');fn(panel,activation,str(output/f'{name}_{activation}'))
if __name__=='__main__':main()
