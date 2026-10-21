#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from common import configure,export,save,style_axis,series_style,INK,MUTED,PAPER

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();configure()
    source=Path(a.input);maps=pd.read_csv(source/'return_map.csv');points=pd.read_csv(source/'return_map_fixed_points.csv');activations=list(maps.activation.drop_duplicates())
    fig,axes=plt.subplots(1,len(activations),figsize=(3.6*len(activations),3.75),squeeze=False)
    kappas=sorted(maps.kappa.unique())
    for i,(ax,activation) in enumerate(zip(axes.flat,activations)):
        panel=maps[maps.activation==activation]
        for kappa,curve in panel.groupby('kappa'):
            sty=series_style(kappas.index(kappa));ax.plot(curve.Z,curve.Z_plus,**sty,label=rf'$\kappa={kappa:g}$')
            fixed=points[(points.activation==activation)&(points.kappa==kappa)]
            for stable,group in fixed.groupby('stable'):
                ax.scatter(group.fixed_point,group.fixed_point,s=25,marker='o' if stable else 'D',
                   facecolor=sty['color'] if stable else PAPER,edgecolor=sty['color'],lw=1,zorder=6)
        high=max(panel.Z.max(),panel.Z_plus.max())*1.02
        ax.plot([0,high],[0,high],ls='--',color=MUTED,lw=.85,label='Identity')
        ax.set_xlim(0,high);ax.set_ylim(0,high);ax.set_aspect('equal',adjustable='box')
        style_axis(ax,'SmoothF' if activation=='smoothf' else 'Tanh',chr(97+i),grid=None)
        ax.set_xlabel('Entry saliency $Z$');ax.set_ylabel('Next entry saliency $Z^+$');ax.legend(loc='lower right')
    fig.legend(handles=[Line2D([],[],marker='o',color=INK,ls='none',ms=4,label='Stable fixed point'),Line2D([],[],marker='D',color=INK,mfc=PAPER,ls='none',ms=4,label='Unstable fixed point')],loc='outside upper center',ncol=2)
    save(fig,a.output);export(maps,a.data_dir,'return_map.csv');export(points,a.data_dir,'return_map_fixed_points.csv')
    x=np.linspace(-3,3,500);af=pd.DataFrame({'x':x,'tanh':np.tanh(x),'smoothf':x/(1+np.abs(x))})
    fig,ax=plt.subplots(figsize=(4.5,2.8))
    ax.axhline(0,color=MUTED,lw=.5,alpha=.4);ax.axvline(0,color=MUTED,lw=.5,alpha=.4)
    for i,(key,label) in enumerate([('tanh','Tanh'),('smoothf','SmoothF')]):ax.plot(x,af[key],**series_style(i),label=label)
    ax.set_xlabel('$x$');ax.set_ylabel(r'$\psi(x)$');style_axis(ax,'Activation response',grid=None);ax.legend(loc='lower right')
    save(fig,Path(a.output).with_name('activation_functions'));export(af,a.data_dir,'activation_functions.csv')
    iz=pd.read_csv(source/'smoothf_implicit_G_Z4.csv');iff=pd.read_csv(source/'smoothf_implicit_G_fixed.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
    for i,(ax,f,col,label,title) in enumerate(zip(axes,[iz,iff],['G','G_fixed'],[r'$G(4,X)$',r'$G(X^2,X)$'],['Escape condition','Fixed-point condition'])):
        ax.plot(f.X,f[col],color=INK,label=label);ax.axhline(float(f.inverse_kappa.iloc[0]),color=MUTED,ls='--',lw=.85,label=r'$1/\kappa$')
        ax.set_xlabel('$X$');ax.set_ylabel('$G$');style_axis(ax,title,chr(97+i));ax.legend(loc='best')
    save(fig,Path(a.output).with_name('smoothf_implicit_curves'))
    export(iz,a.data_dir,'smoothf_implicit_G_Z4.csv');export(iff,a.data_dir,'smoothf_implicit_G_fixed.csv')
    for (activation,kappa),curve in maps.groupby(['activation','kappa']):export(curve[['Z','Z_plus','T_esc']],a.data_dir,f'return_map_{activation}_kappa_{kappa:g}.csv')
if __name__=='__main__':main()
