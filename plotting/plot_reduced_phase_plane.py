#!/usr/bin/env python3
"""Mechanistic portrait of source-branch drift, not a full-network phase plane."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from common import configure,save,MAGNITUDE_CMAP,INK,PAPER,MUTED,CORAL,style_axis

def arrow_along(ax,x,y,fraction=.58,color=PAPER,lw=1.):
    xy=np.column_stack([x,y]);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    if length[-1]<=0:return
    index=min(len(xy)-2,max(0,np.searchsorted(length,fraction*length[-1])))
    end=min(len(xy)-1,index+max(2,len(xy)//30))
    ax.annotate('',xy=xy[end],xytext=xy[index],arrowprops={'arrowstyle':'-|>','mutation_scale':7,'color':color,'lw':lw},zorder=8)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();configure();plt.rcParams['figure.constrained_layout.use']=False
    root=Path(a.input);fields=pd.read_csv(root/'reduced_phase_field.csv');boundaries=pd.read_csv(root/'reduced_phase_boundary.csv');paths=pd.read_csv(root/'reduced_phase_paths.csv')
    for kappa,field in fields.groupby('kappa'):
        activations=list(field.activation.drop_duplicates());fig,axes=plt.subplots(1,len(activations),figsize=(3.6*len(activations),4.5),squeeze=False,sharex=True,sharey=True,layout=None)
        fig.subplots_adjust(left=.075,right=.975,bottom=.27,top=.79,wspace=.25)
        vmax=field.loc[field.valid.astype(bool),'speed'].max();norm=Normalize(0,vmax)
        for i,(ax,activation) in enumerate(zip(axes.flat,activations)):
            f=field[field.activation==activation];x=np.sort(f.source.unique());y=np.sort(f.successor.unique())
            def grid(key):return f.pivot(index='successor',columns='source',values=key).reindex(index=y,columns=x).to_numpy()
            mask=~grid('valid').astype(bool);speed=np.ma.masked_where(mask,grid('speed'))
            image=ax.pcolormesh(x,y,speed,cmap=MAGNITUDE_CMAP,norm=norm,shading='nearest',rasterized=True)
            b=boundaries[(boundaries.activation==activation)&(boundaries.kappa==kappa)]
            ax.fill_between(b.source,np.minimum(b.boundary,y.max()),y.max(),color='#ECEEEA',zorder=2)
            # Field density and weight are controlled; the boundary mask prevents
            # misleading streamlines outside the source-branch validity domain.
            u=np.ma.masked_where(mask,grid('d_source'));v=np.ma.masked_where(mask,grid('d_successor'))
            ax.streamplot(x,y,u,v,density=(.65,.8),color=(1,1,1,.28),linewidth=.45+.35*speed.filled(0)/vmax,arrowsize=.55,minlength=.18,maxlength=2.5,zorder=3)
            visible=b.boundary<=y.max()
            ax.plot(b.loc[visible,'source'],b.loc[visible,'boundary'],color=PAPER,lw=1.4,ls=(0,(4,2)),zorder=5,path_effects=[pe.Stroke(linewidth=2.6,foreground=INK),pe.Normal()])
            pths=paths[(paths.activation==activation)&(paths.kappa==kappa)]
            for pid,path in pths.groupby('path_id'):
                selected=bool(path.selected.iloc[0]);color=PAPER
                ax.plot(path.source,path.successor,color=color,lw=2 if selected else .85,alpha=1 if selected else .52,zorder=7 if selected else 4,
                    path_effects=[pe.Stroke(linewidth=3.2,foreground=INK),pe.Normal()] if selected else None)
                arrow_along(ax,path.source,path.successor,color=color,lw=1 if selected else .65)
                if selected:
                    ax.scatter(path.source.iloc[0],path.successor.iloc[0],s=24,facecolor=INK,edgecolor=PAPER,lw=1.1,zorder=9,clip_on=False)
                    if path.escaped.iloc[-1]:ax.scatter(path.source.iloc[-1],path.successor.iloc[-1],s=26,marker='s',facecolor=CORAL,edgecolor=PAPER,lw=.85,zorder=9)
            ax.set_xlim(x.min(),x.max());ax.set_ylim(0,y.max());ax.set_aspect('equal',adjustable='box')
            style_axis(ax,'SmoothF' if activation=='smoothf' else 'Tanh',chr(97+i),grid=None)
            ax.set_xlabel(r'Source saliency $s=|z_{\rm source}|$')
        axes[0,0].set_ylabel(r'Successor saliency $r=|z_{\rm successor}|$')
        cax=fig.add_axes([.17,.14,.68,.025])
        cb=fig.colorbar(image,cax=cax,orientation='horizontal')
        cb.set_label(r'Reduced saliency speed $\sqrt{\dot{s}^{\,2}+\dot{r}^{\,2}}$');cb.outline.set_visible(False)
        fig.legend(handles=[Line2D([],[],color=INK,ls='--',lw=1,label='Stability boundary'),Line2D([],[],color=INK,marker='o',mfc=PAPER,ls='none',ms=4,label='Entry'),Line2D([],[],color=CORAL,marker='s',ls='none',ms=4,label='Stability loss'),Patch(facecolor='#ECEEEA',edgecolor=MUTED,lw=.5,label='Outside stable branch')],loc='upper center',bbox_to_anchor=(.51,.92),ncol=4,fontsize=6.7)
        fig.suptitle(rf'Reduced source-branch flow · $\kappa={kappa:g}$',fontsize=11,fontweight='bold',y=.985)
        fig.text(.5,.006,'Orthogonal memories · unit gains · singular limit · paths stop at branch loss\nAnalytical reduced model · not a full-network phase portrait',ha='center',fontsize=6.5,color=MUTED,linespacing=1.6)
        save(fig,Path(a.output_dir)/f'reduced_phase_plane_kappa_{kappa:g}',context=False)
if __name__=='__main__':main()
