#!/usr/bin/env python3
"""Parameter heatmaps: conditional escape time and observed transition probability."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from common import configure, export, save, PROBABILITY_CMAP, MAGNITUDE_CMAP, MISSING, MUTED, INK, cell_edges, style_axis

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args()
    frame=pd.read_csv(Path(a.input)/'correlation_phase_plane_summary.csv')
    export(frame,a.data_dir,'correlation_phase_plane.csv');configure()
    eps=np.sort(frame.epsilon.unique());kappas=np.sort(frame.kappa.unique())
    def matrix(column):
        return frame.pivot(index='epsilon',columns='kappa',values=column).reindex(index=eps,columns=kappas).to_numpy(float)
    tesc=matrix('mean_T_esc');probability=matrix('transition_probability')
    no_success=np.isfinite(probability)&(probability<=0)
    missing=~np.isfinite(probability)
    undefined=(~np.isfinite(tesc))&~no_success
    tesc=np.ma.masked_where(no_success|undefined,tesc)
    xe=cell_edges(kappas);ye=cell_edges(eps,singleton_width=.1)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.75),sharey=True)
    finite=tesc.compressed()
    if finite.size:
        low,high=float(finite.min()),float(finite.max())
        pad=max(abs(low)*.02,.01) if low==high else 0
        norm=Normalize(low-pad,high+pad)
    else: norm=Normalize(0,1)
    for i,(ax,data,label,title) in enumerate(zip(axes,[tesc,np.ma.masked_invalid(probability)],
         [r'$T_{\rm esc}$ | correct transition','Transition probability'],['Escape timing','Retrieval reliability'])):
        image=ax.pcolormesh(xe,ye,data,cmap=MAGNITUDE_CMAP if i==0 else PROBABILITY_CMAP,norm=norm if i==0 else Normalize(0,1),shading='flat',rasterized=True)
        mask=(no_success|undefined) if i==0 else (no_success|missing)
        for iy,ix in np.argwhere(mask):
            ax.add_patch(Rectangle((xe[ix],ye[iy]),xe[ix+1]-xe[ix],ye[iy+1]-ye[iy],
                facecolor=MISSING,edgecolor='#A1ABA9',lw=0,hatch='///' if no_success[iy,ix] else '...'))
        # Sparse grids are explicitly shown as sampled cells; no inferred contours.
        if data.size<=64:
            for iy,y in enumerate(eps):
                for ix,x in enumerate(kappas):
                    value=data[iy,ix]
                    if not np.ma.is_masked(value) and np.isfinite(value):
                        level=(norm(value) if i==0 else value)
                        ax.text(x,y,f'{value:.3g}',ha='center',va='center',fontsize=7,color='white' if level<.52 else INK)
        ax.set_xlabel(r'Slow feedback gain $\kappa$');style_axis(ax,title,chr(97+i),grid=None)
        ax.set_xticks(kappas if len(kappas)<=6 else kappas[::max(1,len(kappas)//5)])
        ax.set_yticks(eps if len(eps)<=6 else eps[::max(1,len(eps)//5)])
        cb=fig.colorbar(image,ax=ax,orientation='horizontal',fraction=.085,pad=.10,aspect=30)
        cb.set_label(label);cb.outline.set_visible(False)
        if i==0 and not finite.size: ax.text(.5,.5,'No finite escape times',transform=ax.transAxes,ha='center',color=INK)
    axes[0].set_ylabel(r'Correlation parameter $\epsilon$')
    handles=[]
    if no_success.any(): handles.append(Patch(facecolor=MISSING,edgecolor='#A1ABA9',hatch='///',label='No correct transition'))
    if undefined.any() or missing.any(): handles.append(Patch(facecolor=MISSING,edgecolor='#A1ABA9',hatch='...',label='Unestimated / missing'))
    if handles: fig.legend(handles=handles,loc='outside upper center',ncol=2,fontsize=6.5)
    save(fig,a.output)
if __name__=='__main__':main()
