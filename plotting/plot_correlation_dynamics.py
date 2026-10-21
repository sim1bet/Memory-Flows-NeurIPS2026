#!/usr/bin/env python3
"""Raw correlation dynamics with visible validity limits and missing estimates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import configure,export,save,style_axis,series_style,interval,note,INK,MUTED,CORAL

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--camera-ready',action='store_true',help='Transparent 600 dpi exports with black uppercase panel labels');a=p.parse_args();configure()
    source=Path(a.input);s=pd.read_csv(source/'correlation_dynamics_summary.csv').sort_values('epsilon');theory=json.loads((source/'correlation_dynamics_theory.json').read_text());export(s,a.data_dir,'correlation_dynamics.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),sharex=True);x=s.epsilon.to_numpy(float)
    epsmax=float(theory.get('epsilon_max_interpolated',np.nan));threshold=float(theory['epsilon_max_success_probability'])
    ax=axes[0,0]
    ax.plot(x,s.plateau_fraction,'o',color=INK,label='Observed',ms=3.8)
    ax.plot(x,s.isotonic_sustained_success_fraction,color=INK,lw=1.2,label='Monotone estimate')
    ax.axhline(threshold,color=MUTED,ls='--',lw=.8,label=f'Criterion = {threshold:g}')
    ax.set_ylabel('Sustained-cycle probability');ax.legend(loc='best')
    ax=axes[0,1]
    ax.plot(x,s.mean_kappa_crit,**series_style(0,True),label=r'Estimated $\kappa_{\rm crit}$')
    interval(ax,x,s.kappa_crit_ci_low,s.kappa_crit_ci_high)
    ax.plot(x,s.mean_evaluation_kappa,**series_style(1,True),label=r'Evaluation gain $\kappa$')
    ax.axhline(float(theory['orthogonal_kappa_crit']),color=MUTED,ls=':',lw=.9,label='Orthogonal threshold')
    missing=~np.isfinite(s.mean_kappa_crit)
    if missing.any():ax.plot(x[missing],np.full(missing.sum(),.03),linestyle='none',marker='x',color=CORAL,transform=ax.get_xaxis_transform(),label='Threshold unestimated',clip_on=False)
    ax.set_ylabel(r'Gain $\kappa$');ax.legend(loc='best')
    mask=s.within_epsilon_max.astype(bool).to_numpy()
    for ax,key,ref,lo,hi,ylabel in [
        (axes[1,0],'mean_asymptotic_T_esc','mean_orthogonal_T_esc_matched','asymptotic_T_esc_ci_low','asymptotic_T_esc_ci_high',r'Asymptotic $T_{\rm esc}$'),
        (axes[1,1],'mean_asymptotic_Z','mean_orthogonal_Z_matched','asymptotic_Z_ci_low','asymptotic_Z_ci_high',r'Asymptotic saliency $\bar Z$')]:
        # Keep the full grid with NaNs: excluded points must not be bridged.
        y=np.where(mask,s[key],np.nan);r=np.where(mask,s[ref],np.nan)
        if np.isfinite(y).any() or np.isfinite(r).any():
            ax.plot(x,y,**series_style(0,True),label='Raw correlated')
            ax.plot(x,r,**series_style(1,True),label='Matched orthogonal')
            interval(ax,x,np.where(mask,s[lo],np.nan),np.where(mask,s[hi],np.nan));ax.legend()
        else:note(ax,'No qualified asymptotic estimates\nin this dataset')
        ax.set_ylabel(ylabel);ax.set_xlabel(r'Correlation parameter $\epsilon$')
    titles=['Sustained retrieval','Gain required for retrieval','Escape timing','Asymptotic saliency']
    for i,ax in enumerate(axes.flat):
        style_axis(ax,titles[i],chr(97+i),probability=i==0)
        if np.isfinite(epsmax):ax.axvline(epsmax,color=MUTED,ls=':',lw=.8)
    fig.suptitle('Shading: reported 95% confidence intervals',fontsize=7,color=MUTED)
    if a.camera_ready:
        for ax in axes.flat:
            for text in ax.texts:
                if text.get_text() in ('a','b','c','d'):
                    text.set_text(r'$\mathsf{' + text.get_text().upper() + '}$')
                    text.set_math_fontfamily('cm');text.set_fontweight('normal');text.set_color('black')
        fig.patch.set_facecolor('none')
        for ax in fig.axes:ax.patch.set_facecolor('none')
        output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('pdf','svg','png'):
            fig.savefig(output.with_suffix('.'+ext),dpi=600,bbox_inches='tight',pad_inches=.06,
                        transparent=True,facecolor='none',edgecolor='none')
        plt.close(fig)
    else:
        save(fig,a.output)
if __name__=='__main__':main()
