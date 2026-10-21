#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common import configure,export,save,style_axis,INK,TEAL,CORAL,MUTED,PAPER

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();configure()
    root=Path(a.input);summary=pd.read_csv(root/'kappa_sweep_summary.csv');theory=pd.read_csv(root/'kappa_sweep_theory.csv');critical=pd.read_csv(root/'kappa_critical_point.csv').iloc[0]
    fig,ax=plt.subplots(figsize=(5.6,3.35))
    ax.plot(theory.kappa,theory.T_esc_theory_infty,color=INK,lw=1.8,label='Stable fixed-point theory')
    settled=summary[summary.plateau_ok.astype(bool)];unsettled=summary[~summary.plateau_ok.astype(bool)]
    if len(settled):ax.errorbar(settled.kappa,settled.T_esc_empirical_infty,yerr=settled.T_esc_empirical_window_std,fmt='o',color=TEAL,mfc=PAPER,ms=4,capsize=2,elinewidth=.85,label='Settled simulation ± window SD')
    if len(unsettled):ax.scatter(unsettled.kappa,unsettled.T_esc_empirical_infty,facecolors=PAPER,edgecolors=CORAL,marker='D',s=22,label='Unsettled window',zorder=4)
    ax.axvline(critical.kappa_critical,color=MUTED,ls='--',lw=.8,label=rf'$\kappa_{{\rm crit}}={critical.kappa_critical:.3f}$')
    ax.set_xlabel(r'Slow feedback gain $\kappa$');ax.set_ylabel(r'Asymptotic $T_{\rm esc}$');style_axis(ax,'Gain controls residence time');ax.legend(loc='best')
    save(fig,a.output);export(summary,a.data_dir,'kappa_sweep_empirical.csv');export(theory,a.data_dir,'kappa_sweep_theory.csv')
if __name__=='__main__':main()
