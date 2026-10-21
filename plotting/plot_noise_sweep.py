#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import configure,export,save,style_axis,series_style,interval,MUTED

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();configure()
    s=pd.read_csv(Path(a.input)/'noise_sweep_summary.csv').sort_values('noise_amplitude')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.15));x=s.noise_amplitude
    axes[0].plot(x,s.mean_gap,**series_style(0,True),label='Mean gap')
    interval(axes[0],x,s.mean_gap_ci_low,s.mean_gap_ci_high)
    axes[0].plot(x,s.median_gap,**series_style(1,True),label='Median gap')
    axes[0].axhline(0,color=MUTED,ls=':',lw=.8)
    axes[0].set_ylabel(r'Switching delay $T_{\rm switch}-T_{\rm esc}$')
    axes[1].plot(x,s.post_escape_switch_probability,**series_style(0,True),label='After stability loss')
    axes[1].plot(x,s.premature_switch_probability,**series_style(2,True),label='Premature switch')
    axes[1].set_ylabel('Outcome probability')
    for i,ax in enumerate(axes):
        if np.all(x>0):ax.set_xscale('log')
        elif np.any(x>0):ax.set_xscale('symlog',linthresh=float(x[x>0].min()))
        ax.set_xlabel('Fast-state noise amplitude');style_axis(ax,['Ejection delay','Switching outcomes'][i],chr(97+i),probability=i==1)
        ax.legend(loc='best')
    fig.suptitle('Shading: 95% confidence interval of the mean gap',fontsize=7,color=MUTED)
    save(fig,a.output);export(s,a.data_dir,'noise_sweep.csv')
if __name__=='__main__':main()
