#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common import configure,export,save,style_axis,series_style,interval,MUTED

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();configure()
    f=pd.read_csv(Path(a.input)/'correlation_summary.csv');activations=list(f.activation.drop_duplicates())
    fig,axes=plt.subplots(len(activations),2,figsize=(7.2,2.8*len(activations)),squeeze=False,sharex=True)
    for row,activation in enumerate(activations):
        for rule,curve in f[f.activation==activation].groupby('rule'):
            curve=curve.sort_values('correlation_parameter');sty=series_style(0 if rule=='raw' else 1,True)
            axes[row,0].plot(curve.correlation_parameter,curve.success_probability,**sty,label=rule.upper())
            interval(axes[row,0],curve.correlation_parameter,curve.ci_low,curve.ci_high,sty['color'])
            axes[row,1].plot(curve.correlation_parameter,curve.mean_decoded_distractor,**sty,label=rule.upper())
        for col,ax in enumerate(axes[row]):
            title=('SmoothF' if activation=='smoothf' else 'Tanh')+' · '+['sequence reliability','unintended activation'][col]
            style_axis(ax,title,chr(97+row*2+col),probability=col==0);ax.legend()
        axes[row,0].set_ylabel('Full-cycle probability');axes[row,1].set_ylabel('Maximum unintended coordinate')
    for ax in axes[-1]:ax.set_xlabel('Correlation parameter')
    fig.suptitle('Reliability bands: reported 95% confidence intervals',fontsize=7,color=MUTED)
    save(fig,a.output);export(f,a.data_dir,'correlation_summary.csv')
    for activation,panel in f.groupby('activation'):export(panel,a.data_dir,f'correlation_{activation}.csv')
if __name__=='__main__':main()
