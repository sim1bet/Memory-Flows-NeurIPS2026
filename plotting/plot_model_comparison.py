#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common import configure,export,save,memory_lines,style_axis

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();configure()
    frame=pd.read_csv(Path(a.input)/'model_comparison_timeseries.csv');cases=list(frame.case.drop_duplicates())
    emphasis=frame.groupby('memory')['cosine_overlap'].max().nlargest(5).index
    fig,axes=plt.subplots(len(cases),1,figsize=(7.2,1.75*len(cases)),squeeze=False,sharex=True,sharey=True)
    for i,(ax,case) in enumerate(zip(axes.flat,cases)):
        panel=frame[frame.case==case];first=panel.iloc[0]
        memory_lines(ax,panel,'time','cosine_overlap',emphasis=emphasis)
        title=('One timescale' if first['model']=='one_timescale' else 'Two timescales')+' · '+first['patterns']+' memories'
        style_axis(ax,title,chr(97+i),probability=True)
        ax.set_ylabel('Absolute cosine overlap')
    legend={}
    for ax in axes.flat:
        for handle,label in zip(*ax.get_legend_handles_labels()):legend.setdefault(label,handle)
    fig.legend(list(legend.values()),list(legend),loc='outside upper center',ncol=min(7,len(legend)),fontsize=6.8)
    axes[-1,0].set_xlabel('Physical time $t$');save(fig,a.output)
    export(frame,a.data_dir,'model_comparison.csv')
    for case,panel in frame.groupby('case'):export(panel,a.data_dir,f'model_comparison_{case}.csv')
if __name__=='__main__':main()
