#!/usr/bin/env python3
"""Render every editorial figure from saved data without rerunning simulations."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['smoke','paper'],default='smoke');p.add_argument('--input-root',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--quick',action='store_true',help='110 dpi PNG only');a=p.parse_args()
    raw=a.input_root or ROOT/'data/raw'/a.mode;out=a.output_dir or ROOT/'figures/editorial'/a.mode
    out=out.resolve();out.mkdir(parents=True,exist_ok=True);data=ROOT/'figures/data'/('smoke' if a.mode=='smoke' else '')
    env=dict(os.environ,MPLBACKEND='Agg');env.pop('IDP_SMOKE',None)
    if a.quick:env['IDP_SMOKE']='1'
    env['IDP_FIGURE_CONTEXT']='Reduced validation dataset · not a paper-scale estimate' if a.mode=='smoke' else ''
    jobs=['model_comparison','limit_cycles','escape_time','return_map','kappa_sweep','noise_sweep','capacity','correlation','correlation_dynamics','correlation_phase_plane']
    manifest={'mode':a.mode,'source_root':str(raw.relative_to(ROOT) if raw.is_relative_to(ROOT) else raw),'figures':[],'input_sha256':{}}
    for path in sorted(raw.glob('*/*')):
        if path.is_file():manifest['input_sha256'][str(path.relative_to(raw))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for job in jobs:
        source=raw/job
        if not source.is_dir():raise FileNotFoundError(f'Missing saved dataset: {source}; run the corresponding experiment first.')
        print('Render',job,flush=True)
        dest=['--output-dir',str(out)] if job=='capacity' else ['--output',str(out/('escape_time_validation' if job=='escape_time' else job))]
        subprocess.run([sys.executable,str(ROOT/'plotting'/f'plot_{job}.py'),'--input',str(source),*dest,'--data-dir',str(data)],check=True,env=env)
    # Exact reduced field exported independently of the stochastic/full ODE runs.
    reduced=data/'reduced_phase_plane'
    subprocess.run([sys.executable,str(ROOT/'experiments/export_reduced_phase_plane.py'),'--input',str(raw/'return_map'),'--output',str(reduced)],check=True,env=env)
    env['IDP_FIGURE_CONTEXT']='Analytical reduced model · not a full-network phase portrait'
    subprocess.run([sys.executable,str(ROOT/'plotting/plot_reduced_phase_plane.py'),'--input',str(reduced),'--output-dir',str(out)],check=True,env=env)
    # LaTeX includes the same publication master, avoiding a second styling path.
    wrapper_dir=ROOT/'plotting/latex/generated'/a.mode
    wrapper_dir.mkdir(parents=True,exist_ok=True)
    for pdf in sorted(out.glob('*.pdf')):
        relative=os.path.relpath(pdf,wrapper_dir).replace(os.sep,'/')
        lines=[chr(92)+'documentclass[border=0pt]{standalone}',chr(92)+'usepackage{graphicx}',chr(92)+'begin{document}',chr(92)+'includegraphics{'+relative+'}',chr(92)+'end{document}']
        (wrapper_dir/('include_'+pdf.stem+'.tex')).write_text(chr(10).join(lines)+chr(10))
    manifest['figures']=[p.name for p in sorted(out.glob('*')) if p.suffix in ('.pdf','.svg','.png')]
    (out/'render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Complete: {out}',flush=True)
if __name__=='__main__':main()
