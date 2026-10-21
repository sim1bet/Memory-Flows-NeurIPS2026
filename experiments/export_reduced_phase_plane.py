#!/usr/bin/env python3
"""Export the existing singular-limit source-branch dynamics as figure data.

No full-network simulation is performed. In slow time, s=|z_source| and
r=|z_successor| obey s'=-s, r'=-r+kappa*branch_response(s). Valid only while
s>1 and r<h(s), for unit gains, orthogonal memories, and out-degree <=1.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from idp_hopfield.activations import branch_response,stability_threshold
from idp_hopfield.io import ensure_dir,write_csv,write_json

def integrate_branch(entry,kappa,activation,successor=0.):
    def rhs(t,z):return [-z[0],-z[1]+kappa*branch_response(z[0],activation)]
    def escape(t,z):return z[1]-stability_threshold(z[0],activation)
    escape.terminal=True;escape.direction=1
    end=np.log(entry/1.000001)
    sol=solve_ivp(rhs,[0,end],[entry,successor],events=escape,rtol=1e-9,atol=1e-11,max_step=.02,dense_output=True)
    t=np.linspace(0,sol.t[-1],350);s,r=sol.sol(t)
    return pd.DataFrame({'time_slow':t,'source':s,'successor':r,'escaped':bool(len(sol.t_events[0]))})

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--source-points',type=int,default=401);p.add_argument('--successor-points',type=int,default=361);a=p.parse_args()
    if min(a.source_points,a.successor_points)<3:raise ValueError('Mesh dimensions must be at least 3')
    src=Path(a.input);out=ensure_dir(a.output);maps=pd.read_csv(src/'return_map.csv');points=pd.read_csv(src/'return_map_fixed_points.csv')
    fields=[];boundaries=[];paths=[]
    for (activation,kappa),panel in maps.groupby(['activation','kappa']):
        xmax=max(4.,float(panel.Z.max()));ymax=max(xmax,float(panel.Z_plus.max()) if panel.Z_plus.notna().any() else 0)
        x=np.linspace(1.00001,xmax,a.source_points);y=np.linspace(0,ymax,a.successor_points);s,r=np.meshgrid(x,y)
        h=stability_threshold(s,activation);u=-s;v=-r+kappa*branch_response(s,activation)
        field=pd.DataFrame({'source':s.ravel(),'successor':r.ravel(),'d_source':u.ravel(),'d_successor':v.ravel(),'speed':np.hypot(u,v).ravel(),'valid':(r<h).ravel()})
        field['activation']=activation;field['kappa']=kappa;fields.append(field)
        bx=np.linspace(1.00001,xmax,2001);bound=pd.DataFrame({'source':bx,'boundary':stability_threshold(bx,activation),'nullcline':kappa*branch_response(bx,activation),'activation':activation,'kappa':kappa});boundaries.append(bound)
        stable=points[(points.activation==activation)&(points.kappa==kappa)&points.stable.astype(bool)]
        entry=float(stable.fixed_point.max()) if len(stable) else .8*xmax
        for path_id,start in enumerate([entry,.35*xmax,.60*xmax,.95*xmax]):
            if start<=1:continue
            path=integrate_branch(start,kappa,activation);path['activation']=activation;path['kappa']=kappa;path['path_id']=path_id;path['entry']=start;path['selected']=path_id==0;path['entry_is_return_fixed_point']=bool(len(stable)) and path_id==0;paths.append(path)
    write_csv(pd.concat(fields,ignore_index=True),out/'reduced_phase_field.csv')
    write_csv(pd.concat(boundaries,ignore_index=True),out/'reduced_phase_boundary.csv')
    write_csv(pd.concat(paths,ignore_index=True),out/'reduced_phase_paths.csv')
    write_json({'source_points':a.source_points,'successor_points':a.successor_points,'model':'singular-limit source branch','time':'t/tau_z','unit_gains':True,'validity':'orthogonal memories; out-degree at most one; source>1; successor<h(source)','trajectory_end':'first stability loss, or source branch existence threshold','background':'Euclidean reduced saliency speed; not energy or a Lyapunov function','reference_selection':'stable return-map fixed point if present, otherwise 0.8*domain maximum'},out/'reduced_phase_metadata.json')
if __name__=='__main__':main()
