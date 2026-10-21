#!/usr/bin/env python3
"""Reproducible grayscale and approximate color-vision display diagnostics.

The fixed severity-100 linear RGB transforms below are the commonly used
Machado, Oliveira & Fernandes (2009) endpoint matrices. Diagnostics are not
an accessibility certification. Identity also uses dashes/markers/hatching.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from plotting.common import CMAP,MAGNITUDE_CMAP,_linear,_oklab
MATRICES={
 'Protan approximation':np.array([[.152286,1.052583,-.204868],[.114503,.786281,.099216],[-.003882,-.048116,1.051998]]),
 'Deutan approximation':np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]]),
 'Tritan approximation':np.array([[1.255528,-.076749,-.178779],[-.078411,.930809,.147602],[.004733,.691367,.303900]])}

def encode(rgb):return np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-.055)
def transform(image,kind):
    linear=_linear(np.asarray(image.convert('RGB'))/255.)
    if kind=='Grayscale':linear=np.repeat((linear@np.array([.2126,.7152,.0722]))[...,None],3,axis=-1)
    elif kind in MATRICES:linear=linear@MATRICES[kind].T
    return Image.fromarray(np.uint8(np.clip(encode(linear),0,1)*255+.5))

def main():
    p=argparse.ArgumentParser();p.add_argument('--figures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rgb=CMAP(np.linspace(0,1,256))[:,:3];y=_linear(rgb)@np.array([.2126,.7152,.0722]);d=np.linalg.norm(np.diff(_oklab(rgb),axis=0),axis=1)
    stats={'luminance_range':[float(y.min()),float(y.max())],'minimum_luminance_increment':float(np.diff(y).min()),'oklab_step_coefficient_of_variation':float(d.std()/d.mean()),'cvd_diagnostics':{}}
    for name,m in MATRICES.items():
        linear=np.clip(_linear(rgb)@m.T,0,1);l=linear@np.array([.2126,.7152,.0722]);stats['cvd_diagnostics'][name]={'minimum_luminance_increment':float(np.diff(l).min())}
    rgb_m=MAGNITUDE_CMAP(np.linspace(0,1,MAGNITUDE_CMAP.N))[:,:3]
    lm=_linear(rgb_m);ym=lm@np.array([.2126,.7152,.0722]);dm=np.linalg.norm(np.diff(_oklab(rgb_m),axis=0),axis=1)
    stats['magnitude_palette']={'name':MAGNITUDE_CMAP.name,'luminance_range':[float(ym.min()),float(ym.max())],
        'minimum_luminance_increment':float(np.diff(ym).min()),'oklab_step_coefficient_of_variation':float(dm.std()/dm.mean()),
        'cvd_minimum_luminance_increment':{name:float(np.diff(np.clip(lm@m.T,0,1)@np.array([.2126,.7152,.0722])).min()) for name,m in MATRICES.items()}}
    (a.output/'palette_metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    names=['reduced_phase_plane_kappa_15','tanh_saliency_trajectory','return_map']
    kinds=['Original','Grayscale',*MATRICES]
    sheet=Image.new('RGB',(600*len(kinds),300*len(names)),'#E6E9E6');draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(names):
        source=Image.open(a.figures/(name+'.png')).convert('RGB');source.thumbnail((590,267))
        for j,kind in enumerate(kinds):
            im=source if kind=='Original' else transform(source,kind)
            x=j*600+(600-im.width)//2;y0=i*300+25
            sheet.paste(im,(x,y0));draw.text((j*600+12,i*300+8),kind+' / '+name,fill='#16324F')
    sheet.save(a.output/'accessibility_review.png')
    paths=sorted(a.figures.glob('*.png'))
    for page in range((len(paths)+5)//6):
        contact=Image.new('RGB',(1400,1200),'#E6E9E6');draw=ImageDraw.Draw(contact)
        for i,path in enumerate(paths[page*6:(page+1)*6]):
            im=Image.open(path).convert('RGB');im.thumbnail((685,362));x=(i%2)*700;y0=(i//2)*400
            contact.paste(im,(x+(700-im.width)//2,y0+28));draw.text((x+10,y0+8),path.stem,fill='#16324F')
        contact.save(a.output/f'contact_sheet_{page+1:02d}.png')
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
