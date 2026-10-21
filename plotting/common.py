"""Editorial figure system. Scientific quantities remain in the source tables."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from idp_hopfield.io import ensure_dir, write_csv
INK = '#16324F'
TEAL = '#258477'
CYAN = '#4BA3C3'
GOLD = '#B88637'
CORAL = '#CA6249'
INDIGO = '#5B5F97'
CHARCOAL = '#3A3A3A'
MUTED = '#697781'
PAPER = '#FAFAF8'
GRID = '#DCE1E1'
MISSING = '#E0E2DF'
COLORS = [INK, TEAL, CORAL, INDIGO, GOLD, CYAN]
DASHES = ['-', (0, (5, 1.8)), (0, (2.5, 1.5)), (0, (6, 1.5, 1, 1.5)), ':', (0, (8, 2))]
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

def _linear(rgb):
    rgb = np.asarray(rgb)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)

def _oklab(rgb):
    m = np.array([[.4122214708,.5363325363,.0514459929],[.2119034982,.6806995451,.1073969566],[.0883024619,.2817188376,.6299787005]])
    o = np.array([[.2104542553,.7936177850,-.0040720468],[1.9779984951,-2.4285922050,.4505937099],[.0259040371,.7827717662,-.8086757660]])
    return np.cbrt(_linear(rgb) @ m.T) @ o.T

def _srgb(lab):
    l,a,b = np.moveaxis(np.asarray(lab), -1, 0)
    lms = np.stack([l+.3963377774*a+.2158037573*b,l-.1055613458*a-.0638541728*b,l-.0894841775*a-1.291485548*b],axis=-1)**3
    rgb = lms @ np.array([[4.0767416621,-3.3077115913,.2309699292],[-1.2684380046,2.6097574011,-.3413193965],[-.0041960863,-.7034186147,1.7076147010]]).T
    return np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-.055)

def editorial_cmap():
    """Equal-distance Oklab sampling with increasing measured luminance.

    A dark coral peak would reverse grayscale order after ivory, so the high
    end uses a pale amber/coral tint to retain the requested warm finish.
    """
    anchors = ['#16324F','#286D78','#53A4A4','#B4CDC0','#EBD7A6','#FFF0E5']
    lab = _oklab(np.array([to_rgb(c) for c in anchors]))
    distance = np.r_[0,np.cumsum(np.linalg.norm(np.diff(lab,axis=0),axis=1))]
    samples = np.linspace(0,distance[-1],256)
    rgb = np.clip(_srgb(np.column_stack([np.interp(samples,distance,lab[:,i]) for i in range(3)])),0,1)
    cmap = ListedColormap(rgb,name='memory_tide')
    cmap.set_bad(MISSING)
    return cmap

CMAP = editorial_cmap()


def magnitude_cmap():
    """Slate-violet through rose to warm porcelain, distinct from probability.

    Equal Oklab-distance samples and monotone luminance provide ordered scalar
    encodings without relying on red/green discrimination.
    """
    anchors = ['#292941', '#51466D', '#886784', '#B98E9D', '#DEC4C1', '#FAF2E9']
    lab = _oklab(np.array([to_rgb(c) for c in anchors]))
    distance = np.r_[0,np.cumsum(np.linalg.norm(np.diff(lab,axis=0),axis=1))]
    samples = np.linspace(0,distance[-1],512)
    rgb = np.clip(_srgb(np.column_stack([np.interp(samples,distance,lab[:,i]) for i in range(3)])),0,1)
    cmap = ListedColormap(rgb,name='memory_magnitude')
    cmap.set_bad(MISSING)
    return cmap

PROBABILITY_CMAP = CMAP  # Preserve every original probability color exactly.
MAGNITUDE_CMAP = magnitude_cmap()

def configure():
    for font in (ROOT/'plotting/assets').glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    mpl.rcdefaults()
    for cmap in (PROBABILITY_CMAP,MAGNITUDE_CMAP):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)
    plt.rcParams.update({
        'font.family':'sans-serif', 'font.sans-serif':['Liberation Sans','Inter','Source Sans Pro','DejaVu Sans'],
        'mathtext.fontset':'stix','font.size':8,
        'text.color':CHARCOAL,'axes.labelcolor':CHARCOAL,
        'axes.labelsize':8.5,'axes.titlesize':9,'axes.titleweight':'bold',
        'axes.titlelocation':'left','axes.titlepad':9,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#8D989E','axes.linewidth':.55,
        'axes.prop_cycle':mpl.cycler(color=COLORS),
        'axes.facecolor':PAPER,'figure.facecolor':PAPER,'savefig.facecolor':PAPER,
        'axes.axisbelow':True,'axes.xmargin':.04,'axes.ymargin':.10,
        'lines.linewidth':1.45,'lines.markersize':3.8,'lines.markeredgewidth':.75,
        'lines.solid_capstyle':'round','lines.antialiased':True,'patch.antialiased':True,
        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'xtick.color':MUTED,'ytick.color':MUTED,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.55,'ytick.major.width':.55,
        'xtick.minor.visible':False,'ytick.minor.visible':False,
        'grid.color':GRID,'grid.linewidth':.45,'grid.alpha':.7,
        'legend.fontsize':7,'legend.frameon':False,'legend.handlelength':2.4,
        'legend.labelspacing':.45,'legend.columnspacing':1.25,'legend.borderaxespad':.4,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','path.simplify':False,'figure.dpi':120,
        'figure.constrained_layout.use':True,'figure.constrained_layout.h_pad':.08,
        'figure.constrained_layout.w_pad':.07,'image.cmap':CMAP.name,
    })

def style_axis(ax,title=None,label=None,probability=False,grid='y'):
    if title: ax.set_title(title)
    if label:
        ax.text(-.14,1.075,label,transform=ax.transAxes,fontweight='bold',fontsize=10,color=INK,va='bottom',clip_on=False)
    ax.grid(False)
    if grid: ax.grid(True,axis=grid)
    if ax.get_xscale()=='linear': ax.xaxis.set_major_locator(MaxNLocator(5,min_n_ticks=3))
    if ax.get_yscale()=='linear': ax.yaxis.set_major_locator(MaxNLocator(4,min_n_ticks=3))
    if probability:
        ax.set_ylim(-.04,1.04);ax.set_yticks([0,.5,1])

def series_style(index,markers=False):
    i = int(index)%len(COLORS)
    dash = (int(index)+int(index)//len(COLORS))%len(DASHES)
    result = {'color':COLORS[i],'linestyle':DASHES[dash]}
    if markers: result.update(marker=MARKERS[dash],markerfacecolor=PAPER,markeredgewidth=.85)
    return result

def memory_lines(ax,panel,time,value,max_emphasis=5,emphasis=None):
    """Show every memory; stable IDs determine color, dash, and legend order."""
    scores = panel.groupby('memory')[value].max().sort_values(ascending=False)
    prominent = set(scores.head(max_emphasis).index if emphasis is None else emphasis)
    for memory,curve in panel.groupby('memory',sort=True):
        curve=curve.sort_values(time);active=memory in prominent
        ax.plot(curve[time],curve[value],**(series_style(memory) if active else {'color':MUTED,'linestyle':'-'}),
                alpha=1 if active else .3,lw=1.35 if active else .55,
                label=rf'$\mu={int(memory)}$' if active else None,zorder=3 if active else 1)
    if len(scores)>len(prominent): ax.plot([],[],color=MUTED,lw=.7,alpha=.6,label='other memories')

def interval(ax,x,lo,hi,color=INK):
    ax.fill_between(np.asarray(x,float),np.asarray(lo,float),np.asarray(hi,float),color=color,alpha=.11,linewidth=0,zorder=1)

def cell_edges(values,singleton_width=1.):
    values=np.asarray(values,dtype=float)
    if len(values)==0 or not np.all(np.isfinite(values)) or np.any(np.diff(values)<=0):
        raise ValueError('Cell centers must be finite, strictly increasing, and nonempty')
    if len(values)==1: return np.array([values[0]-singleton_width/2,values[0]+singleton_width/2])
    mid=(values[:-1]+values[1:])/2
    return np.r_[values[0]-(mid[0]-values[0]),mid,values[-1]+(values[-1]-mid[-1])]

def note(ax,text):
    ax.text(.5,.5,text,transform=ax.transAxes,ha='center',va='center',color=MUTED,fontsize=8,linespacing=1.5)

def save(fig,output,context=True):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if context and os.environ.get('IDP_FIGURE_CONTEXT'):
        fig.supxlabel(os.environ['IDP_FIGURE_CONTEXT'],fontsize=6.5,color=MUTED)
    if os.environ.get('IDP_SMOKE')=='1':
        fig.savefig(output.with_suffix('.png'),dpi=110,bbox_inches='tight',pad_inches=.06)
    else:
        for extension in ('pdf','svg','png'):
            fig.savefig(output.with_suffix('.'+extension),dpi=300,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def export(frame,data_dir,name):
    return write_csv(frame,ensure_dir(data_dir)/name)
