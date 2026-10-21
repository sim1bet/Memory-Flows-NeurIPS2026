"""Scientific encoding invariants, rather than pixel-perfect snapshots."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from plotting.common import CMAP,_linear,_oklab,cell_edges,series_style
from experiments.export_reduced_phase_plane import integrate_branch
from idp_hopfield.theory import theoretical_escape_time
from idp_hopfield.activations import stability_threshold

def test_heatmap_edges_preserve_nonuniform_sampling():
    np.testing.assert_allclose(cell_edges([0.,.2,1.]),[-.1,.1,.6,1.4])
    np.testing.assert_allclose(cell_edges([5.],2.),[4.,6.])

def test_palette_is_ordered_in_grayscale():
    rgb=CMAP(np.linspace(0,1,256))[:,:3]
    luminance=_linear(rgb)@np.array([.2126,.7152,.0722])
    assert np.all(np.diff(luminance)>0)
    distance=np.linalg.norm(np.diff(_oklab(rgb),axis=0),axis=1)
    assert distance.std()/distance.mean()<.03

def test_reduced_portrait_matches_existing_escape_theory():
    for activation in ['tanh','smoothf']:
        for entry in [4.,8.,12.]:
            path=integrate_branch(entry,15.,activation)
            time,successor,_=theoretical_escape_time(entry,15.,activation)
            assert np.isfinite(time) and path.escaped.iloc[-1]
            np.testing.assert_allclose(path.time_slow.iloc[-1],time,atol=3e-6,rtol=0)
            np.testing.assert_allclose(path.successor.iloc[-1],successor,atol=3e-6,rtol=0)
            margin=path.successor.to_numpy()-stability_threshold(path.source.to_numpy(),activation)
            assert margin.max()<1e-6
            assert np.all(np.diff(path.source)<0)

def test_identity_styles_do_not_depend_on_display_order():
    ids=[4,0,3,1,2]
    before={i:series_style(i) for i in ids}
    after={i:series_style(i) for i in sorted(ids)}
    assert before==after
    assert len({repr(before[i]['linestyle']) for i in ids})==len(ids)
