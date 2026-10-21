"""Regression tests for denser maps, revised protocols, and timing budgets."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'plotting'))
from idp_hopfield.config import load_config
from experiments import run_correlation_phase_plane as cp
from experiments.run_model_comparison import simulate_case
from plotting.common import PROBABILITY_CMAP,MAGNITUDE_CMAP,editorial_cmap,_linear,_oklab

def test_paper_ranges_and_dense_sampling():
    d=load_config(ROOT/'configs/paper/capacity.yml')
    assert d['fixed_cycle']['alpha_values']==list(range(100,141,5))
    assert (d['scaled_cycle']['alpha_min'],d['scaled_cycle']['alpha_max'])==(.15,.5)
    assert d['fanout_phase']['p_count']==41
    assert d['fanout_phase']['outdegrees']==list(range(2,19))
    d=load_config(ROOT/'configs/paper/correlation_phase_plane.yml')
    assert len(d['epsilons'])==41 and d['kappa_points']==41

def test_probability_palette_unchanged_and_magnitudes_ordered():
    x=np.linspace(0,1,256)
    np.testing.assert_array_equal(PROBABILITY_CMAP(x),editorial_cmap()(x))
    assert not np.allclose(PROBABILITY_CMAP(x),MAGNITUDE_CMAP(x))
    rgb=MAGNITUDE_CMAP(np.linspace(0,1,512))[:,:3]
    assert np.all(np.diff(_linear(rgb)@np.array([.2126,.7152,.0722]))>0)
    distance=np.linalg.norm(np.diff(_oklab(rgb),axis=0),axis=1)
    assert distance.std()/distance.mean()<.03

def test_horizon_accounts_for_escape_switch_and_holding():
    raw=load_config(ROOT/'configs/smoke/correlation_phase_plane.yml')
    expected=3*7.+1.+raw['hold_time']
    assert cp.escape_horizon(raw,[np.nan,4.],[7.])==expected

def test_timed_out_trial_replays_same_seed_with_observed_time():
    raw=load_config(ROOT/'configs/smoke/correlation_phase_plane.yml');cfg=cp.resolved_base(raw)
    calls=[]
    def simulator(patterns,successor,local,seed):
        calls.append((local.max_slow_time_per_transition,seed))
        result={'reference_outcome':'timeout' if len(calls)==1 else 'success',
                'reference_completed_transitions':1,'pending_T_esc':5.}
        return result,[{'T_esc':2.,'T_switch':3.}]
    with patch.object(cp,'simulate_asymptotic_events',side_effect=simulator):
        result,events,history=cp.simulate_with_escape_horizon(None,None,cfg,321,raw,1.)
    assert len(calls)==2 and calls[0][1]==calls[1][1]==321
    assert calls[1][0]>=cp.escape_horizon(raw,[5.],[3.])
    assert result['timing_right_censored']==0 and result['horizon_retries']==1
    assert result['total_time_budget']==calls[-1][0]*cfg.asymptotic_cycles*len(cfg.cycle)

def test_wrong_memory_is_not_retried_and_persistent_timeout_is_censored():
    raw=load_config(ROOT/'configs/smoke/correlation_phase_plane.yml');cfg=cp.resolved_base(raw)
    for outcome in ['wrong_memory','timeout']:
        fake={'reference_outcome':outcome,'reference_completed_transitions':0,'pending_T_esc':np.nan}
        with patch.object(cp,'simulate_asymptotic_events',return_value=(fake,[])) as mock:
            result,_,history=cp.simulate_with_escape_horizon(None,None,cfg,77,raw,1.)
        assert result['reference_outcome']==outcome
        assert result['timing_right_censored']==int(outcome=='timeout')
        assert mock.call_count==(1 if outcome=='wrong_memory' else raw['horizon_max_retries']+1)

def test_noise_is_targeted_seeded_and_leaves_other_cases_unchanged():
    common=load_config(ROOT/'configs/smoke/model_comparison.yml');common['time_horizon']=.2
    assert [c.get('fast_noise',0.) for c in common['cases']]==[0.,1e-4,0.]
    case=common['cases'][1]
    noisy=simulate_case(case,common);repeat=simulate_case(case,common)
    pd.testing.assert_frame_equal(noisy,repeat)
    quiet=simulate_case({**case,'fast_noise':0.},common)
    assert not np.allclose(noisy.cosine_overlap,quiet.cosine_overlap,rtol=0,atol=1e-12)
    for case in [common['cases'][0],common['cases'][2]]:
        pd.testing.assert_frame_equal(simulate_case(case,common),simulate_case({**case,'fast_noise':0.},common))

def test_fanout_all_visible_panels_show_x_tick_numbers_and_labels():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plot_capacity
    frame=pd.DataFrame([{'activation':'tanh','outdegree':d,'N':32,'alpha_actual':a,
        'success_probability':.5,'ci_low':.3,'ci_high':.7} for d in [2,6,10,14,18] for a in [.1,.2]])
    figures=[]
    with patch.object(plot_capacity,'save',side_effect=lambda fig,out:figures.append(fig)):
        plot_capacity.configure();plot_capacity.plot_fanout(frame,'tanh','unused')
    for ax in figures[0].axes[:5]:
        assert ax.get_xlabel() and any(label.get_visible() for label in ax.get_xticklabels())
        assert all(line.get_marker()!='None' for line in ax.lines)
    plt.close(figures[0])
