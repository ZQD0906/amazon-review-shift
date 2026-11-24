import numpy as np
import pandas as pd
import pytest

from review_shift.analysis import (
    COUNT_COLUMNS, bootstrap_stability, compare_windows, decompose_counts,
    evaluate_forecasts, score_distribution, summarize_forecasts, synthetic_panel,
    validate_panel,
)

WINDOW=("2021-10-11","2022-01-03","2022-03-28")
FUTURE="2022-04-25"


def test_mixture_paradox_example_is_exact_and_centered():
    # A 4.6->4.7 and B 3.6->3.7; mixture weights 80/20->20/80.
    a=np.array([[0,0,0,320,480],[0,0,80,120,0]])
    b=np.array([[0,0,0,60,140],[0,0,240,560,0]])
    d=decompose_counts(a,b)
    stars=np.arange(1,6)
    assert np.isclose(d['p_ref']@stars,4.4)
    assert np.isclose(d['p_current']@stars,3.9)
    assert np.isclose(d['within']@stars,.1)
    assert np.isclose(d['composition']@stars,-.6)
    assert np.allclose(d['turnover'],0)
    assert d['identity_error']<1e-12
    assert d['component_mass_error']<1e-12


def test_entry_exit_and_global_weights_reconstruct_exactly():
    a=np.array([[4,2,1,1,2],[0,0,0,0,50],[0,0,0,0,0]])
    b=np.array([[3,3,1,1,2],[0,0,0,0,0],[40,0,0,0,0]])
    d=decompose_counts(a,b)
    assert d['common_coverage_ref']==pytest.approx(1/6)
    assert d['common_coverage_current']==pytest.approx(1/5)
    assert np.allclose(d['overall'],d['within']+d['composition']+d['turnover'])
    assert all(abs(d[k].sum())<1e-12 for k in ('within','composition','turnover'))
    assert np.array_equal(d['new_mask'],[False,False,True])
    assert np.array_equal(d['gone_mask'],[False,True,False])


def test_no_shared_children_is_turnover_not_a_zero_imputed_comparison():
    d=decompose_counts(np.array([[0,0,0,0,10],[0,0,0,0,0]]),
                       np.array([[0,0,0,0,0],[10,0,0,0,0]]))
    assert np.allclose(d['within'],0)
    assert np.allclose(d['composition'],0)
    assert np.allclose(d['turnover'],[-1,0,0,0,1]*np.array([-1]*5))
    assert d['common_coverage_ref']==0


def test_panel_rejects_duplicate_bad_count_and_non_monday():
    p=synthetic_panel().iloc[:5].copy()
    with pytest.raises(ValueError,match='Duplicate'):validate_panel(pd.concat([p,p.iloc[:1]]))
    bad=p.copy();bad.loc[0,'n']+=1
    with pytest.raises(ValueError,match='must equal'):validate_panel(bad)
    bad=p.copy();bad.loc[0,'n1']=-1
    with pytest.raises(ValueError,match='nonnegative'):validate_panel(bad)
    bad=p.copy();bad.loc[0,'week']+=pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='Monday'):validate_panel(bad)


def test_one_period_only_parents_are_preserved_not_dropped():
    p=synthetic_panel()
    extra=p[p.parent_asin=='SIM_PARENT_00'].copy()
    left=extra[extra.week<pd.Timestamp(WINDOW[1])].assign(parent_asin='REF_ONLY')
    right=extra[(extra.week>=pd.Timestamp(WINDOW[1]))&(extra.week<pd.Timestamp(WINDOW[2]))].assign(parent_asin='CURRENT_ONLY')
    result=compare_windows(pd.concat([p,left,right]),*WINDOW,bootstrap_reps=0).set_index('parent_asin')
    assert result.loc['REF_ONLY','status']=='no_current_observations'
    assert result.loc['CURRENT_ONLY','status']=='no_reference_observations'
    assert not result.loc['CURRENT_ONLY','support_eligible']


def test_future_labels_cannot_change_comparison_or_predictions():
    p=synthetic_panel()
    result=compare_windows(p,*WINDOW,bootstrap_reps=60,block_weeks=2)
    future=p.week>=pd.Timestamp(WINDOW[2]);changed=p.copy()
    changed.loc[future,COUNT_COLUMNS]=0
    changed.loc[future,'n1']=changed.loc[future,'n']
    other=compare_windows(changed,*WINDOW,bootstrap_reps=60,block_weeks=2)
    pd.testing.assert_frame_equal(result,other)
    f=evaluate_forecasts(p,result,*WINDOW,FUTURE)
    f2=evaluate_forecasts(changed,other,*WINDOW,FUTURE)
    pd.testing.assert_frame_equal(f[['parent_asin','method','pred_1','pred_2','pred_3','pred_4','pred_5']],
                                  f2[['parent_asin','method','pred_1','pred_2','pred_3','pred_4','pred_5']])
    assert not np.allclose(f.log_loss.fillna(0),f2.log_loss.fillna(0))


def test_bootstrap_reproducible_when_an_unrelated_parent_is_added():
    p=synthetic_panel();one=p[p.parent_asin=='SIM_PARENT_00']
    a=compare_windows(one,*WINDOW,bootstrap_reps=60)
    b=compare_windows(p,*WINDOW,bootstrap_reps=60)
    pd.testing.assert_frame_equal(a.reset_index(drop=True),b[b.parent_asin=='SIM_PARENT_00'].reset_index(drop=True))


def test_bootstrap_lost_shared_child_is_invalid_not_newly_observed():
    a=np.zeros((12,2,5));b=a.copy()
    a[:,0,4]=10;b[:,0,4]=10
    a[0,1,0]=1;b[-1,1,0]=1
    x=bootstrap_stability(a,b,reps=200,block_weeks=4,seed=1)
    assert x['bootstrap_invalid']>0
    assert x['bootstrap_valid_fraction']<.8
    assert not x['stability_available']
    assert np.isnan(x['within_mean_lo'])


def test_empty_calendar_weeks_do_not_create_observed_support():
    p=synthetic_panel();p=p[p.week.isin(pd.to_datetime(['2021-10-11','2022-01-03']))]
    x=compare_windows(p,*WINDOW,bootstrap_reps=0)
    assert (x.weeks_ref==1).all() and (x.weeks_current==1).all()
    assert (x.calendar_weeks_ref==12).all()
    assert not x.support_eligible.any()


def test_missing_future_is_unknown_and_denominator_excludes_it():
    p=synthetic_panel();cmp=compare_windows(p,*WINDOW,bootstrap_reps=0)
    ev=evaluate_forecasts(p,cmp,*WINDOW,FUTURE)
    missing=ev[ev.parent_asin=='SIM_PARENT_17']
    assert (missing.status=='unknown_future').all()
    assert (missing.n_future==0).all() and missing.log_loss.isna().all()
    summary=summarize_forecasts(ev)
    assert (summary.unknown_future_parents==1).all()
    assert (summary.observable_parents==17).all()


def test_rps_is_mean_per_review_score_not_distribution_distance():
    p=np.ones(5)/5;counts=np.array([1,1,1,1,1])
    scores=score_distribution(p,counts)
    fp=np.cumsum(p)[:4]
    expected=np.mean([np.mean((fp-(np.arange(1,5)>=rating))**2) for rating in range(1,6)])
    assert scores['rps']==pytest.approx(expected)
    assert scores['rps']==pytest.approx(.2)
    assert scores['log_loss']==pytest.approx(np.log(5))
    assert scores['rps']>0 # fitted distribution equals observed, but outcome uncertainty remains.


def test_forecasts_use_same_supported_cohort_and_are_probabilities():
    p=synthetic_panel();cmp=compare_windows(p,*WINDOW,bootstrap_reps=0,min_n=10**9)
    ev=evaluate_forecasts(p,cmp,*WINDOW,FUTURE)
    assert not ev.eligible.any() and not ev.observable.any()
    cmp=compare_windows(p,*WINDOW,bootstrap_reps=0)
    ev=evaluate_forecasts(p,cmp,*WINDOW,FUTURE)
    for _,g in ev.groupby('parent_asin'):
        assert g.eligible.nunique()==1 and g.observable.nunique()==1
    assert np.allclose(ev[[f'pred_{k}' for k in range(1,6)]].sum(axis=1),1)


def test_invalid_window_and_forecast_parameters_raise():
    p=synthetic_panel()
    with pytest.raises(ValueError,match='Monday'):compare_windows(p,'2021-10-12',*WINDOW[1:])
    with pytest.raises(ValueError,match='block_weeks'):compare_windows(p,*WINDOW,block_weeks=13)
    with pytest.raises(ValueError,match='Both comparison'):decompose_counts(np.zeros((1,5)),np.ones((1,5)))
    with pytest.raises(ValueError,match='probability'):decompose_counts(np.ones((1,5)),np.ones((1,5)),q=np.ones(5))
    with pytest.raises(ValueError,match='Scoring'):score_distribution(np.zeros(5),np.ones(5))


def test_empty_bucket_has_stable_empty_outputs():
    p=synthetic_panel().iloc[:0]
    cmp=compare_windows(p,*WINDOW,bootstrap_reps=0)
    assert cmp.empty and 'parent_asin' in cmp
    ev=evaluate_forecasts(p,cmp,*WINDOW,FUTURE)
    assert ev.empty and 'log_loss' in ev


def test_user_clusters_are_joint_across_periods_and_guarded():
    from review_shift.analysis import user_cluster_stability
    rows=[]
    for uid in range(60):
        for week in ('2021-10-11','2022-01-03'):
            for child in ['a','b']:
                rows.append({'parent_asin':'p','asin':child,'user_hash':f'u{uid}',
                             'week':week,'rating':1 if uid%3==0 else 5})
    clean=pd.DataFrame(rows)
    result=user_cluster_stability(clean,*WINDOW,reps=80,seed=8)
    assert result['stability_available']
    assert result['overall_mean_lo']==pytest.approx(0)
    assert result['overall_mean_hi']==pytest.approx(0)
    assert result['bootstrap_valid']==80
    blocked=user_cluster_stability(clean,*WINDOW,reps=80,max_users=20)
    assert blocked['stability_reason']=='resource_guard_case_not_resampled'
    assert not blocked['stability_available']


def test_all_bucket_outputs_have_identical_schemas_even_without_support():
    from review_shift.analysis import COMPARISON_COLUMNS,FORECAST_COLUMNS
    panel=synthetic_panel()
    cases=[panel,panel.iloc[:0],panel[panel.week<pd.Timestamp(WINDOW[1])],panel]
    comparison_outputs=[];forecast_outputs=[]
    for idx,case in enumerate(cases):
        comparison=compare_windows(case,*WINDOW,bootstrap_reps=0,min_n=10**9 if idx==3 else 30)
        forecast=evaluate_forecasts(case,comparison,*WINDOW,FUTURE)
        comparison_outputs.append(comparison);forecast_outputs.append(forecast)
    assert all(list(f.columns)==COMPARISON_COLUMNS for f in comparison_outputs)
    assert all(list(f.columns)==FORECAST_COLUMNS for f in forecast_outputs)
    assert comparison_outputs[2].identity_error.isna().all()
    assert forecast_outputs[3].pred_1.isna().all()


def test_empty_boolean_filters_preserve_columns_and_numeric_types():
    empty=synthetic_panel().iloc[:0]
    comp=compare_windows(empty,*WINDOW,bootstrap_reps=0)
    for name in ('comparable','support_eligible','stability_available'):
        assert comp[name].dtype==bool
        assert list(comp[comp[name]].columns)==list(comp.columns)
    assert isinstance(comp.parent_asin.dtype,pd.StringDtype)
    for name in ('n_ref','n_current','weeks_ref','weeks_current'):
        assert pd.api.types.is_numeric_dtype(comp[name].dtype)
    evaluation=evaluate_forecasts(empty,comp,*WINDOW,FUTURE)
    for name in ('eligible','observable'):
        assert evaluation[name].dtype==bool
        assert list(evaluation[evaluation[name]].columns)==list(evaluation.columns)
    assert isinstance(evaluation.parent_asin.dtype,pd.StringDtype)
    assert pd.api.types.is_numeric_dtype(evaluation.n_future.dtype)
