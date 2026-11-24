"""Small regression checks; never rerun the real category jobs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('run_experiments_module', Path(__file__).parents[1]/'scripts/run_experiments.py')
experiments = importlib.util.module_from_spec(spec); spec.loader.exec_module(experiments)


def row(parent, value=1., support=True, comparable=True):
    return {'parent_asin':parent,'support_eligible':support,'comparable':comparable,'n_current':40,
            **{m:value for m in experiments.METRICS}}


def setup_fake(tmp_path,monkeypatch,frames):
    ds=tmp_path/'processed';ds.mkdir();(ds/'audit.json').write_text(json.dumps({'accepted_rows':100,'artifact_sha256':{}}))
    c={'processed_dir':str(ds),'output_dir':str(tmp_path/'results'),
       'windows':{'reference_start':'2021-10-11','current_start':'2022-01-03','current_end':'2022-03-28'},
       'analysis':{'bootstrap_reps':300},'ingestion':{'memory_limit':'128MB','threads':1}}
    monkeypatch.setattr(experiments.data,'list_panel_buckets',lambda _:sorted(frames))
    monkeypatch.setattr(experiments.data,'load_panel_bucket',lambda directory,b:frames[b].copy())
    monkeypatch.setattr(experiments,'compare_windows',lambda f,*a,**k:f.copy())
    return c,ds


def test_empty_support_masks_preserve_schema_and_string_key_join():
    from review_shift.analysis import compare_windows,synthetic_panel
    p=synthetic_panel().iloc[:0]
    empty=compare_windows(p,'2021-10-11','2022-01-03','2022-03-28',bootstrap_reps=0)
    subset=experiments._supported(empty)
    assert list(subset.columns)==list(empty.columns)
    base=pd.DataFrame([row('P')])
    compared=experiments._scale_bucket(base,empty,.1)
    assert list(compared.parent_asin)==['P']
    assert not compared.present.any() and not compared.comparable.any()
    assert compared['abs_diff_within_mean'].isna().all()


def test_blocks_stream_headers_once_and_skip_empty_bucket(tmp_path,monkeypatch):
    empty=pd.DataFrame(columns=list(row('P')))
    c,ds=setup_fake(tmp_path,monkeypatch,{0:empty,1:pd.DataFrame([row('P1')]),2:pd.DataFrame([row('P2')])})
    def compare(f,*args,**kwargs):
        if not len(f):return f.copy()
        result=f.copy()
        if kwargs.get('bootstrap_reps'):
            for key in experiments.BLOCK_COLUMNS:
                if key not in result:result[key]=kwargs.get('block_weeks',0) if key=='block_weeks' else .1
        return result
    monkeypatch.setattr(experiments,'compare_windows',compare)
    result=experiments.run_experiments(c,'blocks')
    table=pd.read_csv(Path(c['output_dir'])/'block_sensitivity.csv')
    assert result['result_rows']==len(table)==8
    assert list(table.columns)==experiments.BLOCK_COLUMNS
    assert set(table.parent_asin)=={'P1','P2'} and set(table.block_weeks)=={1,2,4,8}
    assert not list(Path(c['output_dir']).glob('*.part'))


def test_all_empty_modes_write_schemas_and_zero_error_denominators(tmp_path,monkeypatch):
    empty=pd.DataFrame(columns=list(row('P')))
    c,ds=setup_fake(tmp_path,monkeypatch,{0:empty})
    def sample(origin,target,fraction,**kwargs):
        Path(target).mkdir();(Path(target)/'audit.json').write_text(json.dumps({'accepted_rows':0}))
    monkeypatch.setattr(experiments.data,'sample_clean',sample)
    experiments.run_experiments(c,'users')
    assert list(pd.read_csv(Path(c['output_dir'])/'user_sensitivity.csv').columns)==experiments.USER_COLUMNS
    experiments.run_experiments(c,'scale')
    detail=pd.read_csv(Path(c['output_dir'])/'scale_parent_comparison.csv')
    summary=pd.read_csv(Path(c['output_dir'])/'scale_summary.csv')
    assert not len(detail) and list(detail.columns)==experiments.SCALE_COLUMNS
    assert len(summary)==3 and not summary.base_supported_parents.any()
    assert not summary.within_mean_error_denominator.any()
    assert summary.mean_absolute_within_difference.isna().all()


def test_scale_reports_comparable_denominator_and_keeps_missing_parents(tmp_path,monkeypatch):
    base=pd.DataFrame([row('P1',1.),row('P2',2.)])
    c,ds=setup_fake(tmp_path,monkeypatch,{0:base})
    small=pd.DataFrame([row('P1',1.5)])
    medium=pd.DataFrame([row('P1',1.25),row('P2',np.nan,False,False)])
    def sample(origin,target,fraction,**kwargs):
        Path(target).mkdir();(Path(target)/'audit.json').write_text(json.dumps({'accepted_rows':int(fraction*100)}))
    def load(directory,bucket):
        if str(directory).endswith('_users_10pct'):return small.copy()
        if str(directory).endswith('_users_50pct'):return medium.copy()
        return base.copy()
    monkeypatch.setattr(experiments.data,'sample_clean',sample)
    monkeypatch.setattr(experiments.data,'load_panel_bucket',load)
    experiments.run_experiments(c,'scale')
    summary=pd.read_csv(Path(c['output_dir'])/'scale_summary.csv').set_index('fraction')
    detail=pd.read_csv(Path(c['output_dir'])/'scale_parent_comparison.csv')
    assert len(detail)==6
    assert summary.loc[.1,'base_supported_parents']==2 and summary.loc[.1,'present_parents']==1
    assert summary.loc[.5,'present_parents']==2 and summary.loc[.5,'comparable_parents']==1
    assert summary.loc[.1,'within_mean_error_denominator']==1 and summary.loc[.1,'mean_absolute_within_difference']==.5
    assert summary.loc[.5,'within_mean_error_denominator']==1 and summary.loc[.5,'mean_absolute_within_difference']==.25
    assert summary.loc[1.,'within_mean_error_denominator']==2 and summary.loc[1.,'mean_absolute_within_difference']==0
