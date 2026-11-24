from pathlib import Path
import json
import pandas as pd
import pytest
from review_shift.data import process_sources
from review_shift import __main__ as cli

@pytest.fixture
def configured(tmp_path,monkeypatch):
 raw=tmp_path/'fixture.jsonl'
 rows=[]
 for t,rating in [(1636934400000,4),(1643587200000,3),(1649116800000,2)]:
  rows.append({'parent_asin':'P','asin':'A','user_id':str(t),'timestamp':t,'rating':rating,'text':'fixture'})
 raw.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
 process_sources([{'path':str(raw),'category':'fixture'}],tmp_path/'processed',buckets=1)
 c={'label':'TEST_NOT_REAL_CORPUS','categories':['fixture'],'processed_dir':str(tmp_path/'processed'),'output_dir':str(tmp_path/'out'),
    'windows':{'reference_start':'2021-10-11','current_start':'2022-01-03','current_end':'2022-03-28','future_end':'2022-04-25'},
    'analysis':{'bootstrap_reps':0,'min_n':1,'min_weeks':1,'min_common_coverage':.5}}
 def render(s,examples,path):
  p=Path(path);(p/'summary.json').write_text(json.dumps(s,allow_nan=False));(p/'report.html').write_text('report');(p/'diagnostics.png').write_bytes(b'test fixture')
 monkeypatch.setattr(cli,'write_report',render)
 return c

def test_source_configuration_mismatch_is_rejected(configured):
 configured['categories']=['not_the_processed_source']
 with pytest.raises(ValueError,match='Configured categories'):cli.analyze(configured)

def test_bucket_output_is_separate_and_marked_partial(configured):
 r=cli.analyze(configured,bucket=0)
 assert r['analysis_covers_all_ingested_buckets'] is False
 assert (Path(configured['output_dir'])/'bucket-000/COMPLETE.json').exists()
 assert not (Path(configured['output_dir'])/'comparison.csv').exists()

def test_failure_does_not_replace_previous_completed_report(configured,monkeypatch):
 cli.analyze(configured)
 p=Path(configured['output_dir']);old={n:(p/n).read_bytes() for n in ['comparison.csv','report.html','COMPLETE.json']}
 def fail(*args,**kwargs):raise RuntimeError('intentional computation failure')
 monkeypatch.setattr(cli,'compare_windows',fail)
 with pytest.raises(RuntimeError,match='intentional'):cli.analyze(configured)
 assert all((p/n).read_bytes()==b for n,b in old.items())

def test_empty_window_publishes_empty_typed_tables(configured):
 configured['windows']={'reference_start':'2020-01-06','current_start':'2020-03-30','current_end':'2020-06-22','future_end':'2020-07-20'}
 r=cli.analyze(configured)
 assert r['candidate_parents']==0 and r['forecast_metrics']==[]
 p=Path(configured['output_dir'])
 assert 'parent_asin' in pd.read_csv(p/'comparison.csv').columns
 assert 'pred_1' in pd.read_csv(p/'forecast.csv').columns
 assert r['full_published_reviews_processed'] is False
