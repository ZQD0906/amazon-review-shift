from pathlib import Path
import json, hashlib
import pandas as pd
import pytest
from review_shift.data import (REVISION,process_sources,load_panel_bucket,list_panel_buckets,sample_clean,download_entry,_normalise)

def review(user='u1',asin='A',parent='P',rating=5,timestamp=1609459200000,**extra):
 return dict(user_id=user,asin=asin,parent_asin=parent,rating=rating,timestamp=timestamp,text='a review',verified_purchase=True,**extra)
def source(tmp,name,rows):
 p=tmp/(name+'.jsonl');p.write_text(''.join((r if isinstance(r,str) else json.dumps(r))+'\n' for r in rows));return dict(path=str(p),category=name)
def panels(path):
 frames=[load_panel_bucket(path,b) for b in list_panel_buckets(path)]
 return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def test_global_duplicate_conflict_filter_ledger_and_panel_conservation(tmp_path):
 a=source(tmp_path,'first',[review(),review('u2',rating=1),review('u3',rating=2),review('u3',rating=4),'{bad',review(user=''),review('u5',rating=6),review('u6',timestamp=123)])
 b=source(tmp_path,'second',[review(),review('u7',rating=3),'\n'])
 out=tmp_path/'out';audit=process_sources([a,b],out,buckets=4,batch_size=2,memory_limit='128MB',threads=1)
 assert audit['scanned_rows']==10 and audit['blank_lines']==2
 assert audit['filter_ledger']==dict(invalid_json=1,missing_id=1,invalid_rating=1,invalid_timestamp=1,exact_duplicate=1,identity_conflict=2,sampling_excluded=0,accepted=3)
 assert audit['identity_conflict_groups']==1 and not audit['full_published_reviews_processed']
 p=panels(out);assert len(p)==1 and p.n.sum()==3
 assert list(p.iloc[0][['n1','n2','n3','n4','n5']])==[1,0,1,0,1]
 assert str(p.iloc[0].week)[:10]=='2020-12-28'

def test_staging_and_output_caches_verify_hashes(tmp_path):
 s=source(tmp_path,'cat',[review()]);out=tmp_path/'out'
 first=process_sources([s],out,buckets=2,threads=1)
 assert process_sources([s],out,buckets=2,threads=1)==first
 p=next((out/'panel').rglob('*.parquet'));p.write_bytes(p.read_bytes()+b'altered')
 with pytest.raises(ValueError,match='SHA256'):load_panel_bucket(out,first['panel_buckets'][0])
 with pytest.raises(ValueError,match='SHA256'):process_sources([s],out,buckets=2,threads=1)

def test_changed_input_and_changed_parameters_reject_cache(tmp_path):
 s=source(tmp_path,'cat',[review()]);out=tmp_path/'out';process_sources([s],out,buckets=2,threads=1)
 with pytest.raises(ValueError,match='different sources or parameters'):process_sources([s],out,buckets=3,threads=1)
 Path(s['path']).write_text(json.dumps(review('another'))+'\n')
 with pytest.raises(ValueError,match='Staging cache'):process_sources([s],out,buckets=2,threads=1)

def test_empty_accepted_set_keeps_rejection_ledger(tmp_path):
 s=source(tmp_path,'empty',['bad json',review(rating=0)])
 a=process_sources([s],tmp_path/'out',buckets=2,threads=1)
 assert a['accepted_rows']==0 and a['panel_rows']==0 and a['scanned_rows']==2
 assert not list_panel_buckets(tmp_path/'out') and load_panel_bucket(tmp_path/'out',0).empty

def test_source_size_checked_before_scan_claim(tmp_path):
 s=source(tmp_path,'All_Beauty',[review()]);s.update(expected_bytes=1,revision=REVISION)
 with pytest.raises(ValueError,match='byte count'):process_sources([s],tmp_path/'out')

def test_multiple_parent_mapping_warns_without_silent_merge(tmp_path):
 s=source(tmp_path,'cat',[review('u1','A','P'),review('u2','A','Q'),review('u3','B','P')])
 a=process_sources([s],tmp_path/'out',buckets=3,threads=1)
 assert a['mapping_warning_child_ids_with_multiple_parents']==1 and a['accepted_rows']==3
 assert len(panels(tmp_path/'out'))==3

def test_reviewer_sampling_nested_reaggregates_clean_rows(tmp_path):
 rows=[review(f'u{i}',asin=f'a{i%3}',timestamp=1609459200000+j*86400000) for i in range(100) for j in range(2)]
 s=source(tmp_path,'cat',rows);full=tmp_path/'full';a=process_sources([s],full,buckets=4,threads=1)
 small=sample_clean(full,tmp_path/'small',.25,123,threads=1);large=sample_clean(full,tmp_path/'large',.5,123,threads=1)
 def users(d):
  import pyarrow.parquet as pq
  return {x for p in (d/'clean').rglob('*.parquet') for x in pq.ParquetFile(p).read(columns=['user_hash']).to_pandas().user_hash}
 us=users(tmp_path/'small');ul=users(tmp_path/'large')
 assert us<ul and small['accepted_rows']==2*len(us) and large['accepted_rows']==2*len(ul)
 assert small['scanned_rows']==a['accepted_rows']==200 and panels(tmp_path/'small').n.sum()==small['accepted_rows']
 assert not small['full_published_reviews_processed'] and 'not_raw_jsonl_rescan' in small['coverage']
 with pytest.raises(ValueError,match='unsampled'):sample_clean(tmp_path/'small',tmp_path/'bad',.5)

def test_timestamp_units_and_bool_rating_boundaries():
 assert _normalise(json.dumps(review(timestamp=1609459200)).encode(),'x',1)[1]=='invalid_timestamp'
 assert _normalise(json.dumps(review(rating=True)).encode(),'x',1)[1]=='invalid_rating'
 assert _normalise(json.dumps(review(timestamp=1696118400000)).encode(),'x',1)[1]=='invalid_timestamp'
 r=review();r['sort_timestamp']=r.pop('timestamp');assert _normalise(json.dumps(r).encode(),'x',1)[1]=='accepted_for_dedup'

def test_download_ignored_range_rejected_before_body(tmp_path,monkeypatch):
 e=dict(category='test',url='https://example.test/test.jsonl',expected_bytes=10)
 sig=hashlib.sha256(json.dumps(dict(url=e['url'],bytes=10,revision=None),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 (tmp_path/'test.jsonl.part').write_bytes(b'ab');(tmp_path/'test.jsonl.part.json').write_text(json.dumps(dict(signature=sig)))
 class R:
  status_code=200;headers={}
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def iter_content(self,*args,**kwargs):raise AssertionError('must not read body')
 monkeypatch.setattr('review_shift.data.requests.get',lambda *a,**k:R())
 with pytest.raises(ValueError,match='honor HTTP Range'):download_entry(e,tmp_path)
 assert (tmp_path/'test.jsonl.part').read_bytes()==b'ab'

def test_download_cache_hash_and_complete_response(tmp_path,monkeypatch):
 e=dict(category='test',url='https://example.test/test.jsonl',expected_bytes=4)
 class R:
  status_code=200;headers={'Content-Length':'4'}
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def iter_content(self,*args,**kwargs):yield b'abcd'
 calls=[]
 def get(*a,**k):calls.append(a);return R()
 monkeypatch.setattr('review_shift.data.requests.get',get)
 p=download_entry(e,tmp_path);assert p.read_bytes()==b'abcd'
 assert download_entry(e,tmp_path)==p and len(calls)==1
 p.write_bytes(b'abce')
 with pytest.raises(ValueError,match='verification'):download_entry(e,tmp_path)

def test_tampered_full_claim_and_counts_are_rejected(tmp_path):
 s=source(tmp_path,'cat',[review()]);out=tmp_path/'out'
 a=process_sources([s],out,buckets=2,threads=1)
 path=out/'audit.json';a['full_published_reviews_processed']=True;path.write_text(json.dumps(a))
 with pytest.raises(ValueError,match='Full-corpus'):list_panel_buckets(out)
 a['full_published_reviews_processed']=False;a['accepted_rows']=2;path.write_text(json.dumps(a))
 with pytest.raises(ValueError,match='ledger'):list_panel_buckets(out)

@pytest.mark.parametrize('mutation',['missing','duplicate','extra','string'])
def test_bucket_manifest_cannot_omit_duplicate_or_invent_partitions(tmp_path,mutation):
 s=source(tmp_path,'cat',[review()]);out=tmp_path/'out';a=process_sources([s],out,buckets=4,threads=1)
 b=a['panel_buckets'][0]
 a['panel_buckets']={'missing':[],'duplicate':[b,b],'extra':sorted([b,(b+1)%4]),'string':[str(b)]}[mutation]
 (out/'audit.json').write_text(json.dumps(a))
 with pytest.raises(ValueError,match='panel_buckets'):
  list_panel_buckets(out)
