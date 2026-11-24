"""Extract review evidence for manual checking; no automatic topic labels."""
from pathlib import Path
import argparse,json,sys,hashlib
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from review_shift.__main__ import read_config,source_entries


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/pilot.json');p.add_argument('--parents',type=int,default=5);p.add_argument('--per-parent',type=int,default=8)
 a=p.parse_args();c=read_config(a.config)
 if a.parents<1 or a.per_parent<1:raise ValueError('Positive evidence limits required')
 selected=pd.DataFrame()
 for chunk in pd.read_csv(Path(c['output_dir'])/'comparison.csv',chunksize=100000):
  chunk=chunk.loc[chunk.support_eligible.fillna(False).astype(bool)]
  selected=pd.concat([selected,chunk],ignore_index=True).nlargest(a.parents,'n_current')
 ids=set(selected.parent_asin);counts={x:0 for x in ids};seen=set();rows=[]
 start=int(datetime.fromisoformat(c['windows']['reference_start']).replace(tzinfo=timezone.utc).timestamp()*1000)
 end=int(datetime.fromisoformat(c['windows']['current_end']).replace(tzinfo=timezone.utc).timestamp()*1000)
 # Deterministic file-order illustration; explicitly not representative topic prevalence.
 for source in source_entries(c,download=False):
  with Path(source['path']).open(encoding='utf-8') as f:
   for line in f:
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    parent=r.get('parent_asin');t=r.get('timestamp',r.get('sort_timestamp'))
    if parent not in ids or counts[parent]>=a.per_parent or not isinstance(t,int) or not start<=t<end:continue
    digest=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if digest in seen:continue
    seen.add(digest);counts[parent]+=1
    rows.append({'parent_asin':parent,'asin':r.get('asin'),'date_utc':datetime.fromtimestamp(t/1000,timezone.utc).date().isoformat(),
      'rating':r.get('rating'),'review_text':r.get('text','')[:1500],'record_sha256':digest,'source_category':source['category'],
      'manual_topic':'','manual_validation_status':'unreviewed_raw_candidate_may_include_identity_conflicts',
      'selection_note':'First distinct raw records for past-supported parents; illustrative only, not representative or automatically validated.'})
 out=Path(c['output_dir'])/'manual_evidence.csv'
 pd.DataFrame(rows,columns=['parent_asin','asin','date_utc','rating','review_text','record_sha256','source_category','manual_topic','manual_validation_status','selection_note']).to_csv(out,index=False)
 print(json.dumps({'output':str(out),'selected_parents':len(ids),'evidence_rows':len(rows),'manual_validation_completed':False}))


if __name__=='__main__':main()
