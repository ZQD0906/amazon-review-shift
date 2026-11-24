"""CLI for ingestion, rating diagnostics, and rolling backtests."""
from pathlib import Path
from datetime import datetime,timezone,timedelta,date
import argparse,json,math,sys,time,uuid,shutil
import pandas as pd
from . import __version__
from . import data
from .analysis import compare_windows,evaluate_forecasts,synthetic_panel,COMPARISON_COLUMNS,FORECAST_COLUMNS
from .reporting import write_report


def read_config(path):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    for k in ['processed_dir','output_dir','windows']:
        if k not in c:raise ValueError('Missing configuration: '+k)
    return c


def source_entries(c,download=True):
    m=json.loads(Path(c['manifest']).read_text())
    selected=c.get('categories') or m['expected_review_categories']
    if len(selected)!=len(set(selected)):raise ValueError('Duplicate selected categories')
    by={e['category']:e for e in m['review_entries']}
    if not set(selected)<=set(by):raise ValueError('Unknown source categories')
    result=[]
    for cat in selected:
        e=dict(by[cat]);p=Path(c['raw_dir'])/e['filename']
        if not p.exists():
            if not download:raise FileNotFoundError(p)
            p=data.download_entry(e,c['raw_dir'])
        if p.stat().st_size!=e['expected_bytes']:raise ValueError('Local source size differs from manifest: '+str(p))
        e['path']=str(p);result.append(e)
    return result


def process(c):
    a=data.process_sources(source_entries(c),c['processed_dir'],**c.get('ingestion',{}))
    print(json.dumps({k:a[k] for k in ['scanned_rows','accepted_rows','full_published_reviews_processed','coverage']},ensure_ascii=False),flush=True)
    return a


def _valid_number(x):
    return x is not None and math.isfinite(float(x))


def analyze(c,bucket=None,panel=None,label=None):
    final_out=Path(c['output_dir'])
    if bucket is not None:final_out=final_out/f'bucket-{bucket:03d}'
    final_out.parent.mkdir(parents=True,exist_ok=True)
    out=final_out.parent/f'.{final_out.name}.building-{uuid.uuid4().hex}'
    out.mkdir()
    started=time.monotonic();w=c['windows'];settings=c.get('analysis',{})
    if panel is None:
        ingestion=json.loads((Path(c['processed_dir'])/'audit.json').read_text())
        buckets=data.list_panel_buckets(c['processed_dir'])
        if ingestion.get('sources') is not None:
            expected=set(c.get('categories') or json.loads(Path(c['manifest']).read_text())['expected_review_categories'])
            actual={x['source_category'] for x in ingestion['sources']}
            if actual != expected:raise ValueError('Configured categories differ from processed source audit')
        elif ingestion.get('sampling_kind'):
            raise ValueError('Use the scale experiment entry point for sampled clean-row datasets')
        if bucket is not None:
            if bucket not in buckets:raise ValueError('Selected bucket absent from processed audit')
            buckets=[bucket]
        panels=((b,data.load_panel_bucket(c['processed_dir'],b)) for b in buckets)
    else:
        ingestion={'coverage':'synthetic_panel','scanned_rows':None,'accepted_rows':None,'full_published_reviews_processed':False}
        buckets=[0];panels=iter([(0,panel)])
    summary={'label':label or c['label'],'version':__version__,'created_at_utc':datetime.now(timezone.utc).isoformat(),
             'configuration':c,'windows':w,'ingestion':ingestion,'analyzed_buckets':buckets,
             'source_scanned_rows':ingestion['scanned_rows'],'source_accepted_rows':ingestion['accepted_rows'],
             'full_published_reviews_processed':bool(ingestion.get('full_published_reviews_processed',False)),
             'analysis_covers_all_ingested_buckets':bucket is None,'ingested_bucket_count':len(ingestion.get('panel_buckets',buckets)),'candidate_parents':0,'supported_parents':0,
             'panel_rows_analyzed':0,'max_identity_error':0.,'statuses':{},'forecast_metrics':[],
             'interpretation':'Retrospective observable-review diagnostics; conditional stability only; not production quality or causal impact.'}
    accum={};examples=[];first_comp=True;first_forecast=True
    for b,frame in panels:
        comp=compare_windows(frame,w['reference_start'],w['current_start'],w['current_end'],**settings)
        forecasts=evaluate_forecasts(frame,comp,w['reference_start'],w['current_start'],w['current_end'],w['future_end'],**c.get('forecast',{}))
        comp.to_csv(out/'comparison.csv',index=False,mode='w' if first_comp else 'a',header=first_comp);first_comp=False
        forecasts.to_csv(out/'forecast.csv',index=False,mode='w' if first_forecast else 'a',header=first_forecast);first_forecast=False
        summary['panel_rows_analyzed']+=len(frame);summary['candidate_parents']+=len(comp)
        if len(comp):
            summary['supported_parents']+=int(comp.support_eligible.sum())
            errors=comp.identity_error.dropna()
            if len(errors):summary['max_identity_error']=max(summary['max_identity_error'],float(errors.max()))
            for name,n in comp.status.value_counts().items():summary['statuses'][name]=summary['statuses'].get(name,0)+int(n)
            supported=comp[comp.support_eligible].copy()
            if len(supported):
                supported['_magnitude']=supported.mean_delta.abs()
                examples.append(supported.nlargest(30,'_magnitude'))
        for method,g in forecasts.groupby('method'):
            a=accum.setdefault(method,{'method':method,'eligible_parents':0,'observable_parents':0,'observed_future_reviews':0,
                                      'sum_log_loss':0.,'sum_rps':0.,'weighted_log_loss':0.,'weighted_rps':0.})
            eligible=g[g.eligible.astype(bool)];obs=eligible[eligible.observable.astype(bool)]
            a['eligible_parents']+=len(eligible);a['observable_parents']+=len(obs);a['observed_future_reviews']+=int(obs.n_future.sum())
            for metric in ['log_loss','rps']:
                a['sum_'+metric]+=float(obs[metric].sum());a['weighted_'+metric]+=float((obs[metric]*obs.n_future).sum())
        print(json.dumps({'bucket':b,'candidate_parents':len(comp),'supported_parents':int(comp.support_eligible.sum()) if len(comp) else 0},ensure_ascii=False),flush=True)
    for method,a in sorted(accum.items()):
        k=a['observable_parents'];n=a['observed_future_reviews'];total=a['eligible_parents']
        row={key:a[key] for key in ['method','eligible_parents','observable_parents','observed_future_reviews']}
        row['unknown_future_parents']=total-k;row['observable_fraction']=k/total if total else None
        for metric in ['log_loss','rps']:
            row['parent_weighted_'+metric]=a['sum_'+metric]/k if k else None
            row['review_weighted_'+metric]=a['weighted_'+metric]/n if n else None
        summary['forecast_metrics'].append(row)
    if first_comp:pd.DataFrame(columns=COMPARISON_COLUMNS).to_csv(out/'comparison.csv',index=False)
    if first_forecast:pd.DataFrame(columns=FORECAST_COLUMNS).to_csv(out/'forecast.csv',index=False)
    selected=pd.concat(examples,ignore_index=True).nlargest(30,'_magnitude') if examples else pd.DataFrame()
    summary['elapsed_seconds']=round(time.monotonic()-started,3)
    write_report(summary,selected,out)
    final_out.mkdir(parents=True,exist_ok=True)
    (final_out/'COMPLETE.json').unlink(missing_ok=True)
    (final_out/'report.html').unlink(missing_ok=True)
    files=['comparison.csv','forecast.csv','summary.json','diagnostics.png','report.html']
    for name in files:(out/name).replace(final_out/name)
    receipt={'state':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),'files_sha256':{name:data.sha256_file(final_out/name) for name in files}}
    (out/'COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'COMPLETE.json').replace(final_out/'COMPLETE.json')
    out.rmdir()
    out=final_out
    print(json.dumps({'output_dir':str(out),'candidate_parents':summary['candidate_parents'],'supported_parents':summary['supported_parents'],
                      'full_published_reviews_processed':summary['full_published_reviews_processed']},ensure_ascii=False),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description='Product Review Shift Diagnostics')
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ['process','analyze','run','backtest','sample','download-metadata']:
        s=sub.add_parser(cmd);s.add_argument('--config',default='configs/pilot.json')
        if cmd=='analyze':s.add_argument('--bucket',type=int)
        if cmd=='sample':s.add_argument('--fraction',required=True,type=float);s.add_argument('--out',required=True)
    s=sub.add_parser('manifest');s.add_argument('--source',default='data/source_verification.json');s.add_argument('--out',default='data/manifest.json')
    s=sub.add_parser('demo');s.add_argument('--out',default='results/demo')
    args=p.parse_args()
    if args.command=='manifest':
        m=data.manifest_from_verification(args.source,args.out);print(json.dumps({'review_files':len(m['review_entries']),'review_bytes':m['review_bytes'],'metadata_bytes':m['metadata_bytes']}));return
    if args.command=='demo':
        c=read_config('configs/pilot.json');c['output_dir']=args.out;c['label']='SYNTHETIC_DEMONSTRATION_NOT_REAL_REVIEWS'
        analyze(c,panel=synthetic_panel());return
    c=read_config(args.config)
    if args.command in ['process','run']:process(c)
    if args.command in ['analyze','run']:analyze(c,bucket=getattr(args,'bucket',None))
    if args.command=='sample':
        a=data.sample_clean(c['processed_dir'],args.out,args.fraction,seed=c.get('sample_seed',42),memory_limit=c.get('ingestion',{}).get('memory_limit','4GB'))
        print(json.dumps({'sampling':a.get('sampling_kind'),'accepted_rows':a['accepted_rows'],'out':args.out}));return
    if args.command=='download-metadata':
        m=json.loads(Path(c['manifest']).read_text());wanted=c.get('categories') or m['expected_review_categories'];n=0
        for e in m['metadata_entries']:
            if e['category'] in wanted:data.download_entry(e,Path(c['raw_dir'])/'metadata');n+=1
        print(json.dumps({'metadata_files_downloaded':n,'note':'Downloaded snapshots only; not time-valid historical features.'}));return
    if args.command=='backtest':
        all_results=[]
        for start in c.get('backtest_starts',[]):
            t=date.fromisoformat(start);cfg=json.loads(json.dumps(c))
            cfg['windows']={'reference_start':str(t-timedelta(weeks=12)),'current_start':str(t),'current_end':str(t+timedelta(weeks=12)),'future_end':str(t+timedelta(weeks=16))}
            cfg['output_dir']=str(Path(c['output_dir'])/'rolling'/start)
            r=analyze(cfg);all_results.extend({'current_start':start,**x} for x in r['forecast_metrics'])
        if not all_results:raise ValueError('No backtest_starts configured')
        pd.DataFrame(all_results).to_csv(Path(c['output_dir'])/'rolling_metrics.csv',index=False)


if __name__=='__main__':main()
