"""Inspectable local reports."""
from pathlib import Path
import html,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_report(summary, examples, output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    cols=[c for c in ['parent_asin','status','n_ref','n_current','common_coverage_ref','common_coverage_current','mean_ref','mean_current','mean_delta','within_mean','composition_mean','turnover_mean','bootstrap_valid_fraction'] if c in examples]
    table=examples[cols].to_html(index=False,classes='cards',float_format=lambda x:f'{x:.4f}',escape=True) if len(examples) else '<p>No supported examples in this window.</p>'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Product Review Shift Diagnostics</title><style>body{{font:16px/1.65 system-ui;max-width:1280px;margin:36px auto;padding:0 20px;color:#1b2738;background:#f5f7fa}}h1{{font-size:28px}}section{{background:white;border:1px solid #dce3e9;border-radius:10px;padding:24px;margin:20px 0}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#edf3f8}}.scroll{{overflow:auto}}input{{padding:10px;width:300px}}img{{max-width:100%}}.note{{color:#536273}}</style><h1>Product Review Shift Diagnostics</h1><p>Research prototype for studying changes in published review ratings.</p><section><b>{html.escape(summary['label'])}</b><p>Windows: {html.escape(str(summary['windows']))}</p><p>Candidate product families: {summary['candidate_parents']:,}. With sufficient support: {summary['supported_parents']:,}. Full 34-partition ingestion: {summary['full_published_reviews_processed']}. Buckets analyzed: {len(summary['analyzed_buckets'])}/{summary['ingested_bucket_count']}.</p><p class="note">This report describes observed reviews in the published corpus. Components account for rating shifts descriptively. Resampling ranges measure conditional stability.</p></section><section><h2>Illustrative comparisons</h2><p>Up to 30 supported product families, ranked by absolute mean-rating change. The CSV contains all candidates and the full five-star decomposition.</p><input id="search" placeholder="Filter by parent product ID or status" aria-label="Filter by parent product ID or status"><div class="scroll">{table}</div></section><section><h2>Decomposition and forecast comparison</h2><img src="diagnostics.png" alt="Rating decomposition and time-held-out forecast results"><p><a href="comparison.csv">Comparison and decomposition CSV</a> · <a href="forecast.csv">Forecast CSV</a> · <a href="summary.json">Source, windows, and scores JSON</a></p></section><script>document.getElementById('search').addEventListener('input',e=>{{let q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'':'none')}})</script></html>'''
    (out/'report.html').write_text(body,encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    if len(examples):
        chosen=examples.head(8).iloc[::-1]
        y=np.arange(len(chosen))
        for name,color,offset in [('within_mean','#1f7a8c',-.22),('composition_mean','#c68427',0),('turnover_mean','#7964b5',.22)]:
            axes[0].barh(y+offset,chosen[name],height=.20,label=name.replace('_mean',''),color=color)
        axes[0].set_yticks(y,chosen.parent_asin);axes[0].axvline(0,color='#333',linewidth=.8);axes[0].legend(fontsize=8)
    else:axes[0].text(.5,.5,'No supported comparisons',ha='center',transform=axes[0].transAxes)
    axes[0].set_title('Change accounting: illustrative supported cases');axes[0].set_xlabel('Mean-star contribution (descriptive)')
    forecasts=summary.get('forecast_metrics',[])
    visible=[x for x in forecasts if x.get('review_weighted_rps') is not None]
    if visible:
        axes[1].bar(range(len(visible)),[x['review_weighted_rps'] for x in visible],color=['#667d96','#1f7a8c','#c68427'][:len(visible)])
        axes[1].set_xticks(range(len(visible)),[x['method'].replace('_','\n') for x in visible],fontsize=9)
    else:axes[1].text(.5,.5,'No observable held-out outcomes',ha='center',transform=axes[1].transAxes)
    axes[1].set_title('Future rating prediction: same eligible cohort');axes[1].set_ylabel('Review-weighted ranked probability score (lower better)')
    fig.suptitle(summary['label']+' | published-review research prototype',fontsize=12)
    fig.savefig(out/'diagnostics.png',dpi=170);plt.close(fig)
