#!/usr/bin/env python3
"""Known-truth mechanism experiments, explicitly not Amazon findings.

Run from 完整代码: python scripts/simulate_mechanisms.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review_shift.analysis import bootstrap_stability, decompose_counts


def quantiles(values):
    q=np.quantile(np.asarray(values),[.05,.5,.95])
    return dict(zip(['p05','p50','p95'],map(float,q)))


def mixture_experiments(rng,repetitions):
    p_good=np.array([.02,.03,.05,.2,.7]);p_bad=np.array([.25,.2,.2,.2,.15])
    setups={
      'composition_only':(np.array([480,120]),np.array([120,480]),np.stack([p_good,p_bad]),np.stack([p_good,p_bad])),
      'within_only':(np.array([360,240]),np.array([360,240]),np.stack([p_good,p_bad]),
                     np.stack([np.array([.12,.1,.15,.23,.4]),np.array([.35,.2,.2,.15,.1])])),
      'observed_entry_exit':(np.array([360,240,0]),np.array([360,0,240]),np.stack([p_good,p_good,p_bad]),np.stack([p_good,p_good,p_bad]))}
    output={}
    for name,(n0,n1,p0,p1) in setups.items():
        truth=decompose_counts(n0[:,None]*p0,n1[:,None]*p1)
        errors={part:[] for part in ['within','composition','turnover']}
        max_identity=0
        for _ in range(repetitions):
            a=np.array([rng.multinomial(int(n),p) for n,p in zip(n0,p0)])
            b=np.array([rng.multinomial(int(n),p) for n,p in zip(n1,p1)])
            estimate=decompose_counts(a,b)
            max_identity=max(max_identity,estimate['identity_error'])
            for part in errors:
                errors[part].append(float((estimate[part]-truth[part])@np.arange(1,6)))
        output[name]={
          'n_ref_by_child':n0.tolist(),'n_current_by_child':n1.tolist(),
          'true_probability_ref':p0.tolist(),'true_probability_current':p1.tolist(),
          'true_components':{part:truth[part].tolist() for part in errors},
          'true_mean_components':{part:float(truth[part]@np.arange(1,6)) for part in errors},
          'mean_score_component_error_quantiles':{part:quantiles(values) for part,values in errors.items()},
          'mean_score_component_rmse':{part:float(np.sqrt(np.mean(np.square(values)))) for part,values in errors.items()},
          'maximum_identity_error':float(max_identity),
          'note':'Reference centering q is re-estimated in each independently generated dataset; bootstrap within one dataset freezes q.'}
    return output


def dependence_experiment(rng,repetitions,bootstrap_reps):
    weeks=12;per_week=40;true_rate=.2
    naive_cover=0;block_cover=0;block_available=0;naive_widths=[];block_widths=[];estimates=[]
    for rep in range(repetitions):
        # Independent weeks with Beta-varying probabilities; reviews within a
        # week share a latent probability and are marginally correlated.
        p0=rng.beta(2,8,size=weeks);p1=rng.beta(2,8,size=weeks)
        k0=rng.binomial(per_week,p0);k1=rng.binomial(per_week,p1)
        r0=float(k0.sum()/(weeks*per_week));r1=float(k1.sum()/(weeks*per_week));delta=r1-r0
        stderr=np.sqrt(r0*(1-r0)/(weeks*per_week)+r1*(1-r1)/(weeks*per_week))
        low,high=delta-1.96*stderr,delta+1.96*stderr
        naive_cover+=low<=0<=high;naive_widths.append(high-low);estimates.append(delta)
        a=np.zeros((weeks,1,5));b=a.copy()
        a[:,0,0]=k0;a[:,0,4]=per_week-k0;b[:,0,0]=k1;b[:,0,4]=per_week-k1
        result=bootstrap_stability(a,b,reps=bootstrap_reps,block_weeks=2,seed=100000+rep)
        if result['stability_available']:
            block_available+=1
            lo,hi=result['overall_low_lo'],result['overall_low_hi']
            block_cover+=lo<=0<=hi;block_widths.append(hi-lo)
    def mcse(hits,total):
        p=hits/total;return float(np.sqrt(p*(1-p)/total))
    return {'generator':'12 independent weeks per period; 40 reviews per week; latent week probability Beta(2,8); conditional Binomial counts',
            'true_low_rating_probability':true_rate,'true_period_difference':0.,
            'within_week_intraclass_correlation':1/11,
            'bootstrap_reps_per_dataset':bootstrap_reps,'block_weeks':2,
            'naive_review_iid_wald':{'covered':int(naive_cover),'datasets':repetitions,'coverage':naive_cover/repetitions,
                                    'monte_carlo_standard_error':mcse(naive_cover,repetitions),'width_quantiles':quantiles(naive_widths)},
            'conditional_week_block':{'covered':int(block_cover),'available_datasets':block_available,
                                      'total_datasets':repetitions,'coverage':block_cover/block_available,
                                      'monte_carlo_standard_error':mcse(block_cover,block_available),'width_quantiles':quantiles(block_widths)},
            'observed_difference_quantiles':quantiles(estimates),
            'note':'Simulation coverage is against the repeated-experiment mean difference, not the exact finite-corpus accounting. These weeks have no serial dependence; this does not validate arbitrary weekly/seasonal correlation or user-selection bias.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20261027)
    parser.add_argument('--bootstrap-reps',type=int,default=199)
    parser.add_argument('--out',default='results/mechanism_simulation.json')
    args=parser.parse_args()
    if args.repetitions<1 or args.bootstrap_reps<30:parser.error('Use positive repetitions and at least 30 bootstrap draws')
    rng=np.random.default_rng(args.seed)
    result={'experiment_kind':'PURE_KNOWN_TRUTH_SIMULATION_NOT_AMAZON_NOT_STUDENT_RESULTS',
            'configuration':{'repetitions_per_scenario':args.repetitions,'seed':args.seed,'numpy_version':np.__version__,
                             'generator':'numpy.random.default_rng (PCG64)','error_units':'star-score units; low-rate widths are probability units'},
            'mixture_scenarios':mixture_experiments(rng,args.repetitions),
            'dependence_scenario':dependence_experiment(rng,args.repetitions,args.bootstrap_reps),
            'limitations':[
              'The original corpus has user-centric selection and incomplete item histories absent from these mechanisms.',
              'Exact accounting reconstruction is an algebraic property, not evidence of valid population uncertainty or correct product-quality diagnoses.',
              'Known-truth scenarios are deliberately small with fixed child counts; real sparse shared children can invalidate resampling draws.',
              'Twelve weeks remain a small number of independent units. More reviews or bootstrap draws cannot create additional independent weeks.',
              'No selected-case coverage, simultaneous guarantee, FDR control, causal effect, product improvement, or scaling performance is established.',
              'Component centering is a defined accounting convention; it is not an identified causal attribution.' ]}
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(target),'repetitions_per_scenario':args.repetitions,
                      'max_identity_error':max(x['maximum_identity_error'] for x in result['mixture_scenarios'].values()),
                      'naive_coverage':result['dependence_scenario']['naive_review_iid_wald']['coverage'],
                      'week_block_coverage':result['dependence_scenario']['conditional_week_block']['coverage']},ensure_ascii=False))


if __name__=='__main__':main()
