"""Predeclared stability and CLEAN-row scale checks. No future-label tuning.

Large modes stream one hash bucket at a time. The small user-cluster case study
keeps at most 20 selected parents. Scale errors are conditional on comparable
sampled cases, so each error mean is accompanied by its own denominator.
"""
from pathlib import Path
import sys
import json
import argparse
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import duckdb
from review_shift import data
from review_shift.analysis import compare_windows, user_cluster_stability
from review_shift.__main__ import read_config

BLOCK_COLUMNS = [
    'parent_asin', 'block_weeks', 'bootstrap_valid_fraction', 'stability_available',
    'stability_reason', 'within_mean', 'within_mean_lo', 'within_mean_hi',
    'composition_mean_lo', 'composition_mean_hi', 'turnover_mean_lo', 'turnover_mean_hi']
METRICS = ['within_mean', 'composition_mean', 'turnover_mean', 'mean_delta']
SCALE_COLUMNS = ['fraction', 'parent_asin', 'present', 'support_eligible', 'comparable'] + [
    'abs_diff_' + metric for metric in METRICS]
USER_COLUMNS = [
    'parent_asin', 'selection', 'resampling_unit', 'bootstrap_requested', 'bootstrap_valid',
    'bootstrap_invalid', 'bootstrap_valid_fraction', 'stability_available', 'stability_reason',
    'user_count', 'children_count', 'clean_review_count'] + [
    f'{part}_{unit}_{bound}' for part in ['within', 'composition', 'turnover', 'overall']
    for unit in ['mean', 'low'] for bound in ['lo', 'hi']]
SCALE_SCOPE = ('Frozen full-data-supported cohort; clean-row reviewer thinning, not raw JSONL '
               'end-to-end I/O benchmark; comparison to full corpus is not truth. '
               'present means observed in either comparison window; error means exclude '
               'incomparable/missing cases and report their denominators explicitly.')


def _supported(frame):
    """Empty float/object masks must be cast to bool, not interpreted as columns."""
    return frame.loc[frame['support_eligible'].fillna(False).astype(bool)].copy()


def _begin_csv(path, columns):
    temporary = path.with_name(path.name + '.part')
    pd.DataFrame(columns=columns).to_csv(temporary, index=False)
    return temporary


def _append_csv(path, frame, columns):
    if len(frame):
        frame.reindex(columns=columns).to_csv(path, index=False, mode='a', header=False)
    return len(frame)


def _counter():
    return {'base_supported_parents': 0, 'present_parents': 0,
            'supported_parents': 0, 'comparable_parents': 0,
            'errors': {metric: {'sum': 0., 'denominator': 0} for metric in METRICS}}


def _scale_bucket(base, sampled, fraction):
    """Preserve every fixed-cohort parent, even when absent from the sample."""
    columns = ['parent_asin', *METRICS]
    left = base.reindex(columns=columns).copy()
    right_columns = ['parent_asin', 'n_current', 'support_eligible', 'comparable', *METRICS]
    right = sampled.reindex(columns=right_columns).copy()
    # An all-empty compare_windows result otherwise has a float key; joining it
    # to real string IDs can raise or silently erase the intended fixed cohort.
    left['parent_asin'] = left['parent_asin'].astype('string')
    right['parent_asin'] = right['parent_asin'].astype('string')
    merged = left.merge(right, on='parent_asin', how='left', suffixes=('_full', ''), validate='one_to_one')
    result = pd.DataFrame({'fraction': fraction, 'parent_asin': merged.parent_asin,
                           'present': merged.n_current.notna(),
                           'support_eligible': merged.support_eligible.fillna(False).astype(bool),
                           'comparable': merged.comparable.fillna(False).astype(bool)})
    for metric in METRICS:
        estimate = pd.to_numeric(merged[metric], errors='coerce')
        reference = pd.to_numeric(merged[metric + '_full'], errors='coerce')
        valid = result.comparable & np.isfinite(estimate) & np.isfinite(reference)
        result['abs_diff_' + metric] = (estimate - reference).abs().where(valid, np.nan)
    return result.reindex(columns=SCALE_COLUMNS)


def run_experiments(c, mode):
    w = c['windows']; out = Path(c['output_dir']); out.mkdir(exist_ok=True, parents=True)
    ds = Path(c['processed_dir']); audit = json.loads((ds / 'audit.json').read_text())
    buckets = data.list_panel_buckets(ds)
    kwargs = dict(c['analysis']); kwargs['bootstrap_reps'] = 0

    def compare(frame, **extra):
        return compare_windows(frame, w['reference_start'], w['current_start'],
                               w['current_end'], **{**kwargs, **extra})

    result_rows = 0
    if mode == 'blocks':
        target = out / 'block_sensitivity.csv'
        pending = _begin_csv(target, BLOCK_COLUMNS)
        for bucket in buckets:
            frame = data.load_panel_bucket(ds, bucket)
            base = _supported(compare(frame))
            ids = set(base.parent_asin)
            if ids:
                subset = frame.loc[frame.parent_asin.isin(ids)]
                for length in [1, 2, 4, 8]:
                    result = compare(subset, bootstrap_reps=300, block_weeks=length)
                    result_rows += _append_csv(pending, result, BLOCK_COLUMNS)
            print('block_sensitivity_bucket', bucket, flush=True)
        pending.replace(target)
    elif mode == 'users':
        # Select from observed monitoring support, never future outcomes. Keep
        # only the current global top 20; full-corpus candidates do not accumulate.
        selected = []
        for bucket in buckets:
            frame = data.load_panel_bucket(ds, bucket)
            base = _supported(compare(frame))
            if len(base):
                selected.extend({'bucket': bucket, **row}
                                for row in base.nlargest(20, 'n_current').to_dict('records'))
                selected = sorted(selected, key=lambda r: (-r['n_current'], r['parent_asin']))[:20]
        results = []
        for row in selected:
            paths = [ds / name for name in audit['artifact_sha256']
                     if name.startswith(f"clean/bucket={row['bucket']:03d}/")]
            for path in paths:
                if data.sha256_file(path) != audit['artifact_sha256'][str(path.relative_to(ds))]:
                    raise ValueError('Clean artifact failed SHA256 validation')
            connection = duckdb.connect()
            try:
                # Pull only this parent's required columns and comparison
                # windows, rather than its entire historical record into pandas.
                clean = connection.execute(
                    'SELECT user_hash,parent_asin,asin,week,rating FROM read_parquet(?) '
                    'WHERE parent_asin=? AND week>=CAST(? AS DATE) AND week<CAST(? AS DATE)',
                    [[str(p) for p in paths], row['parent_asin'],
                     w['reference_start'], w['current_end']]).df()
            finally:
                connection.close()
            result = user_cluster_stability(clean, w['reference_start'], w['current_start'],
                                            w['current_end'], reps=300, seed=42)
            results.append({'parent_asin': row['parent_asin'],
                            'selection': 'largest_current_n_among_supported_no_future_labels', **result})
        target = out / 'user_sensitivity.csv'; pending = _begin_csv(target, USER_COLUMNS)
        result_rows = _append_csv(pending, pd.DataFrame(results), USER_COLUMNS)
        pending.replace(target)
    elif mode == 'scale':
        sample_dirs = {1.0: ds}
        for fraction in [.1, .5]:
            target = ds.parent / f'{ds.name}_users_{int(fraction * 100)}pct'
            data.sample_clean(ds, target, fraction, seed=42,
                              memory_limit=c['ingestion']['memory_limit'], threads=c['ingestion']['threads'])
            sample_dirs[fraction] = target
        counters = {fraction: _counter() for fraction in sample_dirs}
        target = out / 'scale_parent_comparison.csv'; pending = _begin_csv(target, SCALE_COLUMNS)
        for bucket in buckets:
            frame = data.load_panel_bucket(ds, bucket)
            base = _supported(compare(frame))
            if not len(base):
                print('scale_bucket', bucket, 'base_supported_parents', 0, flush=True)
                continue
            fixed_ids = set(base.parent_asin)
            for fraction, sample_dir in sorted(sample_dirs.items()):
                sub = data.load_panel_bucket(sample_dir, bucket)
                now = compare(sub.loc[sub.parent_asin.isin(fixed_ids)])
                result = _scale_bucket(base, now, fraction)
                result_rows += _append_csv(pending, result, SCALE_COLUMNS)
                acc = counters[fraction]
                acc['base_supported_parents'] += len(result)
                acc['present_parents'] += int(result.present.sum())
                acc['supported_parents'] += int(result.support_eligible.sum())
                acc['comparable_parents'] += int(result.comparable.sum())
                for metric in METRICS:
                    values = pd.to_numeric(result['abs_diff_' + metric], errors='coerce')
                    valid = values[np.isfinite(values)]
                    acc['errors'][metric]['sum'] += float(valid.sum())
                    acc['errors'][metric]['denominator'] += len(valid)
            print('scale_bucket', bucket, 'base_supported_parents', len(base), flush=True)
        # Only three aggregate records are retained, regardless of corpus size.
        summary = []
        for fraction, acc in sorted(counters.items()):
            sample_audit = json.loads((sample_dirs[fraction] / 'audit.json').read_text())
            row = {'fraction': fraction,
                   **{key: acc[key] for key in ['base_supported_parents', 'present_parents',
                                                'supported_parents', 'comparable_parents']},
                   'accepted_clean_reviews': sample_audit['accepted_rows'], 'scope': SCALE_SCOPE}
            for metric, stats in acc['errors'].items():
                denominator = stats['denominator']
                row['mean_absolute_' + metric + '_difference'] = stats['sum'] / denominator if denominator else np.nan
                row[metric + '_error_denominator'] = denominator
            # Keep the original public field as a backward-compatible alias.
            row['mean_absolute_within_difference'] = row['mean_absolute_within_mean_difference']
            summary.append(row)
        summary_path = out / 'scale_summary.csv'
        summary_pending = summary_path.with_name(summary_path.name + '.part')
        pd.DataFrame(summary).to_csv(summary_pending, index=False)
        pending.replace(target); summary_pending.replace(summary_path)
    else:
        raise ValueError('Unknown experiment mode')
    result = {'mode': mode, 'output_dir': str(out), 'result_rows': result_rows}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--mode', choices=['blocks', 'users', 'scale'], required=True)
    args = parser.parse_args()
    run_experiments(read_config(args.config), args.mode)


if __name__ == '__main__':
    main()
