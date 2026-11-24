"""Auditable five-star distribution decomposition and historical forecasts.

All inference is conditional on the released corpus.
The accounting decomposition is exact; resampling ranges are not population
confidence guarantees or simultaneous/FDR-controlled intervals.
"""
from __future__ import annotations

import hashlib
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

COUNT_COLUMNS = [f"n{i}" for i in range(1, 6)]
KEY_COLUMNS = ["parent_asin", "asin", "week"]
COMPONENTS = ("within", "composition", "turnover")
STARS = np.arange(1.0, 6.0)

# Stable schemas are required when the CLI appends many independent hash buckets.
COMPARISON_COLUMNS = [
    "parent_asin", "reference_start", "current_start", "current_end", "n_ref", "n_current",
    "weeks_ref", "weeks_current", "calendar_weeks_ref", "calendar_weeks_current",
    "children_ref", "children_current", "children_common", "children_newly_observed",
    "children_no_longer_observed", "comparable", "support_eligible", "status",
    "identity_error", "component_mass_error", "common_coverage_ref", "common_coverage_current",
    "mean_ref", "mean_current", "mean_delta", "low_ref", "low_current", "low_delta",
] + [f"{part}_{k}" for part in ("p_ref", "p_current", *COMPONENTS, "overall") for k in range(1,6)] + [
    f"{part}_{unit}" for part in COMPONENTS for unit in ("mean", "low")
] + ["bootstrap_requested", "bootstrap_valid", "bootstrap_invalid", "bootstrap_valid_fraction",
     "stability_available", "stability_reason", "block_weeks"] + [
    f"{part}_{unit}_{bound}" for part in (*COMPONENTS, "overall")
    for unit in ("mean", "low") for bound in ("lo", "hi")]
FORECAST_COLUMNS = ["parent_asin", "method", "eligible", "n_future", "observable", "future_start",
                    "future_end", "status", "log_loss", "rps"] + [f"pred_{k}" for k in range(1,6)]

COMPARISON_INTEGER_COLUMNS = [
    "n_ref", "n_current", "weeks_ref", "weeks_current", "calendar_weeks_ref", "calendar_weeks_current",
    "children_ref", "children_current", "children_common", "children_newly_observed",
    "children_no_longer_observed", "bootstrap_requested", "bootstrap_valid", "bootstrap_invalid", "block_weeks",
]


def _typed_result(frame: pd.DataFrame, columns: list[str], *, integers: list[str],
                  booleans: list[str], strings: list[str]) -> pd.DataFrame:
    """Preserve schema AND boolean filtering behavior for empty hash buckets."""
    result = frame.reindex(columns=columns)
    for column in columns:
        if column in booleans:
            result[column] = result[column].astype("boolean").fillna(False).astype(bool)
        elif column in strings:
            result[column] = result[column].astype("string")
        elif column in integers:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("Int64")
        else:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def _monday(value: Any, label: str) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize() or date.weekday() != 0:
        raise ValueError(f"{label} must be a timezone-naive Monday at midnight")
    return date


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Copy and validate one hash bucket, never the full published corpus."""
    required = KEY_COLUMNS + COUNT_COLUMNS + ["n"]
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(f"Missing panel columns: {missing}")
    frame = panel.loc[:, required].copy()
    for col in ("parent_asin", "asin"):
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identifier: {col}")
        frame[col] = frame[col].astype(str)
    frame["week"] = pd.to_datetime(frame["week"], errors="raise")
    if getattr(frame.week.dt, "tz", None) is not None:
        raise ValueError("Panel weeks must be timezone-naive")
    if (frame.week.isna().any() or frame.week.dt.day_of_week.ne(0).any()
            or frame.week.ne(frame.week.dt.normalize()).any()):
        raise ValueError("Panel weeks must be Monday midnight dates")
    for col in COUNT_COLUMNS + ["n"]:
        values = pd.to_numeric(frame[col], errors="raise")
        if (not np.isfinite(values.to_numpy(dtype=float)).all()
                or (values < 0).any() or (values % 1 != 0).any()):
            raise ValueError("Panel counts must be finite nonnegative integers")
        frame[col] = values.astype("int64")
    if frame["n"].ne(frame[COUNT_COLUMNS].sum(axis=1)).any():
        raise ValueError("Panel n must equal n1+n2+n3+n4+n5")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate parent_asin/asin/week panel keys")
    return frame.loc[frame.n > 0].sort_values(KEY_COLUMNS).reset_index(drop=True)


def decompose_counts(reference: np.ndarray, current: np.ndarray,
                     q: np.ndarray | None = None) -> dict[str, Any]:
    """Exact decomposition of two aligned child-by-five count matrices.

    Zero-count children are unobserved, not five-star distributions of zero.
    Weights always use the entire parent's period count. q defaults to the
    reference parent's distribution and centers composition/turnover terms.
    A supplied q must be a fixed probability vector, not estimated from future.
    """
    left, right = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if left.ndim != 2 or left.shape[1] != 5 or left.shape != right.shape:
        raise ValueError("Counts must be aligned child-by-five matrices")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or (left < 0).any() or (right < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    n0, n1 = left.sum(axis=1), right.sum(axis=1)
    total0, total1 = n0.sum(), n1.sum()
    if total0 <= 0 or total1 <= 0:
        raise ValueError("Both comparison periods must have observations")
    p0, p1 = left.sum(axis=0) / total0, right.sum(axis=0) / total1
    q = p0.copy() if q is None else np.asarray(q, dtype=float)
    if q.shape != (5,) or not np.isfinite(q).all() or (q < 0).any() or not np.isclose(q.sum(), 1):
        raise ValueError("q must be a five-category probability vector")
    common = (n0 > 0) & (n1 > 0)
    new = (n0 == 0) & (n1 > 0)
    gone = (n0 > 0) & (n1 == 0)
    w0, w1 = n0 / total0, n1 / total1
    a = left[common] / n0[common, None]
    b = right[common] / n1[common, None]
    within = (((w0[common] + w1[common]) / 2)[:, None] * (b - a)).sum(axis=0)
    composition = ((w1[common] - w0[common])[:, None] * ((a + b) / 2 - q)).sum(axis=0)
    turnover = ((right[new].sum(axis=0) / total1 - w1[new].sum() * q)
                - (left[gone].sum(axis=0) / total0 - w0[gone].sum() * q))
    overall = p1 - p0
    return {"p_ref": p0, "p_current": p1, "q": q, "within": within,
            "composition": composition, "turnover": turnover, "overall": overall,
            "common_mask": common, "new_mask": new, "gone_mask": gone,
            "common_coverage_ref": float(w0[common].sum()),
            "common_coverage_current": float(w1[common].sum()),
            "identity_error": float(np.max(np.abs(overall - within - composition - turnover))),
            "component_mass_error": float(max(abs(within.sum()), abs(composition.sum()), abs(turnover.sum())))}


def _child_counts(frame: pd.DataFrame, children: list[str]) -> np.ndarray:
    return frame.groupby("asin")[COUNT_COLUMNS].sum().reindex(children, fill_value=0).to_numpy(dtype=float)


def _week_cube(frame: pd.DataFrame, children: list[str], start: pd.Timestamp,
               end: pd.Timestamp) -> np.ndarray:
    weeks = pd.date_range(start, end - pd.Timedelta(weeks=1), freq="W-MON")
    idx = pd.MultiIndex.from_product([weeks, children], names=["week", "asin"])
    grouped = frame.groupby(["week", "asin"])[COUNT_COLUMNS].sum().reindex(idx, fill_value=0)
    return grouped.to_numpy(dtype=float).reshape(len(weeks), len(children), 5)


def _block_weights(weeks: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Non-circular moving blocks; overlap and endpoint effects are retained."""
    starts = rng.integers(0, weeks - block + 1, size=(reps, ceil(weeks / block)))
    indexes = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :weeks]
    weights = np.zeros((reps, weeks), dtype=float)
    np.add.at(weights, (np.arange(reps)[:, None], indexes), 1)
    return weights


def _seed(parent: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{parent}".encode()).digest()[:8], "little")


def bootstrap_stability(reference_cube: np.ndarray, current_cube: np.ndarray, *,
                        reps: int = 300, block_weeks: int = 2, seed: int = 42,
                        min_valid_fraction: float = .8) -> dict[str, Any]:
    """Conditional ranges with frozen q AND original child support classes.

    A draw losing any originally shared child's observations in either period
    is invalid. It is not reclassified as entering/exiting and not zero-filled.
    Calendar zero weeks are included in the input cube. q is never refitted.
    """
    a, b = np.asarray(reference_cube, float), np.asarray(current_cube, float)
    if (a.ndim != 3 or b.ndim != 3 or a.shape[1:] != b.shape[1:] or a.shape[2] != 5
            or not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any()):
        raise ValueError("Bootstrap inputs must be week-by-child-by-five nonnegative cubes")
    if reps < 0 or int(reps) != reps or block_weeks < 1 or int(block_weeks) != block_weeks:
        raise ValueError("Invalid bootstrap repetitions or block length")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    output: dict[str, Any] = {"bootstrap_requested": int(reps), "bootstrap_valid": 0,
                              "bootstrap_invalid": int(reps), "bootstrap_valid_fraction": None,
                              "stability_available": False, "stability_reason": "disabled" if not reps else "not_run",
                              "block_weeks": int(block_weeks)}
    for metric in [f"{part}_{unit}" for part in (*COMPONENTS, "overall") for unit in ("mean", "low")]:
        output[metric + "_lo"] = np.nan; output[metric + "_hi"] = np.nan
    if not reps:
        return output
    if block_weeks > min(len(a), len(b)):
        raise ValueError("block_weeks exceeds the shorter comparison window")
    original = decompose_counts(a.sum(axis=0), b.sum(axis=0))
    common, new, gone, q = (original[k] for k in ("common_mask", "new_mask", "gone_mask", "q"))
    rng = np.random.default_rng(seed)
    da = np.einsum("rw,wck->rck", _block_weights(len(a), block_weeks, reps, rng), a, optimize=True)
    db = np.einsum("rw,wck->rck", _block_weights(len(b), block_weeks, reps, rng), b, optimize=True)
    na, nb = da.sum(axis=2), db.sum(axis=2)
    ta, tb = na.sum(axis=1), nb.sum(axis=1)
    valid = (ta > 0) & (tb > 0) & (na[:, common] > 0).all(axis=1) & (nb[:, common] > 0).all(axis=1)
    nv = int(valid.sum())
    output.update(bootstrap_valid=nv, bootstrap_invalid=int(reps-nv), bootstrap_valid_fraction=nv/reps)
    if nv < max(30, ceil(reps * min_valid_fraction)):
        output["stability_reason"] = "too_many_invalid_draws_or_fewer_than_30"
        return output
    da, db, na, nb, ta, tb = da[valid], db[valid], na[valid], nb[valid], ta[valid], tb[valid]
    wa, wb = na / ta[:, None], nb / tb[:, None]
    pa, pb = da[:, common] / na[:, common, None], db[:, common] / nb[:, common, None]
    within = (((wa[:, common]+wb[:, common])/2)[:, :, None]*(pb-pa)).sum(axis=1)
    composition = ((wb[:, common]-wa[:, common])[:, :, None]*((pa+pb)/2-q)).sum(axis=1)
    turnover = ((db[:, new].sum(axis=1)/tb[:, None]-wb[:, new].sum(axis=1)[:, None]*q)
                -(da[:, gone].sum(axis=1)/ta[:, None]-wa[:, gone].sum(axis=1)[:, None]*q))
    overall = db.sum(axis=1)/tb[:, None]-da.sum(axis=1)/ta[:, None]
    for part, vectors in zip((*COMPONENTS, "overall"), (within, composition, turnover, overall)):
        for unit, values in (("mean", vectors @ STARS), ("low", vectors[:, :2].sum(axis=1))):
            lo, hi = np.quantile(values, [.025, .975])
            output[part + "_" + unit + "_lo"] = float(lo)
            output[part + "_" + unit + "_hi"] = float(hi)
    output.update(stability_available=True, stability_reason="conditional_fixed_q_fixed_support_moving_week_blocks")
    return output


def compare_windows(panel: pd.DataFrame, reference_start: str, current_start: str,
                    current_end: str, *, min_n: int = 30, min_weeks: int = 4,
                    min_common_coverage: float = .5, bootstrap_reps: int = 300,
                    block_weeks: int = 2, seed: int = 42) -> pd.DataFrame:
    """Compare [reference_start,current_start) and [current_start,current_end)."""
    start, middle, end = (_monday(v, k) for v, k in zip(
        (reference_start, current_start, current_end), ("reference_start", "current_start", "current_end")))
    if not start < middle < end:
        raise ValueError("Comparison windows must be ordered and nonempty")
    if min_n < 1 or min_weeks < 1 or not 0 <= min_common_coverage <= 1:
        raise ValueError("Invalid support thresholds")
    if bootstrap_reps and block_weeks > min((middle-start).days//7, (end-middle).days//7):
        raise ValueError("block_weeks exceeds the shorter comparison window")
    frame = validate_panel(panel)
    history = frame[(frame.week >= start) & (frame.week < end)]
    rows = []
    for parent, group in history.groupby("parent_asin", sort=True):
        left, right = group[group.week < middle], group[group.week >= middle]
        children = sorted(group.asin.unique().tolist())
        a, b = _child_counts(left, children), _child_counts(right, children)
        n0, n1 = int(a.sum()), int(b.sum())
        row: dict[str, Any] = {"parent_asin": parent, "reference_start": str(start.date()),
                              "current_start": str(middle.date()), "current_end": str(end.date()),
                              "n_ref": n0, "n_current": n1,
                              "weeks_ref": int(left.week.nunique()), "weeks_current": int(right.week.nunique()),
                              "calendar_weeks_ref": (middle-start).days//7, "calendar_weeks_current": (end-middle).days//7,
                              "children_ref": int((a.sum(axis=1)>0).sum()), "children_current": int((b.sum(axis=1)>0).sum()),
                              "comparable": bool(n0 and n1), "support_eligible": False}
        if not n0 or not n1:
            row.update(status="no_reference_observations" if not n0 else "no_current_observations",
                       stability_available=False, stability_reason="not_comparable", bootstrap_requested=0,
                       bootstrap_valid=0, bootstrap_invalid=0)
            rows.append(row); continue
        d = decompose_counts(a, b)
        for field in ("identity_error", "component_mass_error", "common_coverage_ref", "common_coverage_current"):
            row[field] = d[field]
        row.update(children_common=int(d["common_mask"].sum()), children_newly_observed=int(d["new_mask"].sum()),
                   children_no_longer_observed=int(d["gone_mask"].sum()), mean_ref=float(d["p_ref"]@STARS),
                   mean_current=float(d["p_current"]@STARS), mean_delta=float(d["overall"]@STARS),
                   low_ref=float(d["p_ref"][:2].sum()), low_current=float(d["p_current"][:2].sum()),
                   low_delta=float(d["overall"][:2].sum()))
        for part in ("p_ref", "p_current", "within", "composition", "turnover", "overall"):
            for k, value in enumerate(d[part], 1): row[f"{part}_{k}"] = float(value)
            if part in COMPONENTS:
                row[part+"_mean"] = float(d[part]@STARS)
                row[part+"_low"] = float(d[part][:2].sum())
        reasons = []
        if min(n0, n1) < min_n: reasons.append("insufficient_reviews")
        if min(row["weeks_ref"], row["weeks_current"]) < min_weeks: reasons.append("insufficient_observed_weeks")
        if min(d["common_coverage_ref"], d["common_coverage_current"]) < min_common_coverage:
            reasons.append("insufficient_common_support")
        row["support_eligible"] = not reasons
        row["status"] = "supported" if not reasons else ";".join(reasons)
        if not reasons:
            row.update(bootstrap_stability(_week_cube(left, children, start, middle),
                                           _week_cube(right, children, middle, end), reps=bootstrap_reps,
                                           block_weeks=block_weeks, seed=_seed(parent, seed)))
        else:
            row.update(stability_available=False, stability_reason="support_ineligible", bootstrap_requested=0,
                       bootstrap_valid=0, bootstrap_invalid=0)
        rows.append(row)
    result = _typed_result(pd.DataFrame(rows), COMPARISON_COLUMNS,
                           integers=COMPARISON_INTEGER_COLUMNS,
                           booleans=["comparable", "support_eligible", "stability_available"],
                           strings=["parent_asin", "reference_start", "current_start", "current_end", "status", "stability_reason"])
    result.attrs["interpretation"] = "Exact released-corpus accounting; ranges are conditional stability, not population inference or FDR control."
    return result


def score_distribution(probability: np.ndarray, observed_counts: np.ndarray) -> dict[str, float]:
    """Mean per-review log loss and normalized five-category RPS.

    E[(Fhat_k - I(Y<=k))²] = (Fhat_k-Fobs_k)² + Fobs_k(1-Fobs_k).
    The second term is necessary: a distribution distance is not a proper score.
    """
    p, counts = np.asarray(probability, float), np.asarray(observed_counts, float)
    if (p.shape != (5,) or counts.shape != (5,) or not np.isfinite(p).all() or not np.isfinite(counts).all()
            or (p <= 0).any() or not np.isclose(p.sum(), 1) or (counts < 0).any() or counts.sum() <= 0):
        raise ValueError("Scoring requires positive five-category probabilities and nonempty counts")
    observed = counts / counts.sum()
    fp, fo = np.cumsum(p)[:4], np.cumsum(observed)[:4]
    return {"log_loss": float(-observed @ np.log(p)),
            "rps": float(np.mean((fp-fo)**2 + fo*(1-fo)))}


def evaluate_forecasts(panel: pd.DataFrame, comparison: pd.DataFrame, reference_start: str,
                       current_start: str, current_end: str, future_end: str, *,
                       prior_alpha: float = .5, child_strength: float = 5.) -> pd.DataFrame:
    """Three historical forecasts sharing the same supported parent cohort."""
    dates = [_monday(v, name) for v, name in zip((reference_start,current_start,current_end,future_end),
                                               ("reference_start","current_start","current_end","future_end"))]
    if not all(a < b for a,b in zip(dates, dates[1:])) or prior_alpha <= 0 or child_strength <= 0:
        raise ValueError("Invalid forecast windows or smoothing parameters")
    start, middle, end, future = dates
    frame = validate_panel(panel)
    groups = {str(k):v for k,v in frame[(frame.week>=start)&(frame.week<future)].groupby("parent_asin")}
    methods = ("reference_parent", "current_parent", "current_child_frozen_mix")
    output = []
    for case in comparison.to_dict(orient="records"):
        parent = str(case["parent_asin"])
        group = groups.get(parent, frame.iloc[:0])
        a = group[(group.week>=start)&(group.week<middle)][COUNT_COLUMNS].sum().to_numpy(dtype=float)
        current = group[(group.week>=middle)&(group.week<end)]
        b = current[COUNT_COLUMNS].sum().to_numpy(dtype=float)
        y = group[(group.week>=end)&(group.week<future)][COUNT_COLUMNS].sum().to_numpy(dtype=float)
        eligible = bool(case["support_eligible"]) and a.sum()>0 and b.sum()>0
        forecasts = {}
        if eligible:
            pa=(a+prior_alpha)/(a.sum()+5*prior_alpha)
            pb=(b+prior_alpha)/(b.sum()+5*prior_alpha)
            children=current.groupby("asin")[COUNT_COLUMNS].sum().to_numpy(dtype=float)
            ns=children.sum(axis=1)
            childp=(children+child_strength*pa)/(ns[:,None]+child_strength)
            mix=(ns[:,None]/ns.sum()*childp).sum(axis=0)
            forecasts=dict(zip(methods,(pa,pb,mix)))
        for method in methods:
            row={"parent_asin":parent,"method":method,"eligible":eligible,"n_future":int(y.sum()),
                 "observable":bool(eligible and y.sum()>0),"future_start":str(end.date()),"future_end":str(future.date()),
                 "status":"insufficient_history" if not eligible else ("observed" if y.sum()>0 else "unknown_future"),
                 "log_loss":np.nan,"rps":np.nan}
            if eligible:
                pred=forecasts[method]
                row.update({f"pred_{k}":float(v) for k,v in enumerate(pred,1)})
                if y.sum()>0:row.update(score_distribution(pred,y))
            output.append(row)
    result=_typed_result(pd.DataFrame(output), FORECAST_COLUMNS, integers=["n_future"],
                         booleans=["eligible", "observable"],
                         strings=["parent_asin", "method", "future_start", "future_end", "status"])
    result.attrs["interpretation"]="Held-out released-review probabilities, not diagnosis accuracy or a deployed contemporaneous forecast."
    return result


def summarize_forecasts(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for method,group in evaluation.groupby("method",sort=True):
        eligible=group[group.eligible.astype(bool)];observed=eligible[eligible.observable.astype(bool)]
        row={"method":method,"candidate_parents":len(group),"eligible_parents":len(eligible),
             "observable_parents":len(observed),"unknown_future_parents":len(eligible)-len(observed),
             "observed_future_reviews":int(observed.n_future.sum()),
             "observable_fraction":len(observed)/len(eligible) if len(eligible) else np.nan}
        for metric in ("log_loss","rps"):
            row["macro_"+metric]=float(observed[metric].mean()) if len(observed) else np.nan
            row["review_weighted_"+metric]=float(np.average(observed[metric],weights=observed.n_future)) if len(observed) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def synthetic_panel(seed: int = 42) -> pd.DataFrame:
    """Explicit simulated weekly panel; generated counts are not raw scans."""
    rng=np.random.default_rng(seed);rows=[]
    dates=pd.date_range("2021-10-11",periods=28,freq="W-MON")
    for parent_index in range(18):
        for week_index,week in enumerate(dates):
            for child_index in range(3):
                if parent_index==17 and week_index>=24:continue
                if child_index==2 and ((parent_index%3==2 and week_index<12) or (parent_index%3==1 and week_index>=12)):continue
                n=int(rng.integers(8,25))
                p=np.array([.03,.07,.1,.2,.6]) if child_index==0 else np.array([.18,.15,.2,.22,.25])
                if week_index>=12 and parent_index%3==0:
                    n=n*3 if child_index else n
                if week_index>=12 and parent_index%3==1:
                    p=np.array([.25,.2,.2,.2,.15])
                counts=rng.multinomial(n,p)
                rows.append({"parent_asin":f"SIM_PARENT_{parent_index:02}","asin":f"SIM_{parent_index:02}_{child_index}",
                             "week":week,**dict(zip(COUNT_COLUMNS,counts)),"n":n})
    result=pd.DataFrame(rows)
    result.attrs["source_kind"]="SYNTHETIC_NOT_REAL_AMAZON"
    return result


def user_cluster_stability(clean_parent: pd.DataFrame, reference_start: str, current_start: str,
                           current_end: str, *, reps: int = 300, seed: int = 42,
                           min_valid_fraction: float = .8, max_users: int = 100000,
                           max_children: int = 1000) -> dict[str, Any]:
    """Resample hashed reviewer clusters JOINTLY across both periods.

    Accepts CLEAN individual records for exactly one parent, not a weekly panel.
    A hash is an identifier transformation, not a privacy/anonymity guarantee.
    Large cases are explicitly skipped, never silently subsampled. Ranges do
    not remove user-selection bias or account for cross-reviewer temporal ties.
    """
    from scipy.sparse import coo_matrix
    required={"user_hash","parent_asin","asin","week","rating"}
    if required-set(clean_parent.columns):raise ValueError(f"Missing clean columns: {sorted(required-set(clean_parent.columns))}")
    start,middle,end=[_monday(x,n) for x,n in zip((reference_start,current_start,current_end),
                                                ("reference_start","current_start","current_end"))]
    if not start<middle<end or reps<1 or int(reps)!=reps or not 0<min_valid_fraction<=1:
        raise ValueError("Invalid user bootstrap windows or parameters")
    data=clean_parent[list(required)].copy()
    data["week"]=pd.to_datetime(data.week)
    data=data[(data.week>=start)&(data.week<end)]
    if data.parent_asin.nunique()!=1:raise ValueError("User bootstrap requires exactly one parent with observations")
    if data[["user_hash","asin"]].isna().any().any():raise ValueError("Missing clean identifiers")
    if not data.rating.isin([1,2,3,4,5]).all():raise ValueError("Clean ratings must be integers 1..5")
    users=sorted(data.user_hash.astype(str).unique());children=sorted(data.asin.astype(str).unique())
    output={"resampling_unit":"observed_user_hash_joint_across_periods","bootstrap_requested":int(reps),
            "bootstrap_valid":0,"bootstrap_invalid":int(reps),"bootstrap_valid_fraction":None,
            "stability_available":False,"stability_reason":"not_run","user_count":len(users),
            "children_count":len(children),"clean_review_count":len(data)}
    for metric in [f"{part}_{unit}" for part in (*COMPONENTS,"overall") for unit in ("mean","low")]:
        output[metric+"_lo"]=np.nan;output[metric+"_hi"]=np.nan
    if len(users)>max_users or len(children)>max_children:
        output["stability_reason"]="resource_guard_case_not_resampled";return output
    user_index={u:i for i,u in enumerate(users)};child_index={c:i for i,c in enumerate(children)}
    ui=data.user_hash.astype(str).map(user_index).to_numpy()
    ci=data.asin.astype(str).map(child_index).to_numpy()*5+data.rating.to_numpy(dtype=int)-1
    ref=(data.week<middle).to_numpy();matrices=[]
    for mask in (ref,~ref):
        matrices.append(coo_matrix((np.ones(int(mask.sum())),(ui[mask],ci[mask])),
                                   shape=(len(users),len(children)*5)).tocsr())
    original_counts=[np.asarray(m.sum(axis=0)).reshape(len(children),5) for m in matrices]
    if min(x.sum() for x in original_counts)<=0:
        output["stability_reason"]="not_comparable";return output
    original=decompose_counts(*original_counts);common=original["common_mask"]
    rng=np.random.default_rng(_seed(str(data.parent_asin.iloc[0]),seed));samples=[]
    for offset in range(0,reps,20):
        nr=min(20,reps-offset)
        weights=rng.multinomial(len(users),np.full(len(users),1/len(users)),size=nr)
        draws=[np.asarray(weights@m).reshape(nr,len(children),5) for m in matrices]
        for a,b in zip(*draws):
            if a.sum()<=0 or b.sum()<=0 or (a.sum(axis=1)[common]==0).any() or (b.sum(axis=1)[common]==0).any():continue
            d=decompose_counts(a,b,q=original["q"])
            samples.append({f"{part}_mean":float(d[part]@STARS) for part in (*COMPONENTS,"overall")}
                           |{f"{part}_low":float(d[part][:2].sum()) for part in (*COMPONENTS,"overall")})
    valid=len(samples);output.update(bootstrap_valid=valid,bootstrap_invalid=int(reps-valid),bootstrap_valid_fraction=valid/reps)
    if valid<max(30,ceil(reps*min_valid_fraction)):
        output["stability_reason"]="too_many_invalid_draws_or_fewer_than_30";return output
    for metric in samples[0]:
        low,high=np.quantile([s[metric] for s in samples],[.025,.975])
        output[metric+"_lo"]=float(low);output[metric+"_hi"]=float(high)
    output.update(stability_available=True,stability_reason="conditional_fixed_q_fixed_support_joint_user_clusters")
    return output
