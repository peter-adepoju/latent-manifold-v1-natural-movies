from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind

def bootstrap_ci(x, n_bootstrap: int = 2000, ci: float = 95, rng=None, func=np.nanmean):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    boots = [func(rng.choice(x, size=len(x), replace=True)) for _ in range(int(n_bootstrap))]
    alpha = (100 - ci) / 2
    return {
        "estimate": float(func(x)),
        "ci_low": float(np.nanpercentile(boots, alpha)),
        "ci_high": float(np.nanpercentile(boots, 100 - alpha)),
        "n": int(len(x)),
    }

def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    p = pvals[finite]
    if len(p) == 0:
        return qvals
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty_like(q)
    out[order] = q
    qvals[finite] = out
    return qvals

def paired_effect_table(df: pd.DataFrame, group_col: str, value_col: str, condition_col: str, condition_a: str, condition_b: str) -> pd.DataFrame:
    wide = df.pivot_table(index=group_col, columns=condition_col, values=value_col, aggfunc="mean")
    needed = wide[[condition_a, condition_b]].dropna()
    if needed.empty:
        return pd.DataFrame([{"n_pairs": 0, "mean_delta": np.nan, "t": np.nan, "p": np.nan}])
    delta = needed[condition_a] - needed[condition_b]
    t, p = ttest_rel(needed[condition_a], needed[condition_b]) if len(needed) > 1 else (np.nan, np.nan)
    ci = bootstrap_ci(delta)
    return pd.DataFrame([{**ci, "n_pairs": len(needed), "condition_a": condition_a, "condition_b": condition_b, "mean_delta": float(np.mean(delta)), "t": float(t), "p": float(p)}])

def optional_mixedlm(formula: str, data: pd.DataFrame, group_col: str):
    try:
        import statsmodels.formula.api as smf
        model = smf.mixedlm(formula, data=data, groups=data[group_col])
        return model.fit(reml=False)
    except Exception as exc:
        raise RuntimeError(f"Mixed-effects model failed: {exc}") from exc

def bootstrap_ci_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    n_bootstrap: int = 2000,
    rng=None,
    func=np.nanmean,
) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "value_col": value_col,
            **bootstrap_ci(sub[value_col], n_bootstrap=n_bootstrap, rng=rng, func=func),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def fdr_table(df: pd.DataFrame, p_col: str = "p", q_col: str = "q_fdr") -> pd.DataFrame:
    out = df.copy()
    out[q_col] = benjamini_hochberg(pd.to_numeric(out[p_col], errors="coerce"))
    return out

def mixedlm_summary_table(formula: str, data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    result = optional_mixedlm(formula, data=data, group_col=group_col)
    rows = []
    for name, estimate in result.params.items():
        rows.append({
            "term": name,
            "estimate": float(estimate),
            "std_error": float(result.bse.get(name, np.nan)),
            "z": float(result.tvalues.get(name, np.nan)),
            "p": float(result.pvalues.get(name, np.nan)),
        })
    return fdr_table(pd.DataFrame(rows), p_col="p", q_col="q_fdr")
