"""
loo_crawford.py
===============
Per-mouse, per-miRNA significance using the Crawford & Howell (1998)
leave-one-out t-test on log2 fold changes.

Inputs
------
  Pg_8_weeks.csv            : raw pre-treatment counts (10 mice x 599 miRNAs)
  Pg_16_weeks.csv           : raw post-treatment counts (10 mice x 599 miRNAs)
  deseq2_normalized_counts.csv : DESeq2 size-factor normalized counts
                                 (output of deseq2_analysis.R)
                                 If not available, raw counts are used instead.

Outputs
-------
  group_wilcoxon_results.csv  : paired Wilcoxon per miRNA (fallback if no DESeq2)
  individual_loo_results.csv  : per-mouse x per-miRNA LOO t-test results

Usage
-----
  python loo_crawford.py

Dependencies: numpy, pandas, scipy
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================
PRE_FILE        = "Pg_8_weeks.csv"
POST_FILE       = "Pg_16_weeks.csv"
NORM_FILE       = "deseq2_normalized_counts.csv"   # optional
FDR_ALPHA       = 0.05
PSEUDOCOUNT     = 1.0   # added before log2 to avoid log(0)

# =============================================================================
# Helpers
# =============================================================================

def bh_correction(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Handles NaN by setting q=NaN."""
    n = len(pvals)
    qvals = np.full(n, np.nan)
    finite = ~np.isnan(pvals)
    p = pvals[finite]
    m = len(p)
    order = np.argsort(p)
    ranked = np.empty(m)
    ranked[order] = np.arange(1, m + 1)
    q = p * m / ranked
    # enforce monotonicity (step-down)
    q_sorted = q[order]
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    qvals[finite] = np.minimum(q, 1.0)
    return qvals


def load_counts(pre_file: str, post_file: str):
    pre  = pd.read_csv(pre_file).drop(columns="infected", errors="ignore")
    post = pd.read_csv(post_file).drop(columns="infected", errors="ignore")
    assert pre.shape == post.shape, "pre/post files must have the same shape"
    assert list(pre.columns) == list(post.columns), "Column mismatch between pre/post"
    return pre, post


# =============================================================================
# 1. Group-level: Paired Wilcoxon signed-rank test
#    (Fallback when DESeq2 results are not available.
#     Note: with n=10 and 599 tests, BH-corrected q-values rarely reach 0.05.
#     Treat nominal p < 0.05 as exploratory; q < 0.05 as confirmatory.)
# =============================================================================

def group_wilcoxon(pre_arr: np.ndarray, post_arr: np.ndarray,
                   mirna_cols: list) -> pd.DataFrame:
    n_mirnas = pre_arr.shape[1]
    pvals    = np.ones(n_mirnas)
    stats_w  = np.zeros(n_mirnas)

    for j in range(n_mirnas):
        d = post_arr[:, j] - pre_arr[:, j]
        if (d == 0).all():
            pvals[j] = 1.0
            continue
        try:
            w, p = stats.wilcoxon(d, alternative="two-sided")
            stats_w[j] = w
            pvals[j]   = p
        except Exception:
            pvals[j] = 1.0

    qvals      = bh_correction(pvals)
    lfc        = np.log2((post_arr + PSEUDOCOUNT) / (pre_arr + PSEUDOCOUNT))
    median_lfc = np.median(lfc, axis=0)

    df = pd.DataFrame({
        "miRNA":          mirna_cols,
        "median_log2FC":  np.round(median_lfc, 4),
        "W_statistic":    stats_w,
        "p_value":        np.round(pvals, 6),
        "q_value_BH":     np.round(qvals, 6),
        "sig_nominal":    pvals  < FDR_ALPHA,   # p < 0.05, no correction
        "sig_FDR":        qvals  < FDR_ALPHA,   # q < 0.05 after BH
    }).sort_values("p_value")

    return df


# =============================================================================
# 2. Individual-level: LOO Crawford & Howell (1998) t-test
#    For each mouse i and miRNA j:
#      - compute LFC for all mice
#      - leave mouse i out to form a reference distribution (n=9)
#      - apply the Crawford & Howell SE correction: SE = SD * sqrt(1 + 1/9)
#      - t-test with df = 8
#    Multiple testing: BH FDR across all (n_mice * n_mirnas) tests.
# =============================================================================

def loo_crawford(lfc: np.ndarray, mirna_cols: list) -> pd.DataFrame:
    n_mice, n_mirnas = lfc.shape
    pvals = np.ones((n_mice, n_mirnas))

    for i in range(n_mice):
        others = np.delete(lfc, i, axis=0)        # (n_mice-1, n_mirnas)
        mu     = others.mean(axis=0)
        sd     = others.std(axis=0, ddof=1)

        nonzero = sd > 0
        se      = np.where(nonzero, sd * np.sqrt(1.0 + 1.0 / (n_mice - 1)), np.inf)
        t_stat  = np.where(nonzero, (lfc[i] - mu) / se, 0.0)
        p       = np.where(nonzero,
                           2 * stats.t.sf(np.abs(t_stat), df=n_mice - 2),
                           1.0)
        pvals[i] = p

    flat_p = pvals.flatten()
    flat_q = bh_correction(flat_p)
    qvals  = flat_q.reshape(n_mice, n_mirnas)

    rows = []
    for i in range(n_mice):
        for j, mname in enumerate(mirna_cols):
            rows.append({
                "mouse":       i + 1,
                "miRNA":       mname,
                "log2FC":      round(lfc[i, j], 4),
                "p_value":     round(pvals[i, j], 6),
                "q_value_BH":  round(qvals[i, j], 6),
                "sig_nominal": pvals[i, j] < FDR_ALPHA,
                "sig_FDR":     qvals[i, j] < FDR_ALPHA,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading count data...")
    pre, post = load_counts(PRE_FILE, POST_FILE)
    mirna_cols = list(pre.columns)
    pre_arr    = pre.values.astype(float)
    post_arr   = post.values.astype(float)
    n_mice     = pre_arr.shape[0]

    # ---- Decide which counts to use for LFC ---------------------------------
    if Path(NORM_FILE).exists():
        print(f"Found {NORM_FILE} — using DESeq2 normalized counts for LOO test.")
        norm = pd.read_csv(NORM_FILE, index_col=0)   # rows=miRNAs, cols=samples
        # Expected column order: pre samples (cols 0..n-1), post (cols n..2n-1)
        pre_norm  = norm.iloc[:, :n_mice].values.T    # (n_mice, n_mirnas)
        post_norm = norm.iloc[:, n_mice:].values.T
        # Align to mirna_cols order (DESeq2 may drop low-count miRNAs)
        retained  = [c for c in mirna_cols if c in norm.index]
        idx       = [mirna_cols.index(c) for c in retained]
        pre_arr_n  = pre_arr[:, idx]
        post_arr_n = post_arr[:, idx]
        norm_idx  = [list(norm.index).index(c) for c in retained]
        pre_norm   = norm.iloc[norm_idx, :n_mice].values.T
        post_norm  = norm.iloc[norm_idx, n_mice:].values.T
        lfc        = np.log2((post_norm + PSEUDOCOUNT) / (pre_norm + PSEUDOCOUNT))
        lfc_cols   = retained
        print(f"  miRNAs available after DESeq2 filtering: {len(retained)}")
    else:
        print(f"{NORM_FILE} not found — using raw counts for LOO test.")
        print("  (Run deseq2_analysis.R first for better normalization.)")
        lfc      = np.log2((post_arr + PSEUDOCOUNT) / (pre_arr + PSEUDOCOUNT))
        lfc_cols = mirna_cols

    # ---- Group-level Wilcoxon -----------------------------------------------
    print("\nRunning group-level paired Wilcoxon signed-rank test...")
    grp_df = group_wilcoxon(pre_arr, post_arr, mirna_cols)
    grp_df.to_csv("group_wilcoxon_results.csv", index=False)
    print(f"  p < 0.05 (nominal): {grp_df['sig_nominal'].sum()}")
    print(f"  q < 0.05 (BH FDR):  {grp_df['sig_FDR'].sum()}")
    print("  Saved: group_wilcoxon_results.csv")

    # ---- Individual-level LOO Crawford --------------------------------------
    print("\nRunning per-mouse LOO Crawford & Howell t-test...")
    ind_df = loo_crawford(lfc, lfc_cols)
    ind_df.to_csv("individual_loo_results.csv", index=False)
    sig    = ind_df[ind_df["sig_nominal"]]
    print(f"  Significant hits p < 0.05 (nominal): {len(sig)}")
    print(f"  Significant hits q < 0.05 (BH FDR):  {ind_df['sig_FDR'].sum()}")
    if len(sig):
        print("\n  Nominal hits per mouse:")
        print(sig.groupby("mouse").size().to_string())
    print("  Saved: individual_loo_results.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
