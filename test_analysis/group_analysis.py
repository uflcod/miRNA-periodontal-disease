"""
group_analysis.py
=================
Group-level per-miRNA significance testing for paired pre/post count data.

Usage
-----
  python group_analysis.py --method ttest      --pre Pg_8_weeks.csv --post Pg_16_weeks.csv
  python group_analysis.py --method wilcoxon   --pre Pg_8_weeks.csv --post Pg_16_weeks.csv
  python group_analysis.py --method permutation --pre Pg_8_weeks.csv --post Pg_16_weeks.csv

Options
-------
  --method      ttest | wilcoxon | permutation  (required)
  --pre         CSV file of pre-treatment counts (default: Pg_8_weeks.csv)
  --post        CSV file of post-treatment counts (default: Pg_16_weeks.csv)
  --drop-cols   Comma-separated columns to drop, e.g. infected (default: infected)
  --alpha       Significance threshold (default: 0.05)
  --n-perm      Number of permutations for permutation test (default: 10000)
  --seed        Random seed for permutation test (default: 42)

Output
------
  group_results_{method}.csv  — one row per miRNA, sorted by p-value, with:
    miRNA           : miRNA identifier
    median_log2FC   : median log2 fold change across mice
    mean_log2FC     : mean log2 fold change across mice
    statistic       : test statistic (t / W / observed_mean_lfc)
    p_value         : raw p-value
    q_value_BH      : Benjamini-Hochberg FDR-corrected p-value
    sig_nominal     : p_value < alpha
    sig_FDR         : q_value_BH < alpha

Methods
-------
  ttest       Paired Student's t-test on log2(post+1) - log2(pre+1) differences.
              Assumes differences are approximately normally distributed.
              Reasonable when counts are large; use wilcoxon otherwise.

  wilcoxon    Paired Wilcoxon signed-rank test on raw count differences.
              Non-parametric; no normality assumption.
              Minimum achievable p-value with n=10 is ~0.002.

  permutation Permutation test on mean log2FC.
              Builds empirical null by randomly flipping pre/post labels
              independently per mouse. No distributional assumptions.
              Minimum achievable p-value = 1/2^n_mice (~0.001 for n=10).
              Slower than the other methods; use --n-perm to control precision.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# BH FDR correction
# =============================================================================

def bh_correction(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    qvals = np.full(n, np.nan)
    finite = ~np.isnan(pvals)
    p = pvals[finite]
    m = len(p)
    if m == 0:
        return qvals
    order = np.argsort(p)
    ranked = np.empty(m)
    ranked[order] = np.arange(1, m + 1)
    q = p * m / ranked
    q_sorted = q[order]
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    qvals[finite] = np.minimum(q, 1.0)
    return qvals


# =============================================================================
# Data loading
# =============================================================================

def load_data(pre_file: str, post_file: str, drop_cols: list[str]):
    pre  = pd.read_csv(pre_file).drop(columns=drop_cols, errors="ignore")
    post = pd.read_csv(post_file).drop(columns=drop_cols, errors="ignore")

    if pre.shape != post.shape:
        sys.exit(f"Error: pre {pre.shape} and post {post.shape} have different shapes.")
    if list(pre.columns) != list(post.columns):
        sys.exit("Error: pre and post files have different column names.")

    return (
        pre.values.astype(float),
        post.values.astype(float),
        list(pre.columns),
    )


# =============================================================================
# Summary statistics shared across methods
# =============================================================================

def lfc_summaries(pre_arr: np.ndarray, post_arr: np.ndarray) -> tuple:
    lfc = np.log2((post_arr + 1) / (pre_arr + 1))   # (n_mice, n_mirnas)
    return lfc, np.median(lfc, axis=0), lfc.mean(axis=0)


# =============================================================================
# Method: Paired t-test
# =============================================================================

def run_ttest(pre_arr: np.ndarray, post_arr: np.ndarray,
              mirna_cols: list, alpha: float) -> pd.DataFrame:
    lfc, median_lfc, mean_lfc = lfc_summaries(pre_arr, post_arr)
    n_mirnas = pre_arr.shape[1]

    t_stats = np.zeros(n_mirnas)
    pvals   = np.ones(n_mirnas)

    for j in range(n_mirnas):
        d = lfc[:, j]
        if d.std(ddof=1) == 0:
            continue
        t, p = stats.ttest_1samp(d, popmean=0)
        t_stats[j] = t
        pvals[j]   = p

    qvals = bh_correction(pvals)

    return pd.DataFrame({
        "miRNA":          mirna_cols,
        "median_log2FC":  np.round(median_lfc, 4),
        "mean_log2FC":    np.round(mean_lfc, 4),
        "statistic":      np.round(t_stats, 4),
        "p_value":        np.round(pvals, 6),
        "q_value_BH":     np.round(qvals, 6),
        "sig_nominal":    pvals < alpha,
        "sig_FDR":        qvals < alpha,
    }).sort_values("p_value")


# =============================================================================
# Method: Paired Wilcoxon signed-rank test
# =============================================================================

def run_wilcoxon(pre_arr: np.ndarray, post_arr: np.ndarray,
                 mirna_cols: list, alpha: float) -> pd.DataFrame:
    lfc, median_lfc, mean_lfc = lfc_summaries(pre_arr, post_arr)
    n_mirnas = pre_arr.shape[1]

    w_stats = np.zeros(n_mirnas)
    pvals   = np.ones(n_mirnas)

    for j in range(n_mirnas):
        d = post_arr[:, j] - pre_arr[:, j]
        if (d == 0).all():
            continue
        try:
            w, p = stats.wilcoxon(d, alternative="two-sided")
            w_stats[j] = w
            pvals[j]   = p
        except Exception:
            pass

    qvals = bh_correction(pvals)

    return pd.DataFrame({
        "miRNA":          mirna_cols,
        "median_log2FC":  np.round(median_lfc, 4),
        "mean_log2FC":    np.round(mean_lfc, 4),
        "statistic":      np.round(w_stats, 4),
        "p_value":        np.round(pvals, 6),
        "q_value_BH":     np.round(qvals, 6),
        "sig_nominal":    pvals < alpha,
        "sig_FDR":        qvals < alpha,
    }).sort_values("p_value")


# =============================================================================
# Method: Permutation test
# =============================================================================

def run_permutation(pre_arr: np.ndarray, post_arr: np.ndarray,
                    mirna_cols: list, alpha: float,
                    n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lfc, median_lfc, mean_lfc = lfc_summaries(pre_arr, post_arr)

    n_mice, n_mirnas = lfc.shape
    observed = mean_lfc                                  # (n_mirnas,)

    # Pre-generate all sign-flip matrices: (n_perm, n_mice)
    # Each row is one permutation; each column is +1 or -1 for one mouse.
    # Flipping the sign of lfc[i] is equivalent to swapping pre/post for mouse i.
    print(f"  Running {n_perm:,} permutations across {n_mirnas} miRNAs...")
    sign_flips = rng.choice([-1, 1], size=(n_perm, n_mice))  # (n_perm, n_mice)

    # Vectorised: perm_means[k, j] = mean over mice of (flip[k, i] * lfc[i, j])
    # sign_flips: (n_perm, n_mice)  @  lfc: (n_mice, n_mirnas)  →  (n_perm, n_mirnas)
    perm_means = (sign_flips @ lfc) / n_mice             # (n_perm, n_mirnas)

    # Two-sided p-value: fraction of permutations at least as extreme as observed
    pvals = (np.abs(perm_means) >= np.abs(observed)).mean(axis=0)

    # Minimum achievable p-value = 1/n_perm; avoid exact zeros
    pvals = np.maximum(pvals, 1.0 / n_perm)

    qvals = bh_correction(pvals)

    return pd.DataFrame({
        "miRNA":          mirna_cols,
        "median_log2FC":  np.round(median_lfc, 4),
        "mean_log2FC":    np.round(mean_lfc, 4),
        "statistic":      np.round(observed, 4),   # observed mean LFC
        "p_value":        np.round(pvals, 6),
        "q_value_BH":     np.round(qvals, 6),
        "sig_nominal":    pvals < alpha,
        "sig_FDR":        qvals < alpha,
    }).sort_values("p_value")


# =============================================================================
# Main
# =============================================================================

METHODS = {
    "ttest":       run_ttest,
    "wilcoxon":    run_wilcoxon,
    "permutation": run_permutation,
}

def main():
    parser = argparse.ArgumentParser(
        description="Group-level per-miRNA significance testing for paired pre/post count data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", choices=METHODS.keys(), required=True,
                        help="Statistical test to use.")
    parser.add_argument("--pre",  default="Pg_8_weeks.csv",
                        help="Pre-treatment count CSV (default: Pg_8_weeks.csv)")
    parser.add_argument("--post", default="Pg_16_weeks.csv",
                        help="Post-treatment count CSV (default: Pg_16_weeks.csv)")
    parser.add_argument("--drop-cols", default="infected",
                        help="Comma-separated non-miRNA columns to drop (default: infected)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold (default: 0.05)")
    parser.add_argument("--n-perm", type=int, default=10000,
                        help="Number of permutations (permutation method only, default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (permutation method only, default: 42)")
    parser.add_argument("--output", default=None,
                        help="Output CSV filename (default: group_results_{method}.csv)")
    args = parser.parse_args()

    drop_cols = [c.strip() for c in args.drop_cols.split(",")]

    print(f"Method : {args.method.upper()}")
    print(f"Pre    : {args.pre}")
    print(f"Post   : {args.post}")
    print(f"Alpha  : {args.alpha}")

    pre_arr, post_arr, mirna_cols = load_data(args.pre, args.post, drop_cols)
    n_mice, n_mirnas = pre_arr.shape
    print(f"Mice   : {n_mice}")
    print(f"miRNAs : {n_mirnas}\n")

    if args.method == "permutation":
        out = run_permutation(pre_arr, post_arr, mirna_cols,
                              args.alpha, args.n_perm, args.seed)
    else:
        out = METHODS[args.method](pre_arr, post_arr, mirna_cols, args.alpha)

    outfile = args.output if args.output else f"group_results_{args.method}.csv"
    out.to_csv(outfile, index=False)

    print(f"Significant miRNAs p < {args.alpha} (nominal): {out['sig_nominal'].sum()}")
    print(f"Significant miRNAs q < {args.alpha} (BH FDR):  {out['sig_FDR'].sum()}")
    print(f"\nTop 10 miRNAs:")
    print(out.head(10).to_string(index=False))
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
