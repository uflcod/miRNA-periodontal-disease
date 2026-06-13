"""
interaction_analysis.py
=======================
2x2 repeated measures miRNA analysis for paired infected vs uninfected mice
measured at two timepoints (8 weeks and 16 weeks).

Tests per miRNA
---------------
  1. Interaction  : Is the 8->16 week change different between infected and
                    uninfected mice? (difference of log2FCs, unpaired Wilcoxon
                    rank-sum or permutation test between groups)
  2. Infected     : Does miRNA change 8->16 weeks within infected mice?
                    (paired Wilcoxon signed-rank or paired t-test or permutation)
  3. Uninfected   : Does miRNA change 8->16 weeks within uninfected mice?
                    (same as above)

Usage
-----
  python interaction_analysis.py --method wilcoxon
  python interaction_analysis.py --method ttest
  python interaction_analysis.py --method permutation --n-perm 10000

  # Custom file paths:
  python interaction_analysis.py --method wilcoxon \
      --inf-pre  Pg_8_weeks.csv \
      --inf-post Pg_16_weeks.csv \
      --ctrl-pre  Control_8_weeks.csv \
      --ctrl-post Control_16_weeks.csv

  # Custom output file:
  python interaction_analysis.py --method wilcoxon --output my_results.csv

Options
-------
  --method      ttest | wilcoxon | permutation  (required)
  --inf-pre     Infected pre-treatment CSV   (default: Pg_8_weeks.csv)
  --inf-post    Infected post-treatment CSV  (default: Pg_16_weeks.csv)
  --ctrl-pre    Uninfected pre-treatment CSV (default: Control_8_weeks.csv)
  --ctrl-post   Uninfected post-treatment CSV(default: Control_16_weeks.csv)
  --drop-cols   Comma-separated non-miRNA columns to drop (default: infected)
  --alpha       Significance threshold (default: 0.05)
  --n-perm      Permutations for permutation test (default: 10000)
  --seed        Random seed for permutation test (default: 42)
  --output      Output CSV filename (default: interaction_results_{method}.csv)

Output columns
--------------
  miRNA
  inf_median_log2FC       median LFC across infected mice
  ctrl_median_log2FC      median LFC across uninfected mice
  delta_median_log2FC     inf_median_log2FC - ctrl_median_log2FC (interaction effect size)
  interaction_stat        test statistic for the interaction test
  interaction_p           raw p-value, interaction
  interaction_q           BH-corrected p-value, interaction
  interaction_sig_nominal p < alpha
  interaction_sig_FDR     q < alpha
  infected_stat           test statistic, within-infected test
  infected_p              raw p-value, within-infected
  infected_q              BH-corrected p-value, within-infected
  infected_sig_nominal    p < alpha
  infected_sig_FDR        q < alpha
  ctrl_stat               test statistic, within-uninfected test
  ctrl_p                  raw p-value, within-uninfected
  ctrl_q                  BH-corrected p-value, within-uninfected
  ctrl_sig_nominal        p < alpha
  ctrl_sig_FDR            q < alpha
  pattern                 summary label (see interpretation table below)

Pattern labels
--------------
  infection_specific   interaction sig, uninfected not sig
  age_related          infected and uninfected both sig, interaction not sig
  suppressed           interaction sig, infected not sig, uninfected not sig
  weak_infection       infected sig, interaction not sig, uninfected not sig
  no_change            nothing significant
  other                any other combination
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats


PSEUDOCOUNT = 1.0


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

def load_counts(filepath: str, drop_cols: list[str]) -> np.ndarray:
    df = pd.read_csv(filepath).drop(columns=drop_cols, errors="ignore")
    return df.values.astype(float), list(df.columns)


def load_all(inf_pre: str, inf_post: str,
             ctrl_pre: str, ctrl_post: str,
             drop_cols: list[str]):
    ip_arr, mirna_cols = load_counts(inf_pre,   drop_cols)
    io_arr, cols2      = load_counts(inf_post,  drop_cols)
    cp_arr, cols3      = load_counts(ctrl_pre,  drop_cols)
    co_arr, cols4      = load_counts(ctrl_post, drop_cols)

    for cols, label in [(cols2, inf_post), (cols3, ctrl_pre), (cols4, ctrl_post)]:
        if cols != mirna_cols:
            sys.exit(f"Error: column mismatch between {inf_pre} and {label}")

    shapes = {inf_pre: ip_arr.shape, inf_post: io_arr.shape,
              ctrl_pre: cp_arr.shape, ctrl_post: co_arr.shape}
    if len(set(shapes.values())) != 1:
        sys.exit(f"Error: shape mismatch across files: {shapes}")

    return ip_arr, io_arr, cp_arr, co_arr, mirna_cols


# =============================================================================
# LFC computation
# =============================================================================

def compute_lfc(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    return np.log2((post + PSEUDOCOUNT) / (pre + PSEUDOCOUNT))


# =============================================================================
# Within-group paired tests (same logic as group_analysis.py)
# =============================================================================

def paired_ttest(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lfc = compute_lfc(pre, post)
    n_mirnas = lfc.shape[1]
    t_stats = np.zeros(n_mirnas)
    pvals   = np.ones(n_mirnas)
    for j in range(n_mirnas):
        d = lfc[:, j]
        if d.std(ddof=1) == 0:
            continue
        t, p = stats.ttest_1samp(d, popmean=0)
        t_stats[j] = t
        pvals[j]   = p
    return t_stats, pvals


def paired_wilcoxon(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_mirnas = pre.shape[1]
    w_stats = np.zeros(n_mirnas)
    pvals   = np.ones(n_mirnas)
    for j in range(n_mirnas):
        d = post[:, j] - pre[:, j]
        if (d == 0).all():
            continue
        try:
            w, p = stats.wilcoxon(d, alternative="two-sided")
            w_stats[j] = w
            pvals[j]   = p
        except Exception:
            pass
    return w_stats, pvals


def paired_permutation(pre: np.ndarray, post: np.ndarray,
                       n_perm: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lfc = compute_lfc(pre, post)
    n_mice, n_mirnas = lfc.shape
    observed   = lfc.mean(axis=0)
    sign_flips = rng.choice([-1, 1], size=(n_perm, n_mice))
    perm_means = (sign_flips @ lfc) / n_mice
    pvals      = (np.abs(perm_means) >= np.abs(observed)).mean(axis=0)
    pvals      = np.maximum(pvals, 1.0 / n_perm)
    return observed, pvals


# =============================================================================
# Interaction test
# Two independent groups of per-mouse LFCs:
#   inf_lfc  (n_inf_mice,  n_mirnas)
#   ctrl_lfc (n_ctrl_mice, n_mirnas)
# Test whether the mean LFC differs between groups.
# =============================================================================

def interaction_wilcoxon(inf_lfc: np.ndarray,
                         ctrl_lfc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpaired Wilcoxon rank-sum (Mann-Whitney U) on per-mouse LFCs."""
    n_mirnas = inf_lfc.shape[1]
    u_stats  = np.zeros(n_mirnas)
    pvals    = np.ones(n_mirnas)
    for j in range(n_mirnas):
        a = inf_lfc[:, j]
        b = ctrl_lfc[:, j]
        if np.all(a == b[0]):          # degenerate case
            continue
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            u_stats[j] = u
            pvals[j]   = p
        except Exception:
            pass
    return u_stats, pvals


def interaction_ttest(inf_lfc: np.ndarray,
                      ctrl_lfc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpaired Welch t-test on per-mouse LFCs."""
    n_mirnas = inf_lfc.shape[1]
    t_stats  = np.zeros(n_mirnas)
    pvals    = np.ones(n_mirnas)
    for j in range(n_mirnas):
        a = inf_lfc[:, j]
        b = ctrl_lfc[:, j]
        if a.std(ddof=1) == 0 and b.std(ddof=1) == 0:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        t_stats[j] = t
        pvals[j]   = p
    return t_stats, pvals


def interaction_permutation(inf_lfc: np.ndarray, ctrl_lfc: np.ndarray,
                            n_perm: int,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Permutation test on difference of group mean LFCs.
    Under H0, group labels (infected/uninfected) are exchangeable.
    Each permutation randomly reassigns mice to groups and recomputes
    the difference of means.
    """
    n_inf    = inf_lfc.shape[0]
    combined = np.vstack([inf_lfc, ctrl_lfc])     # (n_inf + n_ctrl, n_mirnas)
    n_total  = combined.shape[0]
    observed = inf_lfc.mean(axis=0) - ctrl_lfc.mean(axis=0)

    perm_diffs = np.zeros((n_perm, inf_lfc.shape[1]))
    for k in range(n_perm):
        idx      = rng.permutation(n_total)
        perm_inf  = combined[idx[:n_inf]]
        perm_ctrl = combined[idx[n_inf:]]
        perm_diffs[k] = perm_inf.mean(axis=0) - perm_ctrl.mean(axis=0)

    pvals = (np.abs(perm_diffs) >= np.abs(observed)).mean(axis=0)
    pvals = np.maximum(pvals, 1.0 / n_perm)
    return observed, pvals


# =============================================================================
# Pattern labelling
# =============================================================================

def label_pattern(inter_sig: bool, inf_sig: bool, ctrl_sig: bool) -> str:
    if inter_sig and inf_sig and not ctrl_sig:
        return "infection_specific"
    if inter_sig and not inf_sig and not ctrl_sig:
        return "suppressed"
    if not inter_sig and inf_sig and ctrl_sig:
        return "age_related"
    if not inter_sig and inf_sig and not ctrl_sig:
        return "weak_infection"
    if not inter_sig and not inf_sig and not ctrl_sig:
        return "no_change"
    return "other"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2x2 paired miRNA interaction analysis: infected vs uninfected, 8 vs 16 weeks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", choices=["ttest", "wilcoxon", "permutation"],
                        required=True,
                        help="Statistical test to apply to all three comparisons.")
    parser.add_argument("--inf-pre",   default="Pg_8_weeks.csv",
                        help="Infected pre-treatment CSV  (default: Pg_8_weeks.csv)")
    parser.add_argument("--inf-post",  default="Pg_16_weeks.csv",
                        help="Infected post-treatment CSV (default: Pg_16_weeks.csv)")
    parser.add_argument("--ctrl-pre",  default="Control_8_weeks.csv",
                        help="Uninfected pre-treatment CSV  (default: Control_8_weeks.csv)")
    parser.add_argument("--ctrl-post", default="Control_16_weeks.csv",
                        help="Uninfected post-treatment CSV (default: Control_16_weeks.csv)")
    parser.add_argument("--drop-cols", default="infected",
                        help="Comma-separated non-miRNA columns to drop (default: infected)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold (default: 0.05)")
    parser.add_argument("--n-perm", type=int, default=10000,
                        help="Number of permutations (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for permutation test (default: 42)")
    parser.add_argument("--output", default=None,
                        help="Output CSV filename (default: interaction_results_{method}.csv)")
    args = parser.parse_args()

    drop_cols = [c.strip() for c in args.drop_cols.split(",")]
    rng       = np.random.default_rng(args.seed)

    print(f"Method    : {args.method.upper()}")
    print(f"Inf  pre  : {args.inf_pre}")
    print(f"Inf  post : {args.inf_post}")
    print(f"Ctrl pre  : {args.ctrl_pre}")
    print(f"Ctrl post : {args.ctrl_post}")
    print(f"Alpha     : {args.alpha}")

    ip_arr, io_arr, cp_arr, co_arr, mirna_cols = load_all(
        args.inf_pre, args.inf_post,
        args.ctrl_pre, args.ctrl_post,
        drop_cols
    )
    n_mice, n_mirnas = ip_arr.shape
    print(f"Mice/group: {n_mice}")
    print(f"miRNAs    : {n_mirnas}\n")

    inf_lfc  = compute_lfc(ip_arr, io_arr)   # (n_mice, n_mirnas)
    ctrl_lfc = compute_lfc(cp_arr, co_arr)

    # ---- Run tests -----------------------------------------------------------
    if args.method == "wilcoxon":
        inter_stat, inter_p = interaction_wilcoxon(inf_lfc, ctrl_lfc)
        inf_stat,   inf_p   = paired_wilcoxon(ip_arr, io_arr)
        ctrl_stat,  ctrl_p  = paired_wilcoxon(cp_arr, co_arr)

    elif args.method == "ttest":
        inter_stat, inter_p = interaction_ttest(inf_lfc, ctrl_lfc)
        inf_stat,   inf_p   = paired_ttest(ip_arr, io_arr)
        ctrl_stat,  ctrl_p  = paired_ttest(cp_arr, co_arr)

    else:  # permutation
        print("Running interaction permutation test...")
        inter_stat, inter_p = interaction_permutation(inf_lfc, ctrl_lfc, args.n_perm, rng)
        print("Running within-infected permutation test...")
        inf_stat,   inf_p   = paired_permutation(ip_arr, io_arr, args.n_perm, rng)
        print("Running within-uninfected permutation test...")
        ctrl_stat,  ctrl_p  = paired_permutation(cp_arr, co_arr, args.n_perm, rng)

    # ---- BH correction (separately per comparison) --------------------------
    inter_q = bh_correction(inter_p)
    inf_q   = bh_correction(inf_p)
    ctrl_q  = bh_correction(ctrl_p)

    # ---- Assemble output -----------------------------------------------------
    inter_sig = inter_p < args.alpha
    inf_sig   = inf_p   < args.alpha
    ctrl_sig  = ctrl_p  < args.alpha

    patterns = [
        label_pattern(inter_sig[j], inf_sig[j], ctrl_sig[j])
        for j in range(n_mirnas)
    ]

    out = pd.DataFrame({
        "miRNA":                  mirna_cols,
        "inf_median_log2FC":      np.round(np.median(inf_lfc,  axis=0), 4),
        "ctrl_median_log2FC":     np.round(np.median(ctrl_lfc, axis=0), 4),
        "delta_median_log2FC":    np.round(
            np.median(inf_lfc, axis=0) - np.median(ctrl_lfc, axis=0), 4),
        "interaction_stat":       np.round(inter_stat, 4),
        "interaction_p":          np.round(inter_p,    6),
        "interaction_q":          np.round(inter_q,    6),
        "interaction_sig_nominal": inter_sig,
        "interaction_sig_FDR":    inter_q < args.alpha,
        "infected_stat":          np.round(inf_stat,   4),
        "infected_p":             np.round(inf_p,      6),
        "infected_q":             np.round(inf_q,      6),
        "infected_sig_nominal":   inf_sig,
        "infected_sig_FDR":       inf_q < args.alpha,
        "ctrl_stat":              np.round(ctrl_stat,  4),
        "ctrl_p":                 np.round(ctrl_p,     6),
        "ctrl_q":                 np.round(ctrl_q,     6),
        "ctrl_sig_nominal":       ctrl_sig,
        "ctrl_sig_FDR":           ctrl_q < args.alpha,
        "pattern":                patterns,
    }).sort_values("interaction_p")

    # ---- Summary -------------------------------------------------------------
    print(f"Interaction significant (nominal p < {args.alpha}): "
          f"{out['interaction_sig_nominal'].sum()}")
    print(f"Interaction significant (BH FDR q < {args.alpha}):  "
          f"{out['interaction_sig_FDR'].sum()}")
    print(f"\nPattern breakdown (nominal thresholds):")
    print(out["pattern"].value_counts().to_string())
    print(f"\nTop 10 miRNAs by interaction p-value:")
    display_cols = ["miRNA", "inf_median_log2FC", "ctrl_median_log2FC",
                    "delta_median_log2FC", "interaction_p", "infected_p",
                    "ctrl_p", "pattern"]
    print(out[display_cols].head(10).to_string(index=False))

    outfile = args.output if args.output else f"interaction_results_{args.method}.csv"
    out.to_csv(outfile, index=False)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
