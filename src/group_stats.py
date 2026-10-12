"""Three-group comparison helpers shared by descriptive_tables.py and paper_figures.py.

Post hoc comparisons follow JASP's one-way ANOVA post hoc output: mean difference
(group 1 - group 2), standard error from the pooled ANOVA error term, Bonferroni-adjusted
p-values over the three pairwise contrasts, Tukey-adjusted 95% confidence intervals, and
Cohen's d (mean difference / pooled ANOVA SD).
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multitest import multipletests

GROUPS = ["N", "A+P-", "A+P+"]


def split_by_group(values, groups):
    values = pd.to_numeric(values, errors="coerce")
    return {g: values[groups == g].dropna().to_numpy(float) for g in GROUPS}


def kruskal_p(by_group):
    samples = list(by_group.values())
    if min(len(s) for s in samples) < 2 or np.var(np.concatenate(samples)) == 0:
        return np.nan
    return stats.kruskal(*samples).pvalue


def posthoc(by_group):
    """JASP-style ANOVA post hoc: Bonferroni p, Tukey 95% CI, pooled SE, Cohen's d."""
    df_within = sum(len(v) for v in by_group.values()) - len(GROUPS)
    mse = sum(((v - v.mean()) ** 2).sum() for v in by_group.values()) / df_within
    t_tukey = qsturng(0.95, len(GROUPS), df_within) / np.sqrt(2)
    n_contrasts = len(GROUPS) * (len(GROUPS) - 1) // 2
    rows = []
    for g1, g2 in itertools.combinations(GROUPS, 2):
        a, b = by_group[g1], by_group[g2]
        diff = a.mean() - b.mean()
        se = np.sqrt(mse * (1 / len(a) + 1 / len(b)))
        p_bonf = min(1.0, n_contrasts * 2 * stats.t.sf(abs(diff / se), df_within))
        rows.append({"group_1": g1, "group_2": g2, "diff": diff,
                     "ci_lower": diff - t_tukey * se, "ci_upper": diff + t_tukey * se,
                     "se": se, "p_bonf": p_bonf, "cohens_d": diff / np.sqrt(mse)})
    return rows


def fdr_q(p_by_feature):
    """Benjamini-Hochberg q for a dict feature -> p (NaN p values are left out)."""
    valid = {f: p for f, p in p_by_feature.items() if not np.isnan(p)}
    q = multipletests(list(valid.values()), method="fdr_bh")[1]
    return dict(zip(valid, q))
