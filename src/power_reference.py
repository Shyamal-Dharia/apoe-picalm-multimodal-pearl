"""Design calculations that the paper reports as a reference (not data analyses).

Smallest standardized mean difference (Cohen's d) that a two-sided two-sample t-test detects with
80% power at the group sizes of each comparison, uncorrected and at the Bonferroni threshold of
the classification tests, and the per-group sample size needed for d = 0.5 (uncorrected).

Run from the repository root:
    python src/power_reference.py
Writes results/tables/minimum_detectable_effect.csv and results/tables/replication_sample_size.csv.
"""
import os

import pandas as pd
from statsmodels.stats.power import TTestIndPower

N_TESTS = 16  # Table 2: four modalities x four comparisons
GROUP_N = {"N": 25, "A+P-": 23, "A+P+": 16}
COMPARISONS = {
    "N vs A+P-": (GROUP_N["N"], GROUP_N["A+P-"]),
    "N vs A+P+": (GROUP_N["N"], GROUP_N["A+P+"]),
    "A+P- vs A+P+": (GROUP_N["A+P-"], GROUP_N["A+P+"]),
    "N vs carriers": (GROUP_N["N"], GROUP_N["A+P-"] + GROUP_N["A+P+"]),
}
OUT_DIR = "results/tables"

power = TTestIndPower()
rows = []
for name, (n1, n2) in COMPARISONS.items():
    for label, alpha in (("uncorrected", 0.05), (f"bonferroni_{N_TESTS}", 0.05 / N_TESTS)):
        d = power.solve_power(effect_size=None, nobs1=n1, ratio=n2 / n1, alpha=alpha, power=0.8,
                              alternative="two-sided")
        rows.append({"comparison": name, "n1": n1, "n2": n2, "alpha": label, "mdes_d_80pct_power": round(d, 2)})
os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/minimum_detectable_effect.csv", index=False)

n = power.solve_power(effect_size=0.5, nobs1=None, ratio=1, alpha=0.05, power=0.8, alternative="two-sided")
pd.DataFrame([{"effect_size_d": 0.5, "alpha": 0.05, "power": 0.8, "n_per_group": int(-(-n // 1))}]).to_csv(
    f"{OUT_DIR}/replication_sample_size.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
