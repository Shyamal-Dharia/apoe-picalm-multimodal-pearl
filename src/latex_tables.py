"""LaTeX table bodies for the paper, built directly from the result files.

Run from the repository root after the analysis scripts:
    python src/latex_tables.py [OUT_DIR]          # default results/latex
Each table is written as a tabular or tabularx environment (no float, no caption) so the
manuscript can \\input it inside its own table environment. The layout follows the original
manuscript (group 1 and group 2 columns, lower and upper CI columns, \\multirow labels, shaded
measure rows) and fits a 345 pt text width (elsarticle preprint); it needs the multirow,
makecell, tabularx and xcolor (table option) packages. fc_pair_summary.csv gives the counts quoted in
the text for the 10 strongest connections of each fMRI state direction.

Mediation tables are written only when the pairwise mediation results exist.
"""
import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/latex"
TEMPLATEFLOW_HOME = os.environ.get("TEMPLATEFLOW_HOME", os.path.expanduser("~/.cache/templateflow"))
SCHAEFER_TSV = os.path.join(TEMPLATEFLOW_HOME, "tpl-MNI152NLin2009cAsym",
                            "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.tsv")
GROUPS = ["N", "A+P-", "A+P+"]

# HCP-MMP1 (Glasser et al., 2016) and AAL3 (Rolls et al., 2020) region names
REGION_NAMES = {
    "PHA1": "Parahippocampal area 1", "PHA2": "Parahippocampal area 2",
    "PeEc": "Perirhinal-ectorhinal cortex", "PreS": "Presubiculum", "43": "Area 43 (subcentral)",
    "pOFC": "Posterior orbitofrontal cortex", "a24": "Area a24 (anterior cingulate)",
    "31pv": "Area 31pv (posterior cingulate)", "AAIC": "Anterior agranular insula complex",
    "8Ad": "Area 8Ad (dorsolateral prefrontal)", "FOP2": "Frontal opercular area 2",
    "7PL": "Area 7PL (superior parietal)", "V2": "Secondary visual cortex",
    "MIP": "Medial intraparietal area", "V1": "Primary visual cortex",
    "23d": "Area 23d (posterior cingulate)", "9p": "Area 9p (dorsolateral prefrontal)",
    "VER10": "Vermis lobule X", "VER9": "Vermis lobule IX",
    "tAV": "Thalamus, anteroventral nucleus", "tVA": "Thalamus, ventral anterior nucleus",
    "PCC": "Posterior cingulate gyrus",
}


NONIMAGING_LABELS = {"eosinophils": ["Eosinophils", r"($10^3/\mu$l)"], "MCH": ["MCH (pg)"]}  # lines of the label
LX = r">{\raggedright\arraybackslash}X"  # tabularx column for names that may wrap


def fmt(x, d=3):
    return f"{x:.{d}f}".replace("-", "$-$")


def fmt_p(p):
    if p < 0.001:
        return "$<$0.001"
    # four decimals where three would hide which side of 0.05 or 0.10 a value falls
    return f"{p:.4f}" if near_threshold(p) else f"{p:.3f}"


def near_threshold(p):
    return 0.0495 <= p < 0.0505 or 0.0995 <= p < 0.1005


def mark(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else r"$^{\dagger}$" if p < 0.10 else ""


def split_region(code):
    """'rPHA1' -> ('PHA1', 'R'); AAL3 vermis labels have no hemisphere."""
    if code[0] in "lr" and code[1:] in REGION_NAMES:
        return code[1:], code[0].upper()
    return code, ""


def write(name, text):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        f.write(text)


os.makedirs(OUT_DIR, exist_ok=True)

# Baseline characteristics (Table 1)
base = pd.read_csv("results/tables/table1_baseline.csv")
demo = pd.read_csv("results/tables/table1_demographics.csv").set_index("group")
cols = GROUPS + ["Overall"]


def counts(variable, level):
    row = base[(base["variable"] == variable) & (base["level"] == level)]
    return [str(row[c].iloc[0]) if len(row) else "0" for c in cols]


def mean_sd(variable):
    row = base[base["variable"] == variable].iloc[0]
    return [row[c] for c in cols]


lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \textbf{N} & \textbf{A+P$-$} & \textbf{A+P+} & \textbf{Overall} \\", r"\midrule",
         "$n$ & " + " & ".join(str(int(demo.loc[c, "n"])) for c in cols) + r" \\",
         "Male / female & " + " & ".join(f"{int(demo.loc[c, 'male'])} / {int(demo.loc[c, 'female'])}" for c in cols) + r" \\",
         "Age, years, mean (SD) & " + " & ".join(mean_sd("age")) + r" \\",
         "BMI, mean (SD) & " + " & ".join(mean_sd("BMI")) + r" \\",
         r"\addlinespace", r"\textit{APOE} haplotype & & & & \\"]
for lvl, label in [("e3/e3", r"$\varepsilon$3/$\varepsilon$3"), ("e3/e4", r"$\varepsilon$3/$\varepsilon$4"),
                   ("e4/e4", r"$\varepsilon$4/$\varepsilon$4")]:
    lines.append(rf"\quad {label} & " + " & ".join(counts("APOE haplotype", lvl)) + r" \\")
lines.append(r"\textit{PICALM} rs3851179 & & & & \\")
for lvl in ["A/A", "G/A", "G/G"]:
    lines.append(rf"\quad {lvl} & " + " & ".join(counts("PICALM rs3851179", lvl)) + r" \\")
edu = "Education (0 primary, 1 secondary, 2 partial higher, 3 higher)"
lines += [r"\addlinespace", r"Education & & & & \\"]
for lvl, label in [("1", "Secondary"), ("2", "Partial higher"), ("3", "Higher"), ("missing", "Missing")]:
    lines.append(rf"\quad {label} & " + " & ".join(counts(edu, lvl)) + r" \\")
lines.append("Hypertension & " + " & ".join(counts("Hypertension (1 = yes)", "1")) + r" \\")
lines.append("Diabetes & " + " & ".join(counts("Diabetes (1 = yes)", "1")) + r" \\")
lines.append(r"Parental dementia & & & & \\")
for lvl, label in [("1", "One parent"), ("2", "Both parents")]:
    lines.append(rf"\quad {label} & " + " & ".join(counts("Parents with dementia (0, 1, 2)", lvl)) + r" \\")
lines.append(r"Smoking & & & & \\")
smoke = "Smoking (0 no, 1 yes, 2 past)"
for lvl, label in [("0", "Never"), ("1", "Current"), ("2", "Former"), ("missing", "Missing")]:
    lines.append(rf"\quad {label} & " + " & ".join(counts(smoke, lvl)) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("table_baseline.tex", "\n".join(lines))

# Permutation tests. The raw table carries the null mean (SD) of each test; tables written before
# those columns existed (the April 2026 per-fold run) are read together with the pipeline log.
perm = pd.read_csv("results/permutation/publication_f1_permutation_table_raw.csv")
# The 75-variable modality is reported as "Non-imaging" (the pipeline calls it "Clinical").
perm["Modality"] = perm["Modality"].replace({"Clinical": "Non-imaging", "Psychometric": "Non-imaging"})
if "Null_Mean_F1_raw" in perm:
    perm = perm.rename(columns={"Null_Mean_F1_raw": "null_mean", "Null_SD_F1_raw": "null_sd"})
else:
    log = open("results/permutation/per_fold_2026-04/f1_perm_table.log").read().splitlines()
    null = []
    for i, line in enumerate(log):
        if line.split()[:3] == ["Modality", "Observed_Mean_F1", "Observed_Std_F1"]:
            for row in log[i + 1:i + 5]:
                f = row.split()
                null.append({"Modality": "Non-imaging" if f[0] == "Psychometric" else f[0],
                             "obs": float(f[1]), "null_mean": float(f[4]), "null_sd": float(f[5])})
    null = pd.DataFrame(null)
    null["Group Comparison"] = np.repeat(perm["Group Comparison"].unique(), 4)
    perm = perm.merge(null, on=["Group Comparison", "Modality"])
    assert np.allclose(perm["obs"], perm["Observed_Mean_F1_raw"], atol=1e-4)
assert len(perm) == 16
COMPARISON_LABELS = {"N vs A+P-": r"\makecell[l]{N vs\\A+P$-$}", "N vs A+P+": r"\makecell[l]{N vs\\A+P+}",
                     "A+P- vs A+P+": r"\makecell[l]{A+P$-$ vs\\A+P+}",
                     "N vs (A+P- and A+P+)": r"\makecell[l]{N vs all\\carriers}"}
lines = [r"\begin{tabular}{llccc}", r"\toprule",
         r"\textbf{Comparison} & \textbf{Modality} & \makecell{\textbf{F1-macro}\\\textbf{mean (SD)}} & "
         r"\makecell{\textbf{Null F1}\\\textbf{mean (SD)}} & \textbf{$p$} \\",
         r"\midrule"]
for k, (comp, part) in enumerate(perm.groupby("Group Comparison", sort=False)):
    if k:
        lines.append(r"\midrule")
    for j, (_, r) in enumerate(part.iterrows()):
        sd = float(str(r["Observed F1"]).split("+/-")[1])
        p = r["p_value_raw"]
        f1 = f"{r['Observed_Mean_F1_raw']:.3f} ({sd:.3f})"
        # smallest attainable p is 1/1001 with 1000 permutations
        cell = f"{p:.4f}{mark(p)}" if near_threshold(p) else f"{p:.3f}{mark(p)}"
        if p < 0.05:
            f1, cell = rf"\textbf{{{f1}}}", rf"\textbf{{{cell}}}"
        label = rf"\multirow{{4}}{{*}}{{{COMPARISON_LABELS[comp]}}}" if j == 0 else ""
        lines.append(f"{label} & {r['Modality']} & {f1} & {r['null_mean']:.3f} ({r['null_sd']:.3f}) & {cell} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("table_permutation.tex", "\n".join(lines))

# Counts quoted in the text: tests below 0.05 and 0.10, and observed F1 within two null SDs of the null mean
z = (perm["Observed_Mean_F1_raw"] - perm["null_mean"]) / perm["null_sd"]
pd.DataFrame([{"n_tests": len(perm), "n_p_below_005": int((perm["p_value_raw"] < 0.05).sum()),
               "n_p_below_010": int((perm["p_value_raw"] < 0.10).sum()),
               "n_within_2_null_sd": int((z < 2).sum()), "null_sd_min": perm["null_sd"].min(),
               "null_sd_max": perm["null_sd"].max(), "lowest_p": perm["p_value_raw"].min()}]
             ).to_csv("results/tables/permutation_summary.csv", index=False)

# Non-imaging post hoc (Table 3): measure label with p_KW over three rows, CI as lower and upper
t6 = pd.read_csv("results/tables/table6_clinical_posthoc.csv")
lines = [r"\begin{tabular}{lllrrrrr}", r"\toprule",
         r"\textbf{Measure} & \textbf{Group 1} & \textbf{Group 2} & \textbf{Diff.} & \multicolumn{2}{c}{\textbf{95\% CI}} & "
         r"\textbf{$d$} & \textbf{$p_{\mathrm{bonf}}$} \\", r"\cmidrule(lr){5-6}",
         r" & & & & \textbf{Lower} & \textbf{Upper} & & \\", r"\midrule"]
for k, (var, part) in enumerate(t6.groupby("variable", sort=False)):
    if k:
        lines.append(r"\midrule")
    name = NONIMAGING_LABELS.get(var, [var.replace("_", " ").replace("MINI-COPE", "Mini-COPE")])
    kw = rf"$p_{{\mathrm{{KW}}}} = {fmt_p(part['p_kruskal'].iloc[0])}$"
    label = rf"\multirow{{{len(part)}}}{{*}}{{\makecell[l]{{" + r"\\".join(name + [kw]) + "}}"
    for j, (_, r) in enumerate(part.iterrows()):
        lines.append(f"{label if j == 0 else ''} & {r['group_1']} & {r['group_2']} & {fmt(r['diff'])} & "
                     f"{fmt(r['ci_lower'])} & {fmt(r['ci_upper'])} & {fmt(r['cohens_d'], 2)} & "
                     f"{fmt_p(r['p_bonf'])}{mark(r['p_bonf'])} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("table_clinical.tex", "\n".join(lines).replace("A+P-", "A+P$-$"))

# sMRI post hoc (supplement): shaded measure rows, region name wraps (tabularx)
t5 = pd.read_csv("results/tables/table5_smri_posthoc.csv")
lines = [rf"\begin{{tabularx}}{{\linewidth}}{{l{LX}llrrrrrr}}", r"\toprule",
         r"\textbf{Region} & \textbf{Name} & \textbf{Group 1} & \textbf{Group 2} & \textbf{Diff.} & "
         r"\multicolumn{2}{c}{\textbf{95\% CI}} & \textbf{$d$} & \textbf{$p_{\mathrm{bonf}}$} & \textbf{$p_{\mathrm{KW}}$} \\",
         r"\cmidrule(lr){6-7}",
         r" & & & & & \textbf{Lower} & \textbf{Upper} & & & \textbf{($q$)} \\", r"\midrule"]
for k, (measure, part) in enumerate(t5.groupby("measure", sort=False)):
    if k:
        lines.append(r"\midrule")
    lines.append(rf"\rowcolor{{gray!10}}\multicolumn{{10}}{{c}}{{\textbf{{{measure}}} ({int(part['n_features_in_family'].iloc[0])} regions)}} \\")
    last = None
    for _, r in part.iterrows():
        code, _ = split_region(r["region"])
        first = r["region"] != last
        region = r["region"] if first else ""  # same codes as the figure and the mediation tables
        name = REGION_NAMES[code] if first else ""
        kw = f"{fmt_p(r['p_kruskal'])} ({r['q_fdr_kw']:.2f})" if first else ""
        lines.append(f"{region} & {name} & {r['group_1']} & {r['group_2']} & {fmt(r['diff'])} & "
                     f"{fmt(r['ci_lower'])} & {fmt(r['ci_upper'])} & {fmt(r['cohens_d'], 2)} & "
                     f"{fmt_p(r['p_bonf'])}{mark(r['p_bonf'])} & {kw} " + r"\\")
        last = r["region"]
lines += [r"\bottomrule", r"\end{tabularx}", ""]
write("table_smri.tex", "\n".join(lines).replace("A+P-", "A+P$-$"))

# fMRI: 10 strongest connections per state direction (supplement)
names = pd.read_csv(SCHAEFER_TSV, sep="\t").set_index("index")["name"]
pretty = lambda i: names[i + 1].replace("7Networks_", "").replace("_", " ")  # 0-based ROI index
fc = pd.read_csv("results/fmri/fMRI_unified_top_fc_pairs_60s_75overlap_multiseed.csv")
lines = [rf"\begin{{tabularx}}{{\linewidth}}{{r{LX}{LX}cc}}", r"\toprule",
         r"\textbf{Rank} & \textbf{Region 1 (ROI)} & \textbf{Region 2 (ROI)} & "
         r"\makecell{\textbf{$r$, carrier-}\\\textbf{dominant}} & \makecell{\textbf{$r$, N-}\\\textbf{dominant}} \\",
         r"\midrule"]
summary = []
for _, r in fc.iterrows():
    a, b = int(r["roi_i"]), int(r["roi_j"])
    cells = [fmt(r[c], 3) if not np.isnan(r[c]) else "---" for c in ("r_risk_higher_N_lt", "r_N_higher_N_gt")]
    lines.append(f"{int(r['rank'])} & {pretty(a)} ({a + 1}) & {pretty(b)} ({b + 1}) & "
                 f"{cells[0]} & {cells[1]} " + r"\\")
for key, col in (("carrier", "r_risk_higher_N_lt"), ("N", "r_N_higher_N_gt")):
    top = fc.dropna(subset=[col]).nlargest(10, col)
    hemis = [(pretty(int(i)).split()[0], pretty(int(j)).split()[0]) for i, j in zip(top["roi_i"], top["roi_j"])]
    nets = pd.Series([pretty(int(x)).split()[1] for x in np.r_[top["roi_i"], top["roi_j"]]]).value_counts()
    summary.append({"direction": key, "n_pairs": len(top),
                    "shared_with_other_direction": int(fc[["r_risk_higher_N_lt", "r_N_higher_N_gt"]].notna().all(axis=1).sum()),
                    "inter_hemispheric": sum(h1 != h2 for h1, h2 in hemis),
                    **{f"endpoints_{n}": int(c) for n, c in nets.items()}})
lines += [r"\bottomrule", r"\end{tabularx}", ""]
write("table_fc_pairs.tex", "\n".join(lines))
pd.DataFrame(summary).fillna(0).to_csv(os.path.join(OUT_DIR, "fc_pair_summary.csv"), index=False)

print(f"Wrote LaTeX tables to {OUT_DIR}/")
