"""Summary of the mediation grid and LaTeX tables for the supplement.

Run from the repository root after src/mediation_grid.R (and, if it ran,
src/run_pairwise_significant_mediations.R), once per mediator set:
    python src/mediation_summary.py                                   # sMRI set
    python src/mediation_summary.py results/mediation/dynamic results/latex dynamic
Arguments: GRID_DIR (default results/mediation), OUT_DIR (default results/latex), NAME (default smri).

Writes
  GRID_DIR/mediation_summary.csv              counts quoted in the text: models fitted, ACME p < .05
                                              (and the number expected by chance), lowest q, q < .05
  OUT_DIR/table_mediation_top_NAME.tex        the 10 omnibus pairs with the lowest ACME p
  OUT_DIR/table_mediation_pairwise_NAME.tex   pairwise contrasts for the FDR-selected pairs, if any
"""
import os
import sys

import pandas as pd

GRID_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/mediation"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/latex"
NAME = sys.argv[3] if len(sys.argv) > 3 else "smri"
GRID_CSV = os.path.join(GRID_DIR, "mediation_grid_all_pairs.csv")
PAIRWISE = {
    "N vs A+P$-$": os.path.join(GRID_DIR, "pairwise_mediation_0_vs_1.csv"),
    "N vs A+P+": os.path.join(GRID_DIR, "pairwise_mediation_0_vs_2.csv"),
    "A+P$-$ vs A+P+": os.path.join(GRID_DIR, "pairwise_mediation_1_vs_2.csv"),
}
MEASURES = {"depth": "Sulcal depth", "fd": "Fractal dimension", "gyrification": "Gyrification",
            "thickness": "Thickness", "grey": "GM volume"}
DYNAMIC_LABELS = {"med_EEG_var_N_states": "EEG HFD variability, N-highest states",
                  "med_EEG_var_AposPminus_states": "EEG HFD variability, A+P$-$-highest states",
                  "med_EEG_var_AposPpos_states": "EEG HFD variability, A+P+-highest states",
                  "med_fMRI_time_N_states": "fMRI dwell time, N-dominant states",
                  "med_fMRI_time_carrier_states": "fMRI dwell time, carrier-dominant states"}
BLOOD = ("leukocytes", "erythrocytes", "hemoglobin", "hematocrit", "MCV", "MCH", "MCHC", "RDW-CV",
         "platelets", "PDW", "MPV", "P-LCR", "neutrophils", "lymphocytes", "monocytes", "eosinophils",
         "basophils", "cholesterol", "triglycerides")
N_TOP = 10
# CVLT scores as described in the OpenNeuro ds004796 participants.json
CVLT_NAMES = {1: "List A trials 1-5", 2: "List A trial 1", 3: "List A trial 5", 4: "List B",
              5: "short-delay free recall", 6: "short-delay cued recall", 7: "long-delay free recall",
              8: "long-delay cued recall", 9: "perseverations", 10: "free-recall intrusions",
              11: "cued-recall intrusions", 12: "recognition hits", 13: "recognition false alarms"}


def mediator_label(m):
    if m in DYNAMIC_LABELS:
        return DYNAMIC_LABELS[m].replace("A+P+", "A+P+")
    measure, region = m.removeprefix("med_").split("_", 1)
    return f"{MEASURES.get(measure, measure)} {region}"


def outcome_label(y):
    """R's check.names turns 'MINI-COPE_6' into 'MINI.COPE_6' and 'eosinophils_%' into 'eosinophils_.'."""
    y = y.removeprefix("y_").replace("MINI.COPE", "Mini-COPE").replace("MINI-COPE", "Mini-COPE").replace("non.HDL", "non-HDL")
    y = y.replace("RDW.CV", "RDW-CV").replace("P.LCR", "P-LCR")
    if y.startswith("CVLT_"):
        return "CVLT " + CVLT_NAMES[int(y.removeprefix("CVLT_"))].replace("-", "--")
    if y.endswith(("_.", "_%")):
        y = y[:-2] + r" (\%)"
    return y.replace("_", " ")


def num(x, d=3):
    return f"{x:.{d}f}".replace("-", "$-$")


def p_cell(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


grid = pd.read_csv(GRID_CSV)
if "acme_p_fdr" not in grid:
    from statsmodels.stats.multitest import multipletests
    grid["acme_p_fdr"] = multipletests(grid["acme_p"], method="fdr_bh")[1]
grid["domain"] = grid["y"].map(lambda y: "blood" if any(b in outcome_label(y) for b in BLOOD) else "psychometric")

summary = {
    "n_models": len(grid),
    "n_outcomes": grid["y"].nunique(),
    "n_mediators": grid["mediator"].nunique(),
    "n_complete_cases_min": int(grid["n"].min()),
    "n_complete_cases_max": int(grid["n"].max()),
    "n_acme_p_lt_05": int((grid["acme_p"] < 0.05).sum()),
    "expected_acme_p_lt_05_under_null": 0.05 * len(grid),
    "min_acme_p": grid["acme_p"].min(),
    "min_acme_q": grid["acme_p_fdr"].min(),
    "n_acme_q_lt_05": int((grid["acme_p_fdr"] < 0.05).sum()),
    "n_acme_q_lt_10": int((grid["acme_p_fdr"] < 0.10).sum()),
    **{f"n_acme_p_lt_05_{d}": int(((grid["acme_p"] < 0.05) & (grid["domain"] == d)).sum())
       for d in ("psychometric", "blood")},
    **{f"n_models_{d}": int((grid["domain"] == d).sum()) for d in ("psychometric", "blood")},
}
# Complete cases per group on gene + covariates (the model sample before outcome/mediator missingness)
COVARIATES = ["age", "sex", "education", "learning_deficits", "HSV_r", "BMI", "hypertension",
              "thyroid_diseases", "other_diseases", "dementia_history_parents"]
inp = pd.read_csv("data/mediation_input_n64.csv")
sample_ids = pd.read_csv("data/participants_n64.csv")["participant_id"]
inp = inp[inp["participant_id"].isin(sample_ids)]
complete = inp.dropna(subset=["gene"] + COVARIATES)
summary.update({f"complete_cases_gene{g}": int((complete["gene"] == g).sum()) for g in (0, 1, 2)})

pd.DataFrame([summary]).to_csv(os.path.join(GRID_DIR, "mediation_summary.csv"), index=False)
print(pd.Series(summary).to_string())

os.makedirs(OUT_DIR, exist_ok=True)
top = grid.nsmallest(N_TOP, "acme_p", keep="all").sort_values(["acme_p", "acme_p_fdr"])  # keeps ties at the cut
# Layout of the original manuscript, fitted to a 345 pt text width: outcome and mediator wrap
# (tabularx), CI as lower and upper columns
LX = r">{\raggedright\arraybackslash}X"
lines = [rf"\begin{{tabularx}}{{\linewidth}}{{{LX}{LX}rrrrrrr}}", r"\toprule",
         r"\textbf{Outcome} & \textbf{Mediator} & \textbf{ACME} & \multicolumn{2}{c}{\textbf{95\% CI}} & \textbf{$p$} & "
         r"\textbf{$q$} & \textbf{ADE} & \textbf{ADE} \\", r"\cmidrule(lr){4-5}",
         r" & & & \textbf{Lower} & \textbf{Upper} & & & & \textbf{$p$} \\", r"\midrule"]
for _, r in top.iterrows():
    lines.append(f"{outcome_label(r['y'])} & {mediator_label(r['mediator'])} & {num(r['acme'])} & "
                 f"{num(r['acme_ci_low'])} & {num(r['acme_ci_high'])} & {p_cell(r['acme_p'])} & "
                 f"{r['acme_p_fdr']:.2f} & {num(r['ade'])} & {p_cell(r['ade_p'])} " + r"\\")
lines += [r"\bottomrule", r"\end{tabularx}", ""]
with open(os.path.join(OUT_DIR, f"table_mediation_top_{NAME}.tex"), "w") as f:
    f.write("\n".join(lines))

pairwise = {k: pd.read_csv(v) for k, v in PAIRWISE.items() if os.path.exists(v)}
pairwise = {k: v for k, v in pairwise.items() if len(v)}
if pairwise:
    lines = [r"\begin{tabular}{lllrrrrr}", r"\toprule",
             r"\textbf{Outcome} & \textbf{Mediator} & \textbf{Contrast} & \textbf{ACME} & \textbf{95\% CI} & "
             r"\textbf{$p$} & \textbf{$a$ ($p$)} & \textbf{$b$ ($p$)} \\", r"\midrule"]
    pairs = pd.concat([v.assign(contrast=k) for k, v in pairwise.items()])
    for (y, m), part in pairs.groupby(["y", "mediator"], sort=False):
        for j, (_, r) in enumerate(part.iterrows()):
            lead = f"{outcome_label(y)} & {mediator_label(m)}" if j == 0 else " & "
            lines.append(f"{lead} & {r['contrast']} & {num(r['acme'])} & [{num(r['acme_ci_low'])}, "
                         f"{num(r['acme_ci_high'])}] & {p_cell(r['acme_p'])} & {num(r['a_hat'])} ({p_cell(r['a_p'])}) & "
                         f"{num(r['b_hat'])} ({p_cell(r['b_p'])}) " + r"\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(os.path.join(OUT_DIR, f"table_mediation_pairwise_{NAME}.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"Pairwise table: {pairs.groupby(['y', 'mediator']).ngroups} pairs")
else:
    print("No pairwise mediation results (no pair passed the FDR step).")
print(f"Wrote {GRID_DIR}/mediation_summary.csv and tables in {OUT_DIR}/")
