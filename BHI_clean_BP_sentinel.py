"""
BHI_clean_BP_sentinel.py
------------------------
Detect and remove the sentinel-value artifact in BP_Systolic_Avg and
BP_Diastolic_Avg, then re-standardize on the remaining valid subjects.

Background
----------
Both BP columns in the cleaned behavioral file contain a 25-subject
artifact cluster: a single identical extreme z-score (≈ +3.31 diastolic,
≈ +3.21 systolic) shared by the SAME 25 subjects in BOTH columns. The
gap between this cluster and the rest of the distribution (no observations
between z = +1 and z = +3) makes a real-biology explanation implausible:
25 different humans do not have identical systolic AND identical diastolic
blood pressures. The pattern is consistent with missing BP measurements
having been filled by a sentinel value (e.g., 999) BEFORE the file was
z-scored — so the placeholder became an extreme outlier shared across
the affected subjects.

Why this matters
----------------
With the sentinel cluster present, any "BP ↔ BAG" association is mostly
driven by those 25 subjects sharing identical extreme predictor values,
producing spurious results in stepwise regression and ridge models alike.

What this script does
---------------------
1. Detects affected subjects = those at the maximum of BP_Diastolic_Avg
   AND simultaneously at the maximum of BP_Systolic_Avg.
   (Both conditions, to avoid catching legitimate top-of-distribution
   outliers that aren't part of the sentinel pattern.)
2. Sets both BP columns to NaN for those subjects.
3. Re-z-scores each BP column using only the valid subjects' values.
4. Writes the cleaned columns + diagnostics to a separate Excel file
   so the user can integrate manually.

Output
------
Results/BHI_BP_Cleaned.xlsx
    BP_Cleaned        – subject_ID, BP_Systolic_Avg, BP_Diastolic_Avg
                        (NaN for the affected subjects, re-standardized
                        on the remaining valid subjects)
    Affected_Subjects – list of subject_IDs whose BP values were
                        flagged as sentinel and removed
    Before_After      – distributional comparison (N, mean, SD,
                        min/max, % at modal value, unique-value count)
    ReadMe            – method note + integration instructions
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = "./Results/BHI_Behavioral_Plus_Questionnaire_Cleaned.xlsx"
OUTPUT_FILE = "./Results/BHI_BP_Cleaned.xlsx"
ID_COL      = "subject_ID"
BP_COLS     = ["BP_Systolic_Avg", "BP_Diastolic_Avg"]

# ── Load ─────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

for c in [ID_COL] + BP_COLS:
    if c not in df.columns:
        raise ValueError(f"Required column '{c}' not found in input.")

bp = df[[ID_COL] + BP_COLS].copy()
for c in BP_COLS:
    bp[c] = pd.to_numeric(bp[c], errors="coerce")

# ── Pre-cleaning snapshot ────────────────────────────────────────────────────
def snapshot(s, label):
    s = s.dropna()
    if len(s) == 0:
        return {"label": label, "N": 0}
    top_pct = s.value_counts().iloc[0] / len(s) * 100
    return {
        "label":       label,
        "N":           int(len(s)),
        "Unique":      int(s.nunique()),
        "Mean":        round(s.mean(), 4),
        "SD":          round(s.std(ddof=0), 4),
        "Min":         round(s.min(), 4),
        "Max":         round(s.max(), 4),
        "Pct_at_modal": round(top_pct, 2),
    }

before_rows = [snapshot(bp[c], f"{c} (before)") for c in BP_COLS]

# ── Detect sentinel cluster ──────────────────────────────────────────────────
# Affected subjects are simultaneously at the max of BOTH columns.
sys_max  = bp["BP_Systolic_Avg"].max()
dia_max  = bp["BP_Diastolic_Avg"].max()
mask_sys = bp["BP_Systolic_Avg"]  == sys_max
mask_dia = bp["BP_Diastolic_Avg"] == dia_max
affected = mask_sys & mask_dia

print(f"\nSentinel detection:")
print(f"  N at BP_Systolic_Avg max  ({sys_max:+.4f}): {int(mask_sys.sum())}")
print(f"  N at BP_Diastolic_Avg max ({dia_max:+.4f}): {int(mask_dia.sum())}")
print(f"  N at BOTH simultaneously (affected)        : {int(affected.sum())}")

if affected.sum() == 0:
    print("\nNo sentinel cluster detected — nothing to clean.")
elif affected.sum() < 5:
    print("\nWarning: very few subjects flagged. Verify the artifact pattern manually before using.")

# ── NaN out the affected subjects in both columns ────────────────────────────
bp_clean = bp.copy()
for c in BP_COLS:
    bp_clean.loc[affected, c] = np.nan

# ── Re-z-score on remaining valid values ─────────────────────────────────────
for c in BP_COLS:
    s = bp_clean[c]
    bp_clean[c] = (s - s.mean()) / s.std(ddof=0)

# Round for clean output
for c in BP_COLS:
    bp_clean[c] = bp_clean[c].round(6)

# ── Post-cleaning snapshot ───────────────────────────────────────────────────
after_rows = [snapshot(bp_clean[c], f"{c} (after)") for c in BP_COLS]
ba_df = pd.DataFrame(before_rows + after_rows)

print("\nBefore vs after:")
print(ba_df.to_string(index=False))

affected_df = pd.DataFrame({
    ID_COL: bp.loc[affected, ID_COL].values,
    "BP_Systolic_Avg_original":  bp.loc[affected, "BP_Systolic_Avg"].values,
    "BP_Diastolic_Avg_original": bp.loc[affected, "BP_Diastolic_Avg"].values,
})

# ── Write output ─────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE} ...")
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
    bp_clean.to_excel(w, sheet_name="BP_Cleaned",        index=False)
    affected_df.to_excel(w, sheet_name="Affected_Subjects", index=False)
    ba_df.to_excel(w, sheet_name="Before_After",        index=False)

    readme = pd.DataFrame({"Notes": [
        "BP_Systolic_Avg and BP_Diastolic_Avg cleaned of sentinel-value artifact.",
        "",
        f"Affected subjects (N = {int(affected.sum())}) are listed in the Affected_Subjects sheet.",
        "Their original z-scored BP values were identical at the column maxima",
        "(≈ +3.21 systolic, ≈ +3.31 diastolic) — almost certainly because their raw",
        "BP measurements were missing and had been replaced by a sentinel value",
        "(e.g., 999) before the file was z-scored.",
        "",
        "Cleaning steps:",
        "  1. Identified subjects sitting simultaneously at the max of BOTH BP columns.",
        "  2. Set their BP_Systolic_Avg and BP_Diastolic_Avg to NaN.",
        "  3. Re-z-scored each column on the remaining valid subjects.",
        "",
        "Coverage after cleaning:",
        f"  BP_Systolic_Avg : {bp_clean['BP_Systolic_Avg'].notna().sum()}/{len(bp_clean)} subjects",
        f"  BP_Diastolic_Avg: {bp_clean['BP_Diastolic_Avg'].notna().sum()}/{len(bp_clean)} subjects",
        "",
        "Implication for downstream pipelines:",
        "  Coverage on the 304-subject BAG anchor will be ~92% (depending on overlap),",
        "  so under stepwise_BAG_*.py with MIN_PRED_COVERAGE = 1.0 these columns will",
        "  be DROPPED. To keep BP in the analysis, lower MIN_PRED_COVERAGE to 0.90.",
        "",
        "Integration:",
        "  1. Open Behavioral_Data_Cleaned.xlsx (and/or",
        "     Results/BHI_Behavioral_Plus_Questionnaire_Cleaned.xlsx).",
        "  2. Replace the existing BP_Systolic_Avg and BP_Diastolic_Avg columns",
        "     with the cleaned versions from the BP_Cleaned sheet of this file,",
        "     joining on subject_ID.",
    ]})
    readme.to_excel(w, sheet_name="ReadMe", index=False)

print("Done.")
