"""
BHI_compute_Rand26_components.py
--------------------------------
Aggregate the 8 RAND-26 subscales into the standard two summary scores:
    Rand26_PCS  – Physical Component Summary
    Rand26_MCS  – Mental Component Summary

Method: Hays simple summary approach (Hays & Morales, 2001,
Annals of Medicine 33(5):350-357) — mean of the within-sample z-scored
subscales for each component. Subscales already on a z-scale are detected
and used as-is; raw subscales are z-scored before averaging.

Component composition (standard SF-36 / RAND-36 grouping):
    PCS  =  mean( PhysicalFunctioning, RoleLim_Physical, Pain, GeneralHealth )
    MCS  =  mean( RoleLim_Emotional, EmotionalWellBeing, EnergyFatigue, SocialFunctioning )

Missing-data rule: a subject's component score is computed if at least
MIN_SUBSCALES of the 4 contributing subscales are non-missing; otherwise NaN.

Output:
    Results/BHI_Rand26_Components.xlsx
        Components       – subject_ID, Rand26_PCS, Rand26_MCS
        Subscales_Used   – the 8 z-scored subscales used to compute the components
        Diagnostics      – per-subscale ceiling %, pairwise correlations,
                           component distributional summary
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = "./Results/BHI_Behavioral_Plus_Questionnaire_Cleaned.xlsx"
OUTPUT_FILE = "./Results/BHI_Rand26_Components.xlsx"
ID_COL      = "subject_ID"
MIN_SUBSCALES = 2     # require ≥ this many non-missing of 4 to compute a component

# Heuristic for "already z-scored" (mean ≈ 0, SD ≈ 1 within tolerance)
ZSCORE_MEAN_TOL = 0.1
ZSCORE_STD_TOL  = 0.1

PCS_SUBSCALES = ["Rand26_PhysicalFunctioning", "Rand26_RoleLim_Physical",
                 "Rand26_Pain",                "Rand26_GeneralHealth"]
MCS_SUBSCALES = ["Rand26_RoleLim_Emotional",   "Rand26_EmotionalWellBeing",
                 "Rand26_EnergyFatigue",       "Rand26_SocialFunctioning"]
ALL_SUBSCALES = PCS_SUBSCALES + MCS_SUBSCALES

# ── Load and validate ────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

if ID_COL not in df.columns:
    raise ValueError(f"Required ID column '{ID_COL}' not found in input.")

missing_cols = [c for c in ALL_SUBSCALES if c not in df.columns]
if missing_cols:
    # Try the alternate naming (RoleLimit vs RoleLim) used by the questionnaire pipeline
    rename_map = {"Rand26_RoleLim_Physical":  "Rand26_RoleLimit_Physical",
                  "Rand26_RoleLim_Emotional": "Rand26_RoleLimit_Emotional"}
    fixed = [c for c in missing_cols if rename_map.get(c) in df.columns]
    if fixed:
        df = df.rename(columns={rename_map[c]: c for c in fixed})
        missing_cols = [c for c in ALL_SUBSCALES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing RAND-26 subscale columns: {missing_cols}")

# Coerce to numeric
sub = df[[ID_COL] + ALL_SUBSCALES].copy()
for c in ALL_SUBSCALES:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

# ── Standardize each subscale if it isn't already on a z-scale ───────────────
def is_zscored(s):
    s = s.dropna()
    if s.std(ddof=0) == 0:
        return False
    return (abs(s.mean()) < ZSCORE_MEAN_TOL and
            abs(s.std(ddof=0) - 1.0) < ZSCORE_STD_TOL)

zscored_status = {}
for c in ALL_SUBSCALES:
    if is_zscored(sub[c]):
        zscored_status[c] = "already_zscored"
    else:
        s = sub[c]
        sub[c] = (s - s.mean()) / s.std(ddof=0)
        zscored_status[c] = "z-scored_now"

print("\nSubscale standardization:")
for c, status in zscored_status.items():
    print(f"  {c:<32} {status}")

# ── Component scores (row-wise mean with missing-data tolerance) ─────────────
def safe_row_mean(rows, min_required):
    out = rows.mean(axis=1)
    enough = rows.notna().sum(axis=1) >= min_required
    return out.where(enough)

pcs = safe_row_mean(sub[PCS_SUBSCALES], MIN_SUBSCALES)
mcs = safe_row_mean(sub[MCS_SUBSCALES], MIN_SUBSCALES)

components = pd.DataFrame({
    ID_COL:       sub[ID_COL],
    "Rand26_PCS": pcs.round(6),
    "Rand26_MCS": mcs.round(6),
})

print(f"\nComputed components for {len(components)} subjects")
print(f"  Rand26_PCS: N={pcs.notna().sum()}  mean={pcs.mean():.3f}  sd={pcs.std():.3f}  "
      f"range=[{pcs.min():.3f}, {pcs.max():.3f}]")
print(f"  Rand26_MCS: N={mcs.notna().sum()}  mean={mcs.mean():.3f}  sd={mcs.std():.3f}  "
      f"range=[{mcs.min():.3f}, {mcs.max():.3f}]")

# ── Diagnostics sheet ────────────────────────────────────────────────────────
diag_rows = []
for c in ALL_SUBSCALES:
    s = sub[c].dropna()
    top_pct = (s.value_counts().iloc[0] / len(s) * 100) if len(s) else float("nan")
    diag_rows.append({
        "Subscale":      c,
        "Component":     "PCS" if c in PCS_SUBSCALES else "MCS",
        "N":             int(s.notna().sum()),
        "Unique_values": int(s.nunique()),
        "Pct_at_modal":  round(top_pct, 1),
        "Mean":          round(s.mean(), 4),
        "SD":            round(s.std(ddof=0), 4),
        "Min":           round(s.min(), 4),
        "Max":           round(s.max(), 4),
        "Standardization": zscored_status[c],
    })
diag_df = pd.DataFrame(diag_rows)

# Pairwise correlations (long form, easier to read in Excel)
corr_long_rows = []
for grp_name, grp_cols in [("PCS", PCS_SUBSCALES), ("MCS", MCS_SUBSCALES)]:
    cm = sub[grp_cols].corr().round(3)
    for i, a in enumerate(grp_cols):
        for j, b in enumerate(grp_cols):
            if j > i:
                corr_long_rows.append(
                    {"Component": grp_name, "Subscale_A": a, "Subscale_B": b, "Pearson_r": cm.loc[a, b]})
corr_df = pd.DataFrame(corr_long_rows)

# Component distribution summary
def comp_summary(name, s):
    return {
        "Component": name,
        "N":          int(s.notna().sum()),
        "Mean":       round(s.mean(), 4),
        "SD":         round(s.std(), 4),
        "Min":        round(s.min(), 4),
        "Max":        round(s.max(), 4),
        "Unique_values": int(s.nunique()),
        "Pct_at_modal":  round(s.value_counts().iloc[0] / s.notna().sum() * 100, 2),
        "p01": round(s.quantile(0.01), 4), "p05": round(s.quantile(0.05), 4),
        "p25": round(s.quantile(0.25), 4), "p50": round(s.quantile(0.50), 4),
        "p75": round(s.quantile(0.75), 4), "p95": round(s.quantile(0.95), 4),
        "p99": round(s.quantile(0.99), 4),
    }
comp_summary_df = pd.DataFrame([comp_summary("Rand26_PCS", pcs),
                                comp_summary("Rand26_MCS", mcs)])

# ── Write output ─────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE} ...")
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
    components.to_excel(w, sheet_name="Components", index=False)
    sub.round(6).to_excel(w, sheet_name="Subscales_Used", index=False)
    diag_df.to_excel(w, sheet_name="Subscale_Diagnostics", index=False)
    corr_df.to_excel(w, sheet_name="Subscale_Correlations", index=False)
    comp_summary_df.to_excel(w, sheet_name="Component_Summary", index=False)

    # README sheet
    readme = pd.DataFrame({"Notes": [
        "Rand26 component scores — Hays simple summary method.",
        "Reference: Hays RD, Morales LS (2001). Annals of Medicine 33(5):350-357.",
        "",
        "PCS = mean of z-scored: " + ", ".join(PCS_SUBSCALES),
        "MCS = mean of z-scored: " + ", ".join(MCS_SUBSCALES),
        "",
        f"Missing-data rule: component computed if >= {MIN_SUBSCALES} of 4 subscales non-missing.",
        "Standardization: subscales detected as already z-scored were used as-is;",
        "  raw subscales were z-scored within sample before averaging.",
        "",
        "To use these in downstream analyses:",
        "  1. Drop the 8 individual Rand26_* subscales from the behavioral file.",
        "  2. Merge in Rand26_PCS and Rand26_MCS from this file on subject_ID.",
    ]})
    readme.to_excel(w, sheet_name="ReadMe", index=False)

print("Done.")
