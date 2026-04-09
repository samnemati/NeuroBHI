"""
Bidirectional Stepwise Regression: Behavioral Predictors → Regional BAG
------------------------------------------------------------------------
- Fixed covariate: Age (always retained, never removed)
- Candidate predictors: all behavioral scores
- Entry criterion:  p < 0.05
- Removal criterion: p > 0.10
- Missing data: listwise deletion (subjects missing any predictor excluded)

Output Excel sheets:
  Summary                  – one row per region (R², Adj-R², F, selected vars)
  <Region>                 – full coefficient table for that region's final model
  Final_Model_Residuals    – residuals from the final model (Age + selected predictors)
  Age_Corrected_Residuals  – residuals from Age-only model (age-partialled BAG)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
P_ENTER  = 0.05
P_REMOVE = 0.10
OUTPUT_FILE = "Stepwise_BAG_Behavioral_Results.xlsx"

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data …")
bag = pd.read_excel("ABC_RegBAG_Clcalc_weightedVol.xlsx")
beh = pd.read_excel("Behavioral_Data_Cleaned.xlsx")

# ── 2. Merge on subject_ID ───────────────────────────────────────────────────
df = bag.merge(beh, on="subject_ID", suffixes=("_bag", "_beh"))
# After merge: Age_bag (from BAG file) and Age_beh (from behavioral file)

# ── 3. Define regions and behavioral predictors ──────────────────────────────
regions = [
    "Language_Specific", "Domain_General", "Frontal", "Temporal",
    "Parietal", "Occipital", "Subcortical", "Cerebellum", "Limbic",
]

non_predictor = {"subject_ID", "Age_bag", "Age_beh", "Age_normalized"} | set(regions)
beh_predictors = [c for c in df.columns if c not in non_predictor]

# Coerce all behavioral columns to numeric (handles object-dtype Excel artefacts)
for col in beh_predictors:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 4. Listwise deletion ─────────────────────────────────────────────────────
all_needed = beh_predictors + regions + ["Age_bag"]
df_clean = df.dropna(subset=all_needed).reset_index(drop=True)
print(f"Subjects: {len(df)} total → {len(df_clean)} after listwise deletion")

# ── 5. Prepare design matrices ───────────────────────────────────────────────
X_fixed      = df_clean[["Age_bag"]].rename(columns={"Age_bag": "Age"})
X_candidates = df_clean[beh_predictors].copy()

# ── 6. Bidirectional stepwise function ───────────────────────────────────────
def stepwise(X_fixed, X_candidates, y, p_enter=P_ENTER, p_remove=P_REMOVE):
    """
    Bidirectional stepwise OLS selection.
    X_fixed is always included and never removed.
    Returns list of selected predictor names from X_candidates.
    """
    selected = []

    while True:
        changed = False

        # ── Forward step: find the best candidate to add ──────────────────
        excluded = [c for c in X_candidates.columns if c not in selected]
        best_pval = p_enter
        best_cand = None

        for cand in excluded:
            X_try = pd.concat(
                [X_fixed, X_candidates[selected + [cand]]], axis=1
            )
            X_try = sm.add_constant(X_try, has_constant="add")
            pval = sm.OLS(y, X_try).fit().pvalues.get(cand, 1.0)
            if pval < best_pval:
                best_pval = pval
                best_cand = cand

        if best_cand is not None:
            selected.append(best_cand)
            changed = True

        # ── Backward step: remove any selected predictor that now exceeds threshold
        if selected:
            X_try = pd.concat(
                [X_fixed, X_candidates[selected]], axis=1
            )
            X_try = sm.add_constant(X_try, has_constant="add")
            pvals = sm.OLS(y, X_try).fit().pvalues
            pvals_sel = {p: pvals[p] for p in selected}
            worst = max(pvals_sel, key=pvals_sel.get)

            if pvals_sel[worst] > p_remove:
                selected.remove(worst)
                changed = True

        if not changed:
            break

    return selected

# ── 7. Run stepwise for each region ──────────────────────────────────────────
results       = {}
summary_rows  = []
final_resid   = df_clean[["subject_ID"]].copy()
age_resid     = df_clean[["subject_ID"]].copy()

for region in regions:
    y = df_clean[region]
    print(f"\n{'─'*60}")
    print(f"Region: {region}")

    # Age-only model → age-corrected residuals
    X_age    = sm.add_constant(X_fixed, has_constant="add")
    age_mod  = sm.OLS(y, X_age).fit()
    age_resid[region] = age_mod.resid.values

    # Bidirectional stepwise
    selected = stepwise(X_fixed, X_candidates, y)
    print(f"  Selected ({len(selected)}): {selected if selected else 'None'}")

    # Final model: Age + selected predictors
    if selected:
        X_final_cols = pd.concat([X_fixed, X_candidates[selected]], axis=1)
    else:
        X_final_cols = X_fixed.copy()

    X_final    = sm.add_constant(X_final_cols, has_constant="add")
    final_mod  = sm.OLS(y, X_final).fit()

    # Final model residuals
    final_resid[region] = final_mod.resid.values

    # Coefficient table
    ci = final_mod.conf_int()
    coef_df = pd.DataFrame({
        "Predictor":   final_mod.params.index,
        "Coefficient": final_mod.params.values.round(6),
        "Std_Error":   final_mod.bse.values.round(6),
        "t_value":     final_mod.tvalues.values.round(4),
        "p_value":     final_mod.pvalues.values.round(6),
        "CI_2.5%":     ci[0].values.round(6),
        "CI_97.5%":    ci[1].values.round(6),
    })

    results[region] = {
        "model":   final_mod,
        "selected": selected,
        "coef_df": coef_df,
    }

    summary_rows.append({
        "Region":               region,
        "N":                    int(final_mod.nobs),
        "R2":                   round(final_mod.rsquared, 4),
        "Adj_R2":               round(final_mod.rsquared_adj, 4),
        "F_statistic":          round(final_mod.fvalue, 4),
        "F_p_value":            round(final_mod.f_pvalue, 6),
        "n_selected_predictors": len(selected),
        "Selected_predictors":  "; ".join(selected) if selected else "None",
    })

    print(f"  R² = {final_mod.rsquared:.4f}  Adj-R² = {final_mod.rsquared_adj:.4f}  "
          f"F({int(final_mod.df_model)},{int(final_mod.df_resid)}) = {final_mod.fvalue:.3f}  "
          f"p = {final_mod.f_pvalue:.4g}")

# ── 8. Write results to Excel ─────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Writing results to {OUTPUT_FILE} …")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    # Summary sheet
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # Per-region coefficient sheets (sheet names ≤ 31 chars)
    for region in regions:
        sheet = region[:31]
        results[region]["coef_df"].to_excel(writer, sheet_name=sheet, index=False)

    # Final model residuals
    final_resid.to_excel(writer, sheet_name="Final_Model_Residuals", index=False)

    # Age-corrected residuals (age-only model residuals)
    age_resid.to_excel(writer, sheet_name="Age_Corrected_Residuals", index=False)

    # ── Auto-fit column widths ────────────────────────────────────────────
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

print(f"Done. Results saved to: {OUTPUT_FILE}")
