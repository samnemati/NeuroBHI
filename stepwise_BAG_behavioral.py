"""
Two-Stage Stepwise Regression: Behavioral Predictors → Age-Corrected Regional BAG
----------------------------------------------------------------------------------
Stage 1 — Age correction (per region):
    Fit:     BAG ~ Age  (OLS)
    Save:    Residuals = age-partialled BAG  (variance due to age removed)

Stage 2 — Behavioral stepwise (per region):
    Outcome: age-partialled BAG residuals from Stage 1
    Predictors: all behavioral scores (Age NOT included — already removed)
    Selection: bidirectional stepwise OLS
      - Entry criterion:  p < 0.05
      - Removal criterion: p > 0.10
    Missing data: listwise deletion (subjects missing any predictor excluded)

Rationale:
    BAG is strongly negatively correlated with chronological age (a known
    regression-to-the-mean artifact in brain age studies). Including Age as
    a covariate in a single-stage model controls for it statistically, but
    Age dominates R² and makes the overall fit appear artificially high
    (R² ~ 0.94–0.999). The two-stage approach first removes age variance
    entirely, then asks: how much do behavioral factors explain of what is
    left? The R² reported here reflects only behavioral variance, making
    results interpretable and comparable across regions.

Multiple Comparisons Correction:
    Two corrections are applied to the Stage 2 F-test p-values across the
    N_REGIONS region models, following Busby et al. (2024):

    1. Bonferroni: corrected threshold = α / N_REGIONS  (0.05 / 12 = 0.0042)
       Conservative; controls family-wise error rate (FWER).
       Busby et al. used this approach for their regional comparisons.

    2. FDR (Benjamini-Hochberg): adjusts p-values to control the false
       discovery rate at 5%. Less conservative than Bonferroni; appropriate
       when region tests are correlated (as brain regions typically are).

    Both are applied to the overall model F-test p-value (does this region's
    behavioral model explain significant variance?). Individual predictor
    p-values within each model are reported as exploratory only — stepwise
    selection already used p-values to choose predictors, so treating those
    same p-values as confirmatory tests would be circular.

Output Excel sheets:
    Summary                  – one row per region; includes raw F p-value,
                               Bonferroni-corrected threshold, FDR-adjusted
                               p-value, and significance flags for both methods
    <Region>                 – full coefficient table for Stage 2 final model
    Age_Corrected_Residuals  – Stage 1 residuals (age-partialled BAG), used as
                               outcome in Stage 2
    Stage2_Model_Residuals   – Stage 2 residuals (unexplained after age + behavior)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
P_ENTER  = 0.05
P_REMOVE = 0.10
OUTPUT_FILE = "Stepwise_BAG_Behavioral_Results.xlsx"

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
bag = pd.read_excel("./Results/BHI_Regional_BrainAgeGap.xlsx")
beh = pd.read_excel("Behavioral_Data_Cleaned.xlsx")

# ── 2. Merge on subject_ID ────────────────────────────────────────────────────
df = bag.merge(beh, on="subject_ID", suffixes=("_bag", "_beh"))

# ── 3. Define regions and behavioral predictors ───────────────────────────────
regions = [
    "LanguageSpecific_Left", "LanguageSpecific_Right", "DomainGeneral_Left", "DomainGeneral_Right", "Frontal_Left", "Frontal_Right", "Temporal_Left",
    "Temporal_Right", "Parietal_Left", "Parietal_Right", "Occipital_Left", "Occipital_Right"
]

# Exclude non-predictor columns from the behavioral candidate list
non_predictor_cols = {"subject_ID", "Age_bag", "Age_beh", "Age_normalized", "BrainAgeR_Global"} | set(regions)
behavioral_predictors = [c for c in df.columns if c not in non_predictor_cols]

# Coerce behavioral columns to numeric (handles object-dtype Excel artefacts)
for col in behavioral_predictors:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 4. Listwise deletion ──────────────────────────────────────────────────────
all_needed = behavioral_predictors + regions + ["Age_bag"]
df_clean = df.dropna(subset=all_needed).reset_index(drop=True)
print(f"Subjects: {len(df)} total → {len(df_clean)} after listwise deletion")

# Rename Age column for clarity
age_series = df_clean[["Age_bag"]].rename(columns={"Age_bag": "Age"})
X_behavioral = df_clean[behavioral_predictors].copy()

# ── 5. Bidirectional stepwise function ────────────────────────────────────────
def stepwise(X_candidates, y, p_enter=P_ENTER, p_remove=P_REMOVE):
    """
    Bidirectional stepwise OLS selection with no fixed covariates.

    Age has already been regressed out of y (Stage 1), so it is NOT included
    here — adding it again would re-introduce the very variance we removed.

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
            X_try = sm.add_constant(X_candidates[selected + [cand]], has_constant="add")
            pval = sm.OLS(y, X_try).fit().pvalues.get(cand, 1.0)
            if pval < best_pval:
                best_pval = pval
                best_cand = cand

        if best_cand is not None:
            selected.append(best_cand)
            changed = True

        # ── Backward step: remove any predictor that now exceeds threshold ─
        if selected:
            X_try = sm.add_constant(X_candidates[selected], has_constant="add")
            pvals = sm.OLS(y, X_try).fit().pvalues
            pvals_sel = {p: pvals[p] for p in selected if p in pvals}
            if pvals_sel:
                worst = max(pvals_sel, key=pvals_sel.get)
                if pvals_sel[worst] > p_remove:
                    selected.remove(worst)
                    changed = True

        if not changed:
            break

    return selected

# ── 6. Run two-stage model for each region ────────────────────────────────────
results        = {}
summary_rows   = []
age_resid_df   = df_clean[["subject_ID"]].copy()   # Stage 1 residuals
stage2_resid_df = df_clean[["subject_ID"]].copy()  # Stage 2 residuals

for region in regions:
    y_bag = df_clean[region]
    print(f"\n{'─'*60}")
    print(f"Region: {region}")

    # ── Stage 1: Regress out Age ──────────────────────────────────────────
    X_age      = sm.add_constant(age_series, has_constant="add")
    age_model  = sm.OLS(y_bag, X_age).fit()
    age_resid  = age_model.resid           # age-partialled BAG
    age_r2     = age_model.rsquared

    age_resid_df[region] = age_resid.values
    print(f"  Stage 1 — Age-only model: R² = {age_r2:.4f}  "
          f"(Age coef = {age_model.params['Age']:.4f}, "
          f"p = {age_model.pvalues['Age']:.2e})")

    # ── Stage 2: Stepwise on age-partialled residuals ─────────────────────
    y_resid  = pd.Series(age_resid.values, index=df_clean.index)
    selected = stepwise(X_behavioral, y_resid)
    print(f"  Stage 2 — Selected ({len(selected)}): {selected if selected else 'None'}")

    # Fit final Stage 2 model
    if selected:
        X_final = sm.add_constant(X_behavioral[selected], has_constant="add")
    else:
        # No predictors selected: intercept-only (R² = 0 by definition)
        X_final = sm.add_constant(pd.DataFrame(np.zeros((len(y_resid), 1)),
                                               columns=["_placeholder"]), has_constant="add")

    final_model = sm.OLS(y_resid, X_final).fit()
    stage2_resid_df[region] = final_model.resid.values

    # Coefficient table (exclude placeholder)
    param_names = [p for p in final_model.params.index if p != "_placeholder"]
    ci = final_model.conf_int()
    coef_df = pd.DataFrame({
        "Predictor":   param_names,
        "Coefficient": final_model.params[param_names].round(6).values,
        "Std_Error":   final_model.bse[param_names].round(6).values,
        "t_value":     final_model.tvalues[param_names].round(4).values,
        "p_value":     final_model.pvalues[param_names].round(6).values,
        "CI_2.5%":     ci.loc[param_names, 0].round(6).values,
        "CI_97.5%":    ci.loc[param_names, 1].round(6).values,
    })

    results[region] = {"model": final_model, "selected": selected, "coef_df": coef_df}

    r2     = final_model.rsquared     if selected else 0.0
    adj_r2 = final_model.rsquared_adj if selected else 0.0
    f_stat = final_model.fvalue       if selected else float("nan")
    f_pval = final_model.f_pvalue     if selected else float("nan")

    print(f"  Stage 2 — Behavioral R² = {r2:.4f}  Adj-R² = {adj_r2:.4f}  "
          f"F = {f_stat:.3f}  p = {f_pval:.4g}")
    print(f"  Interpretation: behavioral predictors explain {r2*100:.1f}% of "
          f"the age-independent BAG variance in {region}")

    summary_rows.append({
        "Region":                    region,
        "N":                         int(final_model.nobs),
        "Stage1_Age_R2":             round(age_r2, 4),
        "Stage2_Behavioral_R2":      round(r2, 4),
        "Stage2_Behavioral_Adj_R2":  round(adj_r2, 4),
        "Stage2_F_statistic":        round(f_stat, 4) if not np.isnan(f_stat) else "n/a",
        "Stage2_F_p_value":          round(f_pval, 6) if not np.isnan(f_pval) else "n/a",
        "n_selected_predictors":     len(selected),
        "Selected_predictors":       "; ".join(selected) if selected else "None",
    })

# ── 7. Multiple comparisons correction across regions ─────────────────────────
summary_df = pd.DataFrame(summary_rows)
n_regions   = len(regions)
alpha       = 0.05

# Extract raw F-test p-values; regions with no selected predictors have "n/a"
raw_pvals = pd.to_numeric(summary_df["Stage2_F_p_value"], errors="coerce")
has_model = raw_pvals.notna()   # False for regions where no predictors were selected

# ── Bonferroni (Busby et al. approach) ───────────────────────────────────────
bonf_threshold = alpha / n_regions
summary_df["Bonferroni_threshold"]    = bonf_threshold
summary_df["Bonferroni_significant"]  = has_model & (raw_pvals < bonf_threshold)

# ── FDR — Benjamini-Hochberg ─────────────────────────────────────────────────
# Run only on regions that have a model (non-NaN p-value)
fdr_adjusted = np.full(n_regions, np.nan)
if has_model.any():
    _, pvals_corrected, _, _ = multipletests(
        raw_pvals[has_model], alpha=alpha, method="fdr_bh"
    )
    fdr_adjusted[has_model.values] = pvals_corrected

summary_df["FDR_adjusted_p"]      = fdr_adjusted.round(6)
summary_df["FDR_significant"]     = ~np.isnan(fdr_adjusted) & (fdr_adjusted < alpha)

# ── Print correction summary ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Multiple comparisons correction (N regions = {n_regions}, α = {alpha})")
print(f"  Bonferroni threshold : {bonf_threshold:.4f}  "
      f"({summary_df['Bonferroni_significant'].sum()} regions survive)")
print(f"  FDR (BH) α = {alpha}    "
      f"({summary_df['FDR_significant'].sum()} regions survive)")
print(f"\n  {'Region':<28} {'Raw p':>10} {'FDR p':>10} {'Bonf.sig':>10} {'FDR sig':>10}")
for _, row in summary_df.iterrows():
    raw_p = row["Stage2_F_p_value"]
    fdr_p = f"{row['FDR_adjusted_p']:.4f}" if not np.isnan(row["FDR_adjusted_p"]) else "n/a"
    bonf  = "YES" if row["Bonferroni_significant"] else "no"
    fdr   = "YES" if row["FDR_significant"]        else "no"
    print(f"  {row['Region']:<28} {str(raw_p):>10} {fdr_p:>10} {bonf:>10} {fdr:>10}")

# ── 8. Write results to Excel ─────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Writing results to {OUTPUT_FILE} ...")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    # Summary sheet — includes raw F p-value, Bonferroni threshold + flag,
    # FDR-adjusted p-value + flag
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # Per-region Stage 2 coefficient sheets
    for region in regions:
        sheet_name = region[:31]
        results[region]["coef_df"].to_excel(writer, sheet_name=sheet_name, index=False)

    # Stage 1 residuals (age-partialled BAG) — the outcome fed into Stage 2
    age_resid_df.to_excel(writer, sheet_name="Age_Corrected_Residuals", index=False)

    # Stage 2 residuals (unexplained after age + behavioral predictors)
    stage2_resid_df.to_excel(writer, sheet_name="Stage2_Model_Residuals", index=False)

    # Auto-fit column widths
    for sname in writer.sheets:
        ws = writer.sheets[sname]
        for col in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

print(f"Done. Results saved to: {OUTPUT_FILE}")
