"""
Regional BAG Predictive Models: Behavior vs. Global-BAG Baseline
-----------------------------------------------------------------
Goal:
    Quantify how well behavioral measures approximate MRI-derived *regional*
    brain age gap, and test whether behavior carries *region-specific* signal
    beyond what global brain aging already explains.

Design:
    For each of 12 regional BAGs, fit two paired 10-fold CV models:

      Model A (behavior)  : 186 behavioral features → age-residualized regional BAG
      Model B (baseline)  : age-residualized global BAG → age-residualized regional BAG

    Identical fold splits for both models → per-fold R² values are paired,
    enabling paired t-tests on ΔR² = R²_A - R²_B per fold.

    Interpretation:
      - Model A > Model B significantly  → behavior carries region-specific info
      - Model A ≈ Model B                → behavior only proxies global brain aging
      - Both near 0                      → this region is not behaviorally predictable

Residualization (Stage 1, matches stepwise_BAG_behavioral.py):
    regional_BAG_resid = OLS(regional_BAG ~ Age).resid
    global_BAG_resid   = OLS((BrainAgeR_Global - Age) ~ Age).resid

Model:
    StandardScaler + RidgeCV (α ∈ {0.01, 0.1, 1, 10, 100}) — α selected by
    inner LOOCV on each training fold, preventing test-fold leakage.

Multiple comparisons across 12 regions:
    - Model A R² vs 0             : one-sample t-test on per-fold R², FDR + Bonferroni
    - ΔR² (A − B) > 0             : one-sample t-test on per-fold ΔR², FDR + Bonferroni

Outputs:
    Results/BHI_Regional_Predictive_Models.xlsx
        Summary            - per-region R²/RMSE/MAE/corr for A and B + ΔR²
        Specificity_Test   - paired comparison A vs B with corrected p-values
        FeatureImportance  - standardized ridge coefficients (wide: features × regions)
        FoldMetrics_A      - per-fold per-region metrics for Model A
        FoldMetrics_B      - per-fold per-region metrics for Model B
        Predictions_A      - subject-level predictions from Model A
        Predictions_B      - subject-level predictions from Model B

    Figures/
        BHI_Regional_Fig1_ModelAvsB_R2.pdf     - paired R² bars across regions
        BHI_Regional_Fig2_FeatureHeatmap.pdf   - top features × region coefficient map
        BHI_Regional_Fig3_Scatter.pdf          - actual vs predicted per region (Model A)
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
N_FOLDS = 10
RANDOM_STATE = 42
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
TOP_FEATURES_FOR_HEATMAP = 20

RESULTS_XLSX = "Results/BHI_Regional_Predictive_Models.xlsx"
FIG_R2       = "Figures/BHI_Regional_Fig1_ModelAvsB_R2.pdf"
FIG_HEATMAP  = "Figures/BHI_Regional_Fig2_FeatureHeatmap.pdf"
FIG_SCATTER  = "Figures/BHI_Regional_Fig3_Scatter.pdf"

REGIONS = [
    "LanguageSpecific_Left", "LanguageSpecific_Right",
    "DomainGeneral_Left",    "DomainGeneral_Right",
    "Frontal_Left",          "Frontal_Right",
    "Temporal_Left",         "Temporal_Right",
    "Parietal_Left",         "Parietal_Right",
    "Occipital_Left",        "Occipital_Right",
]

# ── 1. Load & merge ───────────────────────────────────────────────────────────
print("Loading data ...")
bag = pd.read_excel("Results/BHI_Regional_BrainAgeGap.xlsx")
beh = pd.read_excel("Behavioral_Data_Cleaned.xlsx")

df = bag.merge(beh, on="subject_ID", suffixes=("_bag", "_beh"))

# Behavioral predictors: everything that isn't ID / Age / target columns
non_predictor = {"subject_ID", "Age_bag", "Age_beh", "Age_normalized",
                 "BrainAgeR_Global"} | set(REGIONS)
behavioral_predictors = [c for c in df.columns if c not in non_predictor]

for col in behavioral_predictors:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Listwise deletion on all columns we need
needed = behavioral_predictors + REGIONS + ["Age_bag", "BrainAgeR_Global"]
df = df.dropna(subset=needed).reset_index(drop=True)
print(f"Analytic sample: N = {len(df)}, {len(behavioral_predictors)} behavioral features")

age = df["Age_bag"].values
X_beh = df[behavioral_predictors].values

# ── 2. Stage-1 age residualization ────────────────────────────────────────────
def residualize_against_age(y, age_vec):
    X = sm.add_constant(age_vec, has_constant="add")
    return np.asarray(sm.OLS(y, X).fit().resid)

y_regional_resid = {r: residualize_against_age(df[r].values, age) for r in REGIONS}

# Global BAG = BrainAgeR_Global − Age, then residualize vs Age to remove the
# same age artifact. This is the single-feature baseline for Model B.
global_bag = df["BrainAgeR_Global"].values - age
global_bag_resid = residualize_against_age(global_bag, age)

# ── 3. Cross-validated models (shared folds for paired comparison) ────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_splits = list(kf.split(np.arange(len(df))))

def cv_scores(X, y, splits):
    """Return per-fold R², RMSE, MAE, correlation, and out-of-fold predictions."""
    n = len(y)
    oof_pred = np.full(n, np.nan)
    metrics = []
    for fold_idx, (tr, te) in enumerate(splits):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  RidgeCV(alphas=RIDGE_ALPHAS)),
        ])
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        oof_pred[te] = pred
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean((y[te] - pred) ** 2))
        mae = np.mean(np.abs(y[te] - pred))
        corr = np.corrcoef(y[te], pred)[0, 1] if np.std(pred) > 0 else np.nan
        metrics.append({"fold": fold_idx, "R2": r2, "RMSE": rmse, "MAE": mae, "corr": corr})
    return pd.DataFrame(metrics), oof_pred


summary_rows = []
specificity_rows = []
fold_A_all = []
fold_B_all = []
pred_A = pd.DataFrame({"subject_ID": df["subject_ID"]})
pred_B = pd.DataFrame({"subject_ID": df["subject_ID"]})
coef_records = []

X_B = global_bag_resid.reshape(-1, 1)

for region in REGIONS:
    y = y_regional_resid[region]
    print(f"\n── {region} ──")

    # Model A: behavior → regional BAG residual
    metrics_A, oof_A = cv_scores(X_beh, y, fold_splits)
    metrics_A["region"] = region; metrics_A["model"] = "A_behavior"
    fold_A_all.append(metrics_A)
    pred_A[region + "_actual"] = y
    pred_A[region + "_pred"]   = oof_A

    # Model B: global BAG residual → regional BAG residual
    metrics_B, oof_B = cv_scores(X_B, y, fold_splits)
    metrics_B["region"] = region; metrics_B["model"] = "B_globalBAG"
    fold_B_all.append(metrics_B)
    pred_B[region + "_actual"] = y
    pred_B[region + "_pred"]   = oof_B

    # Summary row
    def mstats(df_, col):
        vals = df_[col].values
        return np.nanmean(vals), np.nanstd(vals)

    rA2, sA2 = mstats(metrics_A, "R2");    rB2, sB2 = mstats(metrics_B, "R2")
    rArmse, _ = mstats(metrics_A, "RMSE"); rBrmse, _ = mstats(metrics_B, "RMSE")
    rAmae, _  = mstats(metrics_A, "MAE");  rBmae, _  = mstats(metrics_B, "MAE")
    rAcorr, _ = mstats(metrics_A, "corr"); rBcorr, _ = mstats(metrics_B, "corr")

    summary_rows.append({
        "region": region,
        "A_R2_mean": rA2, "A_R2_sd": sA2,
        "A_RMSE": rArmse, "A_MAE": rAmae, "A_corr": rAcorr,
        "B_R2_mean": rB2, "B_R2_sd": sB2,
        "B_RMSE": rBrmse, "B_MAE": rBmae, "B_corr": rBcorr,
        "delta_R2_mean": rA2 - rB2,
    })

    # Paired tests across folds
    r2_A = metrics_A["R2"].values
    r2_B = metrics_B["R2"].values
    delta = r2_A - r2_B

    # Test 1: is Model A > 0? (behavior predictive at all)
    t_A, p_A = stats.ttest_1samp(r2_A, 0.0)
    # Test 2: is ΔR² > 0? (behavior beats global-BAG baseline)
    t_D, p_D = stats.ttest_1samp(delta, 0.0)

    specificity_rows.append({
        "region": region,
        "A_R2_mean": rA2, "B_R2_mean": rB2, "delta_R2_mean": rA2 - rB2,
        "delta_R2_sd": np.std(delta, ddof=1),
        "t_A_vs_0": t_A, "p_A_vs_0": p_A,
        "t_delta_vs_0": t_D, "p_delta_vs_0": p_D,
    })

    # Feature importance from Model A refit on full data (standardized coefs)
    pipe_full = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=RIDGE_ALPHAS)),
    ]).fit(X_beh, y)
    coefs = pipe_full.named_steps["ridge"].coef_
    best_alpha = pipe_full.named_steps["ridge"].alpha_
    print(f"   Model A: R² = {rA2:+.3f} ± {sA2:.3f}  |  Model B: R² = {rB2:+.3f}  |  ΔR² = {rA2 - rB2:+.3f}  |  α* = {best_alpha}")
    for feat, c in zip(behavioral_predictors, coefs):
        coef_records.append({"feature": feat, "region": region, "coef": c})

# ── 4. Multiple-comparisons correction across regions ────────────────────────
summary_df = pd.DataFrame(summary_rows)
spec_df    = pd.DataFrame(specificity_rows)

for col, prefix in [("p_A_vs_0", "A_vs_0"), ("p_delta_vs_0", "delta_vs_0")]:
    pvals = spec_df[col].values
    _, fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    spec_df[f"{prefix}_p_FDR"] = fdr
    spec_df[f"{prefix}_sig_FDR"] = fdr < 0.05
    bonf_thresh = 0.05 / len(REGIONS)
    spec_df[f"{prefix}_sig_Bonf"] = pvals < bonf_thresh

# Long → wide coefficient matrix for heatmap (features × regions)
coef_df = pd.DataFrame(coef_records)
coef_wide = coef_df.pivot(index="feature", columns="region", values="coef")
coef_wide = coef_wide.reindex(columns=REGIONS)

# ── 5. Write Excel ────────────────────────────────────────────────────────────
print(f"\nWriting {RESULTS_XLSX} ...")
fold_A_df = pd.concat(fold_A_all, ignore_index=True)
fold_B_df = pd.concat(fold_B_all, ignore_index=True)

with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as w:
    summary_df.to_excel(w, sheet_name="Summary", index=False)
    spec_df.to_excel(w, sheet_name="Specificity_Test", index=False)
    coef_wide.reset_index().to_excel(w, sheet_name="FeatureImportance", index=False)
    fold_A_df.to_excel(w, sheet_name="FoldMetrics_A", index=False)
    fold_B_df.to_excel(w, sheet_name="FoldMetrics_B", index=False)
    pred_A.to_excel(w, sheet_name="Predictions_A", index=False)
    pred_B.to_excel(w, sheet_name="Predictions_B", index=False)

# ── 6. Figure 1: R² bars, Model A vs B per region ────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(REGIONS))
w = 0.38
ax.bar(x - w/2, summary_df["A_R2_mean"], width=w, yerr=summary_df["A_R2_sd"],
       label="Model A (behavior)", color="#c94a4a", capsize=3)
ax.bar(x + w/2, summary_df["B_R2_mean"], width=w, yerr=summary_df["B_R2_sd"],
       label="Model B (global BAG)", color="#4a6fa5", capsize=3)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(REGIONS, rotation=45, ha="right")
ax.set_ylabel("Cross-validated R² (mean ± SD across 10 folds)")
ax.set_title("Regional BAG Prediction: Behavior vs. Global-BAG Baseline")
# Mark regions where ΔR² is FDR-significant
for i, row in spec_df.iterrows():
    if row["delta_vs_0_sig_FDR"]:
        y_mark = max(row["A_R2_mean"], row["B_R2_mean"]) + 0.02
        ax.text(i, y_mark, "*", ha="center", va="bottom", fontsize=14, fontweight="bold")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(FIG_R2)
plt.close(fig)

# ── 7. Figure 2: feature × region coefficient heatmap (top features) ─────────
# Rank features by max abs coefficient across regions
ranked = coef_wide.abs().max(axis=1).sort_values(ascending=False)
top_feats = ranked.head(TOP_FEATURES_FOR_HEATMAP).index.tolist()
heat = coef_wide.loc[top_feats]

fig, ax = plt.subplots(figsize=(10, 0.4 * len(top_feats) + 2))
vmax = np.nanmax(np.abs(heat.values))
sns.heatmap(heat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Standardized ridge coefficient"}, ax=ax)
ax.set_title(f"Top {TOP_FEATURES_FOR_HEATMAP} Behavioral Features × Regional BAG (Model A)")
ax.set_xlabel("Region")
ax.set_ylabel("Behavioral feature")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIG_HEATMAP)
plt.close(fig)

# ── 8. Figure 3: actual vs predicted scatter per region (Model A) ────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
for ax_, region in zip(axes.flat, REGIONS):
    actual = pred_A[region + "_actual"]
    pred   = pred_A[region + "_pred"]
    r2 = summary_df.loc[summary_df["region"] == region, "A_R2_mean"].iloc[0]
    ax_.scatter(actual, pred, s=10, alpha=0.5, color="#333")
    lo, hi = np.nanmin([actual.min(), pred.min()]), np.nanmax([actual.max(), pred.max()])
    ax_.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax_.set_title(f"{region}\nR² = {r2:+.3f}", fontsize=10)
    ax_.set_xlabel("Actual age-residualized BAG")
    ax_.set_ylabel("Predicted")
plt.suptitle("Model A (behavior → regional BAG): out-of-fold predictions", y=1.00)
plt.tight_layout()
plt.savefig(FIG_SCATTER)
plt.close(fig)

# ── 9. Print concise summary ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("REGIONAL PREDICTIVE MODELING SUMMARY")
print("=" * 72)
display_cols = ["region", "A_R2_mean", "B_R2_mean", "delta_R2_mean",
                "p_A_vs_0", "A_vs_0_p_FDR", "A_vs_0_sig_FDR",
                "p_delta_vs_0", "delta_vs_0_p_FDR", "delta_vs_0_sig_FDR"]
print(spec_df[display_cols].to_string(index=False,
      float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else str(v)))
print("\nOutputs:")
print(f"  {RESULTS_XLSX}")
print(f"  {FIG_R2}")
print(f"  {FIG_HEATMAP}")
print(f"  {FIG_SCATTER}")
