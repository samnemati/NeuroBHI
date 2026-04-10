"""
visualize_BAG_results.py
------------------------
Publication-ready PDF figures for the two-stage stepwise BAG analysis.

Figures produced (all saved to ./Figures/):
    Fig1_R2_BarChart.pdf          – Behavioral R² per region with significance flags
    Fig2_Correction_Comparison.pdf– Bonferroni vs FDR p-value comparison
    Fig3_Coefficient_Heatmap.pdf  – Standardized coefficients: predictors × regions
    Fig4_BAG_Distributions.pdf    – Violin plots of age-corrected BAG per region
    Fig5_TopPredictor_Scatter.pdf – Scatter plots: top predictor vs BAG for key regions

DEPENDENCIES
    pip install pandas numpy matplotlib seaborn scipy openpyxl
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_FILE = "Stepwise_BAG_Behavioral_Results.xlsx"
BAG_FILE     = "./Results/BHI_Regional_BrainAgeGap.xlsx"
BEH_FILE     = "Behavioral_Data_Cleaned.xlsx"
OUT_DIR      = "./Figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "pdf.fonttype":     42,   # embeds fonts for journal submission
    "ps.fonttype":      42,
})

PALETTE = {
    "bonf_fdr":   "#2166ac",   # significant by both
    "fdr_only":   "#74add1",   # FDR only
    "neither":    "#d9d9d9",   # not significant
    "left":       "#4393c3",
    "right":      "#d6604d",
    "heatmap_pos":"#d73027",
    "heatmap_neg":"#4575b4",
}

REGION_ORDER = [
    "LanguageSpecific_Left",  "LanguageSpecific_Right",
    "DomainGeneral_Left",     "DomainGeneral_Right",
    "Frontal_Left",           "Frontal_Right",
    "Temporal_Left",          "Temporal_Right",
    "Parietal_Left",          "Parietal_Right",
    "Occipital_Left",         "Occipital_Right",
]

SHORT_LABELS = {
    "LanguageSpecific_Left":  "Lang.Spec. L",
    "LanguageSpecific_Right": "Lang.Spec. R",
    "DomainGeneral_Left":     "Dom.Gen. L",
    "DomainGeneral_Right":    "Dom.Gen. R",
    "Frontal_Left":           "Frontal L",
    "Frontal_Right":          "Frontal R",
    "Temporal_Left":          "Temporal L",
    "Temporal_Right":         "Temporal R",
    "Parietal_Left":          "Parietal L",
    "Parietal_Right":         "Parietal R",
    "Occipital_Left":         "Occipital L",
    "Occipital_Right":        "Occipital R",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading results ...")
xl      = pd.ExcelFile(RESULTS_FILE)
summary = xl.parse("Summary")
summary = summary.set_index("Region").reindex(REGION_ORDER).reset_index()

age_resid = xl.parse("Age_Corrected_Residuals")

coef_data = {}
for region in REGION_ORDER:
    df = xl.parse(region[:31])
    df = df[df["Predictor"] != "const"].copy()
    coef_data[region] = df.set_index("Predictor")

# ── Helper: significance color per row ────────────────────────────────────────
def sig_color(row):
    if row["Bonferroni_significant"] and row["FDR_significant"]:
        return PALETTE["bonf_fdr"]
    elif row["FDR_significant"]:
        return PALETTE["fdr_only"]
    else:
        return PALETTE["neither"]

bar_colors = [sig_color(row) for _, row in summary.iterrows()]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Behavioral R² bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 1: R² bar chart ...")

fig, ax = plt.subplots(figsize=(8, 5))

labels = [SHORT_LABELS[r] for r in summary["Region"]]
r2vals = summary["Stage2_Behavioral_R2"].fillna(0).values
bars   = ax.barh(labels, r2vals, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.65)

# Annotate R² value on each bar
for bar, val, (_, row) in zip(bars, r2vals, summary.iterrows()):
    xpos = val + 0.002
    label = f"{val:.3f}"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=8)

# Reference lines
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Behavioral R² (age-independent variance explained)", labelpad=8)
ax.set_title("Stage 2: Behavioral Predictors → Regional BAG\n"
             "(after age correction, Stage 1)", pad=12)
ax.set_xlim(0, max(r2vals) * 1.25)
ax.invert_yaxis()

# Legend
legend_patches = [
    mpatches.Patch(color=PALETTE["bonf_fdr"], label="Significant: Bonferroni + FDR"),
    mpatches.Patch(color=PALETTE["fdr_only"], label="Significant: FDR only"),
    mpatches.Patch(color=PALETTE["neither"],  label="Not significant"),
]
ax.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=8)

# Bonferroni threshold annotation
bonf = summary["Bonferroni_threshold"].iloc[0]
ax.axvline(bonf, color="#525252", linewidth=1, linestyle="--", alpha=0.6)
ax.text(bonf + 0.001, len(labels) - 0.5,
        f"Bonferroni\nthreshold", fontsize=7, color="#525252", va="top")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Fig1_R2_BarChart.pdf"), bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Bonferroni vs FDR p-value comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 2: Correction comparison ...")

fig, ax = plt.subplots(figsize=(8, 5))

raw_p  = pd.to_numeric(summary["Stage2_F_p_value"], errors="coerce").values
fdr_p  = pd.to_numeric(summary["FDR_adjusted_p"],   errors="coerce").values
labels = [SHORT_LABELS[r] for r in summary["Region"]]

x      = np.arange(len(labels))
width  = 0.35

# Log-transform for display; handle NaN (Frontal_Left: no model)
def safe_neglog10(arr):
    out = np.full(len(arr), np.nan)
    for i, v in enumerate(arr):
        if not np.isnan(v) and v > 0:
            out[i] = -np.log10(v)
    return out

raw_log  = safe_neglog10(raw_p)
fdr_log  = safe_neglog10(fdr_p)
bonf_thr = -np.log10(summary["Bonferroni_threshold"].iloc[0])
fdr_thr  = -np.log10(0.05)

bars1 = ax.bar(x - width/2, raw_log, width, label="Raw p-value",
               color="#6baed6", edgecolor="white")
bars2 = ax.bar(x + width/2, fdr_log, width, label="FDR-adjusted p",
               color="#fd8d3c", edgecolor="white")

ax.axhline(bonf_thr, color=PALETTE["bonf_fdr"], linewidth=1.2,
           linestyle="--", label=f"Bonferroni threshold (α/12)")
ax.axhline(fdr_thr,  color="#d7301f", linewidth=1.2,
           linestyle=":",  label="FDR threshold (α = 0.05)")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_ylabel("−log₁₀(p-value)", labelpad=8)
ax.set_title("Multiple Comparisons Correction: Bonferroni vs. FDR (BH)", pad=12)
ax.legend(frameon=False, fontsize=8)

# Mark Frontal_Left (no model) explicitly
frontal_l_idx = REGION_ORDER.index("Frontal_Left")
ax.text(frontal_l_idx, 0.1, "no model", ha="center", va="bottom",
        fontsize=7, color="#969696", rotation=90)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Fig2_Correction_Comparison.pdf"), bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Coefficient heatmap (predictors × regions)
# Includes only predictors selected in 2+ regions for readability
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 3: Coefficient heatmap ...")

# Count appearances
from collections import Counter
pred_counts = Counter()
for r in REGION_ORDER:
    for p in coef_data[r].index:
        pred_counts[p] += 1

# Keep predictors selected in ≥2 regions
multi_preds = sorted([p for p, c in pred_counts.items() if c >= 2],
                     key=lambda p: -pred_counts[p])

# Build standardized coefficient matrix
heatmap_data = pd.DataFrame(index=multi_preds, columns=REGION_ORDER, dtype=float)
for region in REGION_ORDER:
    df = coef_data[region]
    for pred in multi_preds:
        if pred in df.index:
            heatmap_data.loc[pred, region] = df.loc[pred, "Coefficient"]

# Standardize each row (predictor) so colors reflect direction, not scale
heatmap_std = heatmap_data.copy()
for pred in multi_preds:
    row = heatmap_data.loc[pred].dropna()
    if len(row) > 1:
        rng = row.abs().max()
        if rng > 0:
            heatmap_std.loc[pred] = heatmap_data.loc[pred] / rng

col_labels  = [SHORT_LABELS[r] for r in REGION_ORDER]

fig_h = max(5, len(multi_preds) * 0.45 + 2)
fig, ax = plt.subplots(figsize=(10, fig_h))

sns.heatmap(
    heatmap_std.astype(float),
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor="#dddddd",
    annot=False,
    xticklabels=col_labels,
    yticklabels=multi_preds,
    cbar_kws={"label": "Scaled coefficient\n(+: larger BAG, −: smaller BAG)",
              "shrink": 0.6},
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_title("Behavioral Predictors Selected in ≥2 Brain Regions\n"
             "(color = scaled regression coefficient direction)", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Behavioral predictor", labelpad=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Fig3_Coefficient_Heatmap.pdf"), bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Violin plots of age-corrected BAG distributions
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 4: BAG violin plots ...")

# Melt to long form
bag_long = age_resid.melt(id_vars="subject_ID",
                           value_vars=REGION_ORDER,
                           var_name="Region", value_name="AgeCorr_BAG")
bag_long["Hemisphere"] = bag_long["Region"].apply(
    lambda r: "Left" if r.endswith("_Left") else "Right")
bag_long["RegionShort"] = bag_long["Region"].map(SHORT_LABELS)

fig, ax = plt.subplots(figsize=(12, 5))

hemi_palette = {"Left": PALETTE["left"], "Right": PALETTE["right"]}
region_short_order = [SHORT_LABELS[r] for r in REGION_ORDER]

sns.violinplot(
    data=bag_long,
    x="RegionShort", y="AgeCorr_BAG",
    hue="Hemisphere",
    order=region_short_order,
    hue_order=["Left", "Right"],
    palette=hemi_palette,
    split=True,
    inner="quartile",
    linewidth=0.8,
    ax=ax,
)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel("")
ax.set_ylabel("Age-corrected BAG (years)", labelpad=8)
ax.set_title("Distribution of Age-Corrected Brain Age Gap by Region and Hemisphere", pad=12)
ax.legend(title="Hemisphere", frameon=False, loc="upper right")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Fig4_BAG_Distributions.pdf"), bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Top-predictor scatter plots for the 4 most informative regions
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 5: Top predictor scatter plots ...")

# Pick regions by highest Stage2 R² (excluding Frontal_Left which has no model)
summary_sorted = summary[summary["Stage2_Behavioral_R2"] > 0].sort_values(
    "Stage2_Behavioral_R2", ascending=False)
top_regions = summary_sorted["Region"].head(4).tolist()

# Load behavioral data and merge with age-corrected residuals
beh = pd.read_excel(BEH_FILE)
df_scatter = age_resid.merge(beh, on="subject_ID", how="inner")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, region in zip(axes, top_regions):
    # Top predictor = first entry in coef table (highest |coef| after stepwise)
    top_pred = coef_data[region].index[0]
    coef_val = coef_data[region].loc[top_pred, "Coefficient"]

    y_col = region
    x_col = top_pred

    if x_col not in df_scatter.columns or y_col not in df_scatter.columns:
        ax.text(0.5, 0.5, f"Data not found\n{x_col}", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        continue

    plot_df = df_scatter[[x_col, y_col]].dropna()
    x = pd.to_numeric(plot_df[x_col], errors="coerce")
    y = pd.to_numeric(plot_df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]

    r_val, p_val = stats.pearsonr(x, y)

    hemi_color = PALETTE["left"] if region.endswith("_Left") else PALETTE["right"]

    ax.scatter(x, y, s=18, alpha=0.45, color=hemi_color, linewidths=0)

    # Regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#333333", linewidth=1.5)

    direction = "↑ BAG" if coef_val > 0 else "↓ BAG"
    ax.set_title(f"{SHORT_LABELS[region]}\nTop predictor: {top_pred}  ({direction})",
                 fontsize=9, pad=6)
    ax.set_xlabel(top_pred, fontsize=8)
    ax.set_ylabel("Age-corrected BAG (years)", fontsize=8)
    ax.text(0.97, 0.05, f"r = {r_val:.2f}, p = {p_val:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#444444")

plt.suptitle("Top Behavioral Predictor vs Age-Corrected BAG\n(4 Highest R² Regions)",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Fig5_TopPredictor_Scatter.pdf"),
            bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
print("\nAll figures saved to:", OUT_DIR)
print("  Fig1_R2_BarChart.pdf")
print("  Fig2_Correction_Comparison.pdf")
print("  Fig3_Coefficient_Heatmap.pdf")
print("  Fig4_BAG_Distributions.pdf")
print("  Fig5_TopPredictor_Scatter.pdf")
