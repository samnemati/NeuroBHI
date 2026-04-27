"""
BHI_compute_NIHTB_emotion_composites.py
---------------------------------------
Aggregate the 20 NIH Toolbox Emotion Battery subscales into the three
standard composites defined in:

    Salsman JM, Butt Z, Pilkonis PA, Cyranowski JM, Zill N, Hendrie HC,
    Kupst MJ, Kelly MA, Bode RK, Choi SW, Lai JS, Griffith JW, Stoney CM,
    Brouwers P, Knox SS, Cella D (2013).
    Emotion assessment using the NIH Toolbox.
    Quality of Life Research, 22(7):1843-1858.

The three composites:
    NIHTB_NegAffect_Composite           – ill-being / distress
    NIHTB_PsychWellBeing_Composite      – positive psychological functioning
    NIHTB_SocialRelationship_Composite  – social connectedness

Method: Hays-style simple summary — mean of within-sample z-scored subscales
(parallel to the RAND PCS / MCS script). Subscales already on a z-scale are
detected and used as-is; raw subscales are z-scored within sample first.
Two items are reverse-scored before aggregation:
  - Loneliness            (higher = more lonely → worse social)
  - NIHTB_NegativeAffect  (stored reverse-coded in this dataset relative to
                           its sibling negative-affect items; verified by
                           inter-item correlations — see REVERSE_SCORED comment).

Composite composition:
    NegAffect (10):  Sadness, FearAffect, FearSomaticArousal, AngerAffect,
                     AngerHostility, AngerPhysAggression, NegativeAffect,
                     PerceivedStress, PerceivedRejection, PerceivedHostility
    PsychWellBeing (5): LifeSatisfaction, MeaningPurpose, PositiveAffect,
                        PsychWellBeing, SelfEfficacy
    SocialRelationship (5): EmotionalSupport, InstrumentalSupport,
                            Friendship, Loneliness (REVERSED), SocialSatisfaction

Missing-data rule: a subject's composite is computed if at least
MIN_SUBSCALES_FRAC of the contributing subscales are non-missing
(default 0.5, i.e., at least half of the items).

Output:
    Results/BHI_NIHTB_Emotion_Composites.xlsx
        Composites           – subject_ID + the 3 composite scores
        Subscales_Used       – the z-scored subscales (with Loneliness reversed) used in aggregation
        Subscale_Diagnostics – per-subscale ceiling %, distribution stats
        Subscale_Correlations– pairwise Pearson r within each composite
        Composite_Summary    – distributional summary of the 3 composites
        ReadMe               – method, citation, integration instructions
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = "./Results/BHI_Behavioral_Plus_Questionnaire_Cleaned.xlsx"
OUTPUT_FILE = "./Results/BHI_NIHTB_Emotion_Composites.xlsx"
ID_COL      = "subject_ID"
MIN_SUBSCALES_FRAC = 0.5   # require >= half of the contributing subscales non-missing

# Heuristic for "already z-scored" (mean ≈ 0, SD ≈ 1 within tolerance)
ZSCORE_MEAN_TOL = 0.1
ZSCORE_STD_TOL  = 0.1

NEG_AFFECT = [
    "NIHTB_Sadness",            "NIHTB_FearAffect",
    "NIHTB_FearSomaticArousal", "NIHTB_AngerAffect",
    "NIHTB_AngerHostility",     "NIHTB_AngerPhysAggression",
    "NIHTB_NegativeAffect",     "NIHTB_PerceivedStress",
    "NIHTB_PerceivedRejection", "NIHTB_PerceivedHostility",
]
PSYCH_WB = [
    "NIHTB_LifeSatisfaction", "NIHTB_MeaningPurpose",
    "NIHTB_PositiveAffect",   "NIHTB_PsychWellBeing",
    "NIHTB_SelfEfficacy",
]
SOCIAL = [
    "NIHTB_EmotionalSupport",   "NIHTB_InstrumentalSupport",
    "NIHTB_Friendship",         "NIHTB_Loneliness",        # reverse-scored
    "NIHTB_SocialSatisfaction",
]
REVERSE_SCORED = {
    "NIHTB_Loneliness",       # higher = more lonely (worse social) → flip
    "NIHTB_NegativeAffect",   # In this dataset, NIHTB_NegativeAffect is stored
                              # reverse-coded relative to its 9 sibling items in
                              # the negative-affect cluster: it correlates
                              # negatively with every other neg-affect subscale
                              # (e.g., r = -0.84 with PerceivedStress, mean
                              # r = -0.48 with the other 9), while those 9
                              # correlate positively with each other (mean
                              # r ≈ +0.20). Flipping its sign restores the
                              # expected correlation structure.
}
ALL_SUBSCALES = NEG_AFFECT + PSYCH_WB + SOCIAL

COMPOSITES = {
    "NIHTB_NegAffect_Composite":          NEG_AFFECT,
    "NIHTB_PsychWellBeing_Composite":     PSYCH_WB,
    "NIHTB_SocialRelationship_Composite": SOCIAL,
}

# ── Load and validate ────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

if ID_COL not in df.columns:
    raise ValueError(f"Required ID column '{ID_COL}' not found in input.")

missing_cols = [c for c in ALL_SUBSCALES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing NIHTB Emotion subscale columns: {missing_cols}")

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

# ── Reverse-score items where higher = worse on the composite's direction ────
for c in REVERSE_SCORED:
    sub[c] = -sub[c]
    zscored_status[c] = zscored_status[c] + " + reversed"

print("\nSubscale standardization (and reverse-scoring where applicable):")
for c, status in zscored_status.items():
    print(f"  {c:<32} {status}")

# ── Composite scores (row-wise mean with missing-data tolerance) ─────────────
def safe_row_mean(rows, min_frac):
    out = rows.mean(axis=1)
    enough = rows.notna().sum(axis=1) >= np.ceil(min_frac * rows.shape[1])
    return out.where(enough)

composite_values = {}
for name, items in COMPOSITES.items():
    composite_values[name] = safe_row_mean(sub[items], MIN_SUBSCALES_FRAC).round(6)

composites = pd.DataFrame({ID_COL: sub[ID_COL], **composite_values})

print(f"\nComputed composites for {len(composites)} subjects")
for name in COMPOSITES:
    s = composites[name]
    print(f"  {name:<38} N={s.notna().sum()}  mean={s.mean():.3f}  "
          f"sd={s.std():.3f}  range=[{s.min():.3f}, {s.max():.3f}]  "
          f"unique={s.nunique()}")

# ── Diagnostics: per-subscale stats ──────────────────────────────────────────
diag_rows = []
def composite_of(col):
    for name, items in COMPOSITES.items():
        if col in items:
            return name
    return ""
for c in ALL_SUBSCALES:
    s = sub[c].dropna()
    top_pct = (s.value_counts().iloc[0] / len(s) * 100) if len(s) else float("nan")
    diag_rows.append({
        "Subscale":        c,
        "Composite":       composite_of(c),
        "N":               int(s.notna().sum()),
        "Unique_values":   int(s.nunique()),
        "Pct_at_modal":    round(top_pct, 1),
        "Mean":            round(s.mean(), 4),
        "SD":              round(s.std(ddof=0), 4),
        "Min":             round(s.min(), 4),
        "Max":             round(s.max(), 4),
        "Standardization": zscored_status[c],
    })
diag_df = pd.DataFrame(diag_rows)

# Pairwise correlations within each composite (long form)
corr_long_rows = []
for name, items in COMPOSITES.items():
    cm = sub[items].corr().round(3)
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if j > i:
                corr_long_rows.append(
                    {"Composite": name, "Subscale_A": a, "Subscale_B": b, "Pearson_r": cm.loc[a, b]})
corr_df = pd.DataFrame(corr_long_rows)

# Composite distribution summary
def comp_summary(name, s):
    return {
        "Composite":     name,
        "N":             int(s.notna().sum()),
        "Mean":          round(s.mean(), 4),
        "SD":            round(s.std(), 4),
        "Min":           round(s.min(), 4),
        "Max":           round(s.max(), 4),
        "Unique_values": int(s.nunique()),
        "Pct_at_modal":  round(s.value_counts().iloc[0] / s.notna().sum() * 100, 2),
        "p01": round(s.quantile(0.01), 4), "p05": round(s.quantile(0.05), 4),
        "p25": round(s.quantile(0.25), 4), "p50": round(s.quantile(0.50), 4),
        "p75": round(s.quantile(0.75), 4), "p95": round(s.quantile(0.95), 4),
        "p99": round(s.quantile(0.99), 4),
    }
comp_summary_df = pd.DataFrame([comp_summary(n, composites[n]) for n in COMPOSITES])

# ── Write output ─────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT_FILE} ...")
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
    composites.to_excel(w, sheet_name="Composites", index=False)
    sub.round(6).to_excel(w, sheet_name="Subscales_Used", index=False)
    diag_df.to_excel(w, sheet_name="Subscale_Diagnostics", index=False)
    corr_df.to_excel(w, sheet_name="Subscale_Correlations", index=False)
    comp_summary_df.to_excel(w, sheet_name="Composite_Summary", index=False)

    readme = pd.DataFrame({"Notes": [
        "NIH Toolbox Emotion Battery composite scores — Hays-style simple summary.",
        "Reference: Salsman et al. (2013). Emotion assessment using the NIH Toolbox.",
        "           Quality of Life Research 22(7):1843-1858.",
        "",
        "NIHTB_NegAffect_Composite          = mean of z-scored: " + ", ".join(NEG_AFFECT),
        "NIHTB_PsychWellBeing_Composite     = mean of z-scored: " + ", ".join(PSYCH_WB),
        "NIHTB_SocialRelationship_Composite = mean of z-scored: " + ", ".join(SOCIAL),
        "",
        "Two items reverse-scored (multiplied by -1 after z-scoring) before averaging:",
        "  - Loneliness            (higher = more lonely → worse social)",
        "  - NIHTB_NegativeAffect  (stored reverse-coded in this dataset; verified by",
        "                           inter-item correlations — it has r = -0.84 with",
        "                           PerceivedStress and a mean r = -0.48 with the 9",
        "                           other neg-affect items, while those 9 correlate",
        "                           positively with each other)",
        "After all sign corrections, higher composite values consistently mean:",
        "  NegAffect          : MORE distress / negative affect",
        "  PsychWellBeing     : BETTER psychological well-being",
        "  SocialRelationship : BETTER social relationships",
        "",
        f"Missing-data rule: composite computed if >= {MIN_SUBSCALES_FRAC*100:.0f}% of the contributing",
        "  subscales are non-missing; otherwise NaN.",
        "Standardization: subscales detected as already z-scored were used as-is;",
        "  raw subscales were z-scored within sample before averaging.",
        "",
        "To use these in downstream analyses:",
        "  1. Drop the 20 individual NIHTB Emotion subscales from the behavioral file.",
        "  2. Merge in the 3 composites from this file on subject_ID.",
    ]})
    readme.to_excel(w, sheet_name="ReadMe", index=False)

print("Done.")
