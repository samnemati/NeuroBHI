# NeuroBHI

A pipeline for computing regional Brain Age Gaps (BAGs) from volBrain regional volumes, evaluating their association with behavioral measures, and testing whether behavior can predict regional BAG out-of-sample.

A companion data-prep script (Script 5) extracts BAG-relevant items from the ABC comprehensive health questionnaire and produces an analysis-ready quantitative table to feed into Scripts 3 and 4.

---

## Overview

The pipeline consists of four scripts that run sequentially:

```
volBrain regional volumes (Excel)
            │
            ▼
 [MATLAB] BHI_ROI_brainage_calculation_from_ROIVol.m
            │  Auto-detect TIV and region columns
            │  TIV-normalize volumes
            │  Leave-One-Out ridge regression per subject × region
            │  Predict brain age; compute BAG and proportional BAG
            ▼
 volBrain_Regional_BrainAge.xlsx
 volBrain_ROI_BrainAgeGaps.xlsx
 volBrain_Regional_PropBrainAgeGaps.xlsx
            │
            ▼
 [MATLAB] BHI_compute_regionalBAG_from_roiBAG.m
            │  Fuzzy-match ROI names → volBrain volume columns
            │  Aggregate ROIs into anatomical / functional groups
            │  Compute simple mean + volume-weighted mean BAG per group
            ▼
 Results/BHI_Regional_BrainAgeGap.xlsx
            │
            ├────────────────────────────────────────────┐
            ▼                                            ▼
 [Python] stepwise_BAG_behavioral.py     [Python] BHI_regional_predictive_models.py
            │  Two-stage stepwise:                       │  Ridge 10-fold CV per region:
            │  Stage 1 regresses out Age                 │  Model A: behavior → regional BAG
            │  Stage 2 selects behavioral predictors     │  Model B: global BAG → regional BAG
            │  FDR + Bonferroni across 12 regions        │  Paired ΔR² specificity test
            ▼                                            ▼
 Stepwise_BAG_Behavioral_Results.xlsx    Results/BHI_Regional_Predictive_Models.xlsx
                                         Figures/BHI_Regional_Fig1–3.pdf
```

---

## Script 1 — `BHI_ROI_brainage_calculation_from_ROIVol.m` (MATLAB)

### What it does

Estimates a **predicted brain age** for every subject in every brain region using a Leave-One-Out (LOO) ridge-regularized linear regression trained on healthy controls.

1. **Loads** a volBrain Excel file containing regional volumes (cm³), chronological age, and a `UseForEstimation` flag.
2. **Auto-detects TIV** (Total Intracranial Volume) — first looks for a column containing "intracranial" or "(IC)"; falls back to `Brain(WM+GM) + CSF`.
3. **Auto-detects region columns** — any column whose name contains both "volume" and "cm3".
4. **TIV-normalizes** all volumes (divides by TIV) so brain age reflects relative regional proportions rather than absolute sizes.
5. **Leave-One-Out regression** per subject × per region:
   - Removes the subject from the control pool if they are a control.
   - Fits `Age ~ TIV-normalized volume` on the remaining controls using OLS; switches to ridge regularization (λ = 0.001) when the design matrix is ill-conditioned or underdetermined.
   - Predicts the subject's brain age from their own normalized volume.
   - Computes an **atrophy score** = mean Z-score of the subject's volumes relative to controls.
6. **Computes Brain Age Gap (BAG)** = predicted age − chronological age, and **Proportional BAG** = BAG / chronological age.
7. **Saves** three output Excel files and generates box plots for visual inspection.

### Inputs

| Column | Description |
|---|---|
| `subject_ID` | Subject identifier (required for labelled output) |
| `Age` | Chronological age in years |
| `UseForEstimation` | `"Within Range"` = control, `"Outside Range"` = test |
| TIV column | Auto-detected (intracranial cavity or WM+GM+CSF) |
| `<Region> volume cm3` | One column per brain region (auto-detected) |

### Outputs

| File | Sheet | Contents |
|---|---|---|
| `volBrain_Regional_BrainAge.xlsx` | `Brain_Age` | Predicted brain age [nSubjects × nRegions] |
| `volBrain_ROI_BrainAgeGaps.xlsx` | `Brain_Age_Gap` | BAG = predicted − chronological age |
| `volBrain_Regional_PropBrainAgeGaps.xlsx` | `Prop_Brain_Age_Gap` | Proportional BAG = BAG / chronological age |

Box plots are saved as `Brain_age.png`, `Brain_age_gap.png`, `Brain_age_gap_PBA.png`.

### Requirements

- MATLAB R2019b or later
- No toolboxes required

### Usage

1. Set `datafile` in the USER CONFIGURATION section.
2. Run the script.
3. Check console diagnostics: TIV column detected, number of regions, LOO progress.
4. Inspect the three output Excel files. Feed `volBrain_ROI_BrainAgeGaps.xlsx` into Script 2.

---

## Script 2 — `BHI_compute_regionalBAG_from_roiBAG.m` (MATLAB)

### What it does

Aggregates ROI-level BAG values (from Script 1) into anatomical lobe groups and functional network groups, weighted by regional brain volume.

1. **Loads** the ROI-BAG table (output of Script 1) and the volBrain volume table.
2. **Fuzzy-matches** each ROI-BAG column name to its corresponding volume column by tokenizing names (lowercasing, expanding abbreviations like `L`→`left`, `WM`→`white matter`), scoring by token overlap + hemisphere-match bonus − edit-distance penalty, and selecting the best match per ROI. Falls back to normalized Levenshtein distance when no token overlap is found.
3. **Saves** the ROI → volume column mapping to Excel for inspection and manual correction.
4. **Groups** ROIs into:
   - **Anatomical groups**: Frontal, Temporal, Parietal, Occipital × Left, Right (8 groups)
   - **Functional network groups**: DomainGeneral, LanguageSpecific, CinguloOpercular × Left, Right (6 groups)
5. **Computes** both a simple (unweighted) mean BAG and a volume-weighted mean BAG per group per subject. Missing ROI volumes are imputed with the group mean; if all volumes are missing, falls back to simple mean.
6. **Saves** all results to a single multi-sheet Excel file.

### Inputs

| Variable | Description |
|---|---|
| `roiBAG_file` | ROI-BAG Excel file (`volBrain_ROI_BrainAgeGaps.xlsx` from Script 1) |
| `volumes_file` | volBrain volume Excel file (same as Script 1 input); must contain `subject_ID` |

### Outputs

| File | Sheet | Contents |
|---|---|---|
| `ROI_to_Vol_Mapping.xlsx` | — | ROI-BAG column → matched volume column, match score, edit distance. **Inspect before use.** |
| `Regional_BAGs_weighted_v2.xlsx` | `RegionalBAGs_mean` | Simple mean BAG per group |
| | `RegionalBAGs_weighted` | Volume-weighted mean BAG per group |
| | `ROI_to_Vol_Mapping` | Copy of the mapping table |

### Functional network definitions

| Network | Included regions |
|---|---|
| **DomainGeneral** | Superior/middle frontal, precentral, supramarginal, insula, anterior/posterior cingulate, supplementary motor, frontal operculum |
| **LanguageSpecific** | Inferior frontal (opercular/triangular/orbital), superior/middle temporal, planum temporale/polare, temporal pole, angular, supramarginal, Heschl's gyrus, fusiform |
| **CinguloOpercular** | Anterior/middle cingulate, insula, frontal operculum, supplementary motor |

### Requirements

- MATLAB R2019b or later
- No toolboxes required

### Usage

1. Set the four file paths in the USER CONFIGURATION section.
2. Run the script.
3. Open `ROI_to_Vol_Mapping.xlsx` and verify matches — correct any mismatches manually.
4. Use `Regional_BAGs_weighted_v2.xlsx` (weighted sheet) as input to Script 3.

---

## Script 3 — `stepwise_BAG_behavioral.py` (Python)

### What it does

Runs a **two-stage regression** for each brain region to isolate the behavioral contribution to BAG independently of age.

**Why two stages?**
BAG is strongly negatively correlated with chronological age (a known regression-to-the-mean artifact in brain age studies). Including Age as a single covariate in one model controls for it statistically, but Age dominates R² and makes the overall fit appear artificially high (R² ~ 0.94–0.999). The two-stage approach first removes age variance entirely, then asks: *how much do behavioral factors explain of what remains?* The R² reported is purely behavioral.

**Stage 1 — Age correction (per region):**
1. Fit `BAG ~ Age` (OLS).
2. Save residuals = age-partialled BAG (the outcome for Stage 2).

**Stage 2 — Behavioral stepwise (per region):**
1. Run bidirectional stepwise OLS on the Stage 1 residuals. Age is NOT re-entered — it was already removed.
   - **Forward step**: adds the candidate with the lowest p-value if p < 0.05.
   - **Backward step**: removes any selected predictor if its p-value exceeds 0.10.
   - Repeats until no predictor is added or removed.
2. Fit the final model on the selected behavioral predictors.
3. Report R² — this is the variance explained by behavioral factors alone, above and beyond age.

### Inputs

| File | Description |
|---|---|
| `ABC_RegBAG_Clcalc_weightedVol.xlsx` | Regional BAG file — weighted sheet from Script 2 |
| `Behavioral_Data_Cleaned.xlsx` | Cleaned behavioral scores; must contain `subject_ID` and `Age` |

Tables are merged on `subject_ID`. Listwise deletion is applied — subjects missing any predictor or outcome are excluded.

### Regions modeled

`Language_Specific`, `Domain_General`, `Frontal`, `Temporal`, `Parietal`, `Occipital`, `Subcortical`, `Cerebellum`, `Limbic`

### Stepwise parameters

| Parameter | Value |
|---|---|
| Entry threshold | p < 0.05 |
| Removal threshold | p > 0.10 |
| Age handling | Regressed out in Stage 1; not re-entered in Stage 2 |
| Missing data | Listwise deletion |

### Outputs

All results written to `Stepwise_BAG_Behavioral_Results.xlsx`:

| Sheet | Contents |
|---|---|
| `Summary` | One row per region: N, Stage 1 Age-R², Stage 2 Behavioral R², Adj-R², F-statistic, F p-value, number and names of selected predictors |
| `<Region>` | Full coefficient table for Stage 2: predictor, coefficient, SE, t-value, p-value, 95% CI |
| `Age_Corrected_Residuals` | Stage 1 residuals (age-partialled BAG) — the outcome used in Stage 2 |
| `Stage2_Model_Residuals` | Stage 2 residuals (unexplained after both age and behavioral predictors) |

### Interpreting the Summary sheet

- **Stage1_Age_R2**: how much variance in BAG is due to age alone (expected to be high: 0.83–0.999).
- **Stage2_Behavioral_R2**: how much of the *remaining, age-independent* BAG variance is explained by behavioral predictors. This is the number to report and interpret.

### Requirements

```
pandas
numpy
statsmodels
openpyxl
```

```bash
pip install pandas numpy statsmodels openpyxl
```

### Usage

```bash
python stepwise_BAG_behavioral.py
```

Ensure both input Excel files are in the same directory, or update the file paths in the configuration section at the top.

---

## Script 4 — `BHI_regional_predictive_models.py` (Python)

### What it does

Quantifies how well behavioral measures can **predict** MRI-derived regional BAG out-of-sample, and tests whether behavior carries **region-specific** information beyond what global brain aging already conveys. Where Script 3 asks "which behaviors *associate* with which regions," Script 4 asks "*how accurately* can we reproduce each region's BAG from behavior alone, and does it beat a global-BAG baseline?"

### Design: two paired models per region

For each of 12 regional BAGs, two 10-fold cross-validated ridge regressions are fit on **identical fold splits**, enabling paired per-fold comparison:

| Model | Features → Target |
|---|---|
| **Model A (behavior)** | 187 behavioral features → age-residualized regional BAG |
| **Model B (baseline)** | age-residualized global BAG → age-residualized regional BAG |

Targets are age-residualized via Stage-1 OLS (matching Script 3), so R² reflects the purely non-age variance.

### Interpretation of the paired comparison

- **Model A > Model B significantly** → behavior carries region-specific information beyond global brain aging
- **Model A ≈ Model B** → behavior only proxies global aging; no regional specificity
- **Both near 0** → this region is not behaviorally predictable at all

### Modeling details

| Component | Choice |
|---|---|
| Estimator | `StandardScaler` → `RidgeCV` (α ∈ {0.01, 0.1, 1, 10, 10², 10³, 10⁴, 10⁵}) |
| α selection | Inner LOOCV on each training fold (prevents leakage) |
| CV | 10-fold, `random_state = 42`, same splits for A and B |
| Metrics per fold | R², RMSE, MAE, Pearson r (out-of-fold) |

### Multiple comparisons (across 12 regions)

Both corrections applied via paired one-sample t-tests on per-fold values:

1. **Model A R² vs 0** — does behavior predict regional BAG at all?
2. **ΔR² = R²_A − R²_B vs 0** — does behavior beat the global-BAG baseline?

p-values adjusted with **FDR (Benjamini-Hochberg)** and compared against **Bonferroni threshold (0.05 / 12 = 0.0042)**.

### Inputs

| File | Description |
|---|---|
| `Results/BHI_Regional_BrainAgeGap.xlsx` | Regional BAG table: `subject_ID`, `Age`, `BrainAgeR_Global`, 12 regional BAG columns |
| `Behavioral_Data_Cleaned.xlsx` | Behavioral scores; merged on `subject_ID`; listwise deletion applied |

### Outputs

`Results/BHI_Regional_Predictive_Models.xlsx`:

| Sheet | Contents |
|---|---|
| `Summary` | One row per region: mean ± SD of R²/RMSE/MAE/corr for Models A and B; ΔR² |
| `Specificity_Test` | Paired t-tests (Model A vs 0; ΔR² vs 0) with FDR-adjusted p and Bonferroni flag |
| `FeatureImportance` | Wide matrix — standardized ridge coefficients × region (Model A, full-data refit) |
| `FoldMetrics_A` / `FoldMetrics_B` | Per-fold per-region metrics (10 × 12 rows each) |
| `Predictions_A` / `Predictions_B` | Subject-level out-of-fold predictions vs actual |

Figures (in `Figures/`):

| File | Contents |
|---|---|
| `BHI_Regional_Fig1_ModelAvsB_R2.pdf` | Paired R² bars per region; `*` marks FDR-significant ΔR² |
| `BHI_Regional_Fig2_FeatureHeatmap.pdf` | Top 20 features × 12 regions, coefficient sign/magnitude |
| `BHI_Regional_Fig3_Scatter.pdf` | Model A actual vs predicted scatter, one panel per region |

### Requirements

```
pandas
numpy
scipy
scikit-learn
statsmodels
matplotlib
seaborn
openpyxl
```

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn openpyxl
```

### Usage

```bash
python BHI_regional_predictive_models.py
```

Run from the project root so relative paths to `Results/` and `Figures/` resolve correctly. The console prints a per-region summary table (α selected, R² for A and B, ΔR², corrected p-values).

---

## Script 5 — `BHI_extract_questionnaire.py` (Python)

### What it does

Extracts BAG-relevant items from the ABC comprehensive health questionnaire and produces a single analysis-ready workbook. The source file is a REDCap export with a two-row header (category on row 0, question text on row 1), 345 subjects, and 518 columns spanning 14 instruments. This script selects the columns worth keeping, drops free-text and out-of-scope instruments, and encodes every remaining column numerically with rules that are documented inside the output file.

### Sections kept vs dropped

| Kept | Dropped |
|---|---|
| Health & Physical Activity (Rand26, ABC balance scale, IPAQ) | Reading History (childhood-only items) |
| Alcohol Use | Lifetime Discrimination block — duplicate of Food Security in source |
| COMPASS-31 autonomic symptoms | COVID-19 treatment detail (symptom checkboxes kept) |
| Dietary Screening | Free-text job titles, industries, years-at-job, |
| Food Security | cereal/milk brands, contact initials, |
| Participant Health History (Myself + Family pairs) | PTSD/PSQI "please describe" fields |
| Pearlin Mastery Scale | |
| Periodontal screening | |
| Pittsburgh Sleep Quality Index | |
| PTSD Checklist DSM-5 (PCL-5) | |
| Social Relationship (group participation + person-knowledge flags) | |
| COVID-19 symptom flags (incl. Brain Fog) + positive-test flag | |

### Encoding conventions

| Source pattern | Quantitative encoding |
|---|---|
| Yes / No | 1 / 0 (blank → NaN) |
| Checked / Unchecked | 1 / 0 |
| Ordinal Likert | integer scale (see `Scales` sheet in output) |
| Multi-choice single-select | one-hot columns `<base>__<slug>` |
| Continuous numeric | float (strips `%` and whitespace) |
| Free text | dropped |
| `Don't Know`, `Refused`, `Prefer not to answer`, `Can't choose`, REDCap placeholders | NaN |

Notable scales: `HANDEDNESS_5` (-2..+2), `EDUCATION_5` (1..5), `INCOME_7` (1..7), `NOT_AT_ALL_5` (PCL-5, 0..4), `STRONG_DIS_4` (Pearlin, 1..4; items 4 and 6 positively worded — reverse before summing), `LIKERT_FREQ_4` (PSQI, 0..3), `PSQI_PROBLEM_4`, `PSQI_QUALITY_4`, `CHANGE_6_ALT` (0 = never had the symptom … 6 = much worse — unified positive ordinal so regression coefficients stay interpretable). The full mapping for every scale is written into the output workbook.

### Inputs

| File | Description |
|---|---|
| `Doc/ABC_ComprehensiveQuestionnaire.xlsx` | REDCap export; expects `Study ID` in column 0, category row in row 0, question text in row 1, data from row 2 onward |

### Outputs

`Results/BHI_Questionnaire_Extracted.xlsx` — 5 sheets, all generated in a single run:

| Sheet | Contents |
|---|---|
| `Raw_Selected` | 345 × 336 — original text/number values for selected columns, keyed by `Study_ID` |
| `Quantitative` | 345 × 354 — numerically encoded version suitable for modeling |
| `Encoding_Rules` | 336 rows — for each source column: source index, section, quant name, action, scale name, original question text |
| `Scales` | 154 rows — every ordinal scale with its raw → encoded mapping, plus the Yes/No and Checked/Unchecked rules, plus the full list of missing-value tokens mapped to NaN |
| `ReadMe` | Plain-English summary: sections kept vs dropped, encoding conventions, notable scale choices, data quirks |

### Data quirks

- Source has 345 subjects; `BHI_Regional_BrainAgeGap.xlsx` has 304. **Inner-join on `Study_ID`** before running association or prediction analyses.
- Cols 158–171 in the source are labelled "Lifetime discrimination" but contain an exact duplicate of the Food Security block (cols 144–157). Dropped to avoid double-counting.
- A handful of COMPASS "times per month" cells contain Excel-auto-dated strings like `"4-Mar"` (originally `"3-4"`) — left as NaN. The four garbled times-per-month follow-up columns (raw indices 101, 105, 107, 109) are dropped entirely.

### Requirements

```
pandas
numpy
openpyxl
```

```bash
pip install pandas numpy openpyxl
```

### Usage

```bash
python BHI_extract_questionnaire.py
```

Run from the project root so `Doc/` and `Results/` resolve. The encoding rules live inside the script (plus are mirrored into the output workbook), so re-running after a source-file update regenerates both the data and the reference sheets in sync.

---

## Repository structure

```
NeuroBHI/
├── BHI_ROI_brainage_calculation_from_ROIVol.m   # MATLAB: volBrain volumes → per-ROI brain age & BAG
├── BHI_compute_regionalBAG_from_roiBAG.m        # MATLAB: per-ROI BAGs → weighted regional BAGs
├── stepwise_BAG_behavioral.py                   # Python: regional BAGs → behavioral associations
├── BHI_regional_predictive_models.py            # Python: behavior → regional BAG out-of-sample prediction
├── BHI_extract_questionnaire.py                 # Python: ABC questionnaire → quantitative BAG-relevant features
├── visualize_BAG_results.py                     # Python: figures for Script 3 outputs
├── Doc/                                         # Source questionnaire, reference PDFs, meeting notes
├── Results/                                     # Excel outputs (BAG tables, model results, questionnaire extract)
├── Figures/                                     # PDF figures
└── README.md
```

---

## Citation / Contact

Project: ABC BrainAge / NeuroBHI
Author: Saamnaeh Nemati
