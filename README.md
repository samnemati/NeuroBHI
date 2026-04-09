# NeuroBHI

A pipeline for computing regional Brain Age Gaps (BAGs) from volBrain regional volumes and evaluating their association with behavioral measures via stepwise regression.

---

## Overview

The pipeline consists of three scripts that run sequentially:

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
 Regional_BAGs_weighted_v2.xlsx
            │
            ▼
 [Python] stepwise_BAG_behavioral.py
            │  Merge regional BAGs with behavioral scores
            │  Bidirectional stepwise OLS per region (Age fixed)
            ▼
 Stepwise_BAG_Behavioral_Results.xlsx
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

## Repository structure

```
NeuroBHI/
├── BHI_ROI_brainage_calculation_from_ROIVol.m   # MATLAB: volBrain volumes → per-ROI brain age & BAG
├── BHI_compute_regionalBAG_from_roiBAG.m        # MATLAB: per-ROI BAGs → weighted regional BAGs
├── stepwise_BAG_behavioral.py                   # Python: regional BAGs → behavioral associations
└── README.md
```

---

## Citation / Contact

Project: ABC BrainAge / NeuroBHI
Author: Saamnaeh Nemati
