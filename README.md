# NeuroBHI

A pipeline for computing regional Brain Age Gaps (BAGs) from ROI-level brain age gap values and evaluating their association with behavioral measures via stepwise regression.

---

## Overview

The pipeline consists of two scripts that run sequentially:

```
ROI-level BAGs + volBrain volumes
            │
            ▼
 [MATLAB] BHI_compute_regionalBAG_from_roiBAG.m
            │  Fuzzy-match ROI names → volume columns
            │  Aggregate into anatomical / functional groups
            │  Compute simple mean + volume-weighted mean BAG
            ▼
 Regional_BAGs_weighted_v2.xlsx
            │
            ▼
 [Python] stepwise_BAG_behavioral.py
            │  Merge regional BAGs with behavioral scores
            │  Bidirectional stepwise OLS per region
            │  Age always included as fixed covariate
            ▼
 Stepwise_BAG_Behavioral_Results.xlsx
```

---

## Script 1 — `BHI_compute_regionalBAG_from_roiBAG.m` (MATLAB)

### What it does

1. **Loads** two Excel files: per-ROI BAG scores (one column per brain region, one row per subject) and volBrain regional volumes (in cm³).
2. **Fuzzy-matches** each ROI-BAG column name to its corresponding volume column by tokenizing names (lowercasing, expanding abbreviations like `L`→`left`, `WM`→`white matter`), scoring token overlap + hemisphere-match bonus − edit-distance penalty, and selecting the best match per ROI. Falls back to normalized Levenshtein distance when no token overlap is found.
3. **Saves** the ROI → volume column mapping to Excel so you can inspect and manually correct any wrong matches before trusting the weighted BAGs.
4. **Groups** ROIs by token matching into:
   - **Anatomical groups**: Frontal, Temporal, Parietal, Occipital × Left, Right (8 groups)
   - **Functional network groups**: DomainGeneral, LanguageSpecific, CinguloOpercular × Left, Right (6 groups)
5. **Computes** both a simple (unweighted) mean BAG and a volume-weighted mean BAG per group per subject. When volume data are missing for an individual ROI, the group mean volume is imputed; when all volumes are missing, the simple mean is used as fallback.
6. **Saves** all results to a single multi-sheet Excel file.

### Inputs

| Variable | Description |
|---|---|
| `roiBAG_file` | Excel file with columns `[subject_ID, <ROI>_BAG, ...]` — one row per subject |
| `volumes_file` | volBrain output Excel with `"volume cm3"` columns per region; must contain `subject_ID` |

### Outputs

| File | Sheet | Contents |
|---|---|---|
| `ROI_to_Vol_Mapping.xlsx` | — | Every ROI-BAG column, its matched volume column, match score, and edit distance. **Inspect this before proceeding.** |
| `Regional_BAGs_weighted_v2.xlsx` | `RegionalBAGs_mean` | Simple (unweighted) mean BAG per anatomical/functional group |
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

1. Set the four file paths at the top of the script (`roiBAG_file`, `volumes_file`, `output_mapping_file`, `output_weighted_file`).
2. Run the script.
3. Open `ROI_to_Vol_Mapping.xlsx` and verify matches — correct any mismatches manually in the mapping sheet if needed.
4. Proceed to the Python script using the `RegionalBAGs_weighted` sheet as input.

---

## Script 2 — `stepwise_BAG_behavioral.py` (Python)

### What it does

Runs a **bidirectional stepwise OLS regression** for each brain region, modeling the volume-weighted regional BAG as a function of behavioral scores, with Age always retained as a fixed covariate.

For each region:
1. Fits an **age-only model** and saves its residuals (age-partialled BAG).
2. Runs **bidirectional stepwise selection** over all behavioral predictors:
   - **Forward step**: adds the candidate with the lowest p-value if p < 0.05.
   - **Backward step**: removes any currently-selected predictor if its p-value exceeds 0.10.
   - Repeats until no predictor is added or removed.
3. Fits the **final model** (Age + selected predictors) and saves its residuals and coefficient table.

### Inputs

| File | Description |
|---|---|
| `ABC_RegBAG_Clcalc_weightedVol.xlsx` | Regional BAG file — output of the MATLAB script (weighted sheet) |
| `Behavioral_Data_Cleaned.xlsx` | Cleaned behavioral scores; must contain `subject_ID` and `Age` |

Tables are merged on `subject_ID`. Listwise deletion is applied — subjects missing any predictor or outcome are excluded.

### Regions modeled

`Language_Specific`, `Domain_General`, `Frontal`, `Temporal`, `Parietal`, `Occipital`, `Subcortical`, `Cerebellum`, `Limbic`

### Stepwise parameters

| Parameter | Value |
|---|---|
| Entry threshold (p-enter) | p < 0.05 |
| Removal threshold (p-remove) | p > 0.10 |
| Fixed covariate | Age (always retained, never removed) |
| Missing data | Listwise deletion |

### Outputs

All results are written to `Stepwise_BAG_Behavioral_Results.xlsx`:

| Sheet | Contents |
|---|---|
| `Summary` | One row per region: N, R², Adj-R², F-statistic, F p-value, number of selected predictors, selected predictor names |
| `<Region>` | Full coefficient table for that region's final model: predictor, coefficient, SE, t-value, p-value, 95% CI |
| `Final_Model_Residuals` | Residuals from the final model (Age + selected predictors) per subject per region |
| `Age_Corrected_Residuals` | Residuals from the age-only model (age-partialled BAG) per subject per region |

### Requirements

```
pandas
numpy
statsmodels
openpyxl
```

Install with:
```bash
pip install pandas numpy statsmodels openpyxl
```

### Usage

```bash
python stepwise_BAG_behavioral.py
```

Ensure both input Excel files are in the same directory as the script, or update the file paths in the configuration section at the top.

---

## Repository structure

```
NeuroBHI/
├── BHI_compute_regionalBAG_from_roiBAG.m   # MATLAB: ROI-BAG → weighted regional BAG
├── stepwise_BAG_behavioral.py              # Python: stepwise regression → behavioral associations
└── README.md
```

---

## Citation / Contact

Project: ABC BrainAge / NeuroBHI
Author: Saamnaeh Nemati
