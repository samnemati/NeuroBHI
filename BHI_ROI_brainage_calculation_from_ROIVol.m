%% BHI_ROI_brainage_calculation_from_ROIVol.m
%
% DESCRIPTION
% -----------
% Computes per-region Brain Age, Brain Age Gap (BAG), and Proportional BAG
% for every subject using regional brain volumes from a volBrain output file.
%
% The core method is a Leave-One-Out (LOO) robust linear regression:
%   For each subject i and each brain region j:
%     1. TIV-normalize all regional volumes (volume / TIV).
%     2. Remove subject i from the control pool (if they are a control).
%     3. Fit a linear model:  Age ~ TIV-normalized volume  on the remaining controls.
%        Ridge regularization is applied automatically when the design matrix is
%        ill-conditioned or when the number of predictors exceeds n-1.
%     4. Predict subject i's brain age from their own normalized volume.
%     5. Compute an atrophy score = mean Z-score of subject i's volumes
%        relative to the control mean and SD.
%
% Controls are subjects where UseForEstimation == "Within Range".
% Test subjects are where UseForEstimation == "Outside Range".
%
% TIV DETECTION (automatic, in priority order)
%   1. Column whose name contains "intracranial" or "(IC)"
%   2. Fallback: Brain(WM+GM) column + CSF column
%
% REGION DETECTION (automatic)
%   Any column whose name contains both "volume" and "cm3" (case-insensitive).
%   Falls back to columns containing "volume" only if no cm3 columns are found.
%
% INPUTS (set file paths in the USER CONFIGURATION section below)
% -------
%   datafile : volBrain Excel file with columns:
%              - subject_ID            (required for subject-level output)
%              - Age                   (chronological age in years)
%              - UseForEstimation      ("Within Range" = control, "Outside Range" = test)
%              - TIV / IC column       (auto-detected; see TIV DETECTION above)
%              - <Region> volume cm3   (one column per brain region, auto-detected)
%
% OUTPUTS
% -------
%   volBrain_Regional_BrainAge.xlsx       – Sheet "Brain_Age":
%       Predicted brain age per region per subject [nSubjects x nRegions]
%   volBrain_ROI_BrainAgeGaps.xlsx        – Sheet "Brain_Age_Gap":
%       BAG = predicted age − chronological age [nSubjects x nRegions]
%   volBrain_Regional_PropBrainAgeGaps.xlsx – Sheet "Prop_Brain_Age_Gap":
%       Proportional BAG = BAG / chronological age [nSubjects x nRegions]
%
%   Box plots are generated (and saved as PNG) for each output type.
%   A fallback boxplot of the first 6 regions is shown if create_box_plots fails.
%
% USAGE
% -----
%   1. Set `datafile` in the USER CONFIGURATION section.
%   2. Run the script — no arguments or toolboxes needed.
%   3. Inspect console diagnostics (TIV detected, # regions, LOO summary).
%   4. Check output Excel files for per-region brain age and BAG values.
%
% DEPENDENCIES
% ------------
%   MATLAB R2019b or later. No toolboxes required.
%   (Statistics and Machine Learning Toolbox used only for the boxplot fallback.)
%
% AUTHOR  : Saamnaeh Nemati
% PROJECT : ABC BrainAge / NeuroBHI
% CREATED : ~2025-Q4
% UPDATED : 2026-04-09  (variable rename + full documentation pass)

clear; clc; close all;

%% =========================================================================
%  USER CONFIGURATION — set file paths here before running
%  =========================================================================
datafile                   = '/Users/snemati/Documents/ABC_BrainAge/Output/BrainAge/ABC_Baseline_Regional_volBrain_Volumes.xlsx';
output_brainAge_file       = 'volBrain_Regional_BrainAge.xlsx';
output_BAG_file            = 'volBrain_ROI_BrainAgeGaps.xlsx';
output_proportionalBAG_file = 'volBrain_Regional_PropBrainAgeGaps.xlsx';
%  =========================================================================

%% -------------------------------------------------------------------------
%  SECTION 1 : Load data and validate required columns
%  -------------------------------------------------------------------------
fprintf('Loading data from: %s\n', datafile);
tblVolBrain = readtable(datafile);

allColumnNames = tblVolBrain.Properties.VariableNames;

% Age and UseForEstimation are mandatory
if ~ismember('Age', allColumnNames)
    error('Column "Age" not found. Ensure the input file contains a chronological age column named "Age".');
end
if ~ismember('UseForEstimation', allColumnNames)
    error('Column "UseForEstimation" not found. Expected values: "Within Range" (controls) / "Outside Range" (test).');
end

%% -------------------------------------------------------------------------
%  SECTION 2 : Auto-detect Total Intracranial Volume (TIV)
%  -------------------------------------------------------------------------
% Priority 1: column whose name contains "intracranial" or "(IC)"
isTIV_column = contains(lower(allColumnNames), 'intracranial') | ...
               contains(lower(allColumnNames), 'intracranial cavity') | ...
               contains(allColumnNames, '(IC)');
tivColumnIdx = find(isTIV_column, 1, 'first');

if ~isempty(tivColumnIdx)
    tblVolBrain.TIV = tblVolBrain{:, tivColumnIdx};
    fprintf('Detected TIV column: "%s"\n', allColumnNames{tivColumnIdx});
else
    % Priority 2: Brain (WM+GM) + CSF columns as a proxy for TIV
    isBrainVolume_column = contains(lower(allColumnNames), 'brain (wm+gm)');
    isCSF_column         = contains(lower(allColumnNames), 'cerebro spinal fluid') & ...
                           contains(lower(allColumnNames), 'cm3');
    brainVolumeColumnIdx = find(isBrainVolume_column, 1, 'first');
    csfColumnIdx         = find(isCSF_column, 1, 'first');

    if ~isempty(brainVolumeColumnIdx) && ~isempty(csfColumnIdx)
        tblVolBrain.TIV = tblVolBrain{:, brainVolumeColumnIdx} + tblVolBrain{:, csfColumnIdx};
        fprintf('No IC column found. Fallback TIV = Brain(WM+GM) + CSF: "%s" + "%s"\n', ...
                allColumnNames{brainVolumeColumnIdx}, allColumnNames{csfColumnIdx});
    else
        error(['Could not detect TIV automatically. Add or rename an intracranial volume column ' ...
               '(e.g., "Intracranial Cavity (IC) volume cm3") or adjust the detection logic.']);
    end
end

% Sanity-check: TIV must be numeric and non-trivial
if ~isnumeric(tblVolBrain.TIV)
    error('Detected TIV column is not numeric. Verify the column content.');
end
fprintf('TIV summary — non-missing: %d/%d  |  mean=%.1f  min=%.1f  max=%.1f\n', ...
        sum(~isnan(tblVolBrain.TIV)), height(tblVolBrain), ...
        nanmean(tblVolBrain.TIV), nanmin(tblVolBrain.TIV), nanmax(tblVolBrain.TIV));

%% -------------------------------------------------------------------------
%  SECTION 3 : Auto-detect regional volume columns
%  -------------------------------------------------------------------------
% Primary: columns whose name contains both "volume" and "cm3"
isRegionVolume_column = contains(lower(allColumnNames), 'volume') & ...
                        contains(lower(allColumnNames), 'cm3');
regionColumnIndices   = find(isRegionVolume_column);

if isempty(regionColumnIndices)
    % Fallback: any column containing "volume"
    warning('No "volume cm3" columns found. Falling back to all columns containing "volume".');
    isVolume_column_fallback = contains(lower(allColumnNames), 'volume');
    regionColumnIndices      = find(isVolume_column_fallback);
end

if isempty(regionColumnIndices)
    error('No volume columns detected. Inspect tblVolBrain.Properties.VariableNames and update the detection logic.');
end

regionColumnNames = allColumnNames(regionColumnIndices);
nRegions          = numel(regionColumnIndices);
fprintf('Detected %d region volume columns. First: "%s"\n', nRegions, regionColumnNames{1});

%% -------------------------------------------------------------------------
%  SECTION 4 : Separate controls and test subjects; TIV-normalize volumes
%  -------------------------------------------------------------------------
% Controls: "Within Range" subjects — used to train the age prediction model
% Test:     "Outside Range" subjects — predicted using the control-trained model
isControlSubject = tblVolBrain.UseForEstimation == "Within Range";
isTestSubject    = tblVolBrain.UseForEstimation == "Outside Range";

controlRowIndices = find(isControlSubject);   % row indices in tblVolBrain
testRowIndices    = find(isTestSubject);

% Extract and TIV-normalize control volumes
controlVolumes_raw        = table2array(tblVolBrain(isControlSubject, regionColumnIndices));
controlAges               = tblVolBrain.Age(isControlSubject);
controlTIV                = tblVolBrain.TIV(isControlSubject);
controlTIV                = controlTIV(:);   % ensure column vector
controlVolumes_normalized = controlVolumes_raw ./ controlTIV;   % [nControls x nRegions]

% TIV-normalize test subject volumes (if any)
if ~isempty(testRowIndices)
    testVolumes_raw        = table2array(tblVolBrain(isTestSubject, regionColumnIndices));
    testTIV                = tblVolBrain.TIV(isTestSubject);
    testVolumes_normalized = testVolumes_raw ./ testTIV;   %#ok<NASGU> % reserved for future use
else
    testVolumes_normalized = [];
end

%% -------------------------------------------------------------------------
%  SECTION 5 : Build per-region index sets (one ROI = one region group)
%  -------------------------------------------------------------------------
% Each entry is a single-element cell referencing one column of controlVolumes_normalized.
% This structure mirrors a more general "region group" design — extensible to
% multi-ROI groups (e.g., lobe-level) by changing the index sets.
regionIndexSets  = cell(1, nRegions);
regionGroupNames = cell(1, nRegions);
for regionIdx = 1:nRegions
    regionIndexSets{regionIdx}  = regionIdx;
    regionGroupNames{regionIdx} = regionColumnNames{regionIdx};
end

%% -------------------------------------------------------------------------
%  SECTION 6 : Leave-One-Out (LOO) brain age estimation
%  -------------------------------------------------------------------------
nSubjects             = height(tblVolBrain);
predictedBrainAge     = NaN(nSubjects, nRegions);   % predicted brain age
atrophyScore          = NaN(nSubjects, nRegions);   % mean volume Z-score vs controls
nROIsUsedPerEstimate  = NaN(nSubjects, nRegions);   % # valid predictors used per estimate

% Resolve subject IDs for mapping (fall back to row index if column absent)
if ismember('subject_ID', tblVolBrain.Properties.VariableNames)
    subjectIDs = tblVolBrain.subject_ID;
else
    warning('Column "subject_ID" not found. LOO will use row indices. Add subject_ID for robust mapping.');
    subjectIDs = (1:nSubjects)';
end

fprintf('Starting LOO estimation: %d subjects x %d regions...\n', nSubjects, nRegions);

for subjectIdx = 1:nSubjects

    % Print progress every 50 subjects
    if mod(subjectIdx, 50) == 0
        fprintf('  Processed %d / %d subjects...\n', subjectIdx, nSubjects);
    end

    % --- TIV-normalize this subject's volumes ----------------------------
    subjectTIV = tblVolBrain.TIV(subjectIdx);
    if isempty(subjectTIV) || isnan(subjectTIV) || subjectTIV == 0
        warning('Missing or zero TIV for subject row %d (%s). Skipping.', subjectIdx, string(subjectIDs(subjectIdx)));
        continue;
    end
    subjectRawVolumes        = table2array(tblVolBrain(subjectIdx, regionColumnIndices));
    subjectVolumes_normalized = subjectRawVolumes ./ subjectTIV;   % [1 x nRegions]

    % --- Build LOO control set: remove this subject if they are a control -
    controlVolumes_LOO = controlVolumes_normalized;
    controlAges_LOO    = controlAges;

    subjectPositionInControls = find(controlRowIndices == subjectIdx);
    if ~isempty(subjectPositionInControls)
        % Remove this subject from the training set
        controlVolumes_LOO(subjectPositionInControls, :) = [];
        controlAges_LOO(subjectPositionInControls)       = [];
    end

    % --- Estimate brain age for each region group -------------------------
    for regionIdx = 1:numel(regionIndexSets)
        regionColumnSubset = regionIndexSets{regionIdx};

        % Slice the control matrix and patient vector to this region's columns
        [predictorMatrix_controls, ageVector_controls, predictorVector_patient] = ...
            extract_region_predictors(controlVolumes_LOO, controlAges_LOO, ...
                                      subjectVolumes_normalized, regionColumnSubset);

        % Fit the model and predict brain age
        [predictedAge_region, atrophyScore_region, roiUsageInfo] = ...
            estimate_brain_age(predictorMatrix_controls, ageVector_controls, predictorVector_patient);

        predictedBrainAge(subjectIdx, regionIdx)    = predictedAge_region;
        atrophyScore(subjectIdx, regionIdx)         = atrophyScore_region;

        if isstruct(roiUsageInfo) && isfield(roiUsageInfo, 'count')
            nROIsUsedPerEstimate(subjectIdx, regionIdx) = roiUsageInfo.count;
        end
    end
end

fprintf('LOO estimation complete.\n\n');

%% -------------------------------------------------------------------------
%  SECTION 7 : Compute Brain Age Gap (BAG) and Proportional BAG
%  -------------------------------------------------------------------------
% BAG = predicted brain age − chronological age
% Proportional BAG = BAG / chronological age  (scale-free version)

if isempty(controlRowIndices)
    warning('No controls found (UseForEstimation == "Within Range"). Control-specific gap tables will be empty.');
    BAG_controls             = [];
    proportionalBAG_controls = [];
else
    % Control-only BAG (useful for checking model bias)
    BAG_controls             = predictedBrainAge(controlRowIndices, :) - controlAges;
    proportionalBAG_controls = BAG_controls ./ controlAges;
end

% All-subjects BAG
BAG_allSubjects             = predictedBrainAge - tblVolBrain.Age;
proportionalBAG_allSubjects = BAG_allSubjects   ./ tblVolBrain.Age;

%% -------------------------------------------------------------------------
%  SECTION 8 : Assemble output tables and save to Excel
%  -------------------------------------------------------------------------
validColumnNames = matlab.lang.makeValidName(regionGroupNames);   % MATLAB-safe column headers

% Convert matrices to tables with region column names
tblPredictedBrainAge  = array2table(predictedBrainAge,            'VariableNames', validColumnNames);
tblBAG                = array2table(BAG_allSubjects,               'VariableNames', validColumnNames);
tblProportionalBAG    = array2table(proportionalBAG_allSubjects,   'VariableNames', validColumnNames);

% Attach subject ID column
if ismember('subject_ID', tblVolBrain.Properties.VariableNames)
    subjectIDs_output = tblVolBrain.subject_ID;
else
    subjectIDs_output = (1:nSubjects)';
end
subjectIDcol = table(subjectIDs_output, 'VariableNames', {'subject_ID'});

tblOutput_brainAge       = [subjectIDcol, tblPredictedBrainAge];
tblOutput_BAG            = [subjectIDcol, tblBAG];
tblOutput_proportionalBAG = [subjectIDcol, tblProportionalBAG];

% Write to Excel
writetable(tblOutput_brainAge,        output_brainAge_file,        'Sheet', 'Brain_Age');
writetable(tblOutput_BAG,             output_BAG_file,             'Sheet', 'Brain_Age_Gap');
writetable(tblOutput_proportionalBAG, output_proportionalBAG_file, 'Sheet', 'Prop_Brain_Age_Gap');

fprintf('Saved outputs:\n');
fprintf('  %s  (Brain Age)\n',              output_brainAge_file);
fprintf('  %s  (BAG)\n',                   output_BAG_file);
fprintf('  %s  (Proportional BAG)\n\n',    output_proportionalBAG_file);

%% -------------------------------------------------------------------------
%  SECTION 9 : Diagnostics summary
%  -------------------------------------------------------------------------
fprintf('--- Diagnostics ---\n');
fprintf('  Total subjects     : %d\n', nSubjects);
fprintf('  Controls           : %d (UseForEstimation == "Within Range")\n', numel(controlRowIndices));
fprintf('  Test subjects      : %d (UseForEstimation == "Outside Range")\n', numel(testRowIndices));
fprintf('  Regions detected   : %d\n', nRegions);

% Median number of valid ROIs used per region (first 10 shown)
medianROIsUsed = nanmedian(nROIsUsedPerEstimate, 1);
fprintf('  Median valid ROIs used per region (first 10):\n');
for regionIdx = 1:min(10, nRegions)
    fprintf('    Region %2d (%s): %.1f\n', regionIdx, regionGroupNames{regionIdx}, medianROIsUsed(regionIdx));
end
if nRegions > 10
    fprintf('    ... (%d more regions not shown)\n', nRegions - 10);
end

%% -------------------------------------------------------------------------
%  SECTION 10 : Box plots for visual inspection
%  -------------------------------------------------------------------------
try
    % Uses external create_box_plots if available (must be on MATLAB path)
    create_box_plots(tblPredictedBrainAge, controlAges, 'Brain Age',              [20 100], 'Brain_age.png');
    create_box_plots(tblBAG,               controlAges, 'Brain Age Gap',          [-30 30], 'Brain_age_gap.png');
    create_box_plots(tblProportionalBAG,   controlAges, 'Proportional Brain Age Gap', [-2 2], 'Brain_age_gap_PBA.png');
catch plotError
    % Fallback: built-in boxplot for the first 6 regions
    warning('create_box_plots unavailable (%s). Showing fallback boxplot for first 6 regions.', plotError.message);
    nRegionsToPlot = min(6, nRegions);
    plotData = tblPredictedBrainAge{:, 1:nRegionsToPlot};

    figure('Name', 'Brain Age distributions (first 6 regions)', 'Color', 'w');
    for plotIdx = 1:nRegionsToPlot
        subplot(2, 3, plotIdx);
        boxplot(plotData(:, plotIdx));
        title(validColumnNames{plotIdx}, 'Interpreter', 'none');
        ylim([20 100]);
    end
end


%% =========================================================================
%  LOCAL HELPER FUNCTIONS
%  =========================================================================

function [predictorMatrix_controls, ageVector_controls, predictorVector_patient] = ...
        extract_region_predictors(controlVolumeMatrix, controlAgeVector, ...
                                  subjectVolumeVector, regionSubsetIndices)
% EXTRACT_REGION_PREDICTORS  Slice the control matrix and patient vector to a
%                             specified subset of region columns.
%
%   [X_ctrl, Y_ctrl, X_patient] = extract_region_predictors(
%       controlVolumeMatrix, controlAgeVector,
%       subjectVolumeVector, regionSubsetIndices)
%
%   INPUTS
%     controlVolumeMatrix  : [nControls x nRegions] double — TIV-normalized volumes
%     controlAgeVector     : [nControls x 1] double — chronological ages of controls
%     subjectVolumeVector  : [1 x nRegions] double — TIV-normalized volumes for one subject
%     regionSubsetIndices  : numeric vector — column indices to select
%
%   OUTPUTS
%     predictorMatrix_controls : [nControls x nSelected] — control predictor matrix
%     ageVector_controls       : [nControls x 1]         — control ages (column)
%     predictorVector_patient  : [1 x nSelected]         — subject predictor row vector

    predictorMatrix_controls = controlVolumeMatrix(:, regionSubsetIndices);
    ageVector_controls       = controlAgeVector(:);
    predictorVector_patient  = subjectVolumeVector(regionSubsetIndices);

    % Ensure patient vector is a row vector
    if iscolumn(predictorVector_patient)
        predictorVector_patient = predictorVector_patient';
    end
end


function [predictedBrainAge, meanAtrophyScore, roiUsageInfo] = ...
        estimate_brain_age(predictorMatrix_controls, ageVector_controls, predictorVector_patient)
% ESTIMATE_BRAIN_AGE  Fit a ridge-regularized OLS model on controls and predict
%                     brain age for one subject from their regional volumes.
%
%   [brainAge, atrophy, roiInfo] = estimate_brain_age(X_ctrl, Y_ctrl, X_patient)
%
%   Method:
%     1. Drop any predictor column where the patient value is NaN.
%     2. Drop any predictor column with zero variance across controls.
%     3. Center predictors and response to their control means.
%     4. Fit:  Yc = Xc * beta   using OLS; switch to ridge (lambda=1e-3) if the
%        Gram matrix is ill-conditioned (rcond < 1e-12) or p >= n-1.
%     5. Predict:  brainAge = intercept + X_patient * beta
%     6. Atrophy score = mean Z-score of patient volumes relative to controls.
%
%   INPUTS
%     predictorMatrix_controls : [nControls x nPredictors] — TIV-normalized volumes
%     ageVector_controls       : [nControls x 1]          — chronological ages
%     predictorVector_patient  : [1 x nPredictors]        — patient volumes
%
%   OUTPUTS
%     predictedBrainAge : scalar — estimated brain age (years)
%     meanAtrophyScore  : scalar — mean Z-score of patient volumes vs controls
%     roiUsageInfo      : struct with fields:
%                           .count         — number of valid predictors used
%                           .residuals_RK  — kurtosis of control residuals
%                           .residuals_RM  — mean of control residuals
%                           .residuals_RSTD— SD of control residuals

    ageVector_controls       = ageVector_controls(:);
    predictorVector_patient  = predictorVector_patient(:)';   % ensure row vector

    % --- Step 1: Remove predictors where the patient value is NaN ----------
    hasValidPredictors   = ~isnan(predictorVector_patient);
    predictorMatrix_controls = predictorMatrix_controls(:, hasValidPredictors);
    predictorVector_patient  = predictorVector_patient(hasValidPredictors);

    % Initialize output struct
    roiUsageInfo = struct('count', 0, 'residuals_RK', NaN, 'residuals_RM', NaN, 'residuals_RSTD', NaN);

    if isempty(predictorVector_patient) || isempty(predictorMatrix_controls)
        predictedBrainAge = NaN;
        meanAtrophyScore  = NaN;
        return;
    end

    % --- Step 2: Remove zero-variance predictor columns -------------------
    predictorStdDev      = std(predictorMatrix_controls, 0, 1);
    hasNonZeroVariance   = predictorStdDev > eps;

    if ~all(hasNonZeroVariance)
        predictorMatrix_controls = predictorMatrix_controls(:, hasNonZeroVariance);
        predictorVector_patient  = predictorVector_patient(hasNonZeroVariance);
        predictorStdDev          = predictorStdDev(hasNonZeroVariance);
    end

    if isempty(predictorVector_patient) || size(predictorMatrix_controls, 2) == 0
        predictedBrainAge = NaN;
        meanAtrophyScore  = NaN;
        return;
    end

    % --- Step 3: Center predictors and response around control means ------
    predictorMeans   = mean(predictorMatrix_controls, 1);
    centeredPredictors = bsxfun(@minus, predictorMatrix_controls, predictorMeans);  % [nControls x nPredictors]
    ageMean          = mean(ageVector_controls);
    centeredAge      = ageVector_controls - ageMean;                                % [nControls x 1]

    [nControls, nPredictors] = size(centeredPredictors);
    gramMatrix = centeredPredictors' * centeredPredictors;   % [nPredictors x nPredictors]

    % --- Step 4: Fit OLS or ridge regression ------------------------------
    % Use ridge if the Gram matrix is near-singular or system is underdetermined
    useRidgeRegression = rcond(gramMatrix) < 1e-12 || nPredictors >= (nControls - 1);

    if useRidgeRegression
        ridgePenalty         = 1e-3;
        regressionCoefficients = (gramMatrix + ridgePenalty * eye(size(gramMatrix))) \ ...
                                  (centeredPredictors' * centeredAge);
    else
        regressionCoefficients = gramMatrix \ (centeredPredictors' * centeredAge);
    end

    % Recover intercept from centering:  intercept = ageMean - predictorMeans * beta
    regressionIntercept = ageMean - predictorMeans * regressionCoefficients;

    % --- Step 5: Predict brain age for this subject -----------------------
    predictedBrainAge = regressionIntercept + predictorVector_patient * regressionCoefficients;

    % --- Step 6: Compute atrophy score (mean volume Z-score) --------------
    volumeZScores     = (predictorVector_patient - predictorMeans) ./ predictorStdDev;
    meanAtrophyScore  = mean(volumeZScores(~isnan(volumeZScores)));

    % --- Diagnostic residuals on the training set -------------------------
    predictedAge_controls = predictorMatrix_controls * regressionCoefficients + regressionIntercept;
    controlResiduals      = ageVector_controls - predictedAge_controls;

    roiUsageInfo.count          = nPredictors;
    roiUsageInfo.residuals_RK   = kurtosis(controlResiduals);
    roiUsageInfo.residuals_RM   = mean(controlResiduals);
    roiUsageInfo.residuals_RSTD = std(controlResiduals);
end


function create_box_plots(dataTable, referenceAges, plotTitle, yAxisLimits, outputFilename)
% CREATE_BOX_PLOTS  Generate a boxplot figure for up to 12 columns of a table.
%
%   create_box_plots(dataTable, referenceAges, plotTitle, yAxisLimits, outputFilename)
%
%   INPUTS
%     dataTable      : table — numeric columns to plot (up to 12 shown)
%     referenceAges  : numeric vector — not plotted but reserved for future overlays
%     plotTitle      : char — figure title and y-axis label
%     yAxisLimits    : [ymin ymax] or [] (no clipping)
%     outputFilename : char — path to save figure as PNG (skipped if empty)

    figure('Color', 'w', 'Name', plotTitle);
    nColumnsToPlot = min(12, width(dataTable));
    plotData       = dataTable{:, 1:nColumnsToPlot};

    boxplot(plotData, 'Labels', dataTable.Properties.VariableNames(1:nColumnsToPlot), ...
            'LabelOrientation', 'inline');
    title(plotTitle, 'Interpreter', 'none');
    ylabel(plotTitle);

    if ~isempty(yAxisLimits)
        ylim(yAxisLimits);
    end
    set(gca, 'FontSize', 8);

    if nargin >= 5 && ~isempty(outputFilename)
        try
            saveas(gcf, outputFilename);
        catch
            warning('Failed to save figure to "%s".', outputFilename);
        end
    end
end
