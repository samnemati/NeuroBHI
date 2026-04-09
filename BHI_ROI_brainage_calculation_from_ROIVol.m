%% BHI_ROI_brainage_calculation_from_ROIVol.m
%
% DESCRIPTION
% -----------
%
% For each brain region group (Domain-General, Language-Specific, Frontal,
% Temporal, Parietal, Occipital) × hemisphere (Left, Right):
%   1. Collect TIV-normalized GM volumes of all ROIs in that group.
%   2. Fit one multivariate OLS regression on within-range controls:
%         BrainAgeR_estimated_age ~ [ROI volumes / TIV]
%      Ridge regularization is applied if the design matrix is ill-conditioned.
%   3. Predict each subject's regional brain age from their own ROI volumes.
%      Leave-One-Out (LOO) is applied for within-range controls (subject
%      removed from training before their own prediction).
%      Outside-range subjects are predicted from the full within-range model.
%   4. Brain Age Gap (BAG) = regional brain age − chronological age.
%
% ADAPTATION NOTES (differences from Busby et al.)
% -------------------------------------------------
% • Atlas: volBrain parcellation (not JHU). ROI assignments replicate the
%   Busby group definitions as closely as possible (see ROI DEFINITIONS).
% • Participants: all neurotypical (no stroke lesions). All ROIs in each
%   group are used for every subject (no lesion masking needed).
% • Within-range criterion: BrainAgeR estimate within ±5% of chronological
%   age, exactly as in Busby et al. (n=81 of 304 subjects).
% • LOO applied for within-range controls so that brain age can be estimated
%   for all subjects (Busby et al. could not estimate for within-range
%   controls as there was no LOO; here LOO solves that limitation).
%
% ROI DEFINITIONS (volBrain columns used per group)
% --------------------------------------------------
%   Domain-General (8 ROIs/hemisphere, Busby Fig. 6 explicit):
%     Superior frontal gyrus, Middle frontal gyrus,
%     Orbital inferior frontal gyrus, Posterior cingulate gyrus,
%     Precentral gyrus, Supramarginal gyrus, Insular cortex,
%     Anterior cingulate gyrus
%
%   Language-Specific (9 ROIs/hemisphere, perisylvian language network):
%     Opercular inferior frontal gyrus, Triangular inferior frontal gyrus,
%     Superior temporal gyrus, Middle temporal gyrus, Inferior temporal gyrus,
%     Angular gyrus, Supramarginal gyrus, Temporal pole, Planum temporale
%
%   Frontal (16 ROIs/hemisphere, all volBrain frontal cortical ROIs):
%     Frontal pole, Gyrus rectus, Opercular/Orbital/Triangular IFG,
%     Medial frontal cortex, Middle frontal gyrus,
%     Anterior/Lateral/Medial/Posterior orbital gyri,
%     Precentral gyrus, Precentral gyrus medial segment,
%     Superior frontal gyrus, Superior frontal gyrus medial segment,
%     Supplementary motor cortex
%
%   Temporal (10 ROIs/hemisphere, cortical temporal only):
%     Superior/Middle/Inferior temporal gyrus, Temporal pole,
%     Fusiform gyrus, Planum temporale, Planum polare,
%     Transverse temporal gyrus, Parahippocampal gyrus, Entorhinal area
%
%   Parietal (6 ROIs/hemisphere, all volBrain parietal ROIs):
%     Angular gyrus, Postcentral gyrus, Postcentral gyrus medial segment,
%     Precuneus, Superior parietal lobule, Supramarginal gyrus
%
%   Occipital (8 ROIs/hemisphere, all volBrain occipital ROIs):
%     Calcarine cortex, Cuneus, Lingual gyrus, Occipital fusiform gyrus,
%     Inferior/Middle/Superior occipital gyrus, Occipital pole
%
% INPUTS (set file paths in USER CONFIGURATION)
% -------
%   volumes_file   : volBrain regional volumes Excel
%                    Required columns: subject_ID, Age, Use for estimation,
%                    Intracranial Cavity (IC) volume cm3, + ROI volume columns
%   brainager_file : BrainAgeR global brain age Excel
%                    Required columns: Subject_ID, Brainager_BrainAge
%
% OUTPUTS
% -------
%   BHI_Regional_BrainAge.xlsx        — Sheet "BrainAge":
%       Predicted regional brain age per group per subject
%   BHI_Regional_BrainAgeGap.xlsx     — Sheet "BAG":
%       BAG = predicted regional age − chronological age
%
% REFERENCE
% ---------
%   Busby N, et al. (2024). Regional brain aging: premature aging of the
%   domain general system predicts aphasia severity.
%   Communications Biology, 7:718. https://doi.org/10.1038/s42003-024-06211-8
%
% AUTHOR  : Saamnaeh Nemati
% PROJECT : ABC BrainAge / NeuroBHI
% CREATED : 2026-04-09

clear; clc; close all;

%% =========================================================================
%  USER CONFIGURATION
%  =========================================================================
volumes_file   = '/Users/snemati/Documents/ABC_BrainAge/Output/BrainAge/ABC_Baseline_Regional_VolBrain_Volumes.xlsx';
brainager_file = '/Users/snemati/Documents/ABC_BrainAge/Output/NeuroBHI/ABC_Global_BrainAgeR.xlsx';
output_brainAge_file = '/Users/snemati/Documents/ABC_BrainAge/Output/NeuroBHI/BHI_Regional_BrainAge.xlsx';
output_BAG_file      = '/Users/snemati/Documents/ABC_BrainAge/Output/NeuroBHI/BHI_Regional_BrainAgeGap.xlsx';

%  =========================================================================

%% -------------------------------------------------------------------------
%  SECTION 1 : Load and merge data
%  -------------------------------------------------------------------------
fprintf('Loading volBrain volumes: %s\n', volumes_file);
tblVol = readtable(volumes_file, 'Sheet', 'Sheet1', 'VariableNamingRule', 'preserve');

fprintf('Loading BrainAgeR estimates: %s\n', brainager_file);
tblBrainAgeR = readtable(brainager_file, 'VariableNamingRule', 'preserve');

% Merge on subject_ID (volumes) / Subject_ID (BrainAgeR)
fprintf('Merging tables on subject_ID...\n');
[isMember, brainAgeR_rowIdx] = ismember(tblVol.subject_ID, tblBrainAgeR.Subject_ID);
nUnmatched = sum(~isMember);
if nUnmatched > 0
    warning('%d subjects in volumes file have no BrainAgeR estimate and will be skipped.', nUnmatched);
end

% Attach BrainAgeR estimates to the volumes table
tblVol.BrainAgeR_estimated = NaN(height(tblVol), 1);
tblVol.BrainAgeR_estimated(isMember) = tblBrainAgeR.Brainager_BrainAge(brainAgeR_rowIdx(isMember));

% Exclude subjects with no BrainAgeR estimate
hasEstimate = isMember;
tblVol = tblVol(hasEstimate, :);
fprintf('Subjects with BrainAgeR estimates: %d\n', height(tblVol));

%% -------------------------------------------------------------------------
%  SECTION 2 : Identify control groups and TIV
%  -------------------------------------------------------------------------
% Within-range criterion (Busby et al. 2024):
%   A subject is within-range if their BrainAgeR global estimate is within
%   ±5% of their chronological age:
%       abs(BrainAgeR - Age) / Age <= 0.05
%   These subjects form the normative training set for the regional models.
%   All remaining subjects are outside-range and are predicted from the
%   full within-range model (no LOO).

pctDiff       = abs(tblVol.BrainAgeR_estimated - tblVol.Age) ./ tblVol.Age;
isWithinRange = pctDiff <= 0.05;   % Busby ±5% criterion
isOutsideRange = ~isWithinRange;

nWithin  = sum(isWithinRange);
nOutside = sum(isOutsideRange);
fprintf('Within-range controls (BrainAgeR within +/-5%% of chronological age): %d\n', nWithin);
fprintf('Outside-range subjects: %d\n', nOutside);

% TIV (Intracranial Cavity volume)
TIV_allSubjects = tblVol.("Intracranial Cavity (IC) volume cm3");
if any(isnan(TIV_allSubjects) | TIV_allSubjects == 0)
    warning('%d subjects have missing or zero TIV — they will produce NaN estimates.', ...
            sum(isnan(TIV_allSubjects) | TIV_allSubjects == 0));
end

nSubjects          = height(tblVol);
subjectIDs         = tblVol.subject_ID;
chronologicalAge   = tblVol.Age;
brainAgeR_global   = tblVol.BrainAgeR_estimated;
withinRangeIndices = find(isWithinRange);   % row indices in tblVol

%% -------------------------------------------------------------------------
%  SECTION 3 : Define region groups and their ROI column names
%  -------------------------------------------------------------------------
% Each group is defined for both hemispheres simultaneously.
% Column name template uses '{h}' as a placeholder for 'left' or 'right'.

regionGroups = struct();

regionGroups(1).name = 'DomainGeneral';
regionGroups(1).roi_templates = {
    'Sup. frontal gyrus {h} volume cm3',
    'Middle frontal gyrus {h} volume cm3',
    'Orbital inf. frontal gyrus {h} volume cm3',
    'Posterior cingulate gyrus {h} volume cm3',
    'Precentral gyrus {h} volume cm3',
    'Supramarginal gyrus {h} volume cm3',
    'Insular cortex {h} volume cm3',
    'Anterior cingulate gyrus {h} volume cm3',
};

regionGroups(2).name = 'LanguageSpecific';
regionGroups(2).roi_templates = {
    'Opercular inf. frontal gyrus {h} volume cm3',
    'Triangular inf. frontal gyrus {h} volume cm3',
    'Sup. temporal gyrus {h} volume cm3',
    'Middle temporal gyrus {h} volume cm3',
    'Inf. temporal gyrus {h} volume cm3',
    'Angular gyrus {h} volume cm3',
    'Supramarginal gyrus {h} volume cm3',
    'Temporal pole {h} volume cm3',
    'Planum temporale {h} volume cm3',
};

regionGroups(3).name = 'Frontal';
regionGroups(3).roi_templates = {
    'Frontal pole {h} volume cm3',
    'Gyrus rectus {h} volume cm3',
    'Opercular inf. frontal gyrus {h} volume cm3',
    'Orbital inf. frontal gyrus {h} volume cm3',
    'Triangular inf. frontal gyrus {h} volume cm3',
    'Medial frontal cortex {h} volume cm3',
    'Middle frontal gyrus {h} volume cm3',
    'Anterior orbital gyrus {h} volume cm3',
    'Lateral orbital gyrus {h} volume cm3',
    'Medial orbital gyrus {h} volume cm3',
    'Posterior orbital gyrus {h} volume cm3',
    'Precentral gyrus {h} volume cm3',
    'Precentral gyrus medial segment {h} volume cm3',
    'Sup. frontal gyrus {h} volume cm3',
    'Sup. frontal gyrus medial segment {h} volume cm3',
    'Supplementary motor cortex {h} volume cm3',
};

regionGroups(4).name = 'Temporal';
regionGroups(4).roi_templates = {
    'Sup. temporal gyrus {h} volume cm3',
    'Middle temporal gyrus {h} volume cm3',
    'Inf. temporal gyrus {h} volume cm3',
    'Temporal pole {h} volume cm3',
    'Fusiform gyrus {h} volume cm3',
    'Planum temporale {h} volume cm3',
    'Planum polare {h} volume cm3',
    'Transverse temporal gyrus {h} volume cm3',
    'Parahippocampal gyrus {h} volume cm3',
    'Entorhinal area {h} volume cm3',
};

regionGroups(5).name = 'Parietal';
regionGroups(5).roi_templates = {
    'Angular gyrus {h} volume cm3',
    'Postcentral gyrus {h} volume cm3',
    'Postcentral gyrus medial segment {h} volume cm3',
    'Precuneus {h} volume cm3',
    'Sup. parietal lobule {h} volume cm3',
    'Supramarginal gyrus {h} volume cm3',
};

regionGroups(6).name = 'Occipital';
regionGroups(6).roi_templates = {
    'Calcarine cortex {h} volume cm3',
    'Cuneus {h} volume cm3',
    'Lingual gyrus {h} volume cm3',
    'Occipital fusiform gyrus {h} volume cm3',
    'Inf. occipital gyrus {h} volume cm3',
    'Middle occipital gyrus {h} volume cm3',
    'Sup. occipital gyrus {h} volume cm3',
    'Occipital pole {h} volume cm3',
};

hemispheres     = {'left', 'right'};
hemisphereLabel = {'Left', 'Right'};
nGroups         = numel(regionGroups);

%% -------------------------------------------------------------------------
%  SECTION 4 : Pre-extract and TIV-normalize all ROI volumes
%  -------------------------------------------------------------------------
% Build a struct holding the TIV-normalized volume matrix for each
% region group × hemisphere so we don't re-query the table in the inner loop.

fprintf('\nPre-extracting and TIV-normalizing ROI volumes...\n');

for grpIdx = 1:nGroups
    for hIdx = 1:2
        hemi      = hemispheres{hIdx};
        templates = regionGroups(grpIdx).roi_templates;
        nROIs     = numel(templates);
        colNames  = cellfun(@(t) strrep(t, '{h}', hemi), templates, 'UniformOutput', false);

        % Extract raw volumes [nSubjects x nROIs]
        rawVolumes = zeros(nSubjects, nROIs);
        for roiIdx = 1:nROIs
            rawVolumes(:, roiIdx) = tblVol.(colNames{roiIdx});
        end

        % TIV-normalize: divide each subject's volumes by their TIV
        tivNormalized = rawVolumes ./ TIV_allSubjects;

        regionGroups(grpIdx).normalized_volumes{hIdx} = tivNormalized;
        regionGroups(grpIdx).column_names{hIdx}       = colNames;
    end
end

%% -------------------------------------------------------------------------
%  SECTION 5 : Regional brain age estimation (Busby method)
%  -------------------------------------------------------------------------
% For each region group × hemisphere × subject:
%   - Training set: within-range controls (minus the subject if they are one)
%   - Dependent variable (DV): BrainAgeR global estimated brain age
%   - Independent variables (IVs): TIV-normalized GM volumes of all ROIs
%   - Model: multivariate OLS (ridge if ill-conditioned)
%   - Prediction: apply fitted model to subject's own normalized volumes

% Output matrices [nSubjects x nGroups x 2hemispheres]
predictedRegionalBrainAge = NaN(nSubjects, nGroups, 2);
regionalBAG               = NaN(nSubjects, nGroups, 2);

% BrainAgeR estimates for within-range controls only (the training DV)
brainAgeR_withinRange = brainAgeR_global(isWithinRange);

fprintf('Running LOO regional brain age estimation...\n');
fprintf('  %d region groups x 2 hemispheres x %d subjects\n', nGroups, nSubjects);

for grpIdx = 1:nGroups
    grpName = regionGroups(grpIdx).name;
    fprintf('\nGroup: %s\n', grpName);

    for hIdx = 1:2
        fprintf('  Hemisphere: %s\n', hemisphereLabel{hIdx});

        % Full normalized volume matrix for within-range controls [nWithin x nROIs]
        allNormVol   = regionGroups(grpIdx).normalized_volumes{hIdx};  % [nSubjects x nROIs]
        withinVolumes = allNormVol(isWithinRange, :);   % [nWithin x nROIs]

        for subjIdx = 1:nSubjects

            subjTIV = TIV_allSubjects(subjIdx);
            if isnan(subjTIV) || subjTIV == 0
                continue;  % skip subjects with missing TIV
            end

            % Subject's TIV-normalized ROI volumes [1 x nROIs]
            subjectVolumes = allNormVol(subjIdx, :);

            % ----- Build training set (LOO for within-range controls) -----
            % Find this subject's position in the within-range index list
            positionInWithin = find(withinRangeIndices == subjIdx);

            if ~isempty(positionInWithin)
                % Subject IS a within-range control: remove them (LOO)
                trainVolumes  = withinVolumes;
                trainVolumes(positionInWithin, :) = [];
                trainBrainAge = brainAgeR_withinRange;
                trainBrainAge(positionInWithin)   = [];
            else
                % Subject is outside-range: train on the full within-range set
                trainVolumes  = withinVolumes;
                trainBrainAge = brainAgeR_withinRange;
            end

            % ----- Fit multivariate OLS model and predict ------------------
            predictedAge = fit_and_predict(trainVolumes, trainBrainAge, subjectVolumes);

            predictedRegionalBrainAge(subjIdx, grpIdx, hIdx) = predictedAge;
            regionalBAG(subjIdx, grpIdx, hIdx)               = predictedAge - chronologicalAge(subjIdx);

        end % subject loop

        % Progress: median BAG for this group/hemi
        bagVec = regionalBAG(:, grpIdx, hIdx);
        fprintf('    Done. Median BAG = %.2f years (SD = %.2f)\n', ...
                nanmedian(bagVec), nanstd(bagVec));

    end % hemisphere loop
end % group loop

fprintf('\nEstimation complete.\n');

%% -------------------------------------------------------------------------
%  SECTION 6 : Assemble output tables and save
%  -------------------------------------------------------------------------
% Column name format: <Group>_<Hemisphere>  e.g. DomainGeneral_Left
outputColumnNames = {};
for grpIdx = 1:nGroups
    for hIdx = 1:2
        outputColumnNames{end+1} = sprintf('%s_%s', ...
            regionGroups(grpIdx).name, hemisphereLabel{hIdx}); %#ok<SAGROW>
    end
end

% Reshape 3D matrices to 2D [nSubjects x nGroupsxHemis]
nOutputCols = nGroups * 2;
brainAgeMatrix = NaN(nSubjects, nOutputCols);
bagMatrix      = NaN(nSubjects, nOutputCols);

colOut = 1;
for grpIdx = 1:nGroups
    for hIdx = 1:2
        brainAgeMatrix(:, colOut) = predictedRegionalBrainAge(:, grpIdx, hIdx);
        bagMatrix(:, colOut)      = regionalBAG(:, grpIdx, hIdx);
        colOut = colOut + 1;
    end
end

validColNames = matlab.lang.makeValidName(outputColumnNames);
subjectIDcol  = table(subjectIDs, 'VariableNames', {'subject_ID'});
agecol        = table(chronologicalAge, 'VariableNames', {'Chronological_Age'});
brainAgeRcol  = table(brainAgeR_global, 'VariableNames', {'BrainAgeR_Global'});

tblBrainAge = [subjectIDcol, agecol, brainAgeRcol, ...
               array2table(brainAgeMatrix, 'VariableNames', validColNames)];
tblBAG      = [subjectIDcol, agecol, brainAgeRcol, ...
               array2table(bagMatrix, 'VariableNames', validColNames)];

writetable(tblBrainAge, output_brainAge_file, 'Sheet', 'BrainAge');
writetable(tblBAG,      output_BAG_file,      'Sheet', 'BAG');

fprintf('\nSaved outputs:\n');
fprintf('  %s  (Regional Brain Age)\n', output_brainAge_file);
fprintf('  %s  (Regional Brain Age Gap)\n', output_BAG_file);

%% -------------------------------------------------------------------------
%  SECTION 7 : Diagnostics summary
%  -------------------------------------------------------------------------
fprintf('\n--- Diagnostics ---\n');
fprintf('%-28s %-12s %-12s %-12s\n', 'Region', 'MedianBAG', 'SD_BAG', '%%NaN');
for grpIdx = 1:nGroups
    for hIdx = 1:2
        colLabel = outputColumnNames{(grpIdx-1)*2 + hIdx};
        bagVec   = regionalBAG(:, grpIdx, hIdx);
        pctNaN   = 100 * sum(isnan(bagVec)) / nSubjects;
        fprintf('%-28s %+.3f       %.3f       %.1f%%\n', ...
                colLabel, nanmedian(bagVec), nanstd(bagVec), pctNaN);
    end
end


%% =========================================================================
%  LOCAL HELPER FUNCTIONS
%  =========================================================================

function predictedAge = fit_and_predict(trainVolumes, trainBrainAge, subjectVolumes)
% FIT_AND_PREDICT  Fit a multivariate OLS (or ridge) regression on training
%                  controls and predict brain age for one subject.
%
%   predictedAge = fit_and_predict(trainVolumes, trainBrainAge, subjectVolumes)
%
%   Replicates Busby et al. (2024): one multiple linear regression per region
%   group fitted on within-range controls, then applied to each subject.
%
%   Model:  BrainAgeR_age = intercept + beta * [ROI_vol / TIV]
%
%   Ridge regularization (lambda = 1e-3) is applied automatically when:
%     - The Gram matrix is near-singular (rcond < 1e-12), OR
%     - Number of predictors >= number of training subjects - 1
%
%   Predictors where the subject value is NaN are dropped before fitting.
%   Zero-variance predictors across training subjects are also dropped.
%
%   INPUTS
%     trainVolumes  : [nTrain x nROIs] — TIV-normalized volumes of controls
%     trainBrainAge : [nTrain x 1]     — BrainAgeR estimates of controls
%     subjectVolumes: [1 x nROIs]      — TIV-normalized volumes of subject
%
%   OUTPUT
%     predictedAge  : scalar — predicted regional brain age (years)

    trainBrainAge  = trainBrainAge(:);
    subjectVolumes = subjectVolumes(:)';   % ensure row vector

    % --- Drop predictors where the subject has NaN ----------------------
    validPredictors = ~isnan(subjectVolumes);
    trainVolumes    = trainVolumes(:, validPredictors);
    subjectVolumes  = subjectVolumes(validPredictors);

    if isempty(subjectVolumes) || isempty(trainVolumes)
        predictedAge = NaN;
        return;
    end

    % --- Drop zero-variance predictors (degenerate columns) -------------
    colStdDev      = std(trainVolumes, 0, 1);
    hasVariance    = colStdDev > eps;
    trainVolumes   = trainVolumes(:, hasVariance);
    subjectVolumes = subjectVolumes(hasVariance);

    if isempty(subjectVolumes) || size(trainVolumes, 2) == 0
        predictedAge = NaN;
        return;
    end

    % --- Center predictors and DV around training means -----------------
    % Centering avoids intercept instability and matches standard OLS practice.
    predictorMeans   = mean(trainVolumes, 1);
    centeredTrain    = bsxfun(@minus, trainVolumes, predictorMeans);
    ageMean          = mean(trainBrainAge);
    centeredAge      = trainBrainAge - ageMean;

    [nTrain, nPred] = size(centeredTrain);
    gramMatrix      = centeredTrain' * centeredTrain;

    % --- Choose OLS or ridge based on conditioning ----------------------
    useRidge = (rcond(gramMatrix) < 1e-12) || (nPred >= nTrain - 1);

    if useRidge
        lambda       = 1e-3;
        regrCoeff    = (gramMatrix + lambda * eye(nPred)) \ (centeredTrain' * centeredAge);
    else
        regrCoeff    = gramMatrix \ (centeredTrain' * centeredAge);
    end

    % Recover intercept from centering
    intercept = ageMean - predictorMeans * regrCoeff;

    % --- Predict for this subject ---------------------------------------
    predictedAge = intercept + subjectVolumes * regrCoeff;

end
