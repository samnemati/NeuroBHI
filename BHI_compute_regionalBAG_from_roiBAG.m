%% BHI_compute_regionalBAG_from_roiBAG.m
%
% DESCRIPTION
% -----------
% Computes volume-weighted regional Brain Age Gap (BAG) scores by:
%   (1) Fuzzy-matching each ROI-level BAG column name to its corresponding
%       brain volume column (in cm³) from a volBrain output file.
%   (2) Aggregating matched ROIs into anatomical lobe groups and functional
%       network groups, computing both a simple (unweighted) mean BAG and a
%       volume-weighted mean BAG per group per subject.
%
% INPUTS (set file paths in the USER CONFIGURATION section below)
% -------
%   roi_bag_file  : Excel file with columns [subject_ID, <ROI>_BAG, ...]
%                   One row per subject; each non-ID column is a per-ROI BAG.
%   volumes_file  : Excel file from volBrain with columns containing
%                   "volume cm3" (case-insensitive) for each brain region.
%                   Must contain a subject_ID column for alignment.
%
% OUTPUTS
% -------
%   ROI_to_Vol_Mapping.xlsx        : Mapping of each ROI-BAG column to its
%                                    best-matched volume column, with match
%                                    score and edit distance (inspect and
%                                    manually correct bad matches before
%                                    interpreting weighted BAGs).
%   Regional_BAGs_weighted_v2.xlsx : Three sheets:
%     - RegionalBAGs_mean     : Simple (unweighted) mean BAG per group
%     - RegionalBAGs_weighted : Volume-weighted mean BAG per group
%     - ROI_to_Vol_Mapping    : Copy of the mapping table
%
% ANATOMICAL GROUPS
% -----------------
%   Lobes: Frontal, Temporal, Parietal, Occipital  x  Left, Right
%   ROIs are assigned by matching their tokenized name against lobe and
%   hemisphere keywords.
%
% FUNCTIONAL NETWORK GROUPS
% -------------------------
%   DomainGeneral   x Left, Right  (superior/middle frontal, precentral,
%                                   supramarginal, insula, cingulate, etc.)
%   LanguageSpecific x Left, Right  (inferior frontal, superior/middle temporal,
%                                    planum temporale/polare, angular, etc.)
%   CinguloOpercular x Left, Right  (anterior/middle cingulate, insula,
%                                    frontal operculum, supplementary motor)
%
% MATCHING ALGORITHM
% ------------------
%   Each column name is tokenized (lowercased, punctuation stripped,
%   abbreviations expanded: L→left, R→right, WM→white matter, GM→grey matter,
%   unit words removed). For each ROI-BAG column, every volume column is
%   scored as:
%       score = nSharedTokens + hemisphereMatchBonus - 0.5 * editDistancePenalty
%   The best-scoring volume column is selected. If no token overlap exists,
%   a normalized Levenshtein edit distance fallback is used (threshold < 0.35).
%
% USAGE
% -----
%   1. Set the four file paths in the USER CONFIGURATION section.
%   2. Run the script (no arguments needed).
%   3. Inspect ROI_to_Vol_Mapping.xlsx and correct any wrong matches.
%   4. Use Regional_BAGs_weighted_v2.xlsx for downstream analyses.
%
% DEPENDENCIES
% ------------
%   MATLAB R2019b or later (uses readtable, writetable, contains, intersect).
%   No toolboxes required.
%
% AUTHOR  : Saamnaeh Nemati
% PROJECT : ABC BrainAge / NeuroBHI
% CREATED : ~2025-Q4
% UPDATED : 2026-04-09  (variable rename + full documentation pass)

clear; clc; close all;

%% =========================================================================
%  USER CONFIGURATION — set file paths here before running
%  =========================================================================
roiBAG_file     = '/Users/snemati/Documents/ABC_BrainAge/Output/BrainAge/volBrain_Regional_BrainAgeGaps.xlsx';
volumes_file    = '/Users/snemati/Documents/ABC_BrainAge/Output/BrainAge/ABC_Baseline_Regional_Volumes.xlsx';
output_mapping_file = 'ROI_to_Vol_Mapping.xlsx';
output_weighted_file = 'Regional_BAGs_weighted_v2.xlsx';
%  =========================================================================

%% -------------------------------------------------------------------------
%  SECTION 1 : Load input tables
%  -------------------------------------------------------------------------
fprintf('Loading ROI-BAG file: %s\n', roiBAG_file);
tblROI_BAG = readtable(roiBAG_file);

fprintf('Loading brain volumes file: %s\n', volumes_file);
tblVolumes = readtable(volumes_file);

% --- Identify the subject ID column in the BAG table ---------------------
% The expected column name is "subject_ID"; fall back to the first column.
if ismember('subject_ID', tblROI_BAG.Properties.VariableNames)
    subjectIDs_BAG = tblROI_BAG.subject_ID;
else
    subjectIDs_BAG = tblROI_BAG{:, 1};
    tblROI_BAG.subject_ID = subjectIDs_BAG;  % ensure it is named uniformly
end

% Collect all column names that represent per-ROI BAG values (exclude ID)
allBAGColumnNames  = tblROI_BAG.Properties.VariableNames;
roiBAGColumnNames  = setdiff(allBAGColumnNames, {'subject_ID'});

% --- Identify volume columns in the volumes table ------------------------
% Prefer columns whose name contains both "volume" and "cm3".
allVolumeColumnNames = tblVolumes.Properties.VariableNames;
isVolumeCm3Column = contains(lower(allVolumeColumnNames), 'volume') & ...
                    contains(lower(allVolumeColumnNames), 'cm3');
volumeColumnNames = allVolumeColumnNames(isVolumeCm3Column);

if isempty(volumeColumnNames)
    % Fallback: any column containing "volume" (unit may be implicit)
    isVolumeCm3Column = contains(lower(allVolumeColumnNames), 'volume');
    volumeColumnNames = allVolumeColumnNames(isVolumeCm3Column);
    warning('No "volume cm3" columns found; falling back to columns containing "volume".');
end
fprintf('Detected %d volume columns in the volumes file.\n', numel(volumeColumnNames));

% --- Align subjects between the two tables --------------------------------
% Requires subject_ID in the volumes table.
if ismember('subject_ID', tblVolumes.Properties.VariableNames)
    % libidx(i): row index in tblVolumes for subject i of tblROI_BAG (0 = no match)
    [hasVolumeMatch, volumeRowIndex] = ismember(tblROI_BAG.subject_ID, tblVolumes.subject_ID);
    fprintf('Subjects matched between BAG and volumes: %d / %d\n', ...
            sum(hasVolumeMatch), numel(hasVolumeMatch));
else
    error('tblVolumes must contain a "subject_ID" column to align subjects for volume weighting.');
end

%% -------------------------------------------------------------------------
%  SECTION 2 : Tokenize column names for fuzzy matching
%  -------------------------------------------------------------------------
% Each column name is broken into a bag-of-words token list after lowercasing,
% expanding abbreviations, and stripping uninformative words (see tokenize_name).

nROIs          = numel(roiBAGColumnNames);
nVolumeColumns = numel(volumeColumnNames);

% Token lists for every ROI-BAG column
roiBAG_tokens = cell(nROIs, 1);
for roiIdx = 1:nROIs
    roiBAG_tokens{roiIdx} = tokenize_name(roiBAGColumnNames{roiIdx});
end

% Token lists for every volume column
volume_tokens = cell(nVolumeColumns, 1);
for volIdx = 1:nVolumeColumns
    volume_tokens{volIdx} = tokenize_name(volumeColumnNames{volIdx});
end

%% -------------------------------------------------------------------------
%  SECTION 3 : Compute pairwise matching scores (ROI-BAG vs. volume columns)
%  -------------------------------------------------------------------------
% For each (ROI, volume column) pair, compute:
%   sharedTokenCounts     : number of tokens that appear in both names
%   hemisphereMatchFlags  : 1 if both names share the same hemisphere word
%   normalizedEditDistances: Levenshtein distance between joined token strings,
%                            normalized to [0, 1] (lower = more similar)
%   matchingScores        : composite score = sharedTokens + hemisphereBonus
%                           - 0.5 * editDistancePenalty  (higher = better)

matchingScores          = zeros(nROIs, nVolumeColumns);
sharedTokenCounts       = zeros(nROIs, nVolumeColumns);
hemisphereMatchFlags    = zeros(nROIs, nVolumeColumns);
normalizedEditDistances = ones(nROIs, nVolumeColumns);   % default 1 (worst)

for roiIdx = 1:nROIs
    roiTokens   = roiBAG_tokens{roiIdx};
    roiJoined   = strjoin(roiTokens, ' ');
    roiHasLeft  = any(contains(roiTokens, 'left'));
    roiHasRight = any(contains(roiTokens, 'right'));

    for volIdx = 1:nVolumeColumns
        volTokens   = volume_tokens{volIdx};
        volJoined   = strjoin(volTokens, ' ');
        volHasLeft  = any(contains(volTokens, 'left'));
        volHasRight = any(contains(volTokens, 'right'));

        % Count shared tokens between ROI name and volume column name
        commonTokens = intersect(roiTokens, volTokens);
        nCommonTokens = numel(commonTokens);
        sharedTokenCounts(roiIdx, volIdx) = nCommonTokens;

        % Give a bonus if both names refer to the same hemisphere
        hemisphereBonus = 0;
        if (roiHasLeft && volHasLeft) || (roiHasRight && volHasRight)
            hemisphereBonus = 1;
            hemisphereMatchFlags(roiIdx, volIdx) = 1;
        end

        % Normalized Levenshtein edit distance between full joined strings
        editDist = normalized_edit_dist(roiJoined, volJoined);   % 0..1
        normalizedEditDistances(roiIdx, volIdx) = editDist;

        % Composite score: reward token overlap + hemisphere match,
        % apply a mild edit-distance penalty for tie-breaking
        matchingScores(roiIdx, volIdx) = nCommonTokens + hemisphereBonus - 0.5 * editDist;
    end
end

%% -------------------------------------------------------------------------
%  SECTION 4 : Select best-matching volume column for each ROI-BAG column
%  -------------------------------------------------------------------------
roiToVolumeColumnMap = cell(nROIs, 1);   % matched volume column name (or '' if none)
matchScore           = zeros(nROIs, 1);  % composite score of the match
matchEditDistance    = zeros(nROIs, 1);  % edit distance of the match

for roiIdx = 1:nROIs
    scoresForROI = matchingScores(roiIdx, :);
    [bestScore, bestVolIdx] = max(scoresForROI);

    if bestScore <= 0
        % No token overlap and no hemisphere bonus — try edit-distance fallback
        [minEditDist, closestVolIdx] = min(normalizedEditDistances(roiIdx, :));
        if minEditDist < 0.35
            bestVolIdx  = closestVolIdx;
            bestScore   = 0.1;   % flag as low-confidence
        else
            bestVolIdx  = 0;     % truly unmatched
            bestScore   = 0;
        end
    end

    if bestVolIdx > 0
        roiToVolumeColumnMap{roiIdx} = volumeColumnNames{bestVolIdx};
        matchScore(roiIdx)        = bestScore;
        matchEditDistance(roiIdx) = normalizedEditDistances(roiIdx, bestVolIdx);
    else
        roiToVolumeColumnMap{roiIdx} = '';     % unmatched
        matchScore(roiIdx)        = 0;
        matchEditDistance(roiIdx) = NaN;
    end
end

% --- Matching diagnostics ------------------------------------------------
isHighConfidenceMatch = matchScore >= 1;   % ≥1 token overlapped or hemi boost
isLowConfidenceMatch  = (matchScore > 0 & matchScore < 1) | ...
                        (~cellfun(@isempty, roiToVolumeColumnMap) & isnan(matchEditDistance));
nHighConfidenceMatches = sum(isHighConfidenceMatch);
nLowConfidenceMatches  = sum(isLowConfidenceMatch);
nUnmatchedROIs         = sum(cellfun(@isempty, roiToVolumeColumnMap));

fprintf('Mapping summary: %d high-confidence, %d low-confidence, %d unmatched (out of %d ROIs)\n', ...
        nHighConfidenceMatches, nLowConfidenceMatches, nUnmatchedROIs, nROIs);

% Print up to 10 high-confidence examples
fprintf('\nExamples of high-confidence matches (ROI_BAG -> volume column) [score, editDist]:\n');
exampleCount = 0;
for roiIdx = 1:nROIs
    if ~isempty(roiToVolumeColumnMap{roiIdx}) && matchScore(roiIdx) >= 1 && exampleCount < 10
        exampleCount = exampleCount + 1;
        fprintf('  %-50s ->  %-50s  [score=%.2f, edit=%.3f]\n', ...
                roiBAGColumnNames{roiIdx}, roiToVolumeColumnMap{roiIdx}, ...
                matchScore(roiIdx), matchEditDistance(roiIdx));
    end
end

% Print up to 15 low-confidence or unmatched examples for manual review
if exampleCount < 10
    fprintf('\nLow-confidence / unmatched ROIs (may need manual correction):\n');
    lowConfCount = 0;
    for roiIdx = 1:nROIs
        if isempty(roiToVolumeColumnMap{roiIdx}) || matchScore(roiIdx) < 1
            lowConfCount = lowConfCount + 1;
            fprintf('  %-50s ->  %-50s  [score=%.2f, edit=%.3f]\n', ...
                    roiBAGColumnNames{roiIdx}, roiToVolumeColumnMap{roiIdx}, ...
                    matchScore(roiIdx), matchEditDistance(roiIdx));
            if lowConfCount >= 15, break; end
        end
    end
end

% --- Save mapping table for inspection / manual correction ---------------
mappingROI_names        = roiBAGColumnNames(:);
mappingVolColumn_names  = roiToVolumeColumnMap(:);
mappingScores           = matchScore(:);
mappingEditDistances    = matchEditDistance(:);

tblROI_to_Volume_Mapping = table(mappingROI_names, mappingVolColumn_names, ...
                                  mappingScores, mappingEditDistances, ...
                                  'VariableNames', {'ROI_BAG', 'Matched_VolColumn', 'Score', 'EditDist'});
writetable(tblROI_to_Volume_Mapping, output_mapping_file);
fprintf('\nSaved ROI-to-volume mapping to %s.\n', output_mapping_file);
fprintf('Inspect this file and manually correct any wrong matches before using weighted BAGs.\n\n');

%% -------------------------------------------------------------------------
%  SECTION 5 : Aggregate ROIs into groups and compute regional BAGs
%  -------------------------------------------------------------------------
% Extract the full ROI-BAG data matrix: rows = subjects, cols = ROIs
roiBAG_matrix = tblROI_BAG{:, roiBAGColumnNames};   % [nSubjects x nROIs]

% --- 5a. Anatomical lobe groups (lobe x hemisphere) ----------------------
lobeNames        = {'frontal', 'temporal', 'parietal', 'occipital'};
hemisphereNames  = {'left', 'right'};

anatomicalGroupNames      = {};
anatomicalBAG_simpleMean  = [];   % [nSubjects x nGroups], unweighted mean
anatomicalBAG_weightedMean = [];  % [nSubjects x nGroups], volume-weighted mean

for lobeIdx = 1:numel(lobeNames)
    lobeName = lobeNames{lobeIdx};
    for hemisphereIdx = 1:2
        hemisphereName = hemisphereNames{hemisphereIdx};
        groupName = sprintf('%s_%s', upperFirst(lobeName), upperFirst(hemisphereName));
        anatomicalGroupNames{end+1} = groupName; %#ok<SAGROW>

        % Find ROIs whose token list contains both the lobe name and hemisphere
        roiIsInGroup = false(nROIs, 1);
        for roiIdx = 1:nROIs
            roiToks = roiBAG_tokens{roiIdx};
            if any(contains(roiToks, lobeName)) && any(contains(roiToks, hemisphereName))
                roiIsInGroup(roiIdx) = true;
            end
        end
        groupROI_indices = find(roiIsInGroup);

        if isempty(groupROI_indices)
            % No ROIs matched this group — fill with NaN
            anatomicalBAG_simpleMean(:, end+1)   = NaN(size(roiBAG_matrix, 1), 1); %#ok<SAGROW>
            anatomicalBAG_weightedMean(:, end+1) = NaN(size(roiBAG_matrix, 1), 1); %#ok<SAGROW>
            fprintf('ANAT group %s: no ROIs matched.\n', groupName);
            continue;
        end

        groupBAG_data = roiBAG_matrix(:, groupROI_indices);

        % Simple (unweighted) mean across ROIs in this group
        anatomicalBAG_simpleMean(:, end+1) = nanmean(groupBAG_data, 2); %#ok<SAGROW>

        % Volume-weighted mean: weight each ROI's BAG by its volume
        anatomicalBAG_weightedMean(:, end+1) = compute_volume_weighted_mean( ...
            groupBAG_data, groupROI_indices, roiToVolumeColumnMap, tblVolumes, volumeRowIndex); %#ok<SAGROW>

        fprintf('ANAT group %s: %d ROIs included (volume-weighted mean computed)\n', ...
                groupName, numel(groupROI_indices));
    end
end

% --- 5b. Functional network groups (network x hemisphere) ----------------
% Each network is defined by a set of anatomical keywords; an ROI is
% assigned to a network if any keyword appears in the ROI's token list.

domainGeneralKeywords  = {'superior frontal', 'sup frontal', 'middle frontal', ...
                           'precentral', 'supramarginal', 'insula', 'insular', ...
                           'anterior cingulate', 'posterior cingulate', ...
                           'supplementary motor', 'operculum', 'frontal operculum'};

languageSpecificKeywords = {'inferior frontal', 'opercular', 'triangular', 'orbital', ...
                             'superior temporal', 'middle temporal', 'planum temporale', ...
                             'planum polare', 'temporal pole', 'angular', 'supramarginal', ...
                             'heschl', 'planum', 'fusiform'};

cinguloOpercularKeywords = {'anterior cingulate', 'middle cingulate', 'insula', 'insular', ...
                             'frontal operculum', 'supplementary motor', 'operculum'};

functionalNetworkKeywordSets = {domainGeneralKeywords, languageSpecificKeywords, cinguloOpercularKeywords};
functionalNetworkLabels      = {'DomainGeneral', 'LanguageSpecific', 'CinguloOpercular'};

functionalGroupNames       = {};
functionalBAG_simpleMean   = [];
functionalBAG_weightedMean = [];

for networkIdx = 1:numel(functionalNetworkKeywordSets)
    networkKeywords = functionalNetworkKeywordSets{networkIdx};
    for hemisphereIdx = 1:2
        hemisphereName = hemisphereNames{hemisphereIdx};
        groupName = sprintf('%s_%s', functionalNetworkLabels{networkIdx}, upperFirst(hemisphereName));
        functionalGroupNames{end+1} = groupName; %#ok<SAGROW>

        % An ROI belongs to this group if:
        %   (a) its token list contains the correct hemisphere word, AND
        %   (b) any of the network's keywords appears in its token list
        roiIsInGroup = false(nROIs, 1);
        for roiIdx = 1:nROIs
            roiToks = roiBAG_tokens{roiIdx};
            if any(contains(roiToks, hemisphereName))
                for kwIdx = 1:numel(networkKeywords)
                    if any(contains(roiToks, networkKeywords{kwIdx}))
                        roiIsInGroup(roiIdx) = true;
                        break;
                    end
                end
            end
        end
        groupROI_indices = find(roiIsInGroup);

        if isempty(groupROI_indices)
            functionalBAG_simpleMean(:, end+1)   = NaN(size(roiBAG_matrix, 1), 1); %#ok<SAGROW>
            functionalBAG_weightedMean(:, end+1) = NaN(size(roiBAG_matrix, 1), 1); %#ok<SAGROW>
            fprintf('FUNC group %s: no ROIs matched.\n', groupName);
            continue;
        end

        groupBAG_data = roiBAG_matrix(:, groupROI_indices);

        functionalBAG_simpleMean(:, end+1) = nanmean(groupBAG_data, 2); %#ok<SAGROW>
        functionalBAG_weightedMean(:, end+1) = compute_volume_weighted_mean( ...
            groupBAG_data, groupROI_indices, roiToVolumeColumnMap, tblVolumes, volumeRowIndex); %#ok<SAGROW>

        fprintf('FUNC group %s: %d ROIs included (volume-weighted mean computed)\n', ...
                groupName, numel(groupROI_indices));
    end
end

%% -------------------------------------------------------------------------
%  SECTION 6 : Assemble output tables and save to Excel
%  -------------------------------------------------------------------------
subjectIDs_output = tblROI_BAG.subject_ID;

% Convert numeric matrices to named tables
tblAnatomicalBAG_simple   = array2table(anatomicalBAG_simpleMean,   'VariableNames', matlab.lang.makeValidName(anatomicalGroupNames));
tblAnatomicalBAG_weighted = array2table(anatomicalBAG_weightedMean, 'VariableNames', matlab.lang.makeValidName(anatomicalGroupNames));
tblFunctionalBAG_simple   = array2table(functionalBAG_simpleMean,   'VariableNames', matlab.lang.makeValidName(functionalGroupNames));
tblFunctionalBAG_weighted = array2table(functionalBAG_weightedMean, 'VariableNames', matlab.lang.makeValidName(functionalGroupNames));

% Combine subject IDs with anatomical and functional group columns
tblOutput_simpleMean   = [table(subjectIDs_output, 'VariableNames', {'subject_ID'}), ...
                           tblAnatomicalBAG_simple,   tblFunctionalBAG_simple];
tblOutput_weightedMean = [table(subjectIDs_output, 'VariableNames', {'subject_ID'}), ...
                           tblAnatomicalBAG_weighted, tblFunctionalBAG_weighted];

% Write to Excel: one file, three sheets
writetable(tblOutput_simpleMean,          output_weighted_file, 'Sheet', 'RegionalBAGs_mean');
writetable(tblOutput_weightedMean,        output_weighted_file, 'Sheet', 'RegionalBAGs_weighted');
writetable(tblROI_to_Volume_Mapping,      output_weighted_file, 'Sheet', 'ROI_to_Vol_Mapping');

fprintf('\nSaved regional BAGs (simple and volume-weighted) to %s\n', output_weighted_file);
fprintf('Sheets: RegionalBAGs_mean | RegionalBAGs_weighted | ROI_to_Vol_Mapping\n');
fprintf('\nAll done. Inspect the ROI_to_Vol_Mapping sheet and RegionalBAGs_weighted carefully.\n');


%% =========================================================================
%  LOCAL HELPER FUNCTIONS
%  =========================================================================

function tokens = tokenize_name(columnName)
% TOKENIZE_NAME  Convert a column name into a canonical bag-of-words token list.
%
%   tokens = tokenize_name(columnName)
%
%   Steps:
%     1. Lowercase everything.
%     2. Replace underscores and hyphens with spaces.
%     3. Remove parentheses, dots, commas, percent signs.
%     4. Expand common abbreviations (WM, GM, L, R).
%     5. Remove uninformative unit/meta words (volume, cm3, total, bag).
%     6. Collapse multiple spaces and split into a cell array of tokens.
%
%   INPUT
%     columnName : char or string — a raw column header from a table
%   OUTPUT
%     tokens     : cell array of char — cleaned token words

    s = lower(columnName);

    % Replace delimiter characters with spaces
    s = strrep(s, '_', ' ');
    s = strrep(s, '-', ' ');
    s = regexprep(s, '[\(\)\.,%]', ' ');

    % Expand common abbreviations to full words
    s = regexprep(s, '\bwm\b', 'white matter');
    s = regexprep(s, '\bgm\b', 'grey matter');
    s = regexprep(s, '\br\b',  'right');
    s = regexprep(s, '\bl\b',  'left');

    % Remove words that carry no discriminative information for matching
    s = regexprep(s, '\bvolume\b|\bcm3\b|\bcc\b|\btotal\b', ' ');
    s = regexprep(s, '\bbag\b', ' ');   % "BAG" suffix in ROI column names

    % Collapse whitespace and trim
    s = regexprep(s, '\s+', ' ');
    s = strtrim(s);

    % Split into individual tokens, remove any empty entries
    tokens = strsplit(s, ' ');
    tokens = tokens(~cellfun('isempty', tokens));
end


function dist = normalized_edit_dist(strA, strB)
% NORMALIZED_EDIT_DIST  Compute the Levenshtein edit distance between two
%                        strings, normalized to the range [0, 1].
%
%   dist = normalized_edit_dist(strA, strB)
%
%   dist = 0 means identical strings; dist = 1 means maximally different.
%   Uses classic dynamic-programming Levenshtein algorithm.
%
%   INPUTS
%     strA, strB : char or string — strings to compare
%   OUTPUT
%     dist       : double in [0, 1]

    strA = char(strA);
    strB = char(strB);
    lenA = length(strA);
    lenB = length(strB);

    % Initialize edit-distance matrix
    editDistMatrix = zeros(lenA + 1, lenB + 1);
    editDistMatrix(1, :) = 0:lenB;
    editDistMatrix(:, 1) = 0:lenA;

    % Fill matrix using standard DP recurrence
    for rowIdx = 2:lenA + 1
        for colIdx = 2:lenB + 1
            substitutionCost = ~(strA(rowIdx - 1) == strB(colIdx - 1));   % 0 if equal, 1 if not
            editDistMatrix(rowIdx, colIdx) = min([ ...
                editDistMatrix(rowIdx - 1, colIdx)     + 1, ...   % deletion
                editDistMatrix(rowIdx,     colIdx - 1) + 1, ...   % insertion
                editDistMatrix(rowIdx - 1, colIdx - 1) + substitutionCost]);  % substitution
        end
    end

    levenshteinDist = editDistMatrix(lenA + 1, lenB + 1);
    maxLength = max(lenA, lenB);

    if maxLength == 0
        dist = 0;   % both strings are empty — perfect match
    else
        dist = levenshteinDist / maxLength;   % normalize to [0, 1]
    end
end


function weightedMeanBAG = compute_volume_weighted_mean( ...
    groupBAG_data, groupROI_globalIndices, roiToVolumeColumnMap, tblVolumes, volumeRowIndex)
% COMPUTE_VOLUME_WEIGHTED_MEAN  Compute a per-subject volume-weighted mean BAG
%                                for a set of ROIs within a group.
%
%   weightedMeanBAG = compute_volume_weighted_mean(
%       groupBAG_data, groupROI_globalIndices,
%       roiToVolumeColumnMap, tblVolumes, volumeRowIndex)
%
%   For each subject, the weighted mean is:
%       weightedMeanBAG(s) = sum(BAG(s,k) * volume(s,k)) / sum(volume(s,k))
%   If volume data are missing for all ROIs in a group, the function falls
%   back to a simple (unweighted) nanmean.  Missing volumes for individual
%   ROIs are imputed with the group mean volume across ROIs for that subject.
%
%   INPUTS
%     groupBAG_data           : [nSubjects x nGroupROIs] double — BAG values
%     groupROI_globalIndices  : [nGroupROIs x 1] int — indices into the full
%                               ROI list (roiBAGColumnNames / roiToVolumeColumnMap)
%     roiToVolumeColumnMap    : {nROIs x 1} cell of char — matched volume column
%                               name for each ROI ('' if unmatched)
%     tblVolumes              : table — the brain volumes table
%     volumeRowIndex          : [nSubjects x 1] int — row in tblVolumes for
%                               each subject in tblROI_BAG (0 = not found)
%   OUTPUT
%     weightedMeanBAG         : [nSubjects x 1] double — weighted mean BAG

    nSubjects   = size(groupBAG_data, 1);
    nGroupROIs  = size(groupBAG_data, 2);
    weightedMeanBAG = nan(nSubjects, 1);

    % Build volume matrix: roiVolumeMatrix(s, k) = volume of ROI k for subject s
    roiVolumeMatrix = nan(nSubjects, nGroupROIs);
    for roiIdx = 1:nGroupROIs
        globalROI_idx       = groupROI_globalIndices(roiIdx);
        matchedVolumeColumn = roiToVolumeColumnMap{globalROI_idx};

        if ~isempty(matchedVolumeColumn)
            hasMatchedVolume = volumeRowIndex > 0;   % subjects present in tblVolumes
            volumeValues     = nan(nSubjects, 1);
            volumeValues(hasMatchedVolume) = tblVolumes{volumeRowIndex(hasMatchedVolume), matchedVolumeColumn};
            roiVolumeMatrix(:, roiIdx) = volumeValues;
        end
        % Unmatched ROIs keep NaN volumes and will be excluded from weighting
    end

    % Compute weighted mean per subject
    for subjectIdx = 1:nSubjects
        bagValues   = groupBAG_data(subjectIdx, :);
        validBAGmask = ~isnan(bagValues);

        if ~any(validBAGmask)
            % All BAG values missing for this subject — return NaN
            weightedMeanBAG(subjectIdx) = NaN;
            continue;
        end

        validBAGvalues = bagValues(validBAGmask);
        rawWeights     = roiVolumeMatrix(subjectIdx, validBAGmask);

        if all(isnan(rawWeights)) || nansum(rawWeights) == 0
            % No volume data available — fall back to simple mean
            weightedMeanBAG(subjectIdx) = nanmean(validBAGvalues);
        else
            % Impute missing individual ROI volumes with the group mean volume
            groupMeanVolume = nanmean(rawWeights(~isnan(rawWeights)));
            rawWeights(isnan(rawWeights)) = groupMeanVolume;

            if sum(rawWeights) == 0
                % All imputed volumes are zero — fall back to simple mean
                weightedMeanBAG(subjectIdx) = nanmean(validBAGvalues);
            else
                normalizedWeights = rawWeights ./ sum(rawWeights);
                weightedMeanBAG(subjectIdx) = nansum(validBAGvalues .* normalizedWeights);
            end
        end
    end
end


function result = upperFirst(inputStr)
% UPPERFIRST  Capitalize only the first character of a string.
%
%   result = upperFirst(inputStr)
%
%   INPUT  inputStr : char or string
%   OUTPUT result   : char with first letter uppercased, rest lowercased

    result = strtrim(inputStr);
    if isempty(result)
        return;
    end
    result    = lower(result);
    result(1) = upper(result(1));
end
