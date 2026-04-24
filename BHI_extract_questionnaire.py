"""
Extract BAG-relevant columns from ABC_ComprehensiveQuestionnaire.xlsx and
produce a two-sheet workbook:

  Raw_Selected  - original text/number values for selected columns
  Quantitative  - numerically-encoded version suitable for analysis

Encoding rules (confirmed with user):
  - Yes/No, Checked/Unchecked -> 1/0, blank -> NaN
  - Ordinal Likert scales -> integer scale (lowest -> 0 or 1 by convention below)
  - Categorical single-select with no natural order -> one-hot
  - Continuous totals -> kept as-is (coerced to numeric, '%' stripped)
  - Free-text columns -> dropped
  - "Don't know" / "Refused" / "Prefer not to answer" / "Not applicable" -> NaN

Sections dropped: Reading History; COVID-19 treatment detail (symptom checkboxes kept).
Cols 159-172 (duplicated food-security block mislabeled as Lifetime Discrimination)
are dropped; the first block at 144-157 is kept.
"""

import re
import numpy as np
import pandas as pd

SRC = 'Doc/ABC_ComprehensiveQuestionnaire.xlsx'
DST = 'Results/BHI_Questionnaire_Extracted.xlsx'

MISSING_TOKENS = {
    "don't know", "dont know", "don't know/refused",
    "don't know / not sure", "don't know/not sure",
    "refused", "prefer not to answer",
    "not applicable", "not applicable (the event did not involve the death of a close family member or close friend)",
    "choose this option to enter days per week",
    "choose this option to enter days in past 30 days",
    "choose this option to provide number of drinks",
    "choose to enter number of drinks",
    "choose to enter number of times",
    "can't choose",
    "no drinks in past 30 days",  # handled separately as 0 for drinks-per-week
}


def _norm(s):
    if pd.isna(s):
        return None
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def to_missing(val):
    n = _norm(val)
    return pd.NA if n in MISSING_TOKENS else val


# ---------- encoders ----------

def enc_yesno(s):
    """Yes=1, No=0, missing otherwise."""
    def f(v):
        n = _norm(v)
        if n in ('yes', 'y', 'true', '1'):
            return 1
        if n in ('no', 'n', 'false', '0'):
            return 0
        return np.nan
    return s.map(f)


def enc_checked(s):
    """Checked=1, Unchecked=0."""
    def f(v):
        n = _norm(v)
        if n == 'checked':
            return 1
        if n == 'unchecked':
            return 0
        return np.nan
    return s.map(f)


def enc_ordinal(mapping):
    """Build a function: string -> int via case-insensitive lookup."""
    lut = { _norm(k): v for k, v in mapping.items() }
    def f(v):
        n = _norm(v)
        if n is None or n in MISSING_TOKENS:
            return np.nan
        return lut.get(n, np.nan)
    return f


def enc_numeric(s):
    """Coerce to numeric after stripping '%' and whitespace."""
    def f(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().rstrip('%').strip()
        if s == '' or _norm(s) in MISSING_TOKENS:
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return s.map(f)


def enc_onehot(s, prefix):
    """Return DataFrame of one-hot columns, NaN preserved as all-NaN row."""
    clean = s.map(lambda v: None if pd.isna(v) or _norm(v) in MISSING_TOKENS else str(v).strip())
    cats = sorted(set(c for c in clean if c is not None))
    out = pd.DataFrame(index=s.index)
    for c in cats:
        # make a safe slug from category
        slug = re.sub(r'[^A-Za-z0-9]+', '_', c).strip('_')[:40]
        col = f'{prefix}__{slug}'
        col_vals = clean.map(lambda v: np.nan if v is None else (1 if v == c else 0))
        out[col] = col_vals
    return out


# ---------- named ordinal scales ----------

LIKERT_FREQ_4 = {  # PSQI 5a-5j, etc.
    'not during the past month': 0,
    'less than once a week': 1,
    'once or twice a week': 2,
    'three or more times a week': 3,
}
LIKERT_SEVERITY_3 = {'mild': 1, 'moderate': 2, 'severe': 3}
LIKERT_FREQ_COMPASS = {
    'rarely': 1, 'occasionally': 2, 'frequently': 3, 'almost always': 4,
}
CHANGE_6 = {
    'completely gone': 0,
    'gotten much better': 1,
    'gotten somewhat better': 2,
    'stayed about the same': 3,
    'gotten somewhat worse': 4,
    'gotten much worse': 5,
}
CHANGE_6_ALT = {  # 0..6 positive ordinal; never-had at healthy end
    'i have not had any of these symptoms': 0,
    'completely gone': 1,
    'much better': 2,
    'somewhat better': 3,
    'staying the same': 4,
    'staying about the same': 4,
    'somewhat worse': 5,
    'much worse': 6,
    'getting much better': 2,
    'getting somewhat better': 3,
    'getting somewhat worse': 5,
    'getting much worse': 6,
}
FREQ_RARE_CONST_4 = {  # Rarely/Occasionally/Frequently/Constantly
    'rarely': 1, 'occasionally': 2, 'frequently': 3, 'constantly': 4,
}
FREQ_NEVER_4 = {  # Never/Occasionally/Frequently/Constantly
    'never': 0, 'occasionally': 1, 'frequently': 2, 'constantly': 3,
}
FREQ_NEVER_3 = {  # Never/Occasionally/Constantly (no Frequently)
    'never': 0, 'occasionally': 1, 'constantly': 2,
}
FREQ_NEVER_FREQ_3 = {  # Never/Occasionally/Frequently
    'never': 0, 'occasionally': 1, 'frequently': 2,
}
FREQ_NEVER_SOMETIMES_3 = {  # Never/Sometimes/A lot of the time
    'never': 0, 'sometimes': 1, 'a lot of the time': 2,
}
FREQ_NEVER_SOMETIMES_2 = {  # Never/Sometimes
    'never': 0, 'sometimes': 1,
}
SWEATING_5 = {
    "i sweat much less than i used to": -2,
    "i sweat somewhat less than i used to": -1,
    "i haven't noticed any changes in my sweating": 0,
    "i sweat somewhat more than i used to": 1,
    "i sweat much more than i used to": 2,
}
SATIETY_5 = {
    "i get full a lot less quickly now than i used to": -2,
    "i get full less quickly now than i used to": -1,
    "i haven't noticed any change": 0,
    "i get full more quickly now than i used to": 1,
    "i get full a lot more quickly now than i used to": 2,
}
PSQI_PROBLEM_4 = {
    'no problem at all': 0,
    'only a very slight problem': 1,
    'somewhat of a problem': 2,
    'a very big problem': 3,
}
PSQI_QUALITY_4 = {
    'very good': 0, 'fairly good': 1, 'fairly bad': 2, 'very bad': 3,
}
NEVER_TRUE_3 = {'never true': 0, 'sometimes true': 1, 'often true': 2}
NOT_AT_ALL_5 = {
    'not at all': 0, 'a little bit': 1, 'moderately': 2,
    'quite a bit': 3, 'extremely': 4,
}
STRONG_DIS_4 = {
    'strongly disagree': 1, 'disagree': 2, 'agree': 3, 'strongly agree': 4,
}
# Note: items 4 and 6 of Pearlin Mastery are positively worded and will be
# reverse-scored at the user's discretion; here we preserve the raw 1-4 response.

HANDEDNESS_5 = {
    'always right': 2, 'usually right': 1, 'both equally': 0,
    'usually left': -1, 'always left': -2,
}
EDUCATION_5 = {
    'primary school': 1, 'high school': 2, 'college/university': 3,
    'graduate school': 4, 'post-graduate degree': 5,
}
INCOME_7 = {
    'less than $10,000': 1,
    '$10,000 to $24,999': 2,
    '$25,000 to $49,999': 3,
    '$50,000 to $74,999': 4,
    '$75,000 to $100,000': 5,
    '$100,000 to $150,000': 6,
    'more than $150,000': 7,
    # 'Prefer not to answer' -> NaN via MISSING_TOKENS
}
DAYS_PER_WEEK_IPAQ = {
    'none': 0,
    '1 day per week': 1, '2 days per week': 2, '3 days per week': 3,
    '4 days per week': 4, '5 days per week': 5, '6 days per week': 6,
    '7 days per week': 7,
}
DIET_FREQ_9 = {
    'never': 0,
    '1 time last month': 1,
    '2-3 times last month': 2,
    '1 time per week': 3,
    '2 times per week': 4,
    '3-4 times per week': 5,
    '5-6 times per week': 6,
    '1 time per day': 7,
    '2-3 times per day': 8,
    '2 or more times per day': 8,
    '4-5 times per day': 9,
    '6 or more times per day': 10,
}
FOODSEC_4 = {
    'enough of the kinds of food we want to eat': 0,
    'enough but not always the kinds of food we want': 1,
    'sometime not enough to eat': 2,
    'often not enough to eat': 3,
}
PERIO_SELFRATE = {  # gum health
    'poor': 1, 'fair': 2, 'good': 3, 'very good': 4, 'excellent': 5,
}
PERIO_FREQ_3 = {
    'never': 0, '1-6 times': 1, 'greater than or equal to 7 times': 2,
}
PARTICIPATION_3 = {
    'not at all': 0,
    'infrequent participant (at least one time over the past 6 months)': 1,
    'frequent participant (at least once per month)': 2,
}
TRUST_PEOPLE = {
    "you almost always can't be too careful in dealing with people": 0,
    "you usually can't be too careful in dealing with people": 1,
    'people can usually be trusted': 2,
    'people can almost always be trusted': 3,
    # "Can't choose" -> NaN
}


# ---------- Column plan ----------

# Given row-1 headers, we list the columns we keep and how to encode each.
# 'action' is one of: numeric | yesno | checked | ordinal:<SCALE_NAME> | onehot | drop

# Indices here are the 0-based column indices in the raw (header=None) sheet,
# so col 0 = Study ID. Row 0 is section header, Row 1 is the question text.

# We use column indices rather than names because the question text in row 1
# is very long and sometimes duplicated across sections.

COL_PLAN = []  # list of (idx, short_name, action)

# -- Study ID
COL_PLAN.append((0, 'Study_ID', 'id'))

# ---- Health & Physical Activity (1..49) ----
# Rand26 totals (1..8) are numeric 0-100
rand26 = [
    (1, 'Rand26_Pain'),
    (2, 'Rand26_SocialFunctioning'),
    (3, 'Rand26_GeneralHealth'),
    (4, 'Rand26_EmotionalWellBeing'),
    (5, 'Rand26_EnergyFatigue'),
    (6, 'Rand26_RoleLimit_Emotional'),
    (7, 'Rand26_RoleLimit_Physical'),
    (8, 'Rand26_PhysicalFunctioning'),
]
for i, n in rand26:
    COL_PLAN.append((i, n, 'numeric'))

# ABC Scale items (9..24) are 0-100 confidence; col 25 is total, 26 is % self conf
abc_items = {
    9:  'ABC_walk_around_house', 10: 'ABC_walk_stairs',
    11: 'ABC_bend_pickup',       12: 'ABC_reach_shelf_eye',
    13: 'ABC_reach_above_head',  14: 'ABC_stand_on_chair',
    15: 'ABC_sweep_floor',       16: 'ABC_walk_to_car',
    17: 'ABC_in_out_car',        18: 'ABC_walk_parking_lot',
    19: 'ABC_walk_ramp',         20: 'ABC_walk_crowded_mall',
    21: 'ABC_bumped_into',       22: 'ABC_escalator_railing',
    23: 'ABC_escalator_parcels', 24: 'ABC_walk_icy_sidewalks',
    25: 'ABC_Total',
    26: 'ABC_PctSelfConfidence',
}
for i, n in abc_items.items():
    COL_PLAN.append((i, n, 'numeric'))

# IPAQ physical activity (27..48)
ipaq_days = {
    27: 'IPAQ_vigorous_days',
    30: 'IPAQ_moderate_days',
    33: 'IPAQ_walk_days',
}
for i, n in ipaq_days.items():
    COL_PLAN.append((i, n, 'ordinal:DAYS_PER_WEEK_IPAQ'))
# hours/minutes sub-items and sitting times are numeric
ipaq_numeric = {
    28: 'IPAQ_vigorous_hours',   29: 'IPAQ_vigorous_minutes',
    31: 'IPAQ_moderate_hours',   32: 'IPAQ_moderate_minutes',
    34: 'IPAQ_walk_hours',       35: 'IPAQ_walk_minutes',
    36: 'IPAQ_sit_weekday_hours',37: 'IPAQ_sit_weekday_minutes',
    38: 'IPAQ_sit_weekend_hours',39: 'IPAQ_sit_weekend_minutes',
    40: 'IPAQ_extra1',           41: 'IPAQ_extra2',
    42: 'IPAQ_extra3',           43: 'IPAQ_extra4',
    44: 'IPAQ_extra5',           45: 'IPAQ_extra6',
    46: 'IPAQ_extra7',           47: 'IPAQ_extra8',
    48: 'IPAQ_extra9',           49: 'IPAQ_extra10',
}
for i, n in ipaq_numeric.items():
    COL_PLAN.append((i, n, 'numeric'))

# ---- Alcohol Use (50..59) ----
COL_PLAN.append((50, 'Alc_DrankPast30d_YN', 'yesno'))
# 51,54,56,58 are prompts with option labels only (no useful data) -> drop
COL_PLAN.append((52, 'Alc_DaysPerWeek',    'numeric'))
COL_PLAN.append((53, 'Alc_DaysPast30',     'numeric'))
COL_PLAN.append((55, 'Alc_AvgDrinksPerDay','numeric'))
COL_PLAN.append((57, 'Alc_BingeOccasions', 'numeric'))
COL_PLAN.append((59, 'Alc_MaxDrinksOcc',   'numeric'))

# ---- Reading History (60..77) -> DROP entirely per user agreement ----

# ---- COMPASS-31 autonomic (78..114) ----
# Structure is NOT a uniform gate/freq/severity/change repetition. Mapped per
# actual question after inspection of the source file.
compass_items = [
    (78,  'COMPASS_Orthostatic_YN',          'yesno'),
    (79,  'COMPASS_Orthostatic_freq',        'ordinal:LIKERT_FREQ_COMPASS'),
    (80,  'COMPASS_Orthostatic_severity',    'ordinal:LIKERT_SEVERITY_3'),
    (81,  'COMPASS_Orthostatic_change',      'ordinal:CHANGE_6'),
    (82,  'COMPASS_SkinColor_YN',            'yesno'),
    (83,  'COMPASS_SkinColor_Hands',         'checked'),
    (84,  'COMPASS_SkinColor_Feet',          'checked'),
    (85,  'COMPASS_SkinColor_change',        'ordinal:CHANGE_6_ALT'),
    (86,  'COMPASS_Sweating_change',         'ordinal:SWEATING_5'),
    (87,  'COMPASS_DryEyes_YN',              'yesno'),
    (88,  'COMPASS_DryMouth_YN',             'yesno'),
    (89,  'COMPASS_DryEyesMouth_change',     'ordinal:CHANGE_6_ALT'),
    (90,  'COMPASS_EarlySatiety',            'ordinal:SATIETY_5'),
    (91,  'COMPASS_Bloating',                'ordinal:FREQ_NEVER_SOMETIMES_3'),
    (92,  'COMPASS_Vomit',                   'ordinal:FREQ_NEVER_SOMETIMES_2'),
    (93,  'COMPASS_AbdPain',                 'ordinal:FREQ_NEVER_SOMETIMES_3'),
    (94,  'COMPASS_Diarrhea_YN',             'yesno'),
    (95,  'COMPASS_Diarrhea_freq',           'ordinal:FREQ_RARE_CONST_4'),
    (96,  'COMPASS_Diarrhea_timesPerMonth',  'numeric'),
    (97,  'COMPASS_Diarrhea_severity',       'ordinal:LIKERT_SEVERITY_3'),
    (98,  'COMPASS_Diarrhea_change',         'ordinal:CHANGE_6_ALT'),
    (99,  'COMPASS_Constipation_YN',         'yesno'),
    (100, 'COMPASS_Constipation_freq',       'ordinal:FREQ_RARE_CONST_4'),
    # col 101 is "times per month" but has garbled data (e.g. "2-Jan") -> drop
    (102, 'COMPASS_Constipation_severity',   'ordinal:LIKERT_SEVERITY_3'),
    (103, 'COMPASS_Constipation_change',     'ordinal:CHANGE_6_ALT'),
    (104, 'COMPASS_Bladder_loseControl',     'ordinal:FREQ_NEVER_4'),
    # col 105: times per month (Excel-mangled date strings) -> drop
    (106, 'COMPASS_Bladder_diffPassUrine',   'ordinal:FREQ_NEVER_3'),
    # col 107: empty -> drop
    (108, 'COMPASS_Bladder_incompleteEmpty', 'ordinal:FREQ_NEVER_4'),
    # col 109: times per month (garbled) -> drop
    (110, 'COMPASS_Eye_BrightLight',         'ordinal:FREQ_NEVER_4'),
    (111, 'COMPASS_Eye_BrightLight_severity','ordinal:LIKERT_SEVERITY_3'),
    (112, 'COMPASS_Eye_Focus',               'ordinal:FREQ_NEVER_FREQ_3'),
    (113, 'COMPASS_Eye_Focus_severity',      'ordinal:LIKERT_SEVERITY_3'),
    (114, 'COMPASS_Eye_change',              'ordinal:CHANGE_6_ALT'),
]
for i, n, a in compass_items:
    COL_PLAN.append((i, n, a))

# ---- Dietary Screening (115..143) ----
# 115, 118, 121..143 are food-frequency; 116,117,119,120 are free text
diet_freq_cols = {
    115: 'Diet_Cereals',         118: 'Diet_Milk',
    121: 'Diet_Soda',            122: 'Diet_FruitJuice',
    123: 'Diet_CoffeeTeaSugar',  124: 'Diet_SweetDrinks',
    125: 'Diet_Fruit',           126: 'Diet_GreenSalad',
    127: 'Diet_FriedPotato',     128: 'Diet_OtherPotato',
    129: 'Diet_Beans',           130: 'Diet_WholeGrains',
    131: 'Diet_OtherVeg',        132: 'Diet_Salsa',
    133: 'Diet_Pizza',           134: 'Diet_TomatoSauce',
    135: 'Diet_Cheese',          136: 'Diet_RedMeat',
    137: 'Diet_ProcessedMeat',   138: 'Diet_WholeGrainBread',
    139: 'Diet_Candy',           140: 'Diet_SweetRolls',
    141: 'Diet_CookiesCake',     142: 'Diet_IceCream',
    143: 'Diet_Popcorn',
}
for i, n in diet_freq_cols.items():
    COL_PLAN.append((i, n, 'ordinal:DIET_FREQ_9'))
# col 119 is multi-choice kind-of-milk -> one-hot
COL_PLAN.append((119, 'Diet_MilkKind', 'onehot'))

# ---- Food Security (144..157) ----
COL_PLAN.append((144, 'FoodSec_Statement',     'ordinal:FOODSEC_4'))
COL_PLAN.append((145, 'FoodSec_WorryRunOut',   'ordinal:NEVER_TRUE_3'))
COL_PLAN.append((146, 'FoodSec_DidntLast',     'ordinal:NEVER_TRUE_3'))
COL_PLAN.append((147, 'FoodSec_CouldntAfford', 'ordinal:NEVER_TRUE_3'))
COL_PLAN.append((148, 'FoodSec_CutMeals_YN',   'yesno'))
# 149 is a frequency follow-up (ordinal)
COL_PLAN.append((150, 'FoodSec_AteLess_YN',    'yesno'))
COL_PLAN.append((151, 'FoodSec_Hungry_YN',     'yesno'))
COL_PLAN.append((152, 'FoodSec_LostWeight_YN', 'yesno'))
COL_PLAN.append((153, 'FoodSec_NotEatWholeDay_YN', 'yesno'))
# 154 similar follow-up
COL_PLAN.append((155, 'FoodSec_Kids_FewLowCost','ordinal:NEVER_TRUE_3'))
COL_PLAN.append((156, 'FoodSec_Kids_NoBalancedMeal','ordinal:NEVER_TRUE_3'))
COL_PLAN.append((157, 'FoodSec_Kids_NotEnough',  'ordinal:NEVER_TRUE_3'))

# ---- Lifetime Discrimination (158..171) ----
# IMPORTANT: in the source file these cols duplicate the Food Security questions.
# Treated as a labelling error -> DROP to avoid double-counting.

# ---- Participant Health History (172..368) ----
# Handedness (Writing .2, Throwing, Toothbrush, Spoon) — col indices 172..175
COL_PLAN.append((172, 'Hand_Writing', 'ordinal:HANDEDNESS_5'))
COL_PLAN.append((173, 'Hand_Throwing', 'ordinal:HANDEDNESS_5'))
COL_PLAN.append((174, 'Hand_Toothbrush', 'ordinal:HANDEDNESS_5'))
COL_PLAN.append((175, 'Hand_Spoon', 'ordinal:HANDEDNESS_5'))
# Demographics
COL_PLAN.append((176, 'Education',          'ordinal:EDUCATION_5'))
COL_PLAN.append((177, 'Income_Personal',    'ordinal:INCOME_7'))
COL_PLAN.append((178, 'Income_Household',   'ordinal:INCOME_7'))
COL_PLAN.append((179, 'Adults_in_Household','numeric'))
COL_PLAN.append((180, 'Children_in_Household','numeric'))
COL_PLAN.append((181, 'Employment_Status',  'onehot'))
# 182 specify-other employment -> drop (free text)
# 183 job title -> drop (free text)
# 184 industry -> drop (free text)
# Years-at-job fields are mostly free-text ("35 years", "Retired 6 yrs",
# "<1", "9 months") with many unparseable variants -> drop per free-text rule.
# col 185 = years current job, 188 = years longest, 191 = years 2nd longest

# Health-condition checkboxes (192..368): each question has a pair (Myself, Family).
# Based on the earlier inspection the values are Checked / Unchecked.
# We'll label them with a short slug extracted from the row-1 text.
# These are added dynamically after reading the file (needs access to row 1).


def build_condition_plan(row1):
    """Create (idx, short_name, 'checked') entries for health-condition columns
    in range 192..368, detecting Myself vs Family suffix."""
    entries = []
    for i in range(192, 369):
        txt = row1[i]
        if pd.isna(txt):
            continue
        s = str(txt)
        # only accept entries that follow the choice=<...>Myself|Family pattern
        m = re.match(r'^(.*?)\s*\(choice=<div[^>]*>\s*(Myself|Family Member[^<]*)\s*</div>\)\s*$', s)
        if not m:
            # some columns in this range are non-condition (e.g. age-of-onset,
            # specifics) - treat them generically
            continue
        cond_raw = m.group(1).strip()
        who = 'self' if m.group(2).strip().startswith('Myself') else 'family'
        # slugify
        slug = re.sub(r'[^A-Za-z0-9]+', '_', cond_raw).strip('_')[:40]
        name = f'HealthHist__{slug}__{who}'
        entries.append((i, name, 'checked'))
    return entries


# Age-of-onset fields (numeric) and misc fields inside HealthHist we can keep.
# Scan row1 for "age of onset" / "age when" and keep them numeric.

def build_onset_plan(row1):
    entries = []
    for i in range(192, 369):
        txt = row1[i]
        if pd.isna(txt):
            continue
        s = str(txt).lower()
        if re.match(r'^(.*?)\s*\(choice=<', str(txt)):
            continue  # already handled
        if any(k in s for k in ['age of onset', 'age when',
                                'how old were you',
                                'hearing aid', 'hearing loss',
                                'current medications', 'prescription medications']):
            pass  # skip medication / hearing-aid free text for now
        # keep 'Height', 'Weight', 'BMI', 'heart rate', 'blood pressure' if present
        if any(k in s for k in ['height', 'weight', 'bmi',
                                'waist', 'hip']):
            slug = re.sub(r'[^A-Za-z0-9]+', '_', str(txt))[:40].strip('_')
            entries.append((i, f'HealthHist__{slug}', 'numeric'))
    return entries


# ---- Pearlin Mastery (369..377) ----
COL_PLAN.append((369, 'Pearlin_Total', 'numeric'))
pearlin_items = {
    370: 'Pearlin_Q1_NoWay',
    371: 'Pearlin_Q2_PushedAround',
    372: 'Pearlin_Q3_LittleControl',
    373: 'Pearlin_Q4_CanDoAnything',  # positive (reverse later if desired)
    374: 'Pearlin_Q5_Helpless',
    375: 'Pearlin_Q6_FutureOnMe',     # positive
    376: 'Pearlin_Q7_LittleICanDo',
}
for i, n in pearlin_items.items():
    COL_PLAN.append((i, n, 'ordinal:STRONG_DIS_4'))
COL_PLAN.append((377, 'Social_TrustPeople', 'ordinal:TRUST_PEOPLE'))

# ---- Periodontal (378..385) ----
COL_PLAN.append((378, 'Perio_HaveGumDisease_YN', 'yesno'))
COL_PLAN.append((379, 'Perio_BoneLossTold_YN',  'yesno'))
COL_PLAN.append((380, 'Perio_GumTreatmentEver_YN','yesno'))
COL_PLAN.append((381, 'Perio_LooseTeeth_YN',    'yesno'))
COL_PLAN.append((382, 'Perio_MouthwashFreq',   'ordinal:PERIO_FREQ_3'))
COL_PLAN.append((383, 'Perio_FlossFreq',       'ordinal:PERIO_FREQ_3'))
COL_PLAN.append((384, 'Perio_GumHealthRating', 'ordinal:PERIO_SELFRATE'))
COL_PLAN.append((385, 'Perio_ToothLookWrong_YN','yesno'))

# ---- Pittsburgh Sleep Quality (386..412) ----
COL_PLAN.append((386, 'PSQI_TotalScore',      'numeric'))
# 387 bedtime clock-time -> drop (text like '23:30' needs parsing)
COL_PLAN.append((388, 'PSQI_Q2_SleepLatencyMin','numeric'))
# 389 wake time -> drop
COL_PLAN.append((390, 'PSQI_Q4_SleepHours',     'numeric'))
psqi_freq_items = {
    391: 'PSQI_Q5a_cannotSleep30',
    392: 'PSQI_Q5b_wakeMiddleNight',
    393: 'PSQI_Q5c_bathroom',
    394: 'PSQI_Q5d_cannotBreathe',
    395: 'PSQI_Q5e_coughSnore',
    396: 'PSQI_Q5f_tooCold',
    397: 'PSQI_Q5g_tooHot',
    398: 'PSQI_Q5h_badDreams',
    399: 'PSQI_Q5i_pain',
    # 400 Q5j describe text -> drop
    401: 'PSQI_Q5j_otherFreq',
    402: 'PSQI_Q6_medToSleep',
    403: 'PSQI_Q7_troubleStayingAwake',
    # 404 Q8 uses the PROBLEM_4 scale (below)
    # 405 Q9 uses the QUALITY_4 scale (below)
    # 406 Q10 bed-partner categorical -> onehot (below)
    407: 'PSQI_Q10a_snoreLoud',
    408: 'PSQI_Q10b_pauseBreathing',
    409: 'PSQI_Q10c_legsTwitch',
    410: 'PSQI_Q10d_disorientation',
    # 411 Q10e describe -> drop
    412: 'PSQI_Q10e_otherFreq',
}
for i, n in psqi_freq_items.items():
    COL_PLAN.append((i, n, 'ordinal:LIKERT_FREQ_4'))
COL_PLAN.append((404, 'PSQI_Q8_EnthusiasmProblem', 'ordinal:PSQI_PROBLEM_4'))
COL_PLAN.append((405, 'PSQI_Q9_QualityRating',     'ordinal:PSQI_QUALITY_4'))
COL_PLAN.append((406, 'PSQI_Q10_BedPartner',       'onehot'))

# ---- PTSD Checklist DSM-5 (413..436) ----
COL_PLAN.append((413, 'PTSD_EventDeathInjury_YN', 'yesno'))
COL_PLAN.append((414, 'PTSD_HowExperienced',     'onehot'))
# 415 describe text -> drop
# 416 natural vs accident - three categories -> ordinal (0=N/A,1=natural,2=accident)
#     but safer as onehot
COL_PLAN.append((416, 'PTSD_DeathCause',         'onehot'))
ptsd_items = {
    417: 'PTSD_Q1',  418: 'PTSD_Q2',  419: 'PTSD_Q3',  420: 'PTSD_Q4',
    421: 'PTSD_Q5', 422: 'PTSD_Q6',  423: 'PTSD_Q7',  424: 'PTSD_Q8',
    425: 'PTSD_Q9', 426: 'PTSD_Q10', 427: 'PTSD_Q11', 428: 'PTSD_Q12',
    429: 'PTSD_Q13',430: 'PTSD_Q14', 431: 'PTSD_Q15', 432: 'PTSD_Q16',
    433: 'PTSD_Q17',434: 'PTSD_Q18', 435: 'PTSD_Q19', 436: 'PTSD_Q20',
}
for i, n in ptsd_items.items():
    COL_PLAN.append((i, n, 'ordinal:NOT_AT_ALL_5'))

# ---- Social Relationship (437..458) ----
# 437,438 initials free text -> drop
# 439..450 per-person demographics (Person1/2/3 initials,gender,relationship,edu)
# Drop names/initials, keep count-based summary is not trivial; drop all for now
# 451..453 "Does Person X know Person Y" -> yes/no (network density markers) keep
COL_PLAN.append((451, 'Social_P1knowsP2', 'yesno'))
COL_PLAN.append((452, 'Social_P1knowsP3', 'yesno'))
COL_PLAN.append((453, 'Social_P2knowsP3', 'yesno'))
# 454..458 group participation
participation_cols = {
    454: 'Social_Hobby',
    455: 'Social_Sports',
    456: 'Social_Neighborhood',
    457: 'Social_Church',
    458: 'Social_ArtsMusic',
}
for i, n in participation_cols.items():
    COL_PLAN.append((i, n, 'ordinal:PARTICIPATION_3'))

# ---- COVID-19 (459..518) ----
COL_PLAN.append((459, 'COVID_EverTestedPositive_YN', 'yesno'))
# 460 lab upload url -> drop
covid_symptoms = {
    461: 'COVID_Sym_LossTasteSmell',
    462: 'COVID_Sym_SevereHeadache',
    463: 'COVID_Sym_Fever',
    464: 'COVID_Sym_TroubleBreathing',
    465: 'COVID_Sym_Respiratory',
    466: 'COVID_Sym_ExtremeFatigue',
    467: 'COVID_Sym_BodyAches',
    468: 'COVID_Sym_BrainFog',
}
for i, n in covid_symptoms.items():
    COL_PLAN.append((i, n, 'checked'))
# 469 diagnosis text -> drop
COL_PLAN.append((470, 'COVID_ThoughtHadIt_YN', 'yesno'))
# 471 onward: treatment details -> drop

# -------------------- main --------------------

def main():
    df = pd.read_excel(SRC, sheet_name='Sheet1', header=None)
    row1 = df.iloc[1]
    data = df.iloc[2:].reset_index(drop=True)

    # Build full plan
    plan = list(COL_PLAN)
    plan.extend(build_condition_plan(row1))
    plan.extend(build_onset_plan(row1))

    # Deduplicate on index keeping last (onset plan may overlap with condition plan)
    seen = {}
    for entry in plan:
        seen[entry[0]] = entry
    plan = list(seen.values())
    plan.sort(key=lambda e: e[0])

    # --- Raw_Selected sheet ---
    raw_cols = []
    raw_names = []
    for idx, name, action in plan:
        raw_names.append(name)
        raw_cols.append(data.iloc[:, idx])
    raw_df = pd.concat(raw_cols, axis=1)
    raw_df.columns = raw_names

    # --- Quantitative sheet ---
    scales = {
        'LIKERT_FREQ_4': LIKERT_FREQ_4,
        'LIKERT_SEVERITY_3': LIKERT_SEVERITY_3,
        'LIKERT_FREQ_COMPASS': LIKERT_FREQ_COMPASS,
        'CHANGE_6': CHANGE_6,
        'CHANGE_6_ALT': CHANGE_6_ALT,
        'NEVER_TRUE_3': NEVER_TRUE_3,
        'NOT_AT_ALL_5': NOT_AT_ALL_5,
        'STRONG_DIS_4': STRONG_DIS_4,
        'HANDEDNESS_5': HANDEDNESS_5,
        'EDUCATION_5': EDUCATION_5,
        'INCOME_7': INCOME_7,
        'DAYS_PER_WEEK_IPAQ': DAYS_PER_WEEK_IPAQ,
        'DIET_FREQ_9': DIET_FREQ_9,
        'FOODSEC_4': FOODSEC_4,
        'PERIO_SELFRATE': PERIO_SELFRATE,
        'PERIO_FREQ_3': PERIO_FREQ_3,
        'PARTICIPATION_3': PARTICIPATION_3,
        'TRUST_PEOPLE': TRUST_PEOPLE,
        'FREQ_RARE_CONST_4': FREQ_RARE_CONST_4,
        'FREQ_NEVER_4': FREQ_NEVER_4,
        'FREQ_NEVER_3': FREQ_NEVER_3,
        'FREQ_NEVER_FREQ_3': FREQ_NEVER_FREQ_3,
        'FREQ_NEVER_SOMETIMES_3': FREQ_NEVER_SOMETIMES_3,
        'FREQ_NEVER_SOMETIMES_2': FREQ_NEVER_SOMETIMES_2,
        'SWEATING_5': SWEATING_5,
        'SATIETY_5': SATIETY_5,
        'PSQI_PROBLEM_4': PSQI_PROBLEM_4,
        'PSQI_QUALITY_4': PSQI_QUALITY_4,
    }

    quant_parts = []
    quant_parts.append(data.iloc[:, 0].rename('Study_ID'))  # ID kept as string

    for idx, name, action in plan:
        if action == 'id':
            continue
        s = data.iloc[:, idx]
        if action == 'numeric':
            quant_parts.append(enc_numeric(s).rename(name))
        elif action == 'yesno':
            quant_parts.append(enc_yesno(s).rename(name))
        elif action == 'checked':
            quant_parts.append(enc_checked(s).rename(name))
        elif action.startswith('ordinal:'):
            scale_name = action.split(':', 1)[1]
            scale = scales[scale_name]
            f = enc_ordinal(scale)
            quant_parts.append(s.map(f).rename(name))
        elif action == 'onehot':
            oh = enc_onehot(s, name)
            quant_parts.append(oh)
        else:  # drop
            continue

    quant_df = pd.concat(quant_parts, axis=1)

    # --- Derived summary scores: PCL-5 (PTSD Checklist for DSM-5) ---
    # Each item is 0 (Not at all) .. 4 (Extremely).
    # Cluster mapping per DSM-5 / PCL-5 scoring manual:
    #   B Intrusion         -> Q1..Q5   (range 0..20)
    #   C Avoidance         -> Q6..Q7   (range 0..8)
    #   D Negative cog/mood -> Q8..Q14  (range 0..28)
    #   E Arousal/reactivity-> Q15..Q20 (range 0..24)
    # Total = sum of all 20 items (range 0..80).
    # Using skipna=False so a missing item yields NaN for the sum (no partial
    # scoring), matching standard PCL-5 practice.
    def _pcl(cols):
        return quant_df[cols].sum(axis=1, skipna=False)

    b_items = [f'PTSD_Q{i}' for i in range(1, 6)]
    c_items = [f'PTSD_Q{i}' for i in range(6, 8)]
    d_items = [f'PTSD_Q{i}' for i in range(8, 15)]
    e_items = [f'PTSD_Q{i}' for i in range(15, 21)]
    all_items = b_items + c_items + d_items + e_items

    quant_df['PCL5_B_Intrusion']  = _pcl(b_items)
    quant_df['PCL5_C_Avoidance']  = _pcl(c_items)
    quant_df['PCL5_D_NegCogMood'] = _pcl(d_items)
    quant_df['PCL5_E_Arousal']    = _pcl(e_items)
    quant_df['PCL5_Total']        = _pcl(all_items)

    # --- Encoding_Rules sheet: one row per source column kept ---
    section_ranges = [
        (0, 0, 'StudyID'),
        (1, 49, 'HealthActivity'),
        (50, 59, 'Alcohol'),
        (60, 77, 'Reading (DROPPED)'),
        (78, 114, 'COMPASS31'),
        (115, 143, 'Diet'),
        (144, 157, 'FoodSecurity'),
        (158, 171, 'LifetimeDiscrim_DUP (DROPPED)'),
        (172, 368, 'HealthHistory'),
        (369, 377, 'PearlinMastery'),
        (378, 385, 'Periodontal'),
        (386, 412, 'PSQI'),
        (413, 436, 'PTSD'),
        (437, 458, 'SocialRelationship'),
        (459, 518, 'COVID19'),
    ]

    def section_of(idx):
        for a, b, name in section_ranges:
            if a <= idx <= b:
                return name
        return '?'

    rules_rows = []
    for idx, name, action in plan:
        q_text = str(row1.iloc[idx]) if pd.notna(row1.iloc[idx]) else ''
        q_text = re.sub(r'<[^>]+>', '', q_text)   # strip HTML
        q_text = re.sub(r'\s+', ' ', q_text).strip()
        if action.startswith('ordinal:'):
            act, scale = 'ordinal', action.split(':', 1)[1]
        elif action == 'onehot':
            act, scale = 'one-hot', ''
        else:
            act, scale = action, ''
        rules_rows.append({
            'Source_Col_Idx': idx,
            'Section': section_of(idx),
            'Quant_Column_Name': name,
            'Action': act,
            'Scale_Name': scale,
            'Original_Question': q_text[:250],
        })
    rules_df = pd.DataFrame(rules_rows).sort_values('Source_Col_Idx').reset_index(drop=True)

    # --- Scales sheet: expanded mapping table ---
    scales_rows = []
    for scale_name, mapping in scales.items():
        for raw_v, enc_v in mapping.items():
            scales_rows.append({
                'Scale_Name': scale_name,
                'Raw_Value': raw_v,
                'Encoded_Value': enc_v,
            })
    # Add the implicit rules too
    for raw_v, enc_v in [('Yes', 1), ('No', 0),
                         ('Checked', 1), ('Unchecked', 0)]:
        scales_rows.append({'Scale_Name': 'yesno / checked',
                            'Raw_Value': raw_v, 'Encoded_Value': enc_v})
    for raw_v in sorted(MISSING_TOKENS):
        scales_rows.append({'Scale_Name': '__missing_tokens__',
                            'Raw_Value': raw_v, 'Encoded_Value': 'NaN'})
    scales_df = pd.DataFrame(scales_rows)

    # --- ReadMe sheet: plain-English summary ---
    readme_lines = [
        'BHI_Questionnaire_Extracted.xlsx — source and encoding reference',
        '',
        'Source: Doc/ABC_ComprehensiveQuestionnaire.xlsx (Sheet1, header spans row 0-1).',
        'Generated by BHI_extract_questionnaire.py.',
        '',
        'Sheets:',
        '  Raw_Selected    — original text/number values for selected columns',
        '  Quantitative    — numerically encoded for analysis (ID plus features)',
        '  Encoding_Rules  — row-per-column: action + scale name + original question',
        '  Scales          — every ordinal scale: raw value -> encoded value',
        '  ReadMe          — this sheet',
        '',
        'Sections kept / dropped:',
        '  KEPT: HealthActivity, Alcohol, COMPASS31, Diet, FoodSecurity,',
        '        HealthHistory, PearlinMastery, Periodontal, PSQI, PTSD,',
        '        SocialRelationship, COVID19 (symptom flags + positive-test only)',
        '  DROPPED: Reading History (childhood-only, not BAG-relevant)',
        '  DROPPED: Cols 158-171 labelled "Lifetime discrimination" — they are',
        '           an exact duplicate of FoodSec cols 144-157 (source-file bug).',
        '',
        'Encoding conventions:',
        '  Yes / No                  -> 1 / 0   (blank -> NaN)',
        '  Checked / Unchecked       -> 1 / 0',
        '  Ordinal Likert scales     -> integer (see Scales sheet)',
        '  Multi-choice categorical  -> one-hot columns "<base>__<slug>"',
        '  Continuous numeric        -> float (strip % and whitespace)',
        '  Free text                 -> dropped (job titles, industries,',
        '                               symptom specifications, contact initials,',
        '                               clock times, years-at-job free-form)',
        '',
        "  Missing tokens mapped to NaN: Don't Know, Don't Know/Refused,",
        "  Refused, Prefer not to answer, Not applicable, Can't choose,",
        '  and REDCap "Choose this option to enter ..." placeholders.',
        '',
        'Notable scale choices:',
        '  HANDEDNESS_5           -2 (always left) .. +2 (always right)',
        '  EDUCATION_5            1 (primary) .. 5 (post-graduate degree)',
        '  INCOME_7               1 (<$10k) .. 7 (>$150k); prefer-not-to-answer -> NaN',
        '  NOT_AT_ALL_5  (PCL-5)  0 (not at all) .. 4 (extremely)',
        '  STRONG_DIS_4 (Pearlin) 1 (strongly disagree) .. 4 (strongly agree)',
        '                         Items Pearlin Q4 and Q6 are POSITIVELY worded —',
        '                         reverse-score before summing if desired.',
        '  LIKERT_FREQ_4 (PSQI)   0 (not during past month) .. 3 (3+/week)',
        '  PSQI_QUALITY_4         0 (very good) .. 3 (very bad)',
        '  PSQI_PROBLEM_4         0 (no problem) .. 3 (a very big problem)',
        '  CHANGE_6               0 (completely gone) .. 5 (gotten much worse)',
        '  CHANGE_6_ALT           0 (never had the symptom) .. 6 (much worse)',
        '                         unified 0-6 scale on user request so negatives',
        '                         are avoided in downstream regression.',
        '',
        'Data quirks to be aware of:',
        '  * 345 rows here vs 304 in BHI_Regional_BrainAgeGap.xlsx — inner-join',
        '    on Study_ID before running BAG-association analyses.',
        '  * A few COMPASS "times per month" cells contain Excel-auto-dated',
        '    strings like "4-Mar" (originally "3-4") — left as NaN.',
        '  * COMPASS columns 101, 105, 107, 109 (times-per-month follow-ups)',
        '    were dropped entirely because the raw data is garbled / nearly empty.',
    ]
    readme_df = pd.DataFrame({'Notes': readme_lines})

    # --- DataAvailability sheet ---
    #
    # For every column in the Quantitative sheet:
    #   - N_valid:          subjects with a non-null value (out of 345)
    #   - N_valid_with_BAG: subjects that are also present in Results/BHI_Regional_BrainAgeGap.xlsx
    #                       (intersection with the BAG analysis sample, N=304)
    #   - Pct_of_BAG_N:     100 * N_valid_with_BAG / (size of BAG sample)
    #   - Priority:         High / Medium / Low / Reference
    #   - Reason:           why the variable is plausibly (ir)relevant to BAG
    #
    # Priority tiers are chosen based on published brain-aging literature and
    # are a prioritisation guide, not a hard filter. Re-inspect for your
    # sample if any of the low-N items are central to a specific hypothesis.

    bag_path = 'Results/BHI_Regional_BrainAgeGap.xlsx'
    try:
        bag_df = pd.read_excel(bag_path)
        bag_ids = set(bag_df['subject_ID'].dropna().astype(str))
    except Exception:
        bag_df = None
        bag_ids = set()
    bag_n = max(len(bag_ids), 1)

    # Map a quant-sheet column name to a section tag (based on prefix).
    # Used only inside this sheet; independent of the Encoding_Rules sheet
    # which keys off source column index.
    def section_of_name(col):
        if col == 'Study_ID':
            return 'ID'
        for prefix, sec in [
            ('Rand26_', 'HealthActivity'),
            ('ABC_', 'HealthActivity'),
            ('IPAQ_', 'HealthActivity'),
            ('Alc_', 'Alcohol'),
            ('COMPASS_', 'COMPASS31'),
            ('Diet_', 'Diet'),
            ('FoodSec_', 'FoodSecurity'),
            ('Hand_', 'HealthHistory'),
            ('Employment_Status', 'HealthHistory'),
            ('HealthHist__', 'HealthHistory'),
            ('Pearlin_', 'PearlinMastery'),
            ('Perio_', 'Periodontal'),
            ('PSQI_', 'PSQI'),
            ('PTSD_', 'PTSD'),
            ('PCL5_', 'PTSD'),
            ('Social_', 'SocialRelationship'),
            ('COVID_', 'COVID19'),
        ]:
            if col.startswith(prefix):
                return sec
        if col in {'Education', 'Income_Personal', 'Income_Household',
                   'Adults_in_Household', 'Children_in_Household'}:
            return 'HealthHistory'
        return 'Other'

    HIGH_EXACT = {
        'Education', 'Income_Personal', 'Income_Household',
        'Rand26_GeneralHealth', 'Rand26_PhysicalFunctioning',
        'PSQI_TotalScore', 'PSQI_Q4_SleepHours', 'PSQI_Q9_QualityRating',
        'COVID_Sym_BrainFog', 'ABC_Total', 'ABC_PctSelfConfidence',
        'Alc_DrankPast30d_YN', 'Alc_DaysPerWeek', 'Alc_DaysPast30',
        'Alc_AvgDrinksPerDay', 'Alc_BingeOccasions', 'Alc_MaxDrinksOcc',
        'IPAQ_vigorous_days', 'IPAQ_moderate_days', 'IPAQ_walk_days',
        'IPAQ_sit_weekday_hours', 'IPAQ_sit_weekend_hours',
        'PCL5_Total', 'PCL5_B_Intrusion', 'PCL5_C_Avoidance',
        'PCL5_D_NegCogMood', 'PCL5_E_Arousal',
    }
    MEDIUM_EXACT = {
        'Rand26_Pain', 'Rand26_SocialFunctioning', 'Rand26_EmotionalWellBeing',
        'Rand26_EnergyFatigue', 'Rand26_RoleLimit_Emotional',
        'Rand26_RoleLimit_Physical',
        'Hand_Writing', 'Hand_Throwing', 'Hand_Toothbrush', 'Hand_Spoon',
        'Adults_in_Household', 'Children_in_Household',
        'Pearlin_Total', 'Social_TrustPeople',
        'COVID_EverTestedPositive_YN', 'COVID_ThoughtHadIt_YN',
        'PTSD_EventDeathInjury_YN',
        'PSQI_Q2_SleepLatencyMin', 'PSQI_Q5j_otherFreq', 'PSQI_Q10e_otherFreq',
    }
    HIGH_HEALTHHIST_SELF = {
        'High_blood_pressure', 'High_Cholesterol', 'Hyperlipidemia',
        'Diabetes_mellitus_Type_2_diabetes', 'Gestational_diabetes_diabetes_during',
        'Myocardial_infarction', 'Stroke', 'Transient_ischemic_attack_TIA_or_mini',
        'Congestive_Heart_Failure', 'Peripheral_vascular_disease',
        'Blocked_blood_vessels', 'Irregular_heart_rhythm',
        'Depression', 'Dementia_Chronic_cognitive_deficit',
        'Epilepsy_or_seizures',
    }

    def classify(col):
        if col == 'Study_ID':
            return 'Reference', 'Subject identifier (join key)'
        if col in HIGH_EXACT:
            return 'High', 'Direct brain-aging predictor in published literature'
        if col in MEDIUM_EXACT:
            return 'Medium', 'Plausible contributor'
        # Rand26 already handled
        # ABC items per-item (except Total, PctSelfConfidence already High)
        if col.startswith('ABC_') and col not in HIGH_EXACT:
            return 'Low', 'Per-item balance confidence; subsumed by ABC_Total'
        # IPAQ minutes/hours - subsumed by days; but sit_*_hours is high
        if col.startswith('IPAQ_') and col not in HIGH_EXACT:
            return 'Low', 'Fine-grained IPAQ sub-item; subsumed by days/sit-hours'
        # Alcohol already handled all
        # Employment onehot
        if col.startswith('Employment_Status__'):
            # retired is most relevant; other categories capture SES/activity
            if 'Retired' in col or 'Working_full_time' in col:
                return 'Medium', 'Retirement/employment status (activity, SES)'
            return 'Low', 'Rare employment category'
        # Handedness already Medium
        # Diet items
        if col.startswith('Diet_'):
            cardiometabolic = {'Diet_RedMeat', 'Diet_ProcessedMeat',
                               'Diet_FriedPotato', 'Diet_Soda',
                               'Diet_SweetDrinks', 'Diet_WholeGrains',
                               'Diet_WholeGrainBread', 'Diet_Fruit',
                               'Diet_GreenSalad', 'Diet_OtherVeg'}
            if col in cardiometabolic:
                return 'Medium', 'Cardiometabolic-relevant food-frequency item'
            return 'Low', 'Specific food-frequency item (noisy, weak BAG signal)'
        # Food security
        if col.startswith('FoodSec_'):
            return 'Low', 'Food insecurity (largely captured by Income_*); keep a summary if relevant'
        # Health History
        if col.startswith('HealthHist__'):
            # parse suffix
            if col.endswith('__family'):
                return 'Low', 'Family history — indirect (genetic) BAG predictor'
            if col.endswith('__self'):
                stem = col[len('HealthHist__'):-len('__self')]
                if stem in HIGH_HEALTHHIST_SELF:
                    return 'High', 'Vascular / metabolic / neuropsych condition (self) — direct BAG predictor'
                # Other self conditions: medium
                low_cond = {'Alcoholism', 'Allergies_hay_fever', 'Asthma',
                            'Connective_tissue_disease', 'Peptic_ulcer_disease',
                            'GERD_Reflux', 'Heart_murmur', 'Hepatitis',
                            'Glaucoma', 'AIDS_Autoimmune_Deficiency_Syndrome',
                            'HIV_Human_Immunodeficiency_Syndrome',
                            'Colitis', 'Kidney_stone', 'Kidney_infections',
                            'Other_thyroid_disease', 'Hyperthyroidism',
                            'Hypothyroidism', 'Blood_transfusions',
                            'Cirrhosis_or_liver_disease', 'Liver_disease',
                            'Solid_tumor', 'Cancer', 'Leukemia', 'Lymphoma',
                            'COPD_or_emphysema', 'Hemiplegia',
                            'Moderate_to_severe_chronic_kidney_disease',
                            'Gastrointestinal_disease', 'Anemia'}
                if stem in low_cond:
                    return 'Medium', 'Other self-reported condition (inflammation / metabolic / general health)'
                # handedness items already captured; anything else
                return 'Medium', 'Self-reported health-history item'
            # Height/weight/BMI etc
            if any(k in col for k in ['Height', 'Weight', 'BMI', 'Waist', 'Hip']):
                return 'High', 'Anthropometric; body composition affects brain age'
            return 'Low', 'Misc health-history sub-item'
        # Pearlin items already Medium via Pearlin_Total; individual items:
        if col.startswith('Pearlin_') and col not in MEDIUM_EXACT:
            return 'Medium', 'Individual Pearlin mastery item; Pearlin_Total is the main score'
        # Periodontal
        if col.startswith('Perio_'):
            return 'Medium', 'Periodontal disease (systemic inflammation proxy)'
        # PSQI sub-items
        if col.startswith('PSQI_'):
            return 'Medium', 'PSQI sub-item; PSQI_TotalScore is the main summary'
        # PTSD items
        if col.startswith('PTSD_'):
            if col in MEDIUM_EXACT:
                return 'Medium', 'PTSD screening item'
            if col.startswith('PTSD_Q'):
                return 'Medium', 'PCL-5 item; sum to a PTSD severity score for the main analysis'
            return 'Low', 'PTSD qualitative/descriptor field'
        # Social
        if col.startswith('Social_'):
            if 'knows' in col:
                return 'Low', 'Social network density flag'
            return 'Medium', 'Social / community participation — engagement proxy'
        # COMPASS autonomic
        if col.startswith('COMPASS_'):
            if col.endswith('_YN'):
                return 'Medium', 'COMPASS-31 symptom-gate (autonomic dysfunction)'
            if col.endswith('_timesPerMonth'):
                return 'Low', 'COMPASS follow-up count (sparse / garbled data)'
            return 'Medium', 'COMPASS-31 severity / change item'
        # COVID
        if col.startswith('COVID_Sym_'):
            return 'Medium', 'COVID-19 symptom flag'
        if col.startswith('COVID_'):
            return 'Low', 'COVID-19 auxiliary field'
        # fallback
        return 'Low', 'Unclassified — default low'

    avail_rows = []
    study_ids = quant_df['Study_ID'].astype(str) if 'Study_ID' in quant_df.columns else None
    in_bag = study_ids.isin(bag_ids) if study_ids is not None else None
    for col in quant_df.columns:
        valid = quant_df[col].notna()
        n_valid = int(valid.sum())
        if in_bag is not None:
            n_valid_bag = int((valid & in_bag).sum())
        else:
            n_valid_bag = 0
        priority, reason = classify(col)
        avail_rows.append({
            'Column': col,
            'Section': section_of_name(col),
            'N_valid': n_valid,
            'N_valid_with_BAG': n_valid_bag,
            'Pct_of_BAG_N': round(100 * n_valid_bag / bag_n, 1),
            'Priority': priority,
            'Reason': reason,
        })
    avail_df = pd.DataFrame(avail_rows)

    # --- Write workbook ---
    import os
    os.makedirs('Results', exist_ok=True)
    with pd.ExcelWriter(DST, engine='openpyxl') as w:
        raw_df.to_excel(w,     sheet_name='Raw_Selected',      index=False)
        quant_df.to_excel(w,   sheet_name='Quantitative',      index=False)
        avail_df.to_excel(w,   sheet_name='DataAvailability',  index=False)
        rules_df.to_excel(w,   sheet_name='Encoding_Rules',    index=False)
        scales_df.to_excel(w,  sheet_name='Scales',            index=False)
        readme_df.to_excel(w,  sheet_name='ReadMe',            index=False)

    print(f'Wrote {DST}')
    print(f'  Raw_Selected:     {raw_df.shape[0]} rows x {raw_df.shape[1]} cols')
    print(f'  Quantitative:     {quant_df.shape[0]} rows x {quant_df.shape[1]} cols')
    print(f'  DataAvailability: {avail_df.shape[0]} rows (priority + N counts)')
    print(f'  Encoding_Rules:   {rules_df.shape[0]} rows')
    print(f'  Scales:           {scales_df.shape[0]} rows ({len(scales)} ordinal scales + yesno + missing tokens)')
    tier = avail_df['Priority'].value_counts().to_dict()
    print(f'  Priority tiers:   {tier}')
    print(f'  BAG sample intersect: N={len(bag_ids)} (of 345 questionnaire subjects)')


if __name__ == '__main__':
    main()
