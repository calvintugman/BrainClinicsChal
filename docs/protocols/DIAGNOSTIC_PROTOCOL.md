# DIAGNOSTIC PROTOCOL = AMENDMENT A6 (FROZEN 2026-10-08, before any MDD-vs-healthy model was fit)
EXPLORATORY, bounded, ONE experiment. Not a replication of any published work; not a submission; the 60-recording Dx replication folder is NOT read or used (no labels locally, same rule as earlier stages). Prize context (user's screenshot + rules page, fetch summary): diagnostic balanced accuracy (BA) > 70%, held two months; EEG data alone; best replicated leaderboard entry ~63%. EEG-only inference throughout: no workbook variable is a predictor of the primary model.
Seen before freezing: cohort counts and age/sex descriptives by group only (no model, no EEG-by-label comparison).

## Cohort (fixed)
MDD = workbook formal_status exactly 'MDD' with v2 features: 164 rTMS patients (session-1 row, as in all earlier stages) + 2 pool people = 166. HEALTHY = workbook formal_status 'HEALTHY' with v2 features: 175 pool people (one session each: lowest with EC and EO). 'TINNITUS/MDD' and all other statuses excluded. Exact-duplicate feature vectors dropped. One row per person (no grouping needed).
Same code path for both groups (verified: 14_features_v2.py and 19_pool_features.py both call 03_features.process with clean_v2.clean). Unknown and NOT controllable: recording period/site/clinical-versus-research context differs by group; this is a potential batch confound that no analysis here removes. Pool attrition (~25%) vs patients (~5%) means selection into the healthy group differs; missing-feature fractions are reported per group.
Predictors (primary D1): the 65 E01c columns (v2). Train-fit median imputation; no missingness indicators.

## Models (L2 logistic; train-fit median impute -> StandardScaler; C {0.01,0.1,1,10} by inner 3-fold log loss; outer StratifiedKFold(5, shuffle) seeds 42/43/44; one shared pipeline)
D1 (PRIMARY, EEG only): 65 E01c features. D0 (confound benchmark, NOT EEG-only, never a candidate): true age + sex. No other models, interactions, feature selection or tuning.
Threshold: argmax balanced accuracy on inner out-of-fold probabilities (ties -> threshold closest to 0.5), applied unchanged to the outer test fold. Metrics per repeat on pooled out-of-fold predictions: BA (primary), AUC, sensitivity, specificity, PPV, Brier, TP/FP/TN/FN.

## Inference and confound analyses
a) Paired patient(person) bootstrap, 2000 draws, rng 1, joint across repeats, conditional on saved predictions: BA and AUC intervals for D1; D1-D0 BA difference.
b) One chance check: label permutation of D1 (999 perms, rng 51; folds regenerated from permuted labels; full refit; statistic = mean BA over 3 repeats; plus-one p).
c) Age/sex confound diagnostics (descriptive): D1 out-of-fold score vs age within each class (Spearman, bootstrap interval); D1 BA on an evaluation-only age/sex-matched subset (greedy 1:1 nearest age within 5 years, same sex, no replacement, fixed order by ID hash); report N pairs. Training is not matched; this is conditional on saved predictions.
d) Complete-case sensitivity: refit D1 on people with all 65 features present (saved folds restricted to them).

## Frozen decision rules
- SIGNAL BEYOND DEMOGRAPHICS (exploratory): D1 mean BA >= 0.70 AND permutation p < 0.05 AND (D1 - D0) mean BA >= 0.05 with bootstrap interval excluding 0.
- LEADERBOARD-SCALE (descriptive only): D1 mean BA >= 0.62 AND p < 0.05.
- If D1 mean BA < 0.62, or D0 is within 0.05 of D1 BA: close the diagnostic arm; no new feature sets or models in this stage.
- Any positive result needs independent data; matched-subset and complete-case results qualify but do not change these rules. No post-result changes.
