# CONFOUND PROTOCOL = AMENDMENT A7 (FROZEN 2026-10-08, after the A6 diagnostic result and before any of the analyses below were run)
EXPLORATORY control for the A6 result (MDD vs healthy EEG-only BA 0.755).
Seen before freezing (label-free or descriptive only): recording headers/sidecars are IDENTICAL across MDD and healthy (same amplifier, 500 Hz, 33 ch, filters, reference, line frequency; dates anonymised to 1985). Numeric subject ID is strongly group-structured: healthy IDs mostly 87,958,013-87,973,000 (a block of ~128), MDD IDs all >= 87,999,321; ID alone gives AUC 0.92 for MDD vs healthy. IDs are NOT assumed to be recording dates; they are a cohort/batch/era PROXY, unverified. ID is never a predictor in any EEG-only model; it appears only in diagnostic controls.
Rules context (fetched 2026-10-08): training-time use of clinical labels is not addressed by the rules; external/own data is explicitly allowed for discovery; the diagnostic held-out task is MDD vs healthy controls (N=60), scored by balanced accuracy. User decision: clinical labels may be used for training provided the prediction itself uses only EEG.

## Common pipeline (identical to A6)
65 E01c features, train-fit median impute -> StandardScaler -> L2 logistic, C {0.01,0.1,1,10} by inner 3-fold log loss, outer StratifiedKFold(5, shuffle) seeds 42/43/44, threshold = argmax inner-OOF balanced accuracy (ties closest to 0.5). Models per task: EEG (65 features), IDONLY (numeric ID), DEMO (age+sex), EEG+ID (65 + ID; diagnostic only). Metrics: BA (primary), AUC, sensitivity, specificity. Bootstrap (2000, rng 1, joint across repeats). Adults only (age >= 18) in every task. Exact-duplicate feature vectors dropped.

## Tasks
T-A (ID-window diagnostic): MDD vs HEALTHY restricted to people with ID in [87,990,000, 88,080,000] (the region where both groups exist). Fit within the window. Also report (conditional) BA of the saved A6 out-of-fold predictions restricted to the same people.
T-B (batch signature in healthy people): healthy block A (ID <= 87,973,000) vs healthy block B (ID > 87,973,000); no MDD involved. Measures whether EEG features carry a cohort/era signature independent of diagnosis.
T-C (MDD specificity): MDD vs clinic-recorded NON-MDD pool adults (workbook formal_status in ADHD, BURNOUT, CHRONIC PAIN, INSOMNIA, PARKINSON, TINNITUS; not Dyslexia (children), not TINNITUS/MDD, not UNKNOWN). 
T-D (clinic vs research): HEALTHY vs the same clinic non-MDD adults.
T-E (descriptive, patients only): AUC of ID alone for Remitter and Responder (3 seeds, same splits) plus Spearman of ID with each outcome. Tells whether any outcome analysis could be era-driven.
Permutation (999, rng 61, folds regenerated from permuted labels, full refit, plus-one p, statistic = mean BA of the EEG model) for T-A and T-C only.

## Frozen interpretation
- A6 DIAGNOSTIC SIGNAL SURVIVES the confound control only if ALL hold: T-A EEG mean BA >= 0.65 AND bootstrap lower bound > 0.55 AND T-A permutation p < 0.05 AND T-A EEG BA exceeds T-A IDONLY BA by >= 0.05 AND T-C EEG mean BA >= 0.62 with permutation p < 0.05.
- BATCH-DRIVEN if T-B EEG BA >= 0.70 AND T-A fails: then the A6 result is attributed to cohort/batch and the diagnostic arm is closed as non-evidence.
- Otherwise INCONCLUSIVE (small healthy-in-window n); no claim; no model search to rescue it.
- T-D BA is reported for context: if healthy vs clinic non-MDD is about as separable as MDD vs healthy while T-C is near chance, the signal is "research cohort vs clinic cohort", not MDD.
- No new feature sets/models in this stage; any follow-up needs a new amendment. Nothing here is a submission or clinical claim.
