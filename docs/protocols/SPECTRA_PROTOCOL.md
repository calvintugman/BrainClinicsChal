# SPECTRA PROTOCOL = AMENDMENT A10 (FROZEN 2026-10-08 while label-blind spectra extraction was running, before any model on the new features was fit)
EXPLORATORY. User request: a richer EEG-only age model (A8 ridge on 65 features gave R2 0.39). Rules page (fetched 2026-10-08): age/sex prediction can be unblinded for the TDBRAIN leaderboard on request but is NOT prize-eligible ("The Brainclinics Foundation Prize only sees to predicting treatment response and diagnostic status in MDD"). One prize-relevant secondary test reuses the same spectra.
Extraction (scripts/47_spectra_extract.py, label-blind): v2 cleaning, the same clean 4-s Hann windows as 03_features, mean one-sided PSD (uV^2/Hz) per channel, 0.25 Hz resolution. Feature range fixed at 1-30 Hz (30-45 Hz is muscle-dominated; the rules require signal of cerebral origin).

## Features (fixed)
SPEC = natural log of mean PSD averaged in 1-Hz bins [k, k+1) Hz, k = 1..29, for 26 channels x {EC, EO} = 1508 features. A condition that failed cleaning or had < 20 clean windows is a NaN block (train-fit median imputation, no indicators).
SPECAVG = mean over the 26 channels of the SPEC log values = 29 x 2 = 58 features.
F65 = the 65 E01c features (A8 baseline).

## Primary: age (adults >= 18; patients' session-1 + pool people with workbook age/sex; exact-duplicate F65 vectors dropped; at least one condition spectrum present)
Models (Ridge, alpha {0.1,1,10,100,1e3,1e4,1e5} by inner 5-fold MSE, train-fit median imputation + StandardScaler): S0 = F65 (A8 baseline recomputed on this cohort), S1 = SPEC, S2 = SPEC + F65, S3 = SPECAVG. No other models.
Evaluation: outer KFold(5, shuffle) seeds 42/43/44; R2, MAE, Pearson r on pooled out-of-fold predictions; within-cohort breakdown (MDD patients, HEALTHY, OTHER pool); paired person bootstrap (2000, rng 1, joint across repeats) for R2 differences S1-S0, S2-S0.
Secondary descriptive: all ages including children (S0 and S1), because the leaderboard's age-prediction population is unknown.
Rule: IMPROVED if max(S1, S2) mean R2 >= S0 + 0.05 with bootstrap lower bound > 0 AND R2 >= 0.30 within every cohort subgroup. Otherwise "no gain from richer spectra".

## Secondary (prize-relevant): full-spectrum remission model
K-SPEC: L2 logistic on SPEC (1508) for the 164 patients, C {1e-4,1e-3,1e-2,0.1,1} by inner 3-fold log loss, same saved outer splits (seeds 42/43/44), threshold = max inner-OOF PPV s.t. sensitivity >= 0.5. Remitter primary, Responder descriptive. Funding rule unchanged (mean AUC >= 0.60, gain >= 0.05 over E01c, positive every repeat); a 999-permutation test (rng 71) is run ONLY if the funding rule is met.

## Descriptive, NOT EEG-only: brain-age gap
Patients' out-of-fold EEG age from S1 (adult model) minus true age. Spearman with Remitter/Responder and CV AUC (logistic on the gap, same splits). Uses true age, so not eligible for the prize; reported for scientific context only.
No post-result changes; follow-ups need a new amendment.
