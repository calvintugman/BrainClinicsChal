# PROTOCOL: resting-only pilot (FROZEN 2026-10-05, before any outcome-associated result was computed)

## Amendment A1 (2026-10-05): scope change
Original task-based experiments E02-E05 (oddball rest-to-task reconfiguration cost) are `blocked_missing_rTMS_oddball`: the downloaded
rTMS patients have no oddball EEG (162 oddball files, 129 people, all HEALTHY). Their hypothesis is UNTESTED, not failed. Definitions are
preserved in the original project spec history. This pilot is a conventional resting baseline: not novel, not a test of the oddball idea.

## Cohort and timing
Endpoint: `Remitter` (primary), `Responder` (secondary, same locked design, run after primary). Labels as supplied; no hierarchy enforced.
Source: workbook TDBRAIN_participants_V3.1.xlsx (sha256 in outputs/inventory_summary.json), Dataset == MDD-rTMS.
Timing: ses-1 used when the workbook has a sessID=1 row. Evidence is a DATASET CONVENTION only (data dictionary: sessID = "session number of the
EEG acquisition session (ascending)"; no acquisition dates or treatment dates are available locally). It is NOT individually verified pretreatment
timing; results must be described as "assuming session 1 precedes treatment". The 3 sessID=2-only people are excluded. Replication folders
(104 + 60 recordings, non-workbook IDs): headers only, never used for fitting/tuning/submission.

## Preprocessing (recording-local, label-blind)
- Channels: first 26 = scalp EEG (Fp1..O2, 10-20); VPVA,VNVB,HPHL,HNHR = EOG; Erbs = ECG; Mass = EMG; Status dropped. Units uV (header), reference linked mastoids
  as acquired (A1/A2 not recorded; no re-reference). Aux channels used for ICA correction only, never as predictors.
- Trailing padding: constant-valued tail (all scalp channels flat) trimmed BEFORE filtering (~1 s in most files; the raw tail otherwise causes huge edge artifacts).
- 50 Hz notch (FIR, width 2 Hz; metadata PowerLineFrequency=50 and spectra), then 0.5-45 Hz FIR band-pass (l_trans 0.5 Hz, h_trans 4 Hz, MNE default firwin/zero-phase).
- Bad channels: median 1-s-block std > 3x or < 1/4 of across-channel median -> spherical-spline interpolation (>3 bad = recording unusable). Interpolated channels are
  excluded from ROI means, FAA (F3,F4 both required) and connectivity.
- ICA: fastica, random_state 42, n_components=min(20, 25-n_bad), fit on 1 Hz high-passed copy with 1-s blocks of ptp>200 uV annotated bad; excluded: components
  correlated (MNE z>3) with bipolar VEOG (VPVA-VNVB), HEOG (HPHL-HNHR) or Erbs ECG, max 3. Bipolar pairing is an assumption from channel names (no documentation). Decision procedure label-blind.
- Windows: 4 s Hann, 50% overlap, constant detrend; reject window if any scalp channel ptp > 150 uV. Recording needs >= 20 clean windows else its features are missing (imputed).
- Gamma omitted. No 30-45 Hz feature either.

## Feature schema (all per patient, one row per patient)
Bands [lo,hi) bins: delta 1-4, theta 4-8, alpha 8-12, beta 12-30 Hz. Band power = sum of one-sided PSD (uV^2/Hz) over bins x df (0.25 Hz) -> uV^2; ROI value = mean of channel band powers over
non-interpolated ROI channels; floor 1e-6 uV^2. log-abs = ln(mean over windows); relative = band/(sum of the 4 bands, 1-30 Hz), mean over windows.
ROIs: frontal F7,F3,Fz,F4,F8; central FC3,FCz,FC4,C3,Cz,C4; posterior P3,Pz,P4,O1,Oz,O2.
- E01-PSD (53): {EC,EO} x 3 ROI x 4 bands x {logabs, rel} = 48; IAF EC and EO (peak of posterior-ROI mean PSD, 7-13 Hz, interior local max >= 1.2x mean of 6-7 & 13-14 Hz flanks, else missing = 2);
  EC/EO alpha contrast = logabs_EC - logabs_EO per ROI (3).
- E01 = E01-PSD + FAA EC, FAA EO (ln alpha F4 - ln alpha F3; positive = more alpha at F4) + spectral entropy EC, EO (normalized Shannon entropy of PSD 1-30 Hz per channel, mean over channels)
  + connectivity (8): EO only, debiased squared wPLI (mne-connectivity wpli2_debiased, multitaper), theta 4-8 / alpha 8-12, first 20 clean non-overlapping 4 s epochs (needs >=20 else missing);
  mean off-diagonal over all usable channels, frontal ROI (within), posterior ROI (within), frontal-posterior (between). Negative estimates kept. Total 65.
- Excluded from predictors: BDI, diagnosis/indication, protocol, age/sex, behavior, dates, missingness flags, artifact burden. (Audit only.)
Missing values: training-fold median imputation, no indicators.

## Evaluation
Cohort: all included patients with a row (164 people). Outer: StratifiedKFold(5, shuffle, seed s) on patients (1 row/patient, so no grouping needed); seeds 42 (primary), 43, 44.
Model (identical for E01-PSD and E01): median impute -> StandardScaler -> L2 LogisticRegression(lbfgs, max_iter 2000); C in {0.01, 0.1, 1, 10} chosen by inner StratifiedKFold(3, shuffle, seed s) log loss.
(Grid extended below the spec's example because p~53-65 vs n~131 training patients; chosen a priori, not from outcomes.) E00: training prevalence constant.
Threshold: from inner out-of-fold probabilities on outer-training only (3-fold with selected C): maximize PPV subject to sensitivity >= 0.5, ties -> higher sensitivity; applied to outer test fold unchanged.
Metrics: AUC, PPV, sensitivity, nPPV=100*PPV/prevalence, Brier, balanced accuracy, n positive calls, TP/FP/TN/FN. Computed on pooled out-of-fold predictions per repeat; undefined -> NaN.
Comparisons (same patients/splits): E01 vs E00, E01-PSD vs E00, E01 vs E01-PSD (paired patient bootstrap, 2000 draws, seed 1, same draw for all models; conditional on saved predictions).
Permutation check of E01 AUC (seed-42 split design): 200 label permutations, full pipeline (inner tuning + threshold) refit each time.
Reliability: split-half (first vs second half of clean windows; connectivity: interleaved epoch halves), Spearman r with Spearman-Brown correction, per feature.
Compute budget: CPU only, 12 workers, < 2 h total. No other models, grids or feature additions after seeing results; poor results are reported as such.
