# CLINICAL PROTOCOL = AMENDMENT A4 (FROZEN 2026-10-08, before any clinical-variable x outcome result was computed)
EXPLORATORY confound-and-ceiling check. NOT prize-eligible and not a submission path: the Brainclinics rules state "All predictions must be based on the EEG data alone" (brainclinics.com/research/brainclinics-prize, read 2026-10-08 via fetch summary; wording worth re-checking by the user), and the local replication folders carry no metadata. Same 164 patients already used in every earlier stage; session-1 = pretreatment remains an unverified assumption. This amends PROTOCOL.md, which excluded age/sex/BDI/protocol from predictors; that exclusion still holds for every EEG-only result already reported.
Seen before freezing (label-blind only): availability of workbook columns among the 167 rTMS people, and marginal distributions of age/sex/BDI_pre/protocol. No outcome-by-clinical tabulation or model was run.

## Purpose (user decision 2026-10-08)
(1) Ceiling: how well do a few baseline clinical variables alone predict Remitter/Responder in this cohort? (2) Do EEG features (E01c) add anything beyond them? (3) Does E01c's apparent signal (AUC 0.577) track clinical variables?

## Variables (core four only; chosen by user, fixed now)
age (years), sex (gender 0/1), BDI_pre, rTMS PROTOCOL (1/2/3; indicators for 1, 2, 3; missing (4 people) = all three zero). Row = the workbook row whose sessID equals the ledger's selected session. NEVER used: BDI_post (it defines the outcome), diagnosis fields, NEO-FFI, lifestyle, oddball behaviour, season/time, weight/height, missingness flags. Age/sex/BDI_pre must be complete (asserted); no imputation is expected.

## Models (all: train-fit median impute -> StandardScaler -> L2 logistic, C {0.01,0.1,1,10} by inner 3-fold log loss; same 164 patients, saved outer splits StratifiedKFold(5, shuffle) seeds 42/43/44; threshold = max inner-OOF PPV s.t. sens>=0.5, ties higher sens; one shared penalty over all columns, no per-block tuning)
- K0: age, sex (2).   K1: age, sex, BDI_pre, protocol indicators (6).   K1E: K1 + the 65 E01c features (71).
- Benchmark: saved E01c predictions (identical patients/splits). No other models, interactions, tree models or feature selection. Remitter primary, Responder secondary (descriptive).

## Analyses
a) Descriptive (no inference, no selection): outcome rate by sex, protocol, age tertile, BDI_pre tertile, with N. Nothing is chosen from it.
b) Outcome models above, all repeats shown. Paired patient bootstrap (2000 draws, rng 1, joint across repeats, conditional on saved predictions) of mean-AUC differences: K1-E01c, K1E-K1, K1E-E01c.
c) One chance check: label permutation of K1 on Remitter (999 perms, rng 31, folds regenerated from permuted labels, full refit, plus-one p). Chance association only; not a test of increment.
d) Tracking check (label-free): Spearman correlation of the E01c out-of-fold predicted probability (mean over 3 repeats) with age, sex, BDI_pre, protocol (Kruskal-Wallis), with bootstrap 95% intervals (2000, rng 1).

## Frozen interpretation rules
- CLINICAL SIGNAL PRESENT: K1 Remitter mean AUC >= 0.60 AND permutation p < 0.05. Otherwise "not detected in this cohort".
- EEG ADDS BEYOND CLINICAL: K1E-K1 mean AUC gain >= 0.05, positive in every repeat, AND bootstrap interval excludes 0. Otherwise "not shown".
- E01c SIGNAL TRACKS CLINICAL: any |rho| >= 0.30 (or Kruskal-Wallis p < 0.01) in (d); report as such, no adjustment is claimed.
- No claim of clinical utility; no change to the closed arms (DynaMix, resting-power extension). Nothing here is a funding-rule test, but the old numbers (AUC >= 0.60, gain >= 0.05, positive every repeat) are reported alongside for continuity.
- Post-result changes to variables, models or rules are not allowed; any follow-up needs a new amendment.
