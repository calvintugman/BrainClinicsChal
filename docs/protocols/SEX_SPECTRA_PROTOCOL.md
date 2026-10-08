# SEX FROM SPECTRA = AMENDMENT A11 (FROZEN 2026-10-08 after A10, before any sex model on the spectra was fit)
EXPLORATORY, leaderboard-only category (not prize-eligible). Same adult cohort and SPEC/F65 features as A10 (SPECTRA_PROTOCOL.md).
Models: L2 logistic, C {1e-4,1e-3,1e-2,0.1,1} by inner 5-fold log loss, train-fit median imputation + StandardScaler: X0 = F65 (A8 baseline), X1 = SPEC, X2 = SPEC + F65. Outer StratifiedKFold(5, shuffle) seeds 42/43/44.
Metrics: AUC and balanced accuracy at the 0.5 threshold (no tuning); within-cohort breakdown (MDD, HEALTHY, OTHER).
Rule: USABLE if max(X1, X2) balanced accuracy >= 0.70 within every cohort subgroup; IMPROVED if its overall BA >= X0 + 0.05. No other models.
