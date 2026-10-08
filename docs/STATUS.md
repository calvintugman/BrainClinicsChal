# STATUS (2026-10-08, end of session) - ALL ARMS CLOSED OR NULL; ONE REAL EEG-ONLY RESULT (age)
Prognostic (EEG-only): best E01c Remitter AUC 0.577 (not significant); nothing reaches the funding rule. Clinical ceiling (NOT EEG-only): BDI_pre+age+sex+protocol AUC 0.70.
Diagnostic: EEG-only MDD-vs-healthy BA 0.755 was shown to be cohort/batch-driven (A7) -> non-evidence; do not use.
Age prediction from EEG: R2 0.39, holds within each cohort (A8). Sex BA 0.71 (weak within healthy).
Closed arms: DynaMix, resting-power EC->EO, clinical-assist, diagnostic. Not done: oddball (no data), Bailey/pyhctsa benchmark, independent validation data, author contact, submission.
Plain-language summary: docs/PROJECT_LOG.md (repo). Details: DECISIONS.md D1-D27, EXPERIMENTS.csv.

# STATUS (2026-10-08) - CLINICAL check (A4) COMPLETE; EEG-only arms unchanged (all closed/null)
Clinical (not prize-eligible): age+sex+BDI_pre+protocol Remitter mean AUC 0.704 (perm p=0.001); EEG adds nothing beyond it; age+sex alone 0.542; BDI_pre alone 0.671 (post-hoc). See CLINICAL_PROTOCOL.md, DECISIONS D20-D21, runs/CLIN_*, CLINX_*. Next: user decision.

Done: DEVIATION_PROTOCOL.md frozen; reference diagnostics (F1 pass, F2 info pass); deviation reliability (F3 pass); applicability; models A-D x 3 repeats x 2 endpoints + complete-case refit; 999-permutation chance check for D; DEVIATION_REPORT.html.
Result: learned reference structure present; deviations reliable within session; NO remission prediction (D mean AUC 0.503; gain vs E01c -0.074; vs A +0.046, vs C +0.010, all intervals include 0). Funding rule not met -> closed.
Preserved limits: same 164 patients reused; session-1 timing unverified; v2 cleaning unresolved removals + 8 ICA failures; reference = mixed-diagnosis, cleaner, adult-skewed population.
Closed arms: DynaMix; resting-power EC->EO extension. Not started (user decision): Bailey/pyhctsa, oddball availability, independent data.
Reproduce (from first_tests/): scripts/30_dev_window_powers.py; 29_dev_reference_mapping.py; 31_dev_reliability.py; 32_dev_applicability.py; 33_dev_outcomes.py; 34_dev_permutation.py; 35_dev_analysis.py; 36_dev_report.py.
