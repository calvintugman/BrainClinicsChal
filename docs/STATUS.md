# STATUS (2026-10-08, late) - prize arms null/closed; age-from-EEG improved to R2 0.53
Prognostic (EEG-only): best E01c Remitter AUC 0.577 (not significant); full-spectrum model 0.485. Clinical ceiling (NOT EEG-only) 0.70.
Diagnostic: MDD-vs-healthy BA 0.755 is cohort/batch-driven (A7) -> non-evidence.
Age from EEG (leaderboard-only): full spectra R2 0.528 adults / 0.650 all ages, holds within each cohort (A10). Sex: AUC 0.82, BA 0.73 (weak in healthy).
Next (user decision): whether to generate age predictions for a leaderboard submission (replication folders untouched so far).

# STATUS (2026-10-08) - CLINICAL check (A4) COMPLETE; EEG-only arms unchanged (all closed/null)
Clinical (not prize-eligible): age+sex+BDI_pre+protocol Remitter mean AUC 0.704 (perm p=0.001); EEG adds nothing beyond it; age+sex alone 0.542; BDI_pre alone 0.671 (post-hoc). See CLINICAL_PROTOCOL.md, DECISIONS D20-D21, runs/CLIN_*, CLINX_*. Next: user decision.

Done: DEVIATION_PROTOCOL.md frozen; reference diagnostics (F1 pass, F2 info pass); deviation reliability (F3 pass); applicability; models A-D x 3 repeats x 2 endpoints + complete-case refit; 999-permutation chance check for D; DEVIATION_REPORT.html.
Result: learned reference structure present; deviations reliable within session; NO remission prediction (D mean AUC 0.503; gain vs E01c -0.074; vs A +0.046, vs C +0.010, all intervals include 0). Funding rule not met -> closed.
Preserved limits: same 164 patients reused; session-1 timing unverified; v2 cleaning unresolved removals + 8 ICA failures; reference = mixed-diagnosis, cleaner, adult-skewed population.
Closed arms: DynaMix; resting-power EC->EO extension. Not started (user decision): Bailey/pyhctsa, oddball availability, independent data.
Reproduce (from first_tests/): scripts/30_dev_window_powers.py; 29_dev_reference_mapping.py; 31_dev_reliability.py; 32_dev_applicability.py; 33_dev_outcomes.py; 34_dev_permutation.py; 35_dev_analysis.py; 36_dev_report.py.
