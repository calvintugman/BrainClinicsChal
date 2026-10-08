# ERA PROTOCOL = AMENDMENT A9 (FROZEN 2026-10-08 after A7 result, before the analyses below were run)
DESCRIPTIVE check, patients only (164, session 1). Motivation: A7 T-E found subject ID (a cohort/era proxy, unverified as recording order) correlates with remission (Spearman -0.22 Remitter, -0.17 Responder), and A7 T-B showed EEG features encode ID-block strongly (BA 0.83 within healthy people). Question: is the weak E01c signal (AUC 0.577) tangled with era?
Analyses (no model search, no new features, no claims of causality):
(a) Spearman (with bootstrap 95% interval, 2000 draws, rng 1) between ID and: E01c out-of-fold probability (mean over 3 saved repeats, Remitter run), age, sex, BDI_pre, protocol (Kruskal-Wallis for protocol).
(b) Out-of-fold AUC of ID alone (logistic, same saved outer splits/seeds, no inner tuning needed: single standardised feature, C=1) for Remitter and Responder.
(c) Partial check: AUC of E01c within ID tertiles (evaluation-only on saved predictions), to see whether E01c discriminates within era.
Frozen reading: "E01c tracks era" if |rho(ID, E01c score)| >= 0.30 or the within-tertile AUCs are all <= 0.55 while the pooled AUC is >= 0.57. Otherwise "no evidence of era tracking". Any such finding limits interpretation only; no prognostic model is changed.
