# Project log (plain-language)

Last updated 2026-10-08. For full detail see `DECISIONS.md` (numbered decisions), `EXPERIMENTS.csv` (every run) and the frozen protocols in `protocols/`.

## 1. The goal
The Brainclinics Prize (built on the open TDBRAIN dataset) pays EUR 10,000 for an EEG-only predictor that beats these bars and holds them for two months:
- **Prognostic:** predict which depressed patients respond to rTMS brain stimulation. Needs nPPV above 140% for remission (above 130% for response) with sensitivity above 40%.
- **Diagnostic:** tell MDD patients from healthy controls with balanced accuracy above 70%.

At the time of the leaderboard screenshot the best prognostic entry was 134% nPPV and the best diagnostic entry was 63%.

**Quick glossary.** *Remitter / Responder*: patient whose depression score improved enough (remitter = recovered, responder = improved by 50% or more). *AUC*: how well a score ranks patients; 0.5 is a coin flip, 1.0 is perfect. *PPV*: of the patients we predict will improve, the share who do. *nPPV*: PPV divided by the base rate, times 100, so 100% means no better than guessing. *Balanced accuracy (BA)*: average of sensitivity and specificity. *EC / EO*: eyes-closed / eyes-open resting recordings. *BDI*: a depression questionnaire score.

## 2. The data
- 164 rTMS patients (78 remitters, 97 responders) with eyes-closed and eyes-open resting EEG, one recording per condition (about 2 minutes, 26 scalp channels).
- About 1,000 other people (healthy volunteers and other clinic groups) for reference work.
- The planned rest-to-task ("oddball") experiment could not be run: none of the rTMS patients have task EEG.
- The raw EEG and anything with subject IDs are **not** in this repository.

## 3. How we worked
Every stage had a written protocol that was frozen **before** any result was scored (`protocols/`), the same folds and seeds throughout, and every run is logged. Because the same 164 patients were reused in every stage, all results are exploratory, not confirmed.

## 4. What was done, in order
| # | Stage | Question | Result | Verdict |
|---|---|---|---|---|
| 1 | Baseline | Do standard resting-EEG features predict remission? | AUC 0.49 to 0.54, permutation p = 0.24 | No |
| 2 | Audit | Is there a bug? | No bug that changes the numbers; one caption error fixed | OK |
| 3 | Batch 2 | Do 7 other feature ideas (gradient boosting, spectral parametrisation, covariance geometry, EC/EO eigenvalues) help? | Best AUC 0.517; none met the bar. One idea was confounded by cleaning artefacts | No |
| 4 | Cleaning v2 | Does fixing the artefact-cleaning step change things? | Same features now AUC 0.577 (interval 0.495 to 0.662) | Shows sensitivity to cleaning, not a biomarker |
| 5 | DynaMix | Does a pretrained dynamical-systems model give useful EEG features? | AUC 0.490, p = 0.56 | No |
| 6 | EC to EO deviation | Does "how far the eyes-open EEG departs from what eyes-closed predicts" help? | AUC 0.503, p = 0.42. The mapping itself worked, but did not predict remission | No |
| 7 | Clinical ceiling | How well do age, sex, baseline BDI and protocol predict remission? | AUC 0.704, p = 0.001. EEG added nothing on top | Real, but not EEG-only |
| 8 | Clinical assist | Can EEG-estimated age, sex and BDI help? | EEG estimates age (R2 0.32) and sex (AUC 0.76) but **not** BDI (correlation 0.17); the resulting models were null (0.52 to 0.60) | No |
| 9 | Diagnostic | MDD vs healthy from EEG alone | BA 0.755, AUC 0.84, p = 0.001 | Looked great, later shown invalid (next row) |
| 10 | Confound control | Is the diagnostic result real? | Within an overlapping ID range BA fell to 0.58; the model separates early from late healthy subjects (BA 0.83); it cannot separate MDD from other clinic patients (0.53) but does separate healthy from them (0.76) | **Batch/cohort effect, not MDD. Diagnostic arm closed.** |
| 11 | Era check | Do patient outcomes or our EEG score follow recording order? | Later subject IDs (and later protocols) remit less; our EEG score does not track ID | Context only |
| 12 | Age and sex | Can EEG predict age and sex? | Age R2 0.39, holds in every cohort; sex BA 0.71 (0.63 in healthy only) | Age is real |

## 5. Main lessons
1. **No EEG-only prediction of rTMS outcome was found.** Best AUC 0.577 (not distinguishable from chance), against roughly 0.70 needed for the prize bar. With 78 remitters and 86 non-remitters, only large effects are detectable.
2. **Baseline depression severity does most of the predictive work** in this cohort, and EEG cannot recover it. Any future EEG feature should be judged against this clinical baseline.
3. **Check for cohort effects before trusting a big number.** The first diagnostic result was 12 points above the leaderboard and turned out to reflect which cohort a recording came from. The subject ID number alone separated the groups (AUC 0.92).
4. **Cleaning choices matter.** The same features moved from 0.517 to 0.577 when artefact removal was corrected.
5. **Age can be predicted from EEG** (R2 about 0.39) and this holds within each cohort.

## 6. Dead ends (do not repeat without a new idea)
Standard band-power/asymmetry/connectivity panels; gradient boosting on them; spectral parametrisation; covariance and EC/EO eigenvalue features; the DynaMix embedding; the EC-to-EO deviation features; clinical-assist models using EEG-estimated age/sex/BDI; MDD-vs-healthy classification on this data (batch-confounded).

## 7. Caveats
- The same patients were reused in every stage; there is no untouched validation set.
- That session 1 is before treatment is assumed from the dataset's numbering, not verified per patient.
- Subject ID as a recording-order proxy is unverified; recording dates are removed from the files.
- Some cleaning decisions are unresolved (a few removed components look drift-like rather than eye-related).
- The rules page does not say whether clinical labels may be used for training when the final prediction uses only EEG. We treated it as allowed; this is not confirmed. Nothing here was submitted.

## 8. Ideas not tried
- Independent data (no second labelled cohort was available locally).
- The temporal-feature benchmark from the literature (needs MATLAB tools we did not have).
- Requesting the oddball task recordings from the dataset authors.
- A richer age model (full spectra), which would need re-extracting features.
- A diagnostic design that is not batch-confounded (needs healthy and MDD recordings from the same source and period).

## 9. Repository map
- `docs/` this log, `DECISIONS.md`, `EXPERIMENTS.csv`, `AUDIT.md`, `PIPELINE.md` (how to rerun), `protocols/`
- `scripts/` numbered analysis code (01 to 46) and shared helpers
- `reports/` self-contained HTML reports from earlier stages
- `results/best/` the headline results (see `BEST_RESULTS.md`)
- `results/runs/` aggregate metrics for every run (no subject-level files)
- `results/summaries/` small JSON summaries
