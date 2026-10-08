# Best results

All numbers use the same 164 rTMS patients (78 remitters) unless stated. "Mean AUC" is the average over three cross-validation repeats (seeds 42, 43, 44). Files for each result sit in the folder named below.

## 1. Best EEG-only prediction of rTMS outcome (the prize target) - `prognostic_EEG_only/`
| | Result |
|---|---|
| Model | 65 resting-EEG features (band power, frontal alpha asymmetry, entropy, connectivity), cleaned with the corrected pipeline ("E01c"), logistic regression |
| Remission AUC | **0.577** (95% interval 0.495 to 0.662, so it includes 0.5 = chance) |
| nPPV at the chosen threshold | about 106 to 113 (the prize needs more than 140), sensitivity about 0.55 to 0.62 |
| Verdict | **Not a usable result.** It does not beat chance with confidence and is far from the prize bar. It also depends on the cleaning choices (the same features under the first cleaning scored 0.517). |

## 2. Ceiling from clinical variables (NOT EEG-only, not prize-eligible) - `clinical_ceiling_not_eligible/`
| | Result |
|---|---|
| Model | age, sex, baseline depression score (BDI), rTMS protocol |
| Remission AUC | **0.704** (permutation test p = 0.001) |
| nPPV / sensitivity | 139 to 155 / 0.50 to 0.59 |
| Why it matters | This is the bar any EEG predictor has to beat in this cohort. Adding the EEG features did not improve on it (0.686). Baseline BDI does most of the work (0.67 alone). EEG could not estimate BDI (correlation 0.17). |

## 3. The one clean EEG-only result: predicting age from EEG - `age_from_EEG/`
| | Result |
|---|---|
| Model | same 65 features, ridge regression, 958 adults |
| R-squared | **0.39** overall; 0.36 in MDD patients, 0.42 in healthy people, 0.37 in other clinic patients |
| Average error | about 9.8 years |
| Verdict | **Real and consistent across cohorts.** (Sex from EEG is weaker: balanced accuracy 0.71 overall, 0.63 within healthy people.) Age is not one of the two prize criteria as far as we could tell from the rules page. |

## 4. Kept for the record, but INVALID - `invalid_diagnostic_batch_confounded/`
EEG-only MDD vs healthy reached balanced accuracy **0.755** (AUC 0.84). This looked like the strongest result in the project and is **not** a real finding. The follow-up control tests showed it separates the research cohort from the clinic cohort rather than MDD from healthy:
- the same model separates healthy people from *other* clinic patients just as well (0.756);
- it cannot separate MDD from other clinic patients (0.533, chance);
- within healthy people alone, it separates early-recorded from late-recorded subjects (0.829);
- the subject ID number alone separates MDD from healthy with AUC 0.92.

Do not cite or submit this result.
