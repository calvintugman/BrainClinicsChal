# BrainClinics challenge

Attempt at the Brainclinics Prize using the open TDBRAIN EEG dataset. The prize goal is to predict from resting EEG alone which depressed patients will respond to rTMS (a magnetic brain stimulation treatment), or to tell depressed patients apart from healthy people.


## Results

| Task | Best result |
|---|---|
| Age from EEG | R² 0.53 in adults (n = 957), 8.4 years average error |
| Sex from EEG | AUC 0.82 |
| rTMS remission from EEG | AUC 0.577, not distinguishable from chance |
| Remission from clinical variables (not EEG) | AUC 0.70 |
| Depressed vs healthy from EEG | 0.755 balanced accuracy, but invalid (see below) |

The age result held up across every cohort in the dataset and is about the same as the leaderboard's age entry (0.52). Mine is cross-validated though, while the leaderboard is scored on held-out data.

The depressed-vs-healthy model looked like my best result until I tested it more. It separated healthy people from other clinic patients just as well, couldn't tell depressed patients from other clinic patients, and the subject ID number alone separated the groups with AUC 0.92. It was picking up which cohort a recording came from, not depression, so I dropped it.

## What I tried

Band power, frontal alpha asymmetry, entropy, connectivity, spectral parametrization, covariance and Riemannian features, a pretrained dynamical-systems embedding (DynaMix), and full per-channel spectra. Models were mostly logistic and ridge regression, plus some gradient boosting. Each stage had a protocol written before I looked at results, and every run is logged.

More detail:
- `docs/PROJECT_LOG.md`: plain-language account of every stage
- `results/best/BEST_RESULTS.md`: the headline numbers and where they come from
- `docs/EXPERIMENTS.csv` and `docs/DECISIONS.md`: every run and decision in order

## Running it

The TDBRAIN data isn't included. You can request it from the dataset authors (DOI 10.7303/syn25671079). Subject-level outputs are left out too.

```bash
pip install -r requirements.txt
```

Scripts in `scripts/` are numbered in the order they were run. They expect the dataset folders one level above the repo and write to `outputs/` and `runs/`. See `docs/PIPELINE.md`. A few scripts read their protocol file from the repo root, so copy it over from `docs/protocols/` first.

The same 164 rTMS patients were used in every stage, so all of this is exploratory.
