# BrainClinics Challenge

Attempts at the Brainclinics Prize using the open TDBRAIN EEG dataset: predicting rTMS outcome in depression from resting EEG (prognostic) and telling MDD from healthy controls (diagnostic).

**Short version:** no EEG-only predictor reached the prize bar. The best prognostic result is a remission AUC of 0.577 (not distinguishable from chance), baseline clinical variables reach 0.70 but are not EEG-only, and an apparently strong MDD-vs-healthy result (balanced accuracy 0.755) was shown to be a cohort effect and is not valid. Age can be predicted from EEG (R-squared about 0.39).

## Start here
- [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) - plain-language account of what was done, what worked, what did not
- [`results/best/BEST_RESULTS.md`](results/best/BEST_RESULTS.md) - the headline results and the files behind them
- [`docs/DECISIONS.md`](docs/DECISIONS.md) and [`docs/EXPERIMENTS.csv`](docs/EXPERIMENTS.csv) - every decision and run, in order

## Layout
```
docs/        project log, decisions, experiment ledger, audit, pipeline notes
docs/protocols/   one written protocol per stage, each frozen before scoring
scripts/     numbered analysis code (01 to 46) and shared helpers
reports/     self-contained HTML reports from earlier stages
results/     best/ (headline results), runs/ (aggregate metrics per run), summaries/ (small JSON)
requirements.txt
```

## What is not here
The TDBRAIN recordings and participant workbook are not included (obtain them from the dataset authors; DOI 10.7303/syn25671079). Subject-level files (features, predictions, ID mappings) are also withheld. The scripts expect the dataset folders one level above the repository root and write to `outputs/` and `runs/`; see [`docs/PIPELINE.md`](docs/PIPELINE.md). A few scripts read their protocol file from the root, so copy it from `docs/protocols/` when re-running.

## Status
Exploratory. Every stage reused the same 164 patients, so none of the results is confirmed on independent data.
