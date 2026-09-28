# Amazon Business Entity Resolution

This project matches business records across three independent data sources. Source 1 is the reference set; the pipeline generates candidate matches from Sources 2 and 3, scores each candidate pair, and writes a submission-ready match file.

## Pipeline

The end-to-end pipeline normalizes business names, addresses, and countries; builds country-partitioned inverted indexes for candidate generation; extracts 28 pairwise features; and uses a gradient-boosted classifier to score candidate pairs. XGBoost is preferred. If it is unavailable, the matcher falls back to scikit-learn's `HistGradientBoostingClassifier`.

Training evaluates a validation split and optimizes the macro-averaged $F_{0.5}$ score. Test inference writes both the candidates considered by the model and the final predicted matches.

## Setup

Use Python with `pip`, then install the dependencies from the project root:

```bash
python -m pip install numpy pandas scikit-learn xgboost
```

XGBoost is used by the included trained model. Scikit-learn provides the fallback matcher when training without XGBoost.

## Data

Place the challenge TSV files in these paths, relative to the project root:

```text
train/
  train_source1.tsv
  train_source2.tsv
  train_source3.tsv
  train_ground_truth.tsv
test/
  test_source1.tsv
  test_source2.tsv
  test_source3.tsv
```

Files must be tab-separated. The challenge data is not included in this repository: the large TSV files are excluded by `.gitignore` and must be obtained separately.

## Run

From this directory, train the matcher and evaluate a validation split:

```bash
python run_pipeline.py --mode train
```

By default, training reads the files under `train/`, samples up to 20,000 Source 1 entities, and saves the model to `models/er_matcher.pkl`. Adjust the sample size or validation split with `--n_sample_s1` and `--val_split`.

Generate test predictions with the included model:

```bash
python run_pipeline.py --mode test
```

Test inference reads the files under `test/` and writes:

- `output/matching_results.tsv` with columns `source1_entity_id` and `matched_entity_ids`.
- `output/candidate_pairs.tsv` with columns `source1_entity_id` and `candidate_entity_ids`.

The output directory is created automatically. Paths, model location, and inference threshold can be overridden with the corresponding CLI options; run `python run_pipeline.py --help` for the full list. If the model file is missing, test mode trains a quick model from the training data first.

## Repository Contents

- `src/` contains normalization, tokenization, indexing, blocking, feature extraction, matching, evaluation, and output generation.
- `run_pipeline.py` is the end-to-end training and inference entry point.
- `run_blocking.py` and the other top-level scripts provide focused or optimized workflow entry points.
- `models/` contains serialized matcher models.
- `tests/` contains blocking and end-to-end tests.
- `scratch/` contains optimization experiments and benchmark scripts.

Generated output files, local Python environments, caches, and challenge data are excluded from Git.