# ML Zoomcamp 2026

Notes and exercises from [DataTalksClub's Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

The course notebooks are kept for reference, but most of this repo is a **rewrite of the lessons using `scikit-learn`** — same data, same concepts, but built the way you'd actually build a model: `Pipeline`, `ColumnTransformer`, proper train/validation/test splits, and honest evaluation.

## Progress

| Module | Task | Notebook | Held-out result |
|---|---|---|---|
| 2 — Regression | Car price prediction | [`02-car-price-prediction-sklearn.ipynb`](notebooks/02-car-price-prediction-sklearn.ipynb) | RMSE **0.196** (log1p price) |
| 2 — Regression | Fuel efficiency (extra exercise) | [`02-car-fuel-efficiency.ipynb`](notebooks/02-car-fuel-efficiency.ipynb) | RMSE **1.591** mpg |
| 3 — Classification | Telco customer churn | [`03-churn-telco-customer-prediction-sklearn.ipynb`](notebooks/03-churn-telco-customer-prediction-sklearn.ipynb) | Accuracy **0.812**, ROC-AUC **0.858** |
| 3 — Classification | Credit-card default | [`03-credit-card-default-prediction.ipynb`](notebooks/03-credit-card-default-prediction.ipynb) | Accuracy **0.818**, ROC-AUC **0.770** |

The module 3 churn notebook is written as a reusable **template**: change the
`DATA_PATH`, `TARGET`, `POSITIVE_LABEL`, and the numeric/categorical column lists,
and the rest of the workflow carries over to a new classification dataset.

## Structure

```text
notebooks/
  02-lessons-orig.ipynb                            # original course notebook (from-scratch NumPy)
  02-car-price-prediction.ipynb                    # follow-along notes on the original
  02-car-price-prediction-sklearn.ipynb            # car price, rewritten with scikit-learn
  02-car-fuel-efficiency.ipynb                     # extra regression exercise
  03-churn-telco-customer-prediction.ipynb         # original course notebook
  03-churn-telco-customer-prediction-sklearn.ipynb # churn, rewritten as a reusable template
  03-credit-card-default-prediction.ipynb          # template applied to credit-card default

data/
  car_price.csv                # module 2: car prices
  car_fuel_efficiency.csv      # module 2: fuel efficiency exercise
  telco.csv                    # module 3: telecom churn
  credit_card_default.csv      # module 3: UCI credit-card default
```

Each `NN-*` prefix corresponds to a module/chapter number in the course.

## Approach

Every `*-sklearn` notebook follows the same workflow:

1. **Load and clean** — consistent column/value names, numeric columns as numbers, target as 0/1.
2. **Split** — 60% train / 20% validation / 20% test with a fixed `random_state`.
3. **Explore** — class balance, missing values, feature importance (risk ratio, mutual information, correlation).
4. **Preprocess in a `Pipeline`** — one-hot encode categoricals, scale numerics; no preprocessing fitted outside the pipeline.
5. **Fit and interpret** — linear/logistic regression coefficients, read in feature-name order.
6. **Evaluate honestly** — validation first, one final refit on train+validation, then the untouched test set.
7. **Analyze errors** — confusion matrix, precision/recall/F1, ROC and PR curves, threshold trade-offs, residual plots.
8. **Sanity-check** — dummy baseline and stratified k-fold cross-validation.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (see `.python-version`):

```bash
uv sync
```

`uv sync` installs both the runtime dependencies and the `dev` group
(JupyterLab, ipykernel, nbconvert). Everything is pinned in `uv.lock`.

## Running the notebooks

```bash
uv run jupyter lab
```

Then open any notebook under `notebooks/`. `uv run` puts the project virtualenv
first on `PATH`, so notebooks execute against the pinned packages rather than a
global Python install.

To execute a notebook headlessly (a quick smoke test after changing code or
dependencies):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03-churn-telco-customer-prediction-sklearn.ipynb
```

## Results in a bit more detail

**Churn (Telco).** Logistic regression on the raw features reaches 0.812 accuracy
and 0.858 ROC-AUC. The dummy "always no churn" baseline is 0.753, so the model
adds real signal — but recall is 0.589, meaning ~41% of churners are still missed
at the default 0.5 threshold. Cross-validation is stable at 0.842 ± 0.005 AUC.

**Credit-card default (UCI).** Accuracy 0.818 vs a 0.777 majority-class baseline,
ROC-AUC 0.770. Recall is only 0.340: this dataset is harder and more imbalanced
(~22% defaults), which makes it a good example of why accuracy alone is
misleading. CV AUC is 0.768 ± 0.011.

Both classification notebooks include a section on moving the decision threshold
and on choosing metrics by the real cost of false positives vs false negatives.

## Data

The four CSVs live under `data/` so the notebooks run offline:

- `car_price.csv` and `car_fuel_efficiency.csv` come from the course materials.
- `telco.csv` is the Telco Customer Churn dataset from Kaggle.
- `credit_card_default.csv` was converted from the UCI "Default of Credit Card
  Clients" `.xls` file; the conversion command is documented in the notebook's
  3.2 section.

## Notes

- The environment is pinned to pandas 3 / scikit-learn 1.9. Notebooks use
  version-robust dtype checks (`pd.api.types.is_string_dtype`,
  `select_dtypes(include=['object', 'string'])`) rather than comparing dtype
  names.
- `02-lessons-orig.ipynb` is the original NumPy implementation and still shows
  the intentional `LinAlgError` from inverting `XᵀX` by hand. The sklearn rewrite
  explains the problem and avoids it.
- `main.py` is the default `uv init` entry point and is not used by the notebooks.

## Next steps

- Try tree-based models (`RandomForestClassifier`, `HistGradientBoostingClassifier`)
  and compare ROC-AUC / PR-AUC.
- Handle class imbalance explicitly (`class_weight='balanced'`, resampling).
- Pick the decision threshold on validation data, then apply it once to test.
- Move on to the course modules on evaluation, deployment, and ensembles.
