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
| 4 — Evaluation | Lead scoring (homework) | [`04-lead-scoring-evaluation.ipynb`](notebooks/04-lead-scoring-evaluation.ipynb) | Validation AUC **0.732**, 5-fold std **0.007** |
| 5 — Deployment | Lead-scoring API + container (homework) | [`05-deployment/`](05-deployment/) | Same lead scores **0.770** both via uvicorn and via Docker |
| 6 — Trees | Credit risk scoring | [`06-credit-risk-decision-trees-explained.ipynb`](notebooks/06-credit-risk-decision-trees-explained.ipynb) | Test ROC-AUC **0.834**, 5-fold CV **0.841 ± 0.020** |
| 6 — Trees | Fuel efficiency (homework) | [`06-fuel-efficiency-trees-homework.ipynb`](notebooks/06-fuel-efficiency-trees-homework.ipynb) | Test RMSE **1.656** mpg (XGBoost), **1.671** (random forest) |

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
  04-lead-scoring-evaluation.ipynb                 # module 4 homework: evaluation metrics
  06-credit-risk-decision-trees-explained.ipynb    # module 6: trees, forests, boosting
  06-fuel-efficiency-trees-homework.ipynb          # module 6 homework: trees, forests, xgboost

05-deployment/                                     # module 5 homework: a separate uv project
  pipeline.bin                                     # frozen artifact (checksum verified)
  model.py                                         # feature order + missing-value policy
  predict.py                                       # FastAPI: GET /health, POST /predict
  q3_load_model.py                                 # unpickle and score in process
  Dockerfile                                       # pinned base image + uv sync --locked
  pyproject.toml, uv.lock, .python-version         # scikit-learn 1.7.2, Python 3.11.15
  README.md                                        # answers and walkthrough

data/
  car_price.csv                # module 2: car prices
  car_fuel_efficiency.csv      # module 2: fuel efficiency exercise
  telco.csv                    # module 3: telecom churn
  credit_card_default.csv      # module 3: UCI credit-card default
  course_lead_scoring_2026.csv # module 4: lead scoring (homework)
  credit_scoring.csv           # module 6: credit risk scoring
  car_fuel_efficiency_2026.csv # module 6 homework: pinned 2026 fuel efficiency release
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

`05-deployment/` is deliberately **not** part of this environment — it is its own
`uv` project with its own lockfile, because the module 5 homework needs
scikit-learn 1.7.2 to unpickle the frozen artifact while these notebooks run on
1.9. Set it up separately:

```bash
cd 05-deployment && uv sync --locked
```

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

**Lead scoring (module 4 homework).** Validation ROC-AUC 0.732; the 5-fold CV
standard deviation is 0.007. This notebook is about evaluation rather than a new
model: ROC-AUC feature importance, precision/recall/F1 across thresholds, `KFold`
stability, and tuning `C`.

**Credit risk (module 6).** A handwritten if/else rule scores 0.471 AUC — worse
than random — which motivates learning the splits. A tuned single tree reaches
0.786 validation AUC, a random forest 0.825, and gradient boosting 0.837. The
gradient-boosted model is refit on train + validation and scores **0.834** on the
test set (5-fold CV **0.841 ± 0.020**). The notebook explains Gini impurity with a
worked example, then follows the progression from one tree to bagging and
boosting, and ends by choosing a decision threshold from the cost of false
positives and false negatives.

**Fuel efficiency trees (module 6 homework).** The same three model families on
the pinned 2026 fuel-efficiency release (10,000 cars, target `fuel_efficiency_mpg`,
RMSE in mpg). A depth-1 tree splits on `model_year` and explains 10% of the error
against the 2.96 mean baseline; a 10-tree forest reaches 1.839 (the homework's
1.837); the tuned forest (100 trees, depth 10) 1.744 on validation and **1.671** on
test; XGBoost with `eta=0.1` **1.656**. The notebook works all six answers with the
homework's exact split (`random_state=1`), shows why 100 trees is a tie-break over
150 at three decimals, why depth 10 beats 15–25 at every forest size, why
`vehicle_weight` is the important feature among the four offered, and why a large
learning rate wins early then overfits (its validation curve bottoms at round 11
and climbs). It also cross-checks impurity importance against permutation
importance, which promotes `fuel_type` and explains the disagreement.

**Deployment (module 5 homework).** The 2026 homework ships a frozen
lead-scoring release, so the work is verification rather than training: confirm
the artifact's SHA-256, score a lead in process (**0.533**), serve a second lead
through `POST /predict` (**0.770**), then reproduce that exact number from inside
the Docker image (**0.770**). The container agreeing with the host is the whole
point — `uv sync --locked` in the `Dockerfile` resolves scikit-learn 1.7.2 from
the same lockfile, so the match is a build guarantee instead of luck. Details and
answers are in [`05-deployment/README.md`](05-deployment/README.md).

## Data

The seven CSVs live under `data/` so the notebooks run offline:

- `car_price.csv` and `car_fuel_efficiency.csv` come from the course materials.
- `telco.csv` is the Telco Customer Churn dataset from Kaggle.
- `credit_card_default.csv` was converted from the UCI "Default of Credit Card
  Clients" `.xls` file; the conversion command is documented in the notebook's
  3.2 section.
- `course_lead_scoring_2026.csv` is the pinned 2026 lead-scoring dataset used in
  the module 4 homework.
- `credit_scoring.csv` is the Credit Scoring dataset from Gaston Sanchez's
  [CreditScoring](https://github.com/gastonstat/CreditScoring) project, used in
  module 6.
- `car_fuel_efficiency_2026.csv` is the **pinned 2026 release** of the fuel
  efficiency dataset, used in the module 6 homework. It is a different file from
  `car_fuel_efficiency.csv` (module 2): 10,000 rows, model years 1975–2024, and a
  different distribution, so results are not comparable across the two.

## Notes

- The environment is pinned to pandas 3 / scikit-learn 1.9. Notebooks use
  version-robust dtype checks (`pd.api.types.is_string_dtype`,
  `select_dtypes(include=['object', 'string'])`) rather than comparing dtype
  names.
- `02-lessons-orig.ipynb` is the original NumPy implementation and still shows
  the intentional `LinAlgError` from inverting `XᵀX` by hand. The sklearn rewrite
  explains the problem and avoids it.
- `main.py` is the default `uv init` entry point and is not used by the notebooks.
- `05-deployment/` vendors the frozen 2026 homework release unmodified, including
  `pipeline.bin` and its checksums. Those files are inputs, not outputs — the only
  things added there are `q3_load_model.py`, `.dockerignore`, and the README.
- The module 5 artifact is **not** byte-for-byte reproducible even from the
  pinned environment (retraining matches predictions to ~5e-7 but changes the
  pickle hash), so `05-deployment/README.md` verifies the checksum instead of
  retraining.
- The module 6 **lesson** notebook uses scikit-learn's
  `GradientBoostingClassifier` instead of the course's XGBoost, so it runs without
  an extra dependency; it includes a parameter translation table for following the
  XGBoost version. The module 6 **homework** notebook uses XGBoost for real, so
  `xgboost>=3.2.0` is a declared dependency (on macOS it also needs the OpenMP
  runtime: `brew install libomp`).

## Next steps

- Try LightGBM or CatBoost and compare with XGBoost on the same splits.
- Handle class imbalance explicitly (`class_weight='balanced'`, resampling).
- Calibrate the probabilities and evaluate the cost of mistakes at portfolio level.
- Try nested cross-validation when tuning many hyperparameters at once.
