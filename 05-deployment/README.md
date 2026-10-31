# Module 5 — Deploying machine learning models (homework)

The 2026 homework ships a **frozen release**: a trained artifact, its checksums,
the API, and a `Dockerfile`. The exercise is not to train anything — it is to
verify you can move that artifact from a pickle in a directory, to a served
process, to a container, and get the same number every time.

## What is in this directory

Vendored unmodified from
[`cohorts/2026/homework/05-deployment`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/05-deployment):

| File | Role |
| --- | --- |
| `pipeline.bin` | the frozen `DictVectorizer` + `LogisticRegression` artifact |
| `model_metadata.json` | dataset checksum, imputation values, model checksum |
| `feature_defaults.json` | training-set medians used for missing numerics |
| `model.py` | feature order + the training-time missing-value policy |
| `predict.py` | FastAPI app, `GET /health` and `POST /predict` |
| `train.py` | the script that produced `pipeline.bin` (only used in the optional check) |
| `Dockerfile` | canonical 2026 container definition |
| `pyproject.toml`, `uv.lock`, `.python-version` | the pinned environment |
| `smoke_test.py`, `q6_test.py` | determinism check and API client |

Added by me:

| File | Role |
| --- | --- |
| `q3_load_model.py` | Question 3 — unpickle and score one lead in process |
| `.dockerignore` | keeps `.venv` and `__pycache__` out of the build context |

## Answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | `uv --version` | **`uv 0.11.18`** locally (setup check, not graded); the container pins `uv 0.10.11` |
| 2 | Scikit-Learn in the 2026 reference environment | **`1.7.2`** |
| 3 | `predict_proba` for the first lead, in process | **`0.533`** |
| 4 | `POST /predict` for the second lead, via uvicorn | **`0.770`** |
| 5 | Python base-image tag in the `Dockerfile` | **`python:3.11.15-slim-bookworm`** |
| 6 | `POST /predict` for the second lead, via the container | **`0.770`** |

Full-precision values, in case rounding ever needs to be re-checked:

```text
Q3  paid_ads / north_america lead    0.5329944186432507  -> 0.533
Q4  organic_search / europe lead     0.7697992923369712  -> 0.770
Q6  same lead, inside the container  0.769799            -> 0.770   (API rounds to 6 places)
```

## Setup

```bash
cd 05-deployment
uv sync --locked
uv run python smoke_test.py
shasum -a 256 pipeline.bin
```

The checksum must be
`1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`, which is
also the `model_sha256` recorded in `model_metadata.json`. Verify that before
trusting anything downstream — a stale pickle is the classic way to get a
plausible-but-wrong probability.

## Walking through the questions

### Q2 — the answer is in the lockfile

`pyproject.toml` pins `scikit-learn==1.7.2`, and `uv.lock` resolves it to
`version = "1.7.2"`. The trap is that the module 4 notebooks in this repo run on
a much newer scikit-learn (`1.9.x`); that is fine for coursework but irrelevant
here. `uv sync --locked` installs what the lock says and refuses to re-resolve,
so the artifact is unpickled against the same scikit-learn that wrote it.
Checking `import sklearn; sklearn.__version__` after the sync confirms `1.7.2`.

### Q3 — load it in process

```bash
uv run python q3_load_model.py
```

`pipeline.bin` is a `sklearn.pipeline.Pipeline`, so `pickle.load` gives back
both the vectorizer and the classifier and one `predict_proba` call covers
preprocessing + scoring. The lead is complete, so imputation does not fire, but
the script still routes it through `model.normalize_record` — that is the same
code path the API uses, and it keeps categoricals coerced to the string values
the vectorizer was fitted on.

Result: **0.533**.

### Q4 — serve it

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

```bash
curl -s http://localhost:9696/health
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{"lead_source":"organic_search","industry":"technology",
       "employment_status":"employed","location":"europe",
       "number_of_courses_viewed":4,"annual_income":80304.0,
       "interaction_count":7,"lead_score":0.74}'
```

`/health` returns `status: ok` plus two hashes — the model checksum and the
feature-defaults checksum. Those are the interesting part of the endpoint: they
let a client prove which artifact is actually loaded, which is stronger than a
bare liveness ping.

Response: `{"conversion_probability":0.769799,"conversion":true}` → **0.770**.

### Q5 — the declaration, not the image

```dockerfile
FROM python:3.11.15-slim-bookworm@sha256:d29f48a3...
```

`python:3.11.15-slim-bookworm`, digest-pinned. Note the `slim-bookworm` base is
still ~1 GB on disk once `uv` and the scientific stack are in the `.venv`, so
the tag matters and the reported size does not.

### Q6 — same answer through the container

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
uv run python q6_test.py
```

```text
{'conversion_probability': 0.769799, 'conversion': True}
```

Verified inside the running container:

```text
Python 3.11.15
uv 0.10.11
sklearn 1.7.2
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1  pipeline.bin
```

Q4 and Q6 agree to all printed digits, which is the actual point of the
homework: the container is not a reimplementation, it is the same environment
resolved from the same lockfile. The `Dockerfile` does
`uv sync --locked --no-dev` from the checked-in `uv.lock`, so the container's
scikit-learn is `1.7.2` by construction, not by luck.

## Reading the numbers

- **0.770** for the `organic_search` / `europe` lead is comfortably above the
  `0.5` decision boundary: the API returns `conversion: true`, and this is a
  lead the model would prioritise.
- **0.533** for the `paid_ads` / `north_america` lead sits barely above the
  boundary. It converts to `true` only because the default threshold is `0.5`,
  and a threshold chosen for precision (module 4's trade-off exercise) would
  flip it to `false`. Treat that number as "coin flip", not as a prediction.
- Logistic regression outputs are on their own scale; nothing here is
  calibrated. Two leads landing at `0.533` and `0.770` are ordered correctly
  relative to each other more reliably than they are readable as literal
  probabilities.

## Reproducibility caveats worth knowing

- **Don't regenerate the lock.** `uv sync` without `--locked` would re-resolve
  and could bump scikit-learn, at which point the pickle may load but score
  differently. `--locked` is the whole guarantee.
- **Python is pinned by `.python-version` to `3.11.15` exactly**, and `uv` will
  download that patch release rather than accept a near neighbour. `3.11.16`
  works in practice but is not what the release declares.
- **Hash-pinned base images cost a pull.** The `@sha256:` digests in the
  `Dockerfile` pin the base layers too, so the build is reproducible across
  machines but cannot silently pick up a patched base.

## Optional check: does retraining reproduce the artifact?

Not part of the graded answers, but it answers the question a frozen release
raises — is this pickle *a* model, or *the* model that the data and script
describe?

```bash
mkdir -p /tmp/retrain
uv run python train.py \
  --data ../data/course_lead_scoring_2026.csv \
  --model /tmp/retrain/pipeline.bin
```

Two notes on that command: the vendored `train.py` resolves `--data` relative to
the course repo layout, so from here the path has to be passed explicitly; and
writing to `/tmp` keeps the frozen `pipeline.bin` untouched.

| Check | Result |
| --- | --- |
| `data/course_lead_scoring_2026.csv` SHA-256 | **matches** `dataset_sha256` in `model_metadata.json` |
| `feature_defaults.json` SHA-256 | **identical** — the medians rebuild exactly |
| `pipeline.bin` SHA-256 | **differs** (`ed6be6e3…` vs `1646bbdc…`) |
| Q3 lead, frozen vs retrained | `0.5329944186` vs `0.5329937257` (Δ 6.9e-07) |
| Q4 lead, frozen vs retrained | `0.7697992923` vs `0.7697987926` (Δ 5.0e-07) |

So the data and the imputation values are exactly recoverable, the scores agree
to about six decimals, but the serialized bytes are **not** reproducible — the
same code against the same pinned scikit-learn produces a different pickle. That
is why the homework pins a checksum instead of a training run: given a rebuilt
artifact you cannot tell a benign float-ordering difference from a real change
without re-scoring every lead. Verify the hash, then verify a known prediction;
do not regenerate the lockfile or the model and assume it is equivalent.

