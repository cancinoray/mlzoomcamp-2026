# ML Zoomcamp 2026

Notes and exercises from [DataTalksClub's Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), following along with the course while rewriting the lesson notebooks with `scikit-learn` for clarity.

## Structure

```text
notebooks/
  02-lessons-orig.ipynb                  # original course notebook (from-scratch NumPy)
  02-car-price-prediction.ipynb          # my own follow-along notes on the original
  02-car-price-prediction-sklearn.ipynb  # same lesson, rewritten with scikit-learn
data/
  car_price.csv                          # dataset used in the linear regression lesson
```

Each `NN-*` prefix corresponds to a module/chapter number in the course.

## Why the scikit-learn rewrites?

The original course notebooks implement things like linear regression and
one-hot encoding from scratch with plain NumPy, which is a great way to learn
the underlying math, but not how you'd actually build a model in practice. The
`*-sklearn.ipynb` notebooks cover the same concepts and datasets using
`scikit-learn` (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`,
`LinearRegression`/`Ridge`, etc.), with explanatory markdown aimed at someone
new to the library.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Running the notebooks

```bash
uv run jupyter lab
```

Then open any notebook under `notebooks/`.
