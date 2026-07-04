# RFE-cross-validation

**Recursive Feature Elimination run inside every cross-validation fold — so you
can see not just *how many* features to keep, but *which* ones, and how stable
that choice is.**

## Motivation

scikit-learn gives you two related but incomplete tools:

- [`RFECV`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
  cross-validates the **number** of features to keep, but doesn't tell you how
  stable each individual feature's importance is.
- [`RFE`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
  ranks the features, but on a **single** fit — with no notion of variability.

`rfe_cv` fills the gap: it runs RFE separately in each CV fold and aggregates
the per-fold rankings. A feature that is consistently ranked first across folds
is a robust choice; one whose rank swings wildly is not. That fold-to-fold
variability is what the shaded band communicates.

![Mean rank across folds for each feature](https://user-images.githubusercontent.com/9744889/160865097-9005a1b4-d4f2-4bde-bad6-be07037fe0a7.png)

*Figure: mean RFE rank across folds for each feature (1 = most important). The
band shows the fold-to-fold variability of the rank.*

## How it works

1. **Ranking (Figure 1).** For each CV fold, fit `RFE` and record the ranking of
   every feature. Aggregate across folds into a mean rank (± a band).
2. **Scoring (Figure 2).** Order features by mean rank, then evaluate a
   cross-validated `scoring` metric using the top *n* features, for *n* from 1
   up to `max_features`. This is the classic "score vs. number of features"
   curve, but built from the cross-validated ranking.

## Install

```bash
pip install -r requirements.txt
```

Requires numpy, pandas, scipy, matplotlib and scikit-learn (≥ 1.0).

## Quick start

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from rfe_cv import rfe_cv

# df has feature columns plus a target column 'y'
features = ['x0', 'x1', 'x2', 'x3']

res = rfe_cv(df, features, 'y', RandomForestRegressor(),
             cv=5, scoring='r2')

res['rank_mean']          # mean RFE rank per feature (1 = best)
res['rank_std']           # rank std across folds
res['scores_mean']        # CV score using the n best features
res['selected_features']  # feature names selected at each step
res['fig1'], res['fig2']  # the two matplotlib figures
```

See [`example.py`](example.py) for regression, classification, comparing two
models on the same axes, and `max_features`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `df` | — | pandas DataFrame with the feature and target columns. |
| `vars_x` | — | list of feature column names. |
| `var_y` | — | target column name (`str`) or names (`list`). |
| `estimator` | — | any sklearn estimator exposing `coef_` or `feature_importances_` (required by RFE). |
| `cv` | `5` | number of folds or a cross-validation splitter. |
| `max_features` | `None` | highest number of best features in the score curve (default: all). |
| `scoring` | `'accuracy'` | sklearn scoring metric; must match the estimator (regression vs. classification). |
| `std_scaling` | `False` | standard-scale features; fitted inside each fold (no leakage). |
| `band` | `'sem'` | what the shaded band shows — see below. |
| `figs` | `None` | `[fig1, fig2]` to overlay onto existing figures. |
| `show` | `True` | draw the legend and call `plt.show()`. |
| `figsize` | `(8, 4)` | size of newly created figures. |
| `model_label` | `None` | legend label for this model. |

Returns a `dict` with keys `features`, `rank_mean`, `rank_std`, `scores_mean`,
`scores_std`, `selected_features`, `fig1`, `fig2`.

## The shaded band (`band`)

Both figures draw a band around each curve, built from the standard deviation
across folds. What that band *means* is your choice — and switching costs
nothing extra, since the std is already computed:

| `band` | Band represents | Half-width |
|---|---|---|
| `'sem'` *(default)* | Student-t confidence interval of the **mean** | `t · s / √n` |
| `'std'` | raw **fold-to-fold dispersion** | `s` |
| `'none'` | no band | — |

> ⚠️ With few folds, the std estimate `s` is itself noisy — its relative error
> is about `1 / √(2(n−1))`, roughly **35% at `cv=5`**. Read narrow bands with
> caution; increasing `cv` is the cheapest way to tighten them reliably.

## Comparing models on the same figures

Pass the figures from the first call into the next, and keep `show=False` until
the final model:

```python
res = rfe_cv(df, features, 'y', model_a, model_label='A', show=False)
rfe_cv(df, features, 'y', model_b, model_label='B',
       figs=[res['fig1'], res['fig2']], show=True)
```

## Roadmap

- [x] Separate module file and example script
- [x] Return structured results (a dict)
- [x] Figure parameters (`figsize`, overlay via `figs`)
- [x] English-only comments
- [x] Return figures instead of only plotting
- [x] Input type checks
- [x] Consistent, leak-free scaling between selection and scoring (Pipeline)
- [x] Choice of what the band represents (`band`)
- [ ] Richer example / notebook walkthrough
