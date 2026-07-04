# RFE-cross-validation
RFE (Recursive Feature Elimination) with cross validation

RFECV from scikit-learn optimizes the number of best features [1] but is usual the desire to know the selected variables for other cases. The RFE function [2] addresses this issue but is it not cross-validated.

A cross-validation for feature importance can give the idea of how a given feature is well established in the feature importance rank. The rank variability is shown by the standard deviation of the rank mean. As the next figure shows: 

![image](https://user-images.githubusercontent.com/9744889/160865097-9005a1b4-d4f2-4bde-bad6-be07037fe0a7.png)

Figure: Mean rank across folds for each feature.


## Usage

```python
from rfe_cv import rfe_cv

res = rfe_cv(df, feature_names, 'target', estimator,
             cv=5, scoring='r2', max_features=None, std_scaling=False,
             band='sem', show=True, model_label='my model')

res['rank_mean']          # mean RFE rank per feature (1 = best)
res['rank_std']           # rank variability across folds
res['scores_mean']        # CV score using the n best features
res['selected_features']  # feature names selected at each step
res['fig1'], res['fig2']  # the two matplotlib figures
```

To overlay several models on the same figures, pass `figs=[res['fig1'], res['fig2']]`
and `show=False` until the final call. See `example.py`.

Install dependencies with `pip install -r requirements.txt`.

TO-DO:

- ~~Separate module file and example script;~~
- ~~Prettier output / returning of results (now returns a dict);~~
- ~~Input parameters for figures;~~
- ~~Get rid of commentaries in portuguese;~~
- ~~Return of figures instead of plotting;~~
- ~~Better check of inputs (types);~~
- ~~Consistent scaling between selection and scoring (via Pipeline);~~
- ~~Consider the variability of the std deviation of the mean (`band='sem'|'std'|'none'` + docstring caveat);~~
- Richer example.

The shaded band is controlled by `band`: `'sem'` (Student-t confidence
interval of the mean, the default), `'std'` (raw fold-to-fold dispersion), or
`'none'`. With few folds the std estimate is itself noisy (~35% relative error
at `cv=5`), so read narrow bands with caution — increasing `cv` is the cheapest
way to tighten it.

[1] http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

[2] https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
