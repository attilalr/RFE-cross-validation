# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import t

from sklearn.base import clone, is_classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE
from sklearn.model_selection import check_cv, cross_val_score


def rfe_cv(df, vars_x, var_y, estimator, cv=5, max_features=None,
           scoring='accuracy', std_scaling=False, band='sem',
           figs=None, show=True, figsize=(8, 4), model_label=None):
    """Recursive Feature Elimination evaluated across cross-validation folds.

    RFE is run inside every CV fold to obtain a *ranking* of the features per
    fold. Aggregating those rankings across folds shows how stable each feature
    is in the importance ranking (Figure 1). A second pass evaluates a
    cross-validated score using the ``n`` best features, for ``n`` from 1 up to
    ``max_features`` (Figure 2).

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the feature and target columns.
    vars_x : list of str
        Names of the feature columns.
    var_y : str or list of str
        Name(s) of the target column(s).
    estimator : sklearn estimator
        Model exposing ``coef_`` or ``feature_importances_`` (required by RFE).
    cv : int or cross-validation generator, default 5
        Cross-validation splitting strategy.
    max_features : int, optional
        Highest number of best features to evaluate in the score curve.
        Defaults to all features.
    scoring : str or callable, default 'accuracy'
        Scoring metric; must be compatible with the estimator
        (regression vs. classification).
    std_scaling : bool, default False
        Standard-scale the features. Scaling is fitted inside each fold (both
        for RFE and for scoring) to avoid data leakage.
    band : {'sem', 'std', 'none'}, default 'sem'
        What the shaded band around each curve represents:

        - ``'sem'``: Student-t confidence interval of the mean across folds
          (``t * s / sqrt(n)``), i.e. how precisely the mean is estimated.
        - ``'std'``: raw standard deviation across folds (``s``), i.e. how
          much the rank/score varies from fold to fold.
        - ``'none'``: no band.

        Both use the standard deviation already computed across folds, so
        switching has no extra computational cost. With few folds ``s`` itself
        is a noisy estimate (relative error ~ ``1 / sqrt(2*(n-1))``, ~35% for
        ``n=5``), so read narrow bands with caution.
    figs : list [fig1, fig2], optional
        Existing figures to overlay onto (e.g. to compare several models on the
        same axes). When omitted, new figures are created.
    show : bool, default True
        Draw the legend and call ``plt.show()``. Set to ``False`` when
        overlaying several models before the final call.
    figsize : tuple, default (8, 4)
        Size of newly created figures.
    model_label : str, optional
        Label used in the legend for this model.

    Returns
    -------
    dict
        Keys: ``features``, ``rank_mean``, ``rank_std``, ``scores_mean``,
        ``scores_std``, ``selected_features`` (list, one entry per step),
        ``fig1``, ``fig2``.
    """

    assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'

    if isinstance(var_y, list):
        y_cols = var_y
    elif isinstance(var_y, str):
        y_cols = [var_y]
    else:
        raise TypeError('var_y must be str or list.')

    if not isinstance(vars_x, list):
        raise TypeError('vars_x must be a list of column names.')

    if band not in ('sem', 'std', 'none'):
        raise ValueError("band must be one of 'sem', 'std' or 'none'.")

    # drop rows with NaNs in any of the used columns
    df_ = df[vars_x + y_cols].dropna()
    if df_.empty:
        raise ValueError('No rows left after dropping NaNs.')

    y_target = df_[var_y]

    # scoring estimator: wrap in a Pipeline so scaling is applied consistently
    # (and without leakage) inside cross_val_score.
    if std_scaling:
        scoring_estimator = Pipeline(
            [('scaler', StandardScaler()), ('estimator', clone(estimator))]
        )
    else:
        scoring_estimator = estimator

    # validate that the scoring matches the estimator (regression vs. classif.)
    try:
        cross_val_score(scoring_estimator, df_[vars_x], y_target,
                        scoring=scoring, cv=2)
    except Exception as e:
        raise ValueError(
            'Check that the scoring matches the estimator '
            '(regression vs. classification case).'
        ) from e

    kfold = check_cv(cv=cv, y=y_target, classifier=is_classifier(estimator))
    n_splits = kfold.get_n_splits()

    results = {
        'features': list(vars_x),
        'scores_mean': [],
        'scores_std': [],
        'selected_features': [],
    }

    # ----------------------- RFE per fold ----------------------------------- #
    strat_y = y_target if is_classifier(estimator) else None
    fold_rankings = []  # one ranking vector per fold -> (n_splits, n_features)

    for train_index, _ in kfold.split(df_, y=strat_y):
        X_train = df_[vars_x].values[train_index].copy()
        if len(vars_x) == 1:
            X_train = X_train.reshape(-1, 1)
        y_train = df_[var_y].values[train_index]

        if std_scaling:
            X_train = StandardScaler().fit_transform(X_train)

        selector = RFE(estimator, n_features_to_select=1, step=1)
        selector.fit(X_train, y_train)
        fold_rankings.append(selector.ranking_)

    ranking = np.asarray(fold_rankings)          # (n_splits, n_features)
    rank_mean = ranking.mean(axis=0)
    rank_std = ranking.std(axis=0)
    results['rank_mean'] = rank_mean
    results['rank_std'] = rank_std

    # multiplier applied to the across-folds std to obtain the band half-width
    if band == 'sem':
        # 68.27% confidence interval of the mean (1 std), Student-t corrected
        alpha = 0.3173
        degrees_of_freedom = max(n_splits - 1, 1)
        band_factor = t.ppf(1 - alpha / 2, degrees_of_freedom) / np.sqrt(n_splits)
    elif band == 'std':
        band_factor = 1.0
    else:  # 'none'
        band_factor = None

    # --------------------------- Figure 1 ----------------------------------- #
    own_figs = figs is None
    if own_figs:
        fig1, ax1 = plt.subplots(figsize=figsize)
        fig2, ax2 = plt.subplots(figsize=figsize)
    else:
        fig1, fig2 = figs
        ax1, ax2 = fig1.axes[0], fig2.axes[0]

    line1, = ax1.plot(rank_mean, 'o-', label=model_label)
    if band_factor is not None:
        ax1.fill_between(
            np.arange(len(rank_mean)),
            rank_mean + band_factor * rank_std,
            rank_mean - band_factor * rank_std,
            alpha=0.15,
            color=line1.get_color(),
        )
    ax1.set_xticks(np.arange(len(rank_mean)))
    ax1.set_xticklabels(vars_x, rotation=-40)
    ax1.set_title(f'mean ranking for each feature when modeling {var_y}')
    ax1.set_ylabel('mean RFE rank (1 = best)')

    # ------------------ score using the n best features --------------------- #
    n_features = len(vars_x)
    if not max_features:
        max_features = n_features

    # ascending rank -> best (lowest rank) first; stable sort breaks ties by
    # original order, so no random jitter is needed.
    order = np.argsort(rank_mean, kind='stable')

    for i in range(min(max_features, n_features)):
        selected_idx = order[:i + 1]
        selected_col_names = [str(vars_x[j]) for j in selected_idx]
        results['selected_features'].append(selected_col_names)
        print(f'{i + 1} selected features: {selected_col_names}')

        score_vec = cross_val_score(
            scoring_estimator,
            df_[selected_col_names],
            y_target,
            scoring=scoring,
            cv=check_cv(cv=cv, y=y_target, classifier=is_classifier(estimator)),
        )
        results['scores_mean'].append(score_vec.mean())
        results['scores_std'].append(score_vec.std())

    scores_mean = np.asarray(results['scores_mean'])
    scores_std = np.asarray(results['scores_std'])
    n_steps = np.arange(len(scores_mean)) + 1

    # --------------------------- Figure 2 ----------------------------------- #
    line2, = ax2.plot(n_steps, scores_mean, 'o-', label=model_label)
    if band_factor is not None:
        ax2.fill_between(
            n_steps,
            scores_mean + band_factor * scores_std,
            scores_mean - band_factor * scores_std,
            alpha=0.15,
            color=line2.get_color(),
        )
    ax2.locator_params(axis='x', nbins=len(scores_mean))
    ax2.set_title(f'modeling {var_y}')
    ax2.set_ylabel(str(scoring))
    ax2.set_xlabel('number of best features')

    if show:
        if model_label is not None:
            ax1.legend()
            ax2.legend()
        plt.show()

    results['fig1'] = fig1
    results['fig2'] = fig2
    return results


if __name__ == "__main__":
    print(''' - Module rfe_cv -

 Use:
 import rfe_cv
 and call rfe_cv.rfe_cv()

                 or
 from rfe_cv import rfe_cv
 and call rfe_cv()
        ''')
