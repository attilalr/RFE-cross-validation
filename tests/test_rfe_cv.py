import matplotlib
matplotlib.use('Agg')  # headless: no display needed for tests

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from rfe_cv import rfe_cv


def _make_df(with_nan=False, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(120, 4)
    # 'a' strong, 'c' moderate, 'b'/'d' irrelevant
    y = X @ np.array([3.0, 0.1, -2.0, 0.0]) + rng.randn(120)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
    df['y'] = y
    if with_nan:
        df.loc[0, 'a'] = np.nan
    return df


def _regressor():
    return RandomForestRegressor(n_estimators=30, random_state=0)


def test_returns_expected_keys():
    df = _make_df()
    res = rfe_cv(df, ['a', 'b', 'c', 'd'], 'y', _regressor(),
                 scoring='r2', show=False)
    for key in ('features', 'rank_mean', 'rank_std', 'scores_mean',
                'scores_std', 'selected_features', 'fig1', 'fig2'):
        assert key in res


def test_rank_std_is_not_the_mean():
    # regression guard for the original bug where std == mean
    df = _make_df()
    res = rfe_cv(df, ['a', 'b', 'c', 'd'], 'y', _regressor(),
                 scoring='r2', show=False)
    assert not np.allclose(res['rank_mean'], res['rank_std'])
    assert np.all(res['rank_std'] >= 0)


def test_most_important_feature_selected_first():
    df = _make_df()
    res = rfe_cv(df, ['a', 'b', 'c', 'd'], 'y', _regressor(),
                 scoring='r2', show=False)
    assert res['selected_features'][0] == ['a']


def test_max_features_limits_score_curve():
    df = _make_df()
    res = rfe_cv(df, ['a', 'b', 'c', 'd'], 'y', _regressor(),
                 scoring='r2', max_features=2, show=False)
    assert len(res['scores_mean']) == 2
    assert len(res['selected_features']) == 2


def test_nan_rows_are_dropped():
    df = _make_df(with_nan=True)
    # must not raise; NaN row is dropped internally
    res = rfe_cv(df, ['a', 'b', 'c', 'd'], 'y', _regressor(),
                 scoring='r2', show=False)
    assert len(res['scores_mean']) == 4


@pytest.mark.parametrize('band', ['sem', 'std', 'none'])
def test_band_options(band):
    df = _make_df()
    res = rfe_cv(df, ['a', 'b', 'c', 'd'], 'y', _regressor(),
                 scoring='r2', band=band, show=False)
    n_fill = len(res['fig1'].axes[0].collections)
    assert n_fill == (0 if band == 'none' else 1)


def test_invalid_band_raises():
    df = _make_df()
    with pytest.raises(ValueError):
        rfe_cv(df, ['a', 'b', 'c', 'd'], 'y', _regressor(),
               scoring='r2', band='nope', show=False)


def test_classification_runs():
    rng = np.random.RandomState(1)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=150, n_features=5, random_state=rng)
    cols = [f'f{i}' for i in range(5)]
    df = pd.DataFrame(X, columns=cols)
    df['y'] = y
    clf = RandomForestClassifier(n_estimators=30, random_state=0)
    res = rfe_cv(df, cols, 'y', clf, scoring='accuracy', show=False)
    assert len(res['scores_mean']) == 5


def test_scoring_estimator_mismatch_raises():
    # accuracy on a regressor should be rejected up front
    df = _make_df()
    with pytest.raises(ValueError):
        rfe_cv(df, ['a', 'b', 'c', 'd'], 'y', _regressor(),
               scoring='accuracy', show=False)
