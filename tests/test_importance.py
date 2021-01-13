import pytest
import matplotlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import data_describe as dd

matplotlib.use("Agg")


@pytest.mark.base
@pytest.mark.xfail(reason="Not implemented for modin")
def test_importance(compute_backend_df):
    importance_vals = dd.importance(compute_backend_df, "d", return_values=True)
    assert len(importance_vals) == compute_backend_df.shape[1] - 1 - 1


@pytest.mark.base
@pytest.mark.xfail(reason="Not implemented for modin")
def test_importance_num_only(data, compute_backend_df):
    data = compute_backend_df.select_dtypes(["number"])
    rfr = RandomForestRegressor(random_state=1)
    assert isinstance(
        dd.importance(data, "a", estimator=rfr, return_values=True), np.ndarray
    ), "Importance values not a numpy array"


@pytest.mark.base
def test_importance_cat_only(data, compute_backend_df):
    num_columns = compute_backend_df.select_dtypes(["number"]).columns.values
    data = data[[c for c in data.columns if c not in num_columns]]
    assert (
        len(dd.importance(data, "d", return_values=True)) == data.shape[1] - 2
    ), "Wrong size of importance values"  # f is null column


@pytest.mark.base
@pytest.mark.xfail(reason="Not implemented for modin")
def test_importance_preprocess(data, compute_backend_df):
    def pre(df, target):
        y = df[target]
        df = df.drop(target, axis=1)
        x_num = df.select_dtypes(["number"])
        x_num = x_num.fillna("-1")
        x_cat = df[[c for c in df.columns if c not in x_num.columns]].astype(str)
        x_cat = x_cat.fillna("")
        x_cat_encoded = x_cat.apply(LabelEncoder().fit_transform)
        X = pd.concat([x_num, x_cat_encoded], axis=1)
        return X, y

    fig = dd.importance(compute_backend_df, "d", preprocess_func=pre)
    assert isinstance(fig, matplotlib.artist.Artist)


@pytest.mark.base
@pytest.mark.xfail(reason="Not implemented for modin")
def test_top_feature(compute_backend_df):
    fig = dd.importance(compute_backend_df, "d", top_features=1)
    assert isinstance(fig, matplotlib.artist.Artist)
