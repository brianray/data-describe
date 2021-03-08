import pytest
import numpy as np
import plotly
from pmdarima.arima import AutoARIMA

from data_describe.compat import _is_dataframe, _compat
from data_describe.anomaly.detection import (
    anomaly_detection,
    AnomalyDetectionWidget,
    _pandas_compute_anomaly,
    _pandas_compute_anomalies_stats,
    _stepwise_fit_and_predict,
)


def test_not_df():
    with pytest.raises(ValueError):
        anomaly_detection("this_is_a_string")


def test_method_not_implemented(numeric_data):
    with pytest.raises(NotImplementedError):
        anomaly_detection(
            numeric_data,
            estimator="not implemented",
        )


def test_regression_and_classification_not_implemented(numeric_data):
    with pytest.raises(ValueError):
        anomaly_detection(numeric_data, target="a", date_col=None)


def test_anomaly_widget():
    ad = AnomalyDetectionWidget()
    assert hasattr(ad, "estimator"), "Anomaly Detection Widget missing estimator"
    assert hasattr(
        ad, "time_split_index"
    ), "Anomaly Detection Widget missing time_split_index"
    assert hasattr(ad, "viz_data"), "Anomaly Detection Widget missing viz_data"
    assert hasattr(ad, "ylabel"), "Anomaly Detection Widget missing ylabel"
    assert hasattr(ad, "xlabel"), "Anomaly Detection Widget missing xlabel"
    assert hasattr(ad, "target"), "Anomaly Detection Widget missing target"
    assert hasattr(ad, "date_col"), "Anomaly Detection Widget missing date_col"
    assert hasattr(ad, "n_periods"), "Anomaly Detection Widget missing n_periods"
    assert hasattr(ad, "sigma"), "Anomaly Detection Widget missing sigma"
    assert hasattr(ad, "__repr__"), "Anomaly Detection Widget missing __repr__ method"
    assert hasattr(
        ad, "_repr_html_"
    ), "Anomaly Detection Widget missing _repr_html_ method"
    assert hasattr(ad, "show"), "Anomaly Detection Widget missing show method"


@pytest.fixture
def arima_default(numeric_data, **auto_arima_args):
    return anomaly_detection(
        numeric_data.iloc[:10],
        estimator="arima",
        target="a",
        date_col="index",
        time_split_index=5,
        n_periods=1,
        viz_backend="plotly",
        **auto_arima_args,
    )


@pytest.fixture
def sklearn_default(numeric_data):
    return anomaly_detection(
        numeric_data.head(10),
        estimator="auto",
        date_col="index",
    )


def test_auto(numeric_data, sklearn_default):
    widget = sklearn_default
    isinstance(widget.estimator, list)
    assert len(widget.estimator) == 2


def test_arima_plotly(arima_default):
    figure = arima_default.show(viz_backend="plotly")
    assert isinstance(figure, plotly.graph_objs.Figure)


@pytest.fixture
def monkeypatch_auto_arima(monkeypatch):
    class mock_arima(AutoARIMA):
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict(self, X):
            return np.ones(X.shape[0])

    monkeypatch.setattr("pmdarima.arima.AutoARIMA", mock_arima)


def test_pandas_compute_anomaly_arima_univariate(
    numeric_data,
    monkeypatch_auto_arima,
    **auto_arima_args,
):
    widget_univariate = _pandas_compute_anomaly(
        data=numeric_data[["a"]].head(10),
        target="a",
        date_col="index",
        estimator=["arima"],
        n_periods=1,
        time_split_index=5,
        **auto_arima_args,
    )

    assert _is_dataframe(widget_univariate.viz_data)
    assert isinstance(
        widget_univariate, AnomalyDetectionWidget
    ), "Fit object was not a ClusterWidget"

    assert hasattr(widget_univariate, "estimator"), "Missing pmdarima auto_arima"
    assert hasattr(widget_univariate, "input_data"), "Missing input data"


def test_pandas_compute_anomaly_arima_exogenous(
    numeric_data,
    monkeypatch_auto_arima,
    **auto_arima_args,
):
    widget = _pandas_compute_anomaly(
        data=numeric_data[["a", "b"]].head(10),
        target="a",
        date_col="index",
        estimator=["arima"],
        n_periods=1,
        time_split_index=5,
        **auto_arima_args,
    )

    assert _is_dataframe(widget.viz_data)
    assert isinstance(
        widget, AnomalyDetectionWidget
    ), "Fit object was not a ClusterWidget"

    assert hasattr(widget, "estimator"), "Missing pmdarima auto_arima"
    assert hasattr(widget, "input_data"), "Missing input data"


def test_pandas_compute_anomalies_stats(numeric_data, arima_default):
    ad = arima_default
    assert isinstance(
        ad.show(viz_backend="plotly"), plotly.graph_objs.Figure
    ), "Default show() didn't return a plotly object"

    predictions_df, estimator = _stepwise_fit_and_predict(
        train=numeric_data[["a"]].head(5),
        test=numeric_data[["a"]].head(5),
        estimator=ad.estimator,
        target="a",
    )
    assert _is_dataframe(predictions_df)
    assert predictions_df.shape == (5, 2)
    assert isinstance(estimator, _compat["pmdarima"].arima.ARIMA)  # type: ignore

    predictions_df = _pandas_compute_anomalies_stats(predictions_df, window=1, sigma=2)
    assert _is_dataframe(predictions_df)
    assert predictions_df.shape == (5, 17)
    assert len(predictions_df.columns) == len(ad.viz_data.columns)
