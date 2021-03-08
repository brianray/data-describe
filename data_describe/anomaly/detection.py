from typing import Optional

import numpy as np
import pandas as pd
import sklearn
import plotly.graph_objs as go
import plotly.offline as po
from plotly.subplots import make_subplots

from data_describe.backends import _get_compute_backend, _get_viz_backend
from data_describe.compat import _is_dataframe, _requires, _in_notebook, _compat
from data_describe._widget import BaseWidget


class AnomalyDetectionWidget(BaseWidget):
    """Container for anomaly calculations and visualization.

    This class (object) is returned from the ``anomaly_detection`` function. The
    attributes documented below can be accessed or extracted.

    Attributes:
        estimator: The anomaly detection estimator/model.
        time_split_index (str, optional): The index to split the input data into a training set and testing set.
            Note: Data is not shuffled.
        viz_data (DataFrame): The data used for the default visualization.
        xlabel (str): The x-axis label for the anomaly plot.
        ylabel (str): The y-axis label for the anomaly plot.
        target (str, optional): The target column.
        date_col (str, optional): The date column.
        n_periods (int, optional): The number of periods for timeseries window.
    """

    def __init__(
        self,
        estimator=None,
        viz_data=None,
        time_split_index=None,
        n_periods=None,
        **kwargs,
    ):
        """Anomaly Detection.

        Args:
            estimator: The anomaly detection estimator/model.
            viz_data (DataFrame): The data used for the default visualization.
            time_split_index (str, optional): The index to split the input data into a training set and testing set.
                Note: Data is not shuffled.
            n_periods (int, optional): The number of periods for timeseries window.
            **kwargs: Keyword arguments.
        """
        super(AnomalyDetectionWidget, self).__init__(**kwargs)
        self.estimator = estimator
        self.time_split_index = time_split_index
        self.viz_data = viz_data
        self.n_periods = n_periods
        self.input_data = None
        self.xlabel = None
        self.ylabel = None
        self.target = None
        self.date_col = None
        self.sigma = None

    def __str__(self):
        return "data-describe Anomaly Detection Widget"

    def __repr__(self):
        return "Anomaly Widget"

    def show(self, viz_backend=None, **kwargs):
        """The default display for this output.

        Displays the anomalies, projected as a lineplot or scatterplot, with detected anomalies as red markers

        Args:
            viz_backend: The visualization backend.
            **kwargs: Keyword arguments.

        Raises:
            ValueError: Data to visualize is missing / not calculated.

        Returns:
            The anomaly plot.
        """
        backend = viz_backend or self.viz_backend

        if self.viz_data is None:
            raise ValueError("Could not find data to visualize.")

        return _get_viz_backend(backend).viz_plot_anomaly(
            predictions_df=self.viz_data,
            n_periods=self.n_periods,
            xlabel=self.date_col,
            ylabel=self.target,
            estimator=self.estimator,
            **kwargs,
        )


def anomaly_detection(
    data,
    target: Optional[str] = None,
    date_col: Optional[str] = None,
    estimator=None,
    time_split_index: Optional[int] = None,
    n_periods: Optional[int] = None,
    sigma: float = 2.0,
    contamination="auto",
    compute_backend: Optional[str] = None,
    viz_backend: Optional[str] = None,
    **kwargs,
):
    """Identify and mark anamolies.

    This feature identifies anomalies in timeseries and tabular data using multiple approaches (supervised, unsupervised, and statistical)
    and then projects the data onto a plot for visualization.

    Args:
        data (DataFrame): The dataframe.
        target (str, optional): The target column. Defaults to None. If target is None, unsupervised methods are used.
        date_col (str, optional): The datetime column. If date_col is specified, the data will be treated as timeseries.
            If the data does not contain datetimes, but contains sequences, set date_col = 'index'.
        estimator (optional): Fitted or instantiated estimator with a .predict() and .fit() method. Defaults to None.
            If estimator is instantiated but not fitted, time_split_index must be specified.
        time_split_index (int, optional): Index to split the data into a train set and a test set. Defaults to None.
            time_split_index must be specified if estimator is instantiated but not fitted.
        n_periods (int, optional): Size of the moving window. This is the number of observations used for calculating the statistic. Defaults to None.
        sigma (float, optional): The standard deviation requirement for identifying anomalies. Defaults to 2.
        contamination (float): The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
            Used when fitting to define the threshold on the scores of the samples. If ‘auto’, the threshold is determined as in the original paper.
            If float, the contamination should be in the range [0, 0.5].
        compute_backend (str, optional): The compute backend.
        viz_backend (str, optional): The visualization backend.
        **kwargs: Keyword arguments.

    Return:
        AnomalyWidget
    """
    if not _is_dataframe(data):
        raise ValueError("Data frame required")

    if estimator is None or estimator == "arima":
        if not target:
            raise ValueError("Please specify the target column for the ARIMA model.")
        estimator = ["arima"]

    elif estimator == "auto":
        iforest = sklearn.ensemble.IsolationForest(
            random_state=1, contamination=contamination, n_jobs=-1
        )
        lof = sklearn.neighbors.LocalOutlierFactor(
            contamination=contamination, novelty=True, n_jobs=-1
        )

        estimator = [iforest, lof]

    elif isinstance(estimator, str):
        raise NotImplementedError(
            f"{estimator} is not implementeed. Please choose between 'arima' and 'auto'."
        )

    elif not hasattr(estimator, "predict") and not hasattr(estimator, "fit"):
        raise AttributeError(
            f"{estimator} does not contain the 'predict' or 'fit' method."
        )

    anomalywidget = _get_compute_backend(compute_backend, data).compute_anomaly(
        data=data,
        target=target,
        date_col=date_col,
        estimator=estimator,
        n_periods=n_periods,
        time_split_index=time_split_index,
        sigma=sigma,
        **kwargs,
    )

    return anomalywidget


def _pandas_compute_anomaly(
    data,
    estimator,
    target: Optional[str] = None,
    date_col: Optional[str] = None,
    n_periods=1,
    time_split_index: Optional[int] = None,
    **kwargs,
):
    """Backend implementation of anomaly detection.

    Args:
        data (DataFrame): The dataframe.
        estimator (list): List of model objects.
        target (str, optional): The target column. Defaults to None. If target is None, unsupervised methods are used.
        date_col (str, optional): Datetime column if data is timeseries. Defaults to None. If data is timeseries, date_col must be specified.
            If estimator is instantiated but not fitted, time_split_index must be specified.
        n_periods (int, optional): Number of periods for timeseries anomaly detection. Default is 1.
        time_split_index (int, optional): Index to split the data into a train set and a test set. Defaults to None.
            time_split_index must be specified if estimator is instantiated but not fitted.
        **kwargs: Keyword arguments.

    Raises:
        ValueError: If method is not implemented.

    Returns:
        AnomalyDetectionWidget

    """
    # Timeseries indicator
    if not date_col:
        raise ValueError("Please specify data_col for timeseries anomaly detection.")

    if date_col != "index":
        data.index = pd.to_datetime(data[date_col])

    numeric_data = data.select_dtypes("number")

    if not numeric_data.index.is_monotonic_increasing:
        numeric_data.sort_index(ascending=True, inplace=True)

    trained_models: list = []  # Initialize list to store the trained model objects
    model_results: dict = (
        {}
    )  # Initialize dictionary to store the model results. {<name of model>: predictions}
    predictions_df = None

    train = numeric_data.iloc[:time_split_index] if target else numeric_data
    test = numeric_data.iloc[time_split_index:] if target else None

    if "arima" in estimator:
        predictions_df, clf = _stepwise_fit_and_predict(
            train=train,
            test=test,
            target=target,
            **kwargs,
        )
        predictions_df = _pandas_compute_anomalies_stats(
            predictions_df, window=n_periods
        )
        trained_models.append(clf)
        model_results["arima"] = (
            predictions_df["impact"].map(lambda x: -1 if x == 7 else 1).tolist()
        )

    for model in _compat["tqdm"].tqdm(  # type: ignore
        estimator,
        desc="Fitting models",
    ):
        if model != "arima":
            unsupervised_fit_and_predict(
                model=model,
                train_data=train,
                test_data=test,
                model_results=model_results,
                trained_models=trained_models,
            )

    if predictions_df is None:
        predictions_df = pd.DataFrame(model_results)
        if not target:
            predictions_df.index = train.index

        else:
            predictions_df["actuals"] = test[target].tolist()
            predictions_df.index = test.index

    return AnomalyDetectionWidget(
        estimator=trained_models,
        viz_data=predictions_df,
        time_split_index=time_split_index,
        target=target,
        n_periods=n_periods,
    )


@_requires("tqdm")
def _stepwise_fit_and_predict(train, test, target, **kwargs):
    """Perform stepwise fit and predict for timeseries data.

    Args:
        train (Series): The training data.
        test (Series): The testing data.
        target (str): The target variable name.
        **kwargs: Keyword arguments.

    Returns:
        predictions_df: DataFrame containing the ground truth, predictions, and indexed by the datetime.

    """
    # check if train data has exogenous variables
    if train.shape[1] == 1:
        estimator = _compat["pmdarima"].arima.auto_arima(  # type: ignore
            y=train[[target]],
            x=None,
            random_state=1,
            **kwargs,
        )

        history = train[target].tolist()
        predictions = list()

        for idx in _compat["tqdm"].tqdm(  # type: ignore
            test[target].index,
            desc="Fitting ARIMA model",
        ):
            estimator.fit(history)
            output = estimator.predict(n_periods=1)
            predictions.append(output[0])
            obs = test[target][idx]
            history.append(obs)

    else:
        estimator = _compat["pmdarima"].arima.auto_arima(  # type: ignore
            y=train[[target]],
            X=train.drop(columns=[target]),
            random_state=1,
            **kwargs,
        )

        history_target = train[target].tolist()
        history_features = (
            train.drop(columns=[target]).to_numpy(dtype="object").tolist()
        )

        predictions = list()
        for idx in _compat["tqdm"].tqdm(  # type: ignore
            test[target].index,
            desc="Fitting ARIMA model",
        ):
            record = test.drop(columns=[target]).loc[idx, :]
            estimator.fit(history_target, history_features)
            output = estimator.predict(n_periods=1, X=[record])
            predictions.append(output[0])
            history_features.append(record)
            history_target.append(test[target][idx])

    predictions_df = pd.DataFrame()
    predictions_df["actuals"] = test[target]
    predictions_df["predictions"] = predictions

    return predictions_df, estimator


def _pandas_compute_anomalies_stats(predictions_df, window, sigma=2.0):
    """Detects anomalies based on the statistical profiling of the residuals (actuals - predicted).

    The rolling mean and rolling standard deviation is used to identify points that are more than n standard deviations away from the mean.

    Args:
        predictions_df (DataFrame): The dataframe containing the ground truth and predictions.
        window (int): Size of the moving window. This is the number of observations used for calculating the statistic. Each window will be a fixed size.
        sigma (float). The standard deviation requirement for anomalies.

    Raises:
        ValueError: If 'actuals' and 'predictions' are not found in predictions_df.

    Returns:
        predictions_df: The dataframe containing the predictions and computed statistics.

    """
    predictions_df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    predictions_df.fillna(0, inplace=True)
    predictions_df["error"] = predictions_df["actuals"] - predictions_df["predictions"]
    predictions_df["percentage_change"] = (
        predictions_df["error"] / predictions_df["actuals"]
    ) * 100
    predictions_df["meanval"] = predictions_df["error"].rolling(window=window).mean()
    predictions_df["deviation"] = predictions_df["error"].rolling(window=window).std()
    predictions_df["-3s"] = predictions_df["meanval"] - (
        sigma * predictions_df["deviation"]
    )
    predictions_df["3s"] = predictions_df["meanval"] + (
        sigma * predictions_df["deviation"]
    )
    predictions_df["-2s"] = predictions_df["meanval"] - (
        1.75 * predictions_df["deviation"]
    )
    predictions_df["2s"] = predictions_df["meanval"] + (
        1.75 * predictions_df["deviation"]
    )
    predictions_df["-1s"] = predictions_df["meanval"] - (
        1.5 * predictions_df["deviation"]
    )
    predictions_df["1s"] = predictions_df["meanval"] + (
        1.5 * predictions_df["deviation"]
    )
    cut_list = predictions_df[
        ["error", "-3s", "-2s", "-1s", "meanval", "1s", "2s", "3s"]
    ]

    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)

    if not isinstance(predictions_df.index, pd.core.indexes.datetimes.DatetimeIndex):
        predictions_df.reset_index(inplace=True)

    predictions_df["impact"] = [
        (lambda x: np.where(cut_sort == predictions_df["error"][x])[1][0])(x)
        for x in range(len(predictions_df["error"]))
    ]
    severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
    region = {
        0: "NEGATIVE",
        1: "NEGATIVE",
        2: "NEGATIVE",
        3: "NEGATIVE",
        4: "POSITIVE",
        5: "POSITIVE",
        6: "POSITIVE",
        7: "POSITIVE",
    }
    predictions_df["color"] = predictions_df["impact"].map(severity)
    predictions_df["region"] = predictions_df["impact"].map(region)
    predictions_df["anomaly_points"] = np.where(
        predictions_df["color"] == 3, predictions_df["error"], np.nan
    )

    return predictions_df


@_requires("plotly")
def _plotly_viz_anomaly(
    predictions_df,
    estimator,
    n_periods,
    xlabel="Time",
    ylabel="Target",
    marker_color="red",
):
    """Plotly interactive plots for anomaly detection.

    Args:
        predictions_df (DataFrame): The dataframe containing predictions and statistics.
        estimator (list): List containing trained models.
        n_periods (int): The number of periods to make predictions on. Required for ARIMA based models.
        xlabel (str, optional): The x-axis label. Defaults to "Time".
        ylabel (str, optional): The y-axis label. Defaults to "Target".
        marker_color (str, optional): The color to mark anomalies. Defaults to "red".

    Returns:
        fig: The plotly figure.
    """
    if "actuals" not in predictions_df.columns:
        raise ValueError("Could not plot anomalies. Please specify a target.")

    if isinstance(estimator[0], _compat["pmdarima"].arima.ARIMA):  # type: ignore
        # Note: auto_arima is a wrapper function for pmdarima.arima.Arima
        fig = _plotly_viz_arima(
            predictions_df=predictions_df,
            n_periods=n_periods,
            xlabel=xlabel,
            ylabel=ylabel,
            marker_color=marker_color,
        )

    else:
        fig = _plotly_viz_auto(
            predictions_df,
            n_periods=n_periods,
            marker_color=marker_color,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    return fig


def unsupervised_fit_and_predict(
    model, train_data, test_data, model_results, trained_models, **kwargs
):
    """Train and fit unsupervised models.

    Args:
        model: The estimator object.
        train_data (DataFrame): The train data.
        test_data (DataFrame): The test data.
        model_results (dict): Dictionary to store model results. i.e. {<model name>: predictions}
        trained_models (list): List containing fitted models.
    """
    model_key = str(model).split("(")[0]
    model.fit(train_data, **kwargs)
    if test_data is None:
        preds = model.predict(train_data).tolist()

    else:
        preds = model.predict(test_data).tolist()

    model_results[model_key] = preds
    trained_models.append(model)


def _plotly_viz_auto(
    predictions_df,
    n_periods,
    title="Anomaly Detection Plots",
    marker_color="red",
    xlabel="Time",
    ylabel="Target",
):
    """Plotly visualization for multiple models.

    Args:
        predictions_df (DataFrame: DataFrame containing predictions and computed statistics.
        n_periods (int): The size of the window used by ARIMA. Since ARIMA looksback n periods to make a prediction,
            the first n records will contain null values. n_periods is used to slice the null values, thus the plot for ARIMA will contain n less records than other models.
        title (str, optional): Title of the plot. Defaults to "Anomaly Detection Plots".
        marker_color (str, optional): The marker color for anomalies. Defaults to "red".
        xlabel (str, optional): The x-axis label. Defaults to "Time".
        ylabel (str, optional): The y-axis label. Defaults to "Target".

    Returns:
        fig: Plotly figure
    """
    fig = make_subplots(
        rows=(predictions_df.shape[1] - 1),
        cols=1,
        x_title=xlabel,
        y_title=ylabel,
        shared_xaxes=True,
        subplot_titles=predictions_df.columns[:-1],
    )

    for idx, model in enumerate(predictions_df.columns[:-1]):
        anomaly = predictions_df[predictions_df[model] == -1]

        if model == "arima":
            fig.add_trace(
                go.Scatter(
                    x=predictions_df.index[:-1][(n_periods - 1) :],
                    y=predictions_df.actuals[:-1][(n_periods - 1) :],
                    name="actual",
                    mode="lines",
                    showlegend=False,
                    marker=dict(size=12, line=dict(width=1), color="blue"),
                ),
                row=idx + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=anomaly.index,
                    y=anomaly["actuals"],
                    text=model,
                    showlegend=False,
                    name=model,
                    mode="markers",
                    marker=dict(size=7, color=marker_color),
                ),
                row=idx + 1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=predictions_df.index[:-1],
                    y=predictions_df.actuals[:-1],
                    name="actual",
                    mode="lines",
                    showlegend=False,
                    marker=dict(size=12, line=dict(width=1), color="blue"),
                ),
                row=idx + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=anomaly.index[:-1],
                    y=anomaly["actuals"][:-1],
                    text=model,
                    showlegend=False,
                    name=model,
                    mode="markers",
                    marker=dict(size=7, color=marker_color),
                ),
                row=idx + 1,
                col=1,
            )

    return fig


def _plotly_viz_arima(
    predictions_df, n_periods, xlabel="Time", ylabel="Target", marker_color="red"
):
    """Plotly interactive plot for anomalies detected by ARIMA.

    Args:
        predictions_df (DataFrame): The dataframe containing the ground truth, predictions, and statistics.
        n_periods (int): The number of periods to remove from the start of test set. Since ARIMA looksback n periods to make a prediction,
            the first n records will contain null values. n_periods is used to slice the null values.
        xlabel (str, optional):  The x-axis label. Defaults to "Time".
        ylabel (str, optional): The y-axis label. Defaults to "Target".
        marker_color (str, optional): The color to mark anomalies. Defaults to "red".

    Returns:
        fig: The Plotly figure for the ARIMA model.

    """
    lookback = n_periods - 1
    predictions_df = predictions_df.iloc[lookback:, :]
    bool_array = abs(predictions_df["anomaly_points"]) > 0
    actuals = predictions_df["actuals"][-len(bool_array) :]
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan

    anomalies = go.Scatter(
        name="Anomaly",
        x=predictions_df.index,
        y=predictions_df["anomaly_points"],
        xaxis="x1",
        yaxis="y1",
        mode="markers",
        marker=dict(color="red", size=11, line=dict(color="red", width=2)),
    )

    upper_bound = go.Scatter(
        hoverinfo="skip",
        x=predictions_df.index,
        showlegend=False,
        xaxis="x1",
        yaxis="y1",
        y=predictions_df["3s"],
        marker=dict(color="#444"),
        line=dict(color=("rgb(23, 96, 167)"), width=2, dash="dash"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )

    lower_bound = go.Scatter(
        name="Confidence Interval",
        x=predictions_df.index,
        xaxis="x1",
        yaxis="y1",
        y=predictions_df["-3s"],
        marker=dict(color="#444"),
        line=dict(color=("rgb(23, 96, 167)"), width=2, dash="dash"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )

    actuals = go.Scatter(
        name="Actuals",
        x=predictions_df.index,
        y=predictions_df["actuals"],
        xaxis="x2",
        yaxis="y2",
        marker=dict(size=12, line=dict(width=1), color="blue"),
    )

    predicted = go.Scatter(
        name="Predicted",
        x=predictions_df.index,
        y=predictions_df["predictions"],
        xaxis="x2",
        yaxis="y2",
        marker=dict(size=12, line=dict(width=1), color="orange"),
    )

    # create plot for error...
    error = go.Scatter(
        name="Error",
        x=predictions_df.index,
        y=predictions_df["error"],
        xaxis="x1",
        yaxis="y1",
        marker=dict(size=12, line=dict(width=1), color="red"),
        text="Error",
    )

    anomalies_map = go.Scatter(
        name="Anomaly",
        showlegend=False,
        x=predictions_df.index,
        y=anomaly_points,
        mode="markers",
        xaxis="x2",
        yaxis="y2",
        marker=dict(color="red", size=11, line=dict(color="red", width=2)),
    )

    moving_average = go.Scatter(
        name="Moving Average",
        x=predictions_df.index,
        y=predictions_df["meanval"],
        xaxis="x1",
        yaxis="y1",
        marker=dict(size=12, line=dict(width=1), color="green"),
        text="Moving average",
    )

    axis = dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        ticklen=4,
        gridcolor="#ffffff",
        tickfont=dict(size=10),
    )

    layout = dict(
        width=1000,
        height=865,
        autosize=False,
        showlegend=True,
        xaxis1=dict(axis, **dict(domain=[0, 1], anchor="y1", showticklabels=True)),
        xaxis2=dict(axis, **dict(domain=[0, 1], anchor="y2", showticklabels=True)),
        yaxis1=dict(
            axis,
            **dict(domain=[2 * 0.21 + 0.20 + 0.09, 1], anchor="x1", hoverformat=".2f"),
        ),
        yaxis2=dict(
            axis,
            **dict(
                domain=[0.21 + 0.12, 2 * 0.31 + 0.02], anchor="x2", hoverformat=".2f"
            ),
        ),
    )

    fig = go.Figure(
        data=[
            anomalies,
            anomalies_map,
            upper_bound,
            lower_bound,
            actuals,
            predicted,
            moving_average,
            error,
        ],
        layout=layout,
    )
    fig.update_layout(
        title="ARIMA Anomalies",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend_title="Legend",
    )
    if _in_notebook():
        po.init_notebook_mode(connected=True)
        return po.iplot(fig)
    else:
        return fig
