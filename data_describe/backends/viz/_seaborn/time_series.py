import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from data_describe.config._config import get_option


def viz_plot_time_series(df, col, result=None, decompose=False):
    """Create timeseries visualization.

    Args:
        df: The dataframe
        col (str or [str]): Column of interest. Column datatype must be numerical.
        result: The statsmodels.tsa.seasonal.DecomposeResult object. Defaults to None.
        decompose: Set as True to decompose the timeseries with moving average. result must not be None. Defaults to False.

    Returns:
        fig: The visualization
    """
    fig, ax = plt.subplots(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )

    if isinstance(col, list):
        for i in col:
            fig = sns.lineplot(x=df.index, y=df[i], legend="full", ax=ax)
        ax.legend(labels=col)
    elif isinstance(col, str) and not decompose:
        fig = sns.lineplot(x=df.index, y=df[col], legend="full", ax=ax)
    elif decompose:
        fig = viz_decomposition(df, result)
        plt.close()
    return fig


def viz_decomposition(df, result):
    """Create timeseries decomposition visualization.

    Args:
        df: The dataframe
        result: The statsmodels.tsa.seasonal.DecomposeResult object.

    Returns:
        fig: The visualization
    """
    fig, ax = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        ),
    )
    sns.lineplot(y=result.observed, x=df.index, ax=ax[0])
    sns.lineplot(y=result.trend, x=df.index, ax=ax[1])
    sns.lineplot(y=result.seasonal, x=df.index, ax=ax[2])
    sns.lineplot(y=result.resid, x=df.index, ax=ax[3])
    fig.suptitle("Time Series Decomposition", fontsize=18)

    plt.close()
    return fig


def viz_plot_autocorrelation(
    timeseries, plot_type="acf", n_lags=40, fft=False, **kwargs
):
    """Create timeseries autocorrelation visualization.

    Args:
        timeseries: Series object containing datetime index
        plot_type: Choose between 'acf' or 'pacf. Defaults to "pacf".
        n_lags: Number of lags to return autocorrelation for. Defaults to 40.
        **kwargs: Keyword arguments for plot_acf and plot_pacf

    Returns:
        fig: The visualization
    """
    fig, ax = plt.subplots(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    if plot_type == "acf":
        fig = sm.graphics.tsa.plot_acf(
            timeseries, ax=ax, lags=n_lags, fft=fft, **kwargs
        )
    elif plot_type == "pacf":
        fig = sm.graphics.tsa.plot_pacf(timeseries, ax=ax, lags=n_lags, **kwargs)
    else:
        raise ValueError("Unsupported input data type")
    plt.xlabel("Lags")
    plt.ylabel(plot_type)
    plt.close()
    return fig
