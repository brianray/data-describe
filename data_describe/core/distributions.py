from typing import Dict, Any, Optional

from pandas.api.types import is_numeric_dtype
import matplotlib
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from scipy.stats import binned_statistic

from data_describe.config._config import get_option
from data_describe.metrics.univariate import spikey, skewed
from data_describe._widget import BaseWidget
from data_describe.compat import _is_dataframe
from data_describe.backends import _get_viz_backend, _get_compute_backend


class DistributionWidget(BaseWidget):
    """Container for distributions.

    This class (object) is returned from the ``distribution`` function. The
    attributes documented below can be accessed or extracted.

    Attributes:
        input_data: The input data
        spike_value: Measure of the "spikey"ness metric, which diagnoses spikey
            histograms where the tallest bin is ``n`` times taller than the average bin.
        skew_value: Measure of the skewness metric.
        spike_factor: The threshold factor used to diagnose "spikey"ness.
        skew_factor: The threshold factor used to diagnose skew.
    """

    def __init__(
        self,
        input_data=None,
        spike_value=None,
        skew_value=None,
        spike_factor=None,
        skew_factor=None,
        contrast=None,
        target=None,
        viz_backend=None,
    ):
        """Distribution Plots.

        Args:
            input_data: The input data
            spike_value: Measure of the "spikey"ness metric, which diagnoses spikey
                histograms where the tallest bin is ``n`` times taller than the average bin.
            skew_value: Measure of the skewness metric.
            spike_factor: The threshold factor used to diagnose "spikey"ness.
            skew_factor: The threshold factor used to diagnose skew.
            contrast: The name of the categorical column to use for multiple contrasts.
            target: The name of the target column that will be overlaid as a line plot.
            viz_backend: The visualization backend.
        """
        self.input_data = input_data
        self.spike_value = spike_value
        self.skew_value = skew_value
        self.spike_factor = spike_factor
        self.skew_factor = skew_factor
        self.contrast = contrast
        self.viz_backend = viz_backend

    def show(self, viz_backend=None, **kwargs):
        """The default display for this output.

        Displays a summary of diagnostics.

        Args:
            viz_backend (str, optional): The visualization backend.
            **kwargs: Keyword arguments.
        """
        viz_backend = viz_backend or self.viz_backend

        return self.plot_all(viz_backend=viz_backend, **kwargs)

    def plot_all(
        self,
        contrast: Optional[str] = None,
        target: Optional[str] = None,
        viz_backend: Optional[str] = None,
        **kwargs,
    ):
        """Shows all distribution plots in a faceted grid.

        Numeric features will be visualized using a histogram/violin plot, and any other
        types will be visualized using a categorical bar plot.

        Args:
            x (str, optional): The feature name to plot. If None, will plot all features.
            contrast (str, optional): The feature name to compare histograms by contrast.
            viz_backend (optional): The visualization backend.
            **kwargs: Additional keyword arguments for the visualization backend.

        Returns:
            Histogram plot(s).
        """
        backend = viz_backend or self.viz_backend
        contrast = contrast or self.contrast
        target = target or self.target

        return _get_viz_backend(backend).viz_all_distribution(
            data=self.input_data, contrast=contrast, target=target, **kwargs
        )

    def plot(
        self,
        x: Optional[str] = None,
        contrast: Optional[str] = None,
        target: Optional[str] = None,
        viz_backend: Optional[str] = None,
        **kwargs,
    ):
        """Generate distribution plots.

        Numeric features will be visualized using a histogram/violin plot, and any other
        types will be visualized using a categorical bar plot.

        Args:
            x (str, optional): The feature name to plot. If None, will plot all features.
            contrast (str, optional): The feature name to compare histograms by contrast.
            mode (str): {'combo', 'violin', 'hist'} The type of plot to display.
                Defaults to a combined histogram/violin plot.
            hist_kwargs (dict, optional): Keyword args for seaborn.histplot.
            violin_kwargs (dict, optional): Keyword args for seaborn.violinplot.
            viz_backend (optional): The visualization backend.
            **kwargs: Additional keyword arguments for the visualization backend.

        Returns:
            Histogram plot(s).
        """
        backend = viz_backend or self.viz_backend
        contrast = contrast or self.contrast
        target = target or self.target

        return _get_viz_backend(backend).viz_distribution(
            data=self.input_data, x=x, contrast=contrast, target=target, **kwargs
        )


def distribution(
    data,
    diagnostic=True,
    contrast=None,
    target=None,
    compute_backend=None,
    viz_backend=None,
    **kwargs,
) -> DistributionWidget:
    """Distribution Plots.

    Visualizes univariate distributions. This feature can be used for generating
    various types of plots for univariate distributions, including: histograms, violin
    plots, bar (count) plots.

    Args:
        data: Data Frame
        diagnostic: If True, will run diagnostics to select "interesting" plots.
        compute_backend: The compute backend.
        viz_backend: The visualization backend.
        **kwargs: Keyword arguments.

    Raises:
        ValueError: Invalid input data type.

    Returns:
        DistributionWidget
    """
    if not _is_dataframe(data):
        raise ValueError("DataFrame required.")

    widget = _get_compute_backend(compute_backend, data).compute_distribution(
        data, diagnostic=diagnostic, **kwargs
    )
    widget.contrast = contrast
    widget.target = target
    return widget


def _pandas_compute_distribution(
    data, diagnostic: bool = True, spike_factor=10, skew_factor=3, **kwargs
):
    """Compute distribution metrics.

    Args:
        data (DataFrame): The data
        diagnostic (bool): If True, will compute diagnostics used to select "interesting" plots.
        spike_factor (int): The spikey-ness factor used to flag spikey histograms. Defaults to 10.
        skew_factor (int): The skew-ness factor used to flag skewed histograms. Defaults to 3.
        **kwargs: Keyword arguments.

    Returns:
        DistributionWidget
    """
    num_data = data.select_dtypes("number")

    spike_value = num_data.apply(spikey, axis=0) if diagnostic else None
    skew_value = num_data.apply(skewed, axis=0) if diagnostic else None

    return DistributionWidget(
        input_data=data,
        spike_value=spike_value,
        skew_value=skew_value,
        spike_factor=spike_factor,
        skew_factor=skew_factor,
    )


def _seaborn_viz_distribution(
    data,
    x: str,
    contrast: Optional[str] = None,
    target: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
):
    """Plots the distribution.

    Args:
        data: The data
        x (str): The column to visualize.
        contrast (str, optional): The column to use as a contrast.
        ax (matplotlib.axes.Axes): Pre-existing axes for the plot.
        **kwargs: Keyword arguments passed to underlying plot functions.

    Returns:
        matplotlib.figure.Figure
    """
    if x in data.select_dtypes("number").columns:
        return _seaborn_viz_numeric(
            data, x, contrast=contrast, target=target, ax=ax, **kwargs
        )
    else:
        return _seaborn_viz_categorical(
            data, x, contrast=contrast, target=target, ax=ax, **kwargs
        )


def _seaborn_viz_all_distribution(
    data,
    contrast: Optional[str] = None,
    target: Optional[str] = None,
    mode: str = "combo",
    **kwargs,
):
    """Plots the distribution.

    Args:
        data: The data
        x (str): The column to visualize.
        contrast (str, optional): The column to use as a contrast.
        **kwargs: Keyword arguments passed to underlying plot functions.

    Returns:
        matplotlib.figure.Figure
    """
    hist_kwargs = {**{"legend": False}, **kwargs.get("hist_kwargs", {})}
    # violin_kwargs = {**{'legend': False}, **kwargs.get("hist_kwargs", {})}
    bar_kwargs = {**{"legend": False}, **kwargs.get("bar_kwargs", {})}

    fig = Figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            max(
                get_option("display.matplotlib.fig_width")
                / 4
                * int(np.ceil(data.shape[1] / 4)),
                get_option("display.matplotlib.fig_height"),
            ),
        )
    )
    for i, col in enumerate(data.columns):
        ax = fig.add_subplot(
            int(np.ceil(data.shape[1] / 4)) + 1,
            4,
            i + 1,
            label=col,
        )
        if col in data.select_dtypes(["number", "datetime", "datetimetz"]).columns:
            _seaborn_viz_histogram(
                data, col, contrast=contrast, target=target, ax=ax, **hist_kwargs
            )
        else:
            _seaborn_viz_bar(
                data, col, contrast=contrast, target=target, ax=ax, **bar_kwargs
            )
    fig.tight_layout()
    return fig


def _seaborn_viz_numeric(
    data,
    x: str,
    contrast: Optional[str] = None,
    target: Optional[str] = None,
    mode: str = "combo",
    hist_kwargs: Optional[dict] = None,
    violin_kwargs: Optional[dict] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
):
    """Plots a histogram/violin plot.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.
        mode (str): {'combo', 'violin', 'hist'} The type of plot to display.
            Defaults to a combined histogram/violin plot.
        hist_kwargs (dict, optional): Keyword args for seaborn.histplot.
        violin_kwargs (dict, optional): Keyword args for seaborn.violinplot.
        **kwargs: Keyword args to be passed to all underlying plotting functions.

    Raises:
        ValueError: Unknown plot mode.

    Returns:
        Matplotlib figure
    """
    hist_kwargs = hist_kwargs or {}
    violin_kwargs = violin_kwargs or {}
    fig = Figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    if mode == "combo":
        gs = GridSpec(nrows=5, ncols=1)
        ax1 = fig.add_subplot(gs.new_subplotspec((0, 0), 1, 1))
        ax2 = fig.add_subplot(gs.new_subplotspec((1, 0), 4, 1))

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        _seaborn_viz_histogram(
            data, x, contrast=contrast, target=target, ax=ax1, **hist_kwargs
        )
        _seaborn_viz_violin(
            data, x, contrast=contrast, target=target, ax=ax2, **violin_kwargs
        )
        ax1.set_title(x)
        return fig
    elif mode == "hist":
        ax = fig.add_subplot()
        _seaborn_viz_histogram(
            data, x, contrast=contrast, target=target, ax=ax, **hist_kwargs, **kwargs
        )
        ax.set_title(x)
        return fig
    elif mode == "violin":
        ax = fig.add_subplot()
        _seaborn_viz_violin(
            data, x, contrast=contrast, target=target, ax=ax, **violin_kwargs, **kwargs
        )
        ax.set_title(x)
        return fig
    else:
        raise ValueError("Unknown value for 'mode' plot type")


def _seaborn_viz_categorical(
    data, x: str, contrast: Optional[str] = None, target: Optional[str] = None, **kwargs
):
    """Plots a bar count plot for a categorical feature.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.
        **kwargs: Keyword args for seaborn.countplot.

    Returns:
        Matplotlib figure
    """
    bar_kwargs = kwargs or {}
    fig = Figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    ax = fig.add_subplot()
    _seaborn_viz_bar(data, x, contrast=contrast, target=target, ax=ax, **bar_kwargs)
    ax.set_title(x)
    return fig


def _seaborn_viz_histogram(
    data,
    x: str,
    contrast: Optional[str] = None,
    target: Optional[str] = None,
    bins=10,
    **kwargs,
):
    """Plot a single histogram.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.
        target (str, optional): The name of the target column that will be overlaid as a line plot.
            If target is numeric: Uses the average target value for each bin
            If the target is binary (category): Converts to a pandas category code (`pandas.Series.cat.codes`)
        **kwargs: Keyword arguments passed to seaborn.distplot

    Raises:
        ValueError: Not a numeric column.

    Returns:
        Seaborn Axis Object
    """
    if x not in data.select_dtypes(["number", "datetime", "datetimetz"]).columns:
        raise ValueError("x must be numeric column")

    default_hist_kwargs: Dict[str, Any] = {"bins": bins}
    hist_kwargs = {**default_hist_kwargs, **(kwargs or {})}
    if contrast and contrast != x:
        data[contrast] = data[contrast].astype("category")
        ax = sns.histplot(x=x, hue=contrast, data=data, **hist_kwargs)
    else:
        ax = sns.histplot(data[x], **hist_kwargs)

    if target and target != x:
        if is_numeric_dtype(data[x]):
            ax2 = ax.twinx()
            if is_numeric_dtype(data[target]):
                estimator = sns._statistics.Histogram()
                bin_edges = estimator.define_bin_edges(
                    data[x]  # , bins=hist_kwargs.get("bins")
                )
                statistic, _, _ = binned_statistic(
                    data[x], data[target], bins=bin_edges
                )
                ax = sns.lineplot(
                    x=bin_edges[:-1] + np.diff(bin_edges) / 2,
                    y=statistic,
                    marker="o",
                    color="black",
                    ax=ax2,
                )
                ax2.set_ylabel(target, rotation=270, labelpad=10)
            elif data[target].nunique() <= 2:
                target_ = data[target].astype("category").cat.codes
                estimator = sns._statistics.Histogram()
                bin_edges = estimator.define_bin_edges(
                    data[x]  # , bins=hist_kwargs.get("bins")
                )
                statistic, _, _ = binned_statistic(data[x], target_, bins=bin_edges)
                ax = sns.lineplot(
                    x=bin_edges[:-1] + np.diff(bin_edges) / 2,
                    y=statistic,
                    marker="o",
                    color="black",
                    ax=ax2,
                )
                ax2.set_ylabel(target, rotation=270, labelpad=10)
            else:
                raise NotImplementedError(
                    "`target` not implemented for more categories with more than 2 levels."
                )

    ax.set_title(x)
    ax.set_xlabel("")
    return ax


def _seaborn_viz_violin(
    data, x: str, contrast: Optional[str] = None, target: Optional[str] = None, **kwargs
):
    """Plot a single violin plot.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.
        target (str, optional): Not implemented.
        **kwargs: Keyword arguments passed to seaborn.violinplot

    Raises:
        ValueError: Not a numeric column.

    Returns:
        Seaborn Axis Object
    """
    if x not in data.select_dtypes("number").columns:
        raise ValueError("x must be numeric column")

    default_violin_kwargs = {"cut": 0}
    violin_kwargs = {**default_violin_kwargs, **(kwargs or {})}
    if contrast and contrast != x:
        data[contrast] = data[contrast].astype("category")
        ax = sns.violinplot(x=x, y=contrast, data=data, **violin_kwargs)
    else:
        ax = sns.violinplot(x=x, data=data, **violin_kwargs)

    ax.set_title(x)
    ax.set_ylabel("")
    ax.set_xlabel("")

    return ax


def _seaborn_viz_bar(
    data, x: str, contrast: Optional[str] = None, target: Optional[str] = None, **kwargs
):
    """Plot a bar chart (count plot) for categorical features.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the column to use for multiple histograms.
        target (str, optional): The name of the target column that will be overlaid as a line plot.
            If target is numeric: Uses the average target value for each bin
            If the target is binary (category): Converts to a pandas category code (`pandas.Series.cat.codes`)
        **kwargs: Keyword arguments passed to seaborn.countplot

    Returns:
        Seaborn Axis Object
    """
    default_bar_kwargs = {
        "orient": "h",
        "alpha": 0.75,
        "edgecolor": matplotlib.rcParams["patch.edgecolor"],
    }

    bar_kwargs = {**default_bar_kwargs, **(kwargs or {})}

    legend = bar_kwargs.pop("legend", True)

    if contrast and contrast != x:
        ax = sns.countplot(x=x, hue=contrast, data=data, **bar_kwargs)
    else:
        ax = sns.countplot(x=x, data=data, **bar_kwargs)

    if not legend:
        ax.legend().remove()

    if target and target != x:
        ax2 = ax.twinx()
        if is_numeric_dtype(data[target]):
            ax = sns.lineplot(
                x=data[x],
                y=data[target],
                marker="o",
                color="black",
                err_style="bars",
                ax=ax2,
            )
        elif data[target].nunique() <= 2:
            target_ = data[target].astype("category").cat.codes
            ax = sns.lineplot(
                x=data[x],
                y=target_,
                marker="o",
                color="black",
                err_style="bars",
                ax=ax2,
            )

    ax.set_title(x)
    ax.set_ylabel("")
    ax.set_xlabel("")

    return ax
