from typing import List
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
from pyscagnostics import scagnostics

from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_STATIC_TYPE


def viz_data_heatmap(
    data: _DATAFRAME_STATIC_TYPE, colnames: List[str], missing: bool = False, **kwargs
):
    """Plots the data heatmap

    Args:
        data: The dataframe
        colnames: The column names, used for tick labels
        missing: If True, plots missing values instead
        kwargs: Keyword arguments passed to seaborn.heatmap
    """
    plot_options = {
        "cmap": "PuRd" if missing else "viridis",
        "robust": True,
        "center": 0,
        "xticklabels": False,
        "yticklabels": colnames,
        "cbar_kws": {"shrink": 0.5},
    }

    plot_options.update(kwargs)

    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    heatmap = sns.heatmap(data, **plot_options)
    plt.title("Data Heatmap")
    plt.ylabel("Variable")
    plt.xlabel("Record #")

    return heatmap


def viz_scatter_plot(data, mode, sample, threshold, **kwargs):
    if mode == "matrix":
        fig = sns.pairplot(data)
        return fig
    elif mode == "all":
        col_pairs = combinations(data.columns, 2)
        fig = []
        for p in col_pairs:
            fig.append(_scatter_plot(data, p[0], p[1], **kwargs))
        return fig
    elif mode == "diagnostic":
        diagnostics = scagnostics(data)

        if threshold is not None:
            diagnostics = _filter_threshold(diagnostics, threshold)

        if len(diagnostics) == 0:
            raise UserWarning("No plots identified by diagnostics")

        fig = []
        for d in diagnostics:
            fig.append(_scatter_plot(data, d[0], d[1], **kwargs))

        return fig
    else:
        raise ValueError(f"Unknown plot mode: {mode}")


def _scatter_plot(data, xname, yname, **kwargs):
    """Generate one scatter (joint) plot

    Args:
        data: A Pandas data frame
        xname: The x-axis column name
        yname: The y-axis column name
        kwargs: Keyword arguments

    Returns:
        The Seaborn figure
    """
    default_joint_kwargs = {
        "height": max(get_option("display.fig_width"), get_option("display.fig_height"))
    }
    default_scatter_kwargs = {}
    default_dist_kwargs = {"kde": False, "rug": False}
    default_joint_kwargs.update(kwargs.get("joint_kwargs", {}))
    default_scatter_kwargs.update(kwargs.get("scatter_kwargs", {}))
    default_dist_kwargs.update(kwargs.get("dist_kwargs", {}))

    g = sns.JointGrid(data[xname], data[yname], **default_joint_kwargs)
    g = g.plot_joint(sns.scatterplot, **default_scatter_kwargs)
    g = g.plot_marginals(sns.distplot, **default_dist_kwargs)
    return g


def _filter_threshold(diagnostics, threshold=0.85):
    """Filter the plots by scatter plot diagnostic threshold

    Args:
        diagnostics: The diagnostics generator from pyscagnostics
        threshold: The scatter plot diagnostic threshold value [0,1] for returning a plot
            If a number: Returns all plots where at least one metric is above this threshold
            If a dictionary: Returns plots where the metric is above its threshold
            For example, {"Outlier": 0.9} returns plots with outlier metrics above 0.9
            The available metrics are: Outlier, Convex, Skinny, Skewed, Stringy, Straight, Monotonic, Clumpy, Striated

    Returns:
        A dictionary of pairs that match the filter
    """
    if isinstance(threshold, dict):
        return [
            d
            for d in diagnostics
            if all([d[2][0][m] >= threshold[m] for m in threshold.keys()])
        ]
    else:
        return [
            d for d in diagnostics if any([v > threshold for v in d[2][0].values()])
        ]
