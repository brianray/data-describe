from typing import List, Optional
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

from data_describe._widget import BaseWidget
from data_describe.config._config import get_option
from data_describe.compat import _is_dataframe, _requires, _in_notebook
from data_describe.backends import _get_viz_backend, _get_compute_backend


class HeatmapWidget(BaseWidget):
    """Container for data heatmap calculation and visualization.

    This class (object) is returned from the ``data_heatmap`` function. The
    attributes documented below can be accessed or extracted.


    Attributes:
        input_data: The input data.
        colnames ([str]): Names of numeric columns.
        std_data: The transposed, standardized data after scaling.
        missing (bool): If True, the heatmap shows missing values as indicators
            instead of standardized values.
        missing_data: The missing value indicator data.
    """

    def __init__(
        self,
        input_data=None,
        colnames: Optional[List] = None,
        std_data=None,
        missing: bool = False,
        missing_data: Optional[bool] = None,
        **kwargs,
    ):
        """Data heatmap.

        Args:
            input_data: The input data.
            colnames ([str]): Names of numeric columns.
            std_data: The transposed, standardized data after scaling.
            missing (bool): If True, the heatmap shows missing values as indicators
                instead of standardized values.
            missing_data (bool): The missing value indicator data.
        """
        super(HeatmapWidget, self).__init__(**kwargs)
        self.input_data = input_data
        self.colnames = colnames
        self.std_data = std_data
        self.missing = missing
        self.missing_data = missing_data
        self.viz_data = missing_data if missing else std_data

    def __str__(self):
        return "data-describe Heatmap Widget"

    def __repr__(self):
        mode = "missing" if self.missing else "standardized"
        return f"Heatmap Widget showing {mode} values."

    def show(self, viz_backend: Optional[str] = None, **kwargs):
        """The default display for this output.

        Shows the data heatmap plot.

        Args:
            viz_backend (str): The visualization backend.
            **kwargs: Keyword arguments.

        Raises:
            ValueError: Computed data is missing.

        Returns:
            The correlation matrix plot.
        """
        backend = viz_backend or self.viz_backend

        if self.viz_data is None:
            raise ValueError("Could not find data to visualize.")

        return _get_viz_backend(backend).viz_data_heatmap(
            self.viz_data, colnames=self.colnames, missing=self.missing, **kwargs
        )


def data_heatmap(
    data,
    missing: bool = False,
    compute_backend: Optional[str] = None,
    viz_backend: Optional[str] = None,
    **kwargs,
) -> HeatmapWidget:
    """Visualizes data patterns in the entire dataset by visualizing as a heatmap.

    This feature operates in two modes.

    (Default): A data heatmap showing standardized values (bounded to [-3, 3]). This
    visualization is useful for showing unusual, ordered patterns in the data that
    would otherwise be unnoticeable in summary statistics or distribution plots.

    Missing: Visualize only missing values.

    Args:
        data: A pandas data frame
        missing (bool): If True, show only missing values
        compute_backend (str): The compute backend.
        viz_backend (str): The visualization backend.
        **kwargs: Keyword arguments

    Returns:
        The data heatmap.
    """
    hwidget = _get_compute_backend(compute_backend, data).compute_data_heatmap(
        data, missing=missing, **kwargs
    )
    hwidget.viz_backend = viz_backend
    return hwidget


def _pandas_compute_data_heatmap(
    data, missing: bool = False, **kwargs
) -> HeatmapWidget:
    """Pre-processes data for the data heatmap.

    Values are standardized (removing the mean and scaling to unit variance).
    If `missing` is set to True, the dataframe flags missing records using 1/0.

    Args:
        data: The dataframe
        missing (bool): If True, uses missing values instead
        **kwargs: Keyword arguments.

    Raises:
        ValueError: Invalid input data type.

    Returns:
        HeatmapWidget
    """
    if not _is_dataframe(data):
        raise ValueError("Unsupported input data type")

    if missing:
        missing_data = data.isna().astype(int)
        colnames = data.columns.values
        return HeatmapWidget(
            input_data=data,
            colnames=colnames,
            missing=True,
            missing_data=missing_data.transpose(),
        )
    else:
        data = data.select_dtypes(["number"])
        colnames = data.columns.values
        scaler = StandardScaler()
        std_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        return HeatmapWidget(
            input_data=data, colnames=colnames, std_data=std_data.transpose()
        )


@_requires("plotly")
def _plotly_viz_data_heatmap(
    data, colnames: List[str], missing: bool = False, **kwargs
):
    """Plots the data heatmap.

    Args:
        data: The dataframe
        colnames (List[str]): The column names, used for tick labels
        missing (bool): If True, plots missing values instead
        **kwargs: Keyword arguments.

    Returns:
        The data heatmap as a Plotly figure.
    """
    data_fig = go.Heatmap(
        z=np.flip(data.values, axis=0),
        x=list(range(data.shape[0])),
        y=list(colnames[::-1]),
        ygap=1,
        zmin=-3 if not missing else 0,
        zmax=3 if not missing else 1,
        colorscale="viridis" if not missing else "greys",
        colorbar={"title": "z-score (bounded)" if not missing else "Missing"},
    )

    figure = go.Figure(
        data=[data_fig],
        layout=go.Layout(
            autosize=False,
            title={
                "text": "Data Heatmap",
                "font": {"size": get_option("display.plotly.title_size")},
            },
            width=get_option("display.plotly.fig_width"),
            height=get_option("display.plotly.fig_height"),
            xaxis=go.layout.XAxis(ticks="", title="Record #", showgrid=False),
            yaxis=go.layout.YAxis(
                ticks="", title="Variable", automargin=True, showgrid=False
            ),
            plot_bgcolor="rgb(0,0,0,0)",
            paper_bgcolor="rgb(0,0,0,0)",
        ),
    )

    if _in_notebook():
        init_notebook_mode(connected=True)
        return iplot(figure, config={"displayModeBar": False})
    else:
        return figure


def _seaborn_viz_data_heatmap(
    data, colnames: List[str], missing: bool = False, **kwargs
):
    """Plots the data heatmap.

    Args:
        data: The dataframe
        colnames (List[str]): The column names, used for tick labels
        missing (bool): If True, plots missing values instead
        **kwargs: Keyword arguments passed to seaborn.heatmap

    Returns:
        The seaborn figure
    """
    cmap = (
        copy.copy(plt.get_cmap("viridis"))
        if not missing
        else copy.copy(plt.get_cmap("Greys"))
    )
    cmap.set_bad(color="white")

    plot_options = {
        "cmap": cmap,
        "robust": True,
        "center": 0 if not missing else 0.5,
        "xticklabels": False,
        "yticklabels": colnames,
        "cbar_kws": {"shrink": 0.5, "label": "z-score (bounded)"},
        "vmin": -3 if not missing else 0,
        "vmax": 3 if not missing else 1,
    }

    plot_options.update(kwargs)

    fig = Figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    ax = fig.add_subplot(111)
    ax = sns.heatmap(data, ax=ax, **plot_options)
    ax.set_title("Data Heatmap")
    ax.set_xlabel("Record #")
    ax.set_ylabel("Variable")
    ax.set_yticklabels(colnames, rotation=0)

    return fig
