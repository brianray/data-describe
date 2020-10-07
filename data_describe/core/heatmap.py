from typing import Tuple, List, Any

import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, iplot
from IPython import get_ipython
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

from data_describe._widget import BaseWidget
from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_viz_backend, _get_compute_backend


def data_heatmap(data, missing=False, compute_backend=None, viz_backend=None, **kwargs):
    """Generate a data heatmap showing standardized data or missing values.

    The data heatmap shows an overview of numeric features that have been standardized.

    Args:
        data: A pandas data frame
        missing (bool): If True, show only missing values
        compute_backend: The compute backend.
        viz_backend: The visualization backend.
        **kwargs: Keyword arguments

    Returns:
        The data heatmap.
    """
    return _get_compute_backend(compute_backend, data).compute_data_heatmap(
        data, missing=missing, **kwargs
    )


class HeatmapWidget(BaseWidget):
    """Heatmap Widget.

    Attributes:
        input_data: The input data.
        colnames: Names of numeric columns.
        std_data: The transposed, standardized data after scaling.
        missing: If True, the heatmap shows missing values as indicators
            instead of standardized values.
        missing_data: The missing value indicator data.
    """

    def __init__(
        self,
        input_data=None,
        colnames=None,
        std_data=None,
        missing=False,
        missing_data=None,
        **kwargs,
    ):
        """Data heatmap.

        Args:
            input_data: The input data.
            colnames: Names of numeric columns.
            std_data: The transposed, standardized data after scaling.
            missing (bool): If True, the heatmap shows missing values as indicators
                instead of standardized values.
            missing_data: The missing value indicator data.
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

    def show(self, viz_backend=None, **kwargs):
        """Show the data heatmap plot.

        Args:
            viz_backend: The visualization backend.
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


def _pandas_compute_data_heatmap(
    data, missing: bool = False, **kwargs
) -> Tuple[Any, List[str]]:
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
    if not isinstance(data, _DATAFRAME_TYPE):
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

    if get_ipython() is not None:
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
        colnames: The column names, used for tick labels
        missing: If True, plots missing values instead
        kwargs: Keyword arguments passed to seaborn.heatmap

    Returns:
        The seaborn figure
    """
    plot_options = {
        "cmap": "viridis" if not missing else "Greys",
        "robust": True,
        "center": 0 if not missing else 0.5,
        "xticklabels": False,
        "yticklabels": colnames,
        "cbar_kws": {"shrink": 0.5},
        "vmin": -3 if not missing else 0,
        "vmax": 3 if not missing else 1,
    }

    plot_options.update(kwargs)

    plt.figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    heatmap = sns.heatmap(data, **plot_options)
    plt.title("Data Heatmap")
    plt.ylabel("Variable")
    plt.xlabel("Record #")

    return heatmap
