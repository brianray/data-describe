from typing import Dict, Callable, Optional
import warnings

import pandas as pd
from pandas.core.dtypes.common import is_float
import numpy as np

from data_describe._widget import BaseWidget
from data_describe.compat import _requires, _compat, _is_series, _is_dataframe
from data_describe.backends._backends import _get_compute_backend


class SummaryWidget(BaseWidget):
    """Container for data summary.

    This class (object) is returned from the ``data_summary`` function. The
    attributes documented below can be accessed or extracted.

    Attributes:
        input_data: The input data.
        info_data: Information about the data shape and size.
        summary_data (DataFrame): The summary statistics.
        as_percentage (bool): If True, display counts as percentage over total
        auto_float (bool): If True, apply formatting to float values
    """

    def __init__(
        self,
        input_data=None,
        info_data=None,
        summary_data=None,
        as_percentage: Optional[bool] = False,
        auto_float: Optional[bool] = True,
        **kwargs,
    ):
        """Data heatmap.

        Args:
            input_data: The input data.
            info_data: Information about the data shape and size.
            summary_data: The summary statistics.
            as_percentage (bool): If True, display counts as percentage over total
            auto_float (bool): If True, apply formatting to float values
        """
        super(SummaryWidget, self).__init__(**kwargs)
        self.input_data = input_data
        self.info_data = info_data
        self.summary_data = summary_data
        self.as_percentage = as_percentage
        self.auto_float = auto_float

    def __str__(self):
        return "data-describe Summary Widget"

    def __repr__(self):
        return "data-describe Summary Widget"

    def show(
        self,
        viz_backend=None,
        as_percentage: Optional[bool] = None,
        auto_float: Optional[bool] = None,
        **kwargs,
    ):
        """The default display for this output.

        Displays the summary information.

        Args:
            viz_backend: The visualization backend.
            as_percentage (bool): If True, display counts as percentage over total
            auto_float (bool): If True, apply formatting to float values
            **kwargs: Keyword arguments.

        Returns:
            The correlation matrix plot.
        """
        try:
            from IPython.display import display

            view = display
        except ImportError:
            view = print

        view(self.info_data)

        summary_data = self.summary_data
        format_dict: Dict[str, Callable] = {}

        as_percentage = as_percentage or self.as_percentage
        if as_percentage:
            for col in ["Zeros", "Nulls", "Top Frequency"]:
                summary_data[col] = summary_data[col] / self.input_data.shape[0]
                format_dict[col] = "{:.1%}".format

        summary_data.fillna("", inplace=True)
        auto_float = auto_float or self.auto_float
        if auto_float:
            for col in summary_data.columns:
                if col not in format_dict.keys():
                    format_dict[col] = _value_formatter

        if len(format_dict) > 0:
            view(summary_data.style.format(format_dict))
        else:
            view(summary_data)


def data_summary(
    data, as_percentage: bool = False, auto_float: bool = True, compute_backend=None
):
    """Summary statistics and data description.

    Args:
        data: The dataframe
        as_percentage (bool): If True, display counts as percentage over total
        auto_float (bool): If True, apply formatting to float values
        compute_backend: The compute backend.

    Returns:
        The dataframe with metrics in rows
    """
    widget = _get_compute_backend(
        backend=compute_backend, df=data
    ).compute_data_summary(data)
    widget.as_percentage = as_percentage
    widget.auto_float = auto_float
    return widget


def mode1(x):
    """Mode (counts only) by Warren Weckesser.

    https://stackoverflow.com/questions/46365859/what-is-the-fastest-way-to-get-the-mode-of-a-numpy-array

    Args:
        x: Input array
    """
    _, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return counts[m]


def _pandas_compute_data_summary(data):
    """Perform computation for summary statistics and data description.

    Args:
        data: The dataframe

    Raises:
        ValueError: Invalid input data type.

    Returns:
        The Pandas dataframe with metrics in rows
    """
    if _is_series(data):
        data = pd.DataFrame(data, columns=[data.name])

    if not _is_dataframe(data):
        raise ValueError("Data must be a Pandas DataFrame")

    info_data = pd.DataFrame(
        {
            "Info": [
                data.shape[0],
                data.shape[1],
                _sizeof_fmt(data.memory_usage().sum()),
            ]
        },
        index=["Rows", "Columns", "Size in Memory"],
    )

    columns = data.columns
    val = data.values
    num_columns = data.select_dtypes("number").columns
    num_ind = np.nonzero([c in num_columns for c in columns])[0]
    date_columns = data.select_dtypes(["datetime", "datetimetz"]).columns
    date_ind = np.nonzero([c in date_columns for c in columns])[0]
    other_columns = data.select_dtypes(
        exclude=["number", "datetime", "datetimetz"]
    ).columns
    other_ind = np.nonzero([c in other_columns for c in columns])[0]
    order = np.concatenate([num_ind, date_ind, other_ind], axis=0)

    dtypes = data.dtypes[order]
    s_mean = np.pad(
        np.mean(val[:, num_ind], axis=0),
        (0, len(data.columns) - num_ind.size),
        constant_values=np.nan,
    )
    s_sd = np.pad(
        np.std(val[:, num_ind].astype(np.float), axis=0),
        (0, len(data.columns) - num_ind.size),
        constant_values=np.nan,
    )
    s_med = np.pad(
        np.median(val[:, num_ind], axis=0),
        (0, len(data.columns) - num_ind.size),
        constant_values=np.nan,
    )
    s_min = np.pad(
        np.min(val[:, np.concatenate([num_ind, date_ind])], axis=0),
        (0, len(data.columns) - num_ind.size - date_ind.size),
        constant_values=np.nan,
    )
    s_max = np.pad(
        np.max(val[:, np.concatenate([num_ind, date_ind])], axis=0),
        (0, len(data.columns) - num_ind.size - date_ind.size),
        constant_values=np.nan,
    )
    s_zero = (data == 0).sum()[order]
    s_null = data.isnull().sum().astype(int)[order]
    s_unique = data.nunique()[order]
    s_freq = np.apply_along_axis(mode1, 0, val.astype("str"))[order]

    summary_data = pd.DataFrame(
        np.vstack(
            [
                dtypes,
                s_null,
                s_zero,
                s_min,
                s_med,
                s_max,
                s_mean,
                s_sd,
                s_unique,
                s_freq,
            ]
        ).transpose()[np.argsort(order), :],
        columns=[
            "Data Type",
            "Nulls",
            "Zeros",
            "Min",
            "Median",
            "Max",
            "Mean",
            "Standard Deviation",
            "Unique",
            "Top Frequency",
        ],
        index=data.columns,
    )

    return SummaryWidget(data, info_data, summary_data)


@_requires("modin")
def _modin_compute_data_summary(data):
    """Perform computation for summary statistics and data description.

    Args:
        data: The dataframe

    Raises:
        ValueError: Invalid input data type.

    Returns:
        The Modin dataframe with metrics in rows
    """
    if _is_series(data):
        data = _compat["modin.pandas"].DataFrame(data)

    if not _is_dataframe(data):
        raise ValueError("Data must be a Modin DataFrame")

    info_data = pd.DataFrame(
        {
            "Info": [
                data.shape[0],
                data.shape[1],
                _sizeof_fmt(data.memory_usage().sum()),
            ]
        },
        index=["Rows", "Columns", "Size in Memory"],
    )

    # Save column order
    columns = data.columns

    dtypes = data.dtypes.to_numpy()
    s_mean = data.mean(numeric_only=True).reindex(columns).to_numpy()
    s_sd = data.std(numeric_only=True).reindex(columns).to_numpy()
    s_med = data.median(numeric_only=True).reindex(columns).to_numpy()
    s_min = data.min(numeric_only=True).reindex(columns).to_numpy()
    s_max = data.max(numeric_only=True).reindex(columns).to_numpy()
    s_zero = data[data == 0].fillna(0).sum().astype(int).to_numpy()
    s_null = data.isnull().sum().astype(int).to_numpy()
    s_unique = data.nunique().to_numpy()
    s_freq = data.apply(lambda x: mode1(x.astype("str"))).to_numpy()

    summary_data = pd.DataFrame(
        np.vstack(
            [
                dtypes,
                s_null,
                s_zero,
                s_min,
                s_med,
                s_max,
                s_mean,
                s_sd,
                s_unique,
                s_freq,
            ]
        ).transpose(),
        columns=[
            "Data Type",
            "Nulls",
            "Zeros",
            "Min",
            "Median",
            "Max",
            "Mean",
            "Standard Deviation",
            "Unique",
            "Top Frequency",
        ],
        index=columns,
    )

    return SummaryWidget(data, info_data, summary_data)


def _get_precision(x, margin: int = 1) -> int:
    """Get the minimum precision for a value.

    Used to determine how to display and format floats.

    Args:
        x: Input value
        margin (int): Added to the calculated precision

    Returns:
        int: Number of decimal places
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            x = (x - np.trunc(x)).astype(np.float)
            magnitude = np.log10(x[x != 0])
            return int(np.ceil(np.abs(magnitude))) + margin
        except (TypeError, ValueError):
            return 0


def _value_formatter(x, precision=None):
    """Formatter for displaying values of mixed type and precision.

    Args:
        x: Input value
        precision (int, optional): Precision for floats.
    """
    if is_float(x):
        precision = precision or _get_precision(x)
        try:
            return f"{{:.{precision}f}}".format(x)
        except ValueError:
            pass
    else:
        return x


def _sizeof_fmt(num):
    """Format byte size to human-readable format.

    https://web.archive.org/web/20111010015624/http://blogmag.net/blog/read/38/Print_human_readable_file_size

    Args:
        num (float): Number of bytes
    """
    for x in ["bytes", "KB", "MB", "GB", "TB", "PB"]:
        if num < 1024.0:
            return f"{num:3.1f} {x}"
        num /= 1024.0
