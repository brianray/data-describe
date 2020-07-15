import matplotlib
import pytest
from plotly.graph_objects import Figure

import data_describe as dd

matplotlib.use("Agg")


def test_heatmap_matplotlib(data, compute_backend):
    fig = dd.data_heatmap(data)
    assert isinstance(fig, matplotlib.artist.Artist)


def test_heatmap_missing(data, compute_backend):
    fig = dd.data_heatmap(data, missing=True)
    assert isinstance(fig, matplotlib.artist.Artist)


def test_heatmap_plotly(data, compute_backend):
    fig = dd.data_heatmap(data, viz_backend="plotly")
    assert isinstance(fig, Figure)


def test_heatmap_invalid(compute_backend):
    data = [1, 2, 3, 4]
    with pytest.raises((ValueError, ModuleNotFoundError)):
        dd.data_heatmap(data)
