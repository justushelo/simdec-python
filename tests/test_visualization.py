import pytest
import pandas as pd
import matplotlib.pyplot as plt
import simdec as sd


def test_visualization_single_output():
    bins = pd.DataFrame({"s1": [1, 2], "s2": [3, 4]})
    palette = [[1, 0, 0, 1], [0, 1, 0, 1]]

    ax = sd.visualization(bins=bins, palette=palette, kind="histogram")
    assert isinstance(ax, plt.Axes)

    ax_box = sd.visualization(bins=bins, palette=palette, kind="boxplot")
    assert isinstance(ax_box, plt.Axes)


def test_visualization_two_outputs():
    bins = pd.DataFrame({"s1": [1, 2]})
    bins2 = pd.DataFrame({"s1": [5, 6]})
    palette = [[1, 0, 0, 1]]

    ax = sd.visualization(bins=bins, bins2=bins2, palette=palette)

    assert ax.get_xlabel() == "Output 1"
    assert len(ax.figure.axes) == 4


def test_visualization_invalid_kind():
    bins = pd.DataFrame({"s1": [1]})
    with pytest.raises(ValueError, match="'kind' can only be 'histogram' or 'boxplot'"):
        sd.visualization(bins=bins, palette=[[1, 0, 0, 1]], kind="invalid")
