import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting import save_and_close_figure


def test_save_and_close_figure_releases_matplotlib_figure(tmp_path):
    fig = plt.figure()
    output = tmp_path / "figure.png"

    save_and_close_figure(fig, output)

    assert output.is_file()
    assert not plt.fignum_exists(fig.number)
