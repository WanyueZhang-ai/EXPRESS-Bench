"""Small helpers for saving plots without retaining pyplot figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from src.efficiency import profiled_operation


@profiled_operation("visualization.save_figure", "visualization")
def save_and_close_figure(fig, output_path):
    """Save a matplotlib figure and always release it from pyplot."""
    try:
        fig.tight_layout()
        fig.savefig(Path(output_path))
    finally:
        plt.close(fig)
