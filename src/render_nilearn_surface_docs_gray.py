from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf


DEFAULT_OUTPUT = Path("figures/nilearn_surface_docs_gray_grid.png")
VIEWS = ["lateral", "dorsal"]
HEMISPHERES = ["left", "right", "both"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a docs-style Nilearn surface grid in grayscale."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )
    return parser.parse_args()


def add_both_panel(
    fig: plt.Figure,
    parent_ax,
    fsaverage_sulcal,
    view: str,
) -> None:
    # Nilearn 0.11.1 rejects hemi='both' in plot_surf. Emulate the "both"
    # column by placing left and right hemispheres inside the same subplot cell.
    bbox = parent_ax.get_position()
    parent_ax.remove()

    gap = 0.02 * bbox.width
    hemi_width = (bbox.width - gap) / 2.0
    left_rect = [bbox.x0, bbox.y0, hemi_width, bbox.height]
    right_rect = [bbox.x0 + hemi_width + gap, bbox.y0, hemi_width, bbox.height]

    for rect, hemi in zip([left_rect, right_rect], ["left", "right"], strict=False):
        ax = fig.add_axes(rect, projection="3d")
        plot_surf(
            surf_map=fsaverage_sulcal,
            hemi=hemi,
            view=view,
            figure=fig,
            axes=ax,
            colorbar=False,
            symmetric_cmap=False,
            bg_on_data=True,
            bg_map=fsaverage_sulcal,
            cmap="gray",
        )
        ax.set_axis_off()

    fig.text(
        bbox.x0 + bbox.width / 2.0,
        bbox.y1 + 0.01,
        f"both - {view}",
        ha="center",
        va="bottom",
        fontsize=12,
    )


def render_docs_gray_grid(output_path: Path, dpi: int) -> Path:
    fsaverage_sulcal = load_fsaverage_data(data_type="sulcal")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(VIEWS),
        ncols=len(HEMISPHERES),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(HEMISPHERES), 4 * len(VIEWS)),
        dpi=dpi,
    )
    axes = np.atleast_2d(axes)

    for view, ax_row in zip(VIEWS, axes, strict=False):
        for ax, hemi in zip(ax_row, HEMISPHERES, strict=False):
            if hemi == "both":
                add_both_panel(fig=fig, parent_ax=ax, fsaverage_sulcal=fsaverage_sulcal, view=view)
                continue

            plot_surf(
                surf_map=fsaverage_sulcal,
                hemi=hemi,
                view=view,
                figure=fig,
                axes=ax,
                title=f"{hemi} - {view}",
                colorbar=False,
                symmetric_cmap=False,
                bg_on_data=True,
                bg_map=fsaverage_sulcal,
                cmap="gray",
            )
            ax.set_axis_off()

    fig.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.95, wspace=0.06, hspace=0.12)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = render_docs_gray_grid(output_path=args.output, dpi=args.dpi)
    print(f"Saved docs-style grayscale surface grid to: {output_path}")


if __name__ == "__main__":
    main()
