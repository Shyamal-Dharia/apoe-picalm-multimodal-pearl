from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.plotting import plot_surf


DEFAULT_OUTPUT = Path("figures/nilearn_surface_gray_hd.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a plain grayscale cortical surface image with nilearn.plotting.plot_surf."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--mesh",
        choices=["fsaverage", "fsaverage5"],
        default="fsaverage5",
        help="Surface mesh resolution. fsaverage is slower but denser.",
    )
    parser.add_argument(
        "--surface",
        choices=["pial", "infl"],
        default="pial",
        help="Surface geometry to render.",
    )
    parser.add_argument(
        "--layout",
        choices=["quad", "dorsal_pair"],
        default="dorsal_pair",
        help="Surface view layout.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Output DPI.",
    )
    parser.add_argument(
        "--darkness",
        type=float,
        default=0.6,
        help="Sulcal shading strength used by plot_surf.",
    )
    return parser.parse_args()


def render_surface(
    output_path: Path,
    mesh: str,
    surface: str,
    layout: str,
    dpi: int,
    darkness: float,
) -> Path:
    fsavg = datasets.fetch_surf_fsaverage(mesh=mesh)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_size = (12, 12) if layout == "quad" else (12, 6)
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    fig.patch.set_facecolor("white")

    if layout == "quad":
        views = [
            ("left", "lateral", getattr(fsavg, f"{surface}_left"), fsavg.sulc_left, (2, 2, 1)),
            ("left", "medial", getattr(fsavg, f"{surface}_left"), fsavg.sulc_left, (2, 2, 2)),
            ("right", "medial", getattr(fsavg, f"{surface}_right"), fsavg.sulc_right, (2, 2, 3)),
            ("right", "lateral", getattr(fsavg, f"{surface}_right"), fsavg.sulc_right, (2, 2, 4)),
        ]
        axes_mode = "subplot"
    else:
        views = [
            ("left", "dorsal", getattr(fsavg, f"{surface}_left"), fsavg.sulc_left, [0.02, 0.02, 0.51, 0.96]),
            ("right", "dorsal", getattr(fsavg, f"{surface}_right"), fsavg.sulc_right, [0.47, 0.02, 0.51, 0.96]),
        ]
        axes_mode = "manual"

    for hemi, view, surf_mesh, bg_map, subplot in views:
        if axes_mode == "subplot":
            ax = fig.add_subplot(*subplot, projection="3d")
        else:
            ax = fig.add_axes(subplot, projection="3d")
        plot_surf(
            surf_mesh=surf_mesh,
            surf_map=None,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            engine="matplotlib",
            colorbar=False,
            alpha=1,
            darkness=darkness,
            axes=ax,
            figure=fig,
        )
        ax.set_axis_off()

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.0, hspace=0.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = render_surface(
        output_path=args.output,
        mesh=args.mesh,
        surface=args.surface,
        layout=args.layout,
        dpi=args.dpi,
        darkness=args.darkness,
    )
    print(f"Saved grayscale surface image to: {output_path}")


if __name__ == "__main__":
    main()
