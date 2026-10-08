from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.image import load_img, resample_to_img
from nilearn.plotting import plot_roi


DEFAULT_OUTPUT = Path("figures/nilearn_harvard_oxford_atlas_hd.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a high-definition brain atlas image using Nilearn."
    )
    parser.add_argument(
        "--atlas",
        choices=["harvard_oxford", "schaefer_100", "schaefer_200"],
        default="harvard_oxford",
        help="Atlas to render.",
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
        default=400,
        help="Output DPI. Higher values increase detail and file size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.78,
        help="Atlas overlay opacity between 0 and 1.",
    )
    return parser.parse_args()


def fetch_atlas_image(atlas_name: str):
    if atlas_name == "harvard_oxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
        return load_img(atlas.maps), "Harvard-Oxford Cortical Atlas (1 mm)"

    if atlas_name == "schaefer_100":
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=100,
            yeo_networks=7,
            resolution_mm=1,
        )
        return load_img(atlas.maps), "Schaefer 2018 Atlas (100 parcels, 7 networks, 1 mm)"

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=200,
        yeo_networks=7,
        resolution_mm=1,
    )
    return load_img(atlas.maps), "Schaefer 2018 Atlas (200 parcels, 7 networks, 1 mm)"


def render_hd_atlas(atlas_name: str, output_path: Path, dpi: int, alpha: float) -> Path:
    atlas_img, atlas_title = fetch_atlas_image(atlas_name)
    bg_img = datasets.load_mni152_template(resolution=1)
    atlas_resampled = resample_to_img(
        atlas_img,
        bg_img,
        interpolation="nearest",
        copy_header=True,
        force_resample=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.975,
        atlas_title,
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.947,
        f"Rendered with Nilearn at {dpi} DPI",
        ha="center",
        va="top",
        fontsize=11,
        color="#4b5563",
    )

    ortho_display = plot_roi(
        atlas_resampled,
        bg_img=bg_img,
        figure=fig,
        axes=(0.04, 0.40, 0.92, 0.48),
        display_mode="ortho",
        cut_coords=(0, -24, 18),
        cmap="nipy_spectral",
        alpha=alpha,
        dim=-0.3,
        black_bg=False,
        annotate=True,
        draw_cross=False,
        colorbar=False,
    )

    axial_display = plot_roi(
        atlas_resampled,
        bg_img=bg_img,
        figure=fig,
        axes=(0.05, 0.08, 0.90, 0.20),
        display_mode="z",
        cut_coords=12,
        cmap="nipy_spectral",
        alpha=alpha,
        dim=-0.35,
        black_bg=False,
        annotate=False,
        draw_cross=False,
        colorbar=False,
    )

    fig.text(0.5, 0.31, "Axial slice montage", ha="center", va="bottom", fontsize=13)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    ortho_display.close()
    axial_display.close()
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = render_hd_atlas(
        atlas_name=args.atlas,
        output_path=args.output,
        dpi=args.dpi,
        alpha=args.alpha,
    )
    print(f"Saved HD atlas image to: {output_path}")


if __name__ == "__main__":
    main()
