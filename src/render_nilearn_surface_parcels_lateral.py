from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import datasets, image, surface
from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf


DEFAULT_OUTPUT = Path("figures/nilearn_surface_schaefer400_lateral_colors.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a light left/right lateral cortical parcellation with bright colors."
    )
    parser.add_argument(
        "--n-rois",
        type=int,
        choices=[300, 400],
        default=400,
        help="Schaefer atlas resolution available in nilearn. 360 is not available here.",
    )
    parser.add_argument(
        "--mesh",
        choices=["fsaverage5", "fsaverage"],
        default="fsaverage5",
        help="Surface mesh resolution.",
    )
    parser.add_argument(
        "--surface",
        choices=["pial", "infl"],
        default="pial",
        help="Surface geometry to render. Use pial to avoid the inflated smooth look.",
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
        help="Output DPI.",
    )
    parser.add_argument(
        "--darkness",
        type=float,
        default=0.12,
        help="Sulcal shading strength. Lower values produce a lighter look.",
    )
    return parser.parse_args()


def build_bright_region_cmap(n_rois: int, seed: int = 7) -> ListedColormap:
    rng = np.random.default_rng(seed)
    hues = np.linspace(0.0, 1.0, n_rois, endpoint=False)
    rng.shuffle(hues)

    colors: list[np.ndarray] = []
    for hue in hues:
        saturation = 0.62 + 0.22 * rng.random()
        value = 0.88 + 0.10 * rng.random()
        colors.append(matplotlib.colors.hsv_to_rgb((hue, saturation, value)))

    # Label 0 stays near-white for background / unlabeled vertices.
    return ListedColormap(np.vstack([[0.97, 0.97, 0.97], colors]))


def project_volume_labels_to_surface(
    atlas_img,
    pial_mesh,
    white_mesh,
    n_rois: int,
) -> np.ndarray:
    surf_values = surface.vol_to_surf(
        atlas_img,
        surf_mesh=pial_mesh,
        inner_mesh=white_mesh,
        kind="depth",
        depth=[0.5],
        interpolation="nearest",
    )
    surf_labels = np.rint(surf_values).astype(int)
    surf_labels[(surf_labels < 0) | (surf_labels > n_rois)] = 0
    return surf_labels


def render_lateral_parcels(
    n_rois: int,
    mesh: str,
    surface: str,
    output_path: Path,
    dpi: int,
    darkness: float,
) -> Path:
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=7,
        resolution_mm=1,
    )
    atlas_img = image.load_img(atlas.maps)
    fsavg = datasets.fetch_surf_fsaverage(mesh=mesh)
    sulcal = load_fsaverage_data(data_type="sulcal")
    cmap = build_bright_region_cmap(n_rois=n_rois)

    left_labels = project_volume_labels_to_surface(
        atlas_img=atlas_img,
        pial_mesh=fsavg.pial_left,
        white_mesh=fsavg.white_left,
        n_rois=n_rois,
    )
    right_labels = project_volume_labels_to_surface(
        atlas_img=atlas_img,
        pial_mesh=fsavg.pial_right,
        white_mesh=fsavg.white_right,
        n_rois=n_rois,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5), dpi=dpi)
    fig.patch.set_facecolor("white")

    panels = [
        (
            "left",
            left_labels,
            getattr(fsavg, f"{surface}_left"),
            sulcal.data.parts["left"],
            [0.01, 0.04, 0.485, 0.92],
        ),
        (
            "right",
            right_labels,
            getattr(fsavg, f"{surface}_right"),
            sulcal.data.parts["right"],
            [0.505, 0.04, 0.485, 0.92],
        ),
    ]

    for hemi, surf_map, surf_mesh, bg_map, rect in panels:
        ax = fig.add_axes(rect, projection="3d")
        plot_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            hemi=hemi,
            view="lateral",
            figure=fig,
            axes=ax,
            colorbar=False,
            cmap=cmap,
            bg_map=bg_map,
            bg_on_data=True,
            darkness=darkness,
            vmin=0,
            vmax=n_rois,
            avg_method="median",
        )
        ax.set_axis_off()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = render_lateral_parcels(
        n_rois=args.n_rois,
        mesh=args.mesh,
        surface=args.surface,
        output_path=args.output,
        dpi=args.dpi,
        darkness=args.darkness,
    )
    print(f"Saved lateral parcel image to: {output_path}")


if __name__ == "__main__":
    main()
