"""
Plotting helpers for images and segmentation masks.

Displays a segmentation mask next to its image or overlaid on it,
coloured with CLASS_CMAP and labelled with a legend of the classes
it contains, and compares a predicted mask with its reference mask.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from oc_p02_fashion_segmentation.constants import ID_TO_CLASS, NUM_CLASSES

# Clothes use vivid colours of the Tailwind CSS palette.
# Background and body parts use neutral shades, so the clothes stand out.
CLASS_COLORS = {
    "Background": "#e0f2fe",  # sky-100
    "Hat": "#4ade80",  # green-400
    "Hair": "#fef9c3",  # yellow-100
    "Sunglasses": "#fde047",  # yellow-300
    "Upper-clothes": "#c084fc",  # purple-400
    "Skirt": "#ef4444",  # red-500
    "Pants": "#4f46e5",  # indigo-600
    "Dress": "#bef264",  # lime-300
    "Belt": "#67e8f9",  # cyan-300
    "Left-shoe": "#f472b6",  # pink-400
    "Right-shoe": "#60a5fa",  # blue-400
    "Face": "#ffe4e6",  # rose-100
    "Left-leg": "#a3a3a3",  # neutral-400
    "Right-leg": "#e5e5e5",  # neutral-200
    "Left-arm": "#e3e7e8",  # mist-200
    "Right-arm": "#9ca8ab",  # mist-400
    "Bag": "#fb923c",  # orange-400
    "Scarf": "#0f766e",  # teal-700
}

CLASS_CMAP = ListedColormap(
    [CLASS_COLORS[ID_TO_CLASS[class_id]] for class_id in range(NUM_CLASSES)]
)

# A single vivid colour for the pixels where two masks disagree
DISAGREEMENT_COLOR = "#e6ff00"  # neon yellow


def display_segmentation(image_array, mask_array):
    """
    Display an image and its segmentation mask side by side.

    Args:
        image_array (np.ndarray): Image to display.
        mask_array (np.ndarray): Segmentation mask with class indices.
    """
    _fig, (ax_image, ax_mask) = plt.subplots(1, 2, figsize=(10, 7))

    ax_image.imshow(image_array)
    ax_image.set_title("Image")
    ax_image.axis("off")

    ax_mask.imshow(
        mask_array,
        cmap=CLASS_CMAP,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        interpolation="nearest",
    )
    ax_mask.set_title("Mask")
    ax_mask.axis("off")

    ax_mask.legend(
        handles=class_legend_handles(np.unique(mask_array)),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    plt.tight_layout()
    plt.show()


def display_overlay(image_array, mask_array, alpha=0.6):
    """
    Display a segmentation mask overlaid on its image.

    Args:
        image_array (np.ndarray): Image to display.
        mask_array (np.ndarray): Segmentation mask with class indices,
            same height and width as the image.
        alpha (float): Opacity of the mask, from 0 (invisible) to 1 (opaque).
            Defaults to 0.6, as in the Hugging Face widget.
    """
    _fig, ax = plt.subplots(figsize=(7, 7))

    draw_overlay(ax, image_array, mask_array, alpha)

    ax.legend(
        handles=class_legend_handles(np.unique(mask_array)),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    plt.tight_layout()
    plt.show()


def display_comparison(
    image_array, reference_mask, predicted_mask, alpha=0.6, image_dim=0.4
):
    """
    Compare a predicted mask with its reference mask, in four panels.

    From left to right: the image, the reference mask and the predicted
    mask overlaid on it, and the pixels where the two masks disagree,
    highlighted with DISAGREEMENT_COLOR on a dimmed copy of the image.
    The share of these pixels is shown in the title of the last panel.

    Args:
        image_array (np.ndarray): Image to display.
        reference_mask (np.ndarray): Mask annotated by hand, with class indices.
        predicted_mask (np.ndarray): Mask predicted by the model, with class indices.
        alpha (float): Opacity of the masks, from 0 (invisible) to 1 (opaque).
            Defaults to 0.6, as in display_overlay.
        image_dim (float): How much the image is dimmed under the
            disagreement, from 0 (unchanged) to 1 (black). Defaults to 0.4.

    Raises:
        ValueError: If the two masks do not have the same shape.
    """
    if reference_mask.shape != predicted_mask.shape:
        raise ValueError(
            f"Masks must have the same shape, got {reference_mask.shape} "
            f"and {predicted_mask.shape}"
        )

    disagreement = reference_mask != predicted_mask

    fig, (ax_image, ax_reference, ax_predicted, ax_disagreement) = plt.subplots(
        1, 4, figsize=(16, 6)
    )

    ax_image.imshow(image_array, cmap="gray" if image_array.ndim == 2 else None)
    ax_image.set_title("Image")
    ax_image.axis("off")

    draw_overlay(ax_reference, image_array, reference_mask, alpha)
    ax_reference.set_title("Reference mask")

    draw_overlay(ax_predicted, image_array, predicted_mask, alpha)
    ax_predicted.set_title("Predicted mask")

    ax_disagreement.imshow(image_array, cmap="gray" if image_array.ndim == 2 else None)
    # A semi-transparent black layer dims the image, whatever its dtype or number of channels
    ax_disagreement.imshow(
        np.zeros(disagreement.shape),
        cmap=ListedColormap(["black"]),
        alpha=image_dim,
    )
    # Masked where the masks agree, so only the disagreeing pixels are coloured.
    # Opaque, so the colour stays vivid on the dimmed image.
    ax_disagreement.imshow(
        np.ma.masked_where(~disagreement, disagreement),
        cmap=ListedColormap([DISAGREEMENT_COLOR]),
        interpolation="nearest",
    )
    ax_disagreement.set_title(f"Disagreement ({disagreement.mean():.1%} of pixels)")
    ax_disagreement.axis("off")

    # One shared legend for the classes of both masks
    legend_handles = class_legend_handles(
        set(np.unique(reference_mask)) | set(np.unique(predicted_mask))
    )
    plt.tight_layout()
    # Anchored below the axes: the inline backend crops the figure to fit it
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(len(legend_handles), 9),
        frameon=False,
    )
    plt.show()


def draw_overlay(ax, image_array, mask_array, alpha):
    """
    Draw a segmentation mask overlaid on its image, on the given axes.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        image_array (np.ndarray): Image to display.
        mask_array (np.ndarray): Segmentation mask with class indices,
            same height and width as the image.
        alpha (float): Opacity of the mask, from 0 (invisible) to 1 (opaque).
    """
    # A grayscale image is a 2D array: without cmap="gray", imshow would colour it with viridis
    ax.imshow(image_array, cmap="gray" if image_array.ndim == 2 else None)
    ax.imshow(
        mask_array,
        cmap=CLASS_CMAP,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        alpha=alpha,
        interpolation="nearest",
    )
    ax.axis("off")


def class_legend_handles(class_ids):
    """
    Build legend entries for the given classes.

    Args:
        class_ids (Iterable[int]): Class indices to show in the legend.

    Returns:
        list[Patch]: One entry per class, in class order.
    """
    return [
        Patch(color=CLASS_COLORS[ID_TO_CLASS[class_id]], label=ID_TO_CLASS[class_id])
        for class_id in sorted(class_ids)
    ]
