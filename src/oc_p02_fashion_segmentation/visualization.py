"""
Plotting helpers for images and segmentation masks.

Displays an image next to its segmentation mask, coloured with
CLASS_CMAP and labelled with a legend of the classes it contains.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from oc_p02_fashion_segmentation.constants import ID_TO_CLASS, NUM_CLASSES

CLASS_CMAP = ListedColormap(plt.cm.tab20.colors[:NUM_CLASSES])


def display_segmentation(image_array, mask_array):
    """
    Display an image and its segmentation mask side by side.

    Args:
        image_array (np.ndarray): Image to display.
        mask_array (np.ndarray): Segmentation mask with class indices.
    """
    fig, (ax_image, ax_mask) = plt.subplots(1, 2, figsize=(10, 7))

    ax_image.imshow(image_array)
    ax_image.set_title("Image")
    ax_image.axis("off")

    ax_mask.imshow(mask_array, cmap=CLASS_CMAP, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
    ax_mask.set_title("Mask")
    ax_mask.axis("off")

    legend_handles = [
        Patch(color=CLASS_CMAP(class_id), label=ID_TO_CLASS[class_id])
        for class_id in np.unique(mask_array)
    ]
    ax_mask.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    plt.show()
