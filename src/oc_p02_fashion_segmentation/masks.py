"""
Utilities for handling segmentation masks returned by the Hugging
Face API.

Decodes the base64-encoded masks for each class and combines them
into a single mask of class IDs, using the IDs defined in CLASS_MAPPING,
then saves it as a PNG file.

get_image_dimensions, decode_base64_mask and create_masks are copied from
the notebook huggingface_api_cloth_seg.ipynb, provided with the project
instructions.
"""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from oc_p02_fashion_segmentation.constants import CLASS_MAPPING, NUM_CLASSES


def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size


def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)


def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros(
        (height, width), dtype=np.uint8
    )  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result["label"]
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result["mask"], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0  # Class ID for Background is 0

    return combined_mask


def save_mask(mask_array: np.ndarray, mask_path: str | Path):
    """
    Save a segmentation mask as a single-channel PNG image.

    PNG is lossless, so every pixel keeps its exact class ID. A lossy
    format like JPEG would alter pixels near class edges and turn them
    into other classes.

    Args:
        mask_array (np.ndarray): 2D segmentation mask with class indices.
        mask_path (str | Path): Destination file, with a .png extension.
            Missing parent folders are created.

    Raises:
        ValueError: If the path does not end in .png, the mask is not 2D,
            or it contains values that are not class IDs.
    """
    mask_path = Path(mask_path)
    if mask_path.suffix.lower() != ".png":
        raise ValueError(f"Masks must be saved as PNG, got: {mask_path.name}")
    if mask_array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask_array.shape}")
    if mask_array.min() < 0 or mask_array.max() >= NUM_CLASSES:
        raise ValueError(f"Mask values must be class IDs from 0 to {NUM_CLASSES - 1}")

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_array.astype(np.uint8)).save(mask_path, format="PNG")


def mask_path_for(image_path: Path, output_dir: Path) -> Path:
    """Name the mask like the dataset masks: image_0.png -> mask_0.png."""
    return output_dir / image_path.name.replace("image_", "mask_")
