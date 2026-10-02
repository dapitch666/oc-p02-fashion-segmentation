import io
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from PIL import Image

WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
SEGMENTATION_URL = (
    "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
# The model preprocessor resizes images to 512 x 512, larger images only cost bandwidth.
MAX_IMAGE_SIZE = 512
JPEG_QUALITY = 90
SENDABLE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


class HFAPIError(Exception):
    """
    Raised when the Hugging Face token or API request is unusable.

    Attributes:
        status_code (int | None): HTTP status code of the response, or None
            if no response was received.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class HFAuthError(HFAPIError):
    """Raised when the token is missing, invalid or not allowed. Retrying is useless."""


class HFTransientError(HFAPIError):
    """
    Raised on a temporary failure (timeout, connection error, rate limit,
    model loading or unavailable). The same request may succeed later.

    Attributes:
        retry_after (float | None): Delay in seconds requested by the API
            through the Retry-After header, or None if it was not provided.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message, status_code)
        self.retry_after = retry_after


@dataclass
class SegmentationResult:
    """
    Predictions of the segmentation model, with information on the request.

    Attributes:
        predictions (list[dict]): One dictionary per detected class, with a
            'label' key (class name) and a 'mask' key (base64-encoded PNG mask).
        api_time_s (float): Time between sending the request and receiving
            the response headers, in seconds.
        request_id (str | None): Value of the x-request-id response header.
        response_bytes (int): Size of the response body, in bytes.
    """

    predictions: list[dict]
    api_time_s: float
    request_id: str | None
    response_bytes: int


def get_hf_token() -> str:
    """Return the Hugging Face token from the environment variables."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise HFAuthError("Hugging Face token is not set in the environment variables.")
    return token


def build_auth_headers(token: str) -> dict[str, str]:
    """Return the Hugging Face header for API requests."""
    return {"Authorization": f"Bearer {token}"}


def whoami(token: str, timeout: float = 5) -> dict:
    """Query the Hugging Face whoami endpoint and return the account info."""
    try:
        response = requests.get(
            WHOAMI_URL,
            headers=build_auth_headers(token),
            timeout=timeout,
        )
    except requests.Timeout as e:
        raise HFTransientError("The request timed out") from e
    except requests.ConnectionError as e:
        raise HFTransientError(f"Could not connect to the API: {e}") from e
    except requests.RequestException as e:
        raise HFAPIError(f"Error occurred while making the request: {e}") from e

    check_status_code(response)
    return response.json()


def segment_image(
    token: str,
    image_path: str | Path,
    timeout: float = 30,
    resize: bool = False,
    to_jpeg: bool = False,
) -> list[dict]:
    """
    Send an image to the segmentation model and return its predictions.

    Same as request_segmentation, without the information on the request.

    Args:
        token (str): Hugging Face token.
        image_path (str | Path): Path to the image to segment.
        timeout (float): Maximum time to wait for the API, in seconds.
        resize (bool): Downscale the image to fit in 512 x 512 before sending it.
        to_jpeg (bool): Convert the image to JPEG before sending it.

    Returns:
        list[dict]: One dictionary per detected class, with a 'label' key
            (class name) and a 'mask' key (base64-encoded PNG mask).

    Raises:
        HFAuthError: If the token is invalid or not allowed.
        HFTransientError: If the request times out, cannot connect, or the
            API is rate limited or temporarily unavailable.
        HFAPIError: If the request fails for another reason, returns an
            unexpected status code or a response that is not a list of
            predictions.
    """
    return request_segmentation(
        token, image_path, timeout, resize=resize, to_jpeg=to_jpeg
    ).predictions


def prepare_image_for_api(
    image_path: str | Path,
    resize: bool = False,
    to_jpeg: bool = False,
    max_size: int = MAX_IMAGE_SIZE,
    quality: int = JPEG_QUALITY,
) -> tuple[bytes, str]:
    """
    Read an image and optionally resize it and convert it to JPEG.

    The format is detected from the file content. With both options off,
    the file bytes are returned untouched. Smaller payloads reduce the
    upload bandwidth, but the model resizes images to 512 x 512 anyway
    and both options slightly change the predictions, see
    notebooks/04_resize_and_jpeg.ipynb.

    Args:
        image_path (str | Path): Path to the image.
        resize (bool): Downscale the image to fit in max_size x max_size,
            keeping the aspect ratio. Images are never enlarged. The format
            is kept, so a PNG stays a lossless PNG, unless to_jpeg is set.
        to_jpeg (bool): Re-encode the image as JPEG, which is lossy.
        max_size (int): Maximum width and height when resizing, in pixels.
        quality (int): JPEG and WebP quality, from 1 to 95.

    Returns:
        tuple[bytes, str]: The image bytes and their MIME type, for example
            'image/png'.

    Raises:
        OSError: If the file cannot be read or is not an image.
    """
    with Image.open(image_path) as image:
        image_format = "JPEG" if to_jpeg else image.format
        if image_format not in SENDABLE_FORMATS:
            image_format = "PNG"  # Lossless fallback, e.g. for GIF or TIFF.
        needs_resize = resize and max(image.size) > max_size
        if image_format == image.format and not needs_resize:
            return Path(image_path).read_bytes(), Image.MIME[image_format]

        if image.mode not in ("RGB", "RGBA", "L"):
            has_alpha = "A" in image.mode or "transparency" in image.info
            image = image.convert("RGBA" if has_alpha else "RGB")
        if image_format in ("JPEG", "BMP") and image.mode == "RGBA":
            image = image.convert("RGB")
        if needs_resize:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format=image_format, quality=quality)
    return buffer.getvalue(), Image.MIME[image_format]


def request_segmentation(
    token: str,
    image_path: str | Path,
    timeout: float = 30,
    resize: bool = False,
    to_jpeg: bool = False,
) -> SegmentationResult:
    """
    Send an image to the segmentation model.

    The image is sent as is, with its format detected from the file content,
    unless an option is set. The returned masks have the size of the sent
    image, which differs from the original when resize is set.

    Args:
        token (str): Hugging Face token.
        image_path (str | Path): Path to the image to segment.
        timeout (float): Maximum time to wait for the API, in seconds.
        resize (bool): Downscale the image to fit in 512 x 512 before sending it.
        to_jpeg (bool): Convert the image to JPEG before sending it.

    Returns:
        SegmentationResult: The predictions, with the request time, the
            request id and the response size.

    Raises:
        HFAuthError: If the token is invalid or not allowed.
        HFTransientError: If the request times out, cannot connect, or the
            API is rate limited or temporarily unavailable.
        HFAPIError: If the request fails for another reason, returns an
            unexpected status code or a response that is not a list of
            predictions.
    """
    try:
        image_bytes, content_type = prepare_image_for_api(
            image_path, resize=resize, to_jpeg=to_jpeg
        )
    except OSError as e:
        raise HFAPIError(f"Could not read or encode the image: {e}") from e
    try:
        response = requests.post(
            SEGMENTATION_URL,
            headers={"Content-Type": content_type, **build_auth_headers(token)},
            data=image_bytes,
            timeout=timeout,
        )
    except requests.Timeout as e:
        raise HFTransientError("The request timed out") from e
    except requests.ConnectionError as e:
        raise HFTransientError(f"Could not connect to the API: {e}") from e
    except requests.RequestException as e:
        raise HFAPIError(f"Error occurred while making the request: {e}") from e

    check_status_code(response)
    try:
        predictions = response.json()
    except requests.JSONDecodeError as e:
        raise HFAPIError(
            f"The API response is not valid JSON: {response.text[:200]}",
            response.status_code,
        ) from e
    if not isinstance(predictions, list):
        raise HFAPIError(
            f"Unexpected API response: {predictions}", response.status_code
        )
    return SegmentationResult(
        predictions=predictions,
        api_time_s=response.elapsed.total_seconds(),
        request_id=response.headers.get("x-request-id"),
        response_bytes=len(response.content),
    )


def parse_retry_after(response: requests.Response) -> float | None:
    """
    Return the delay in seconds from the Retry-After header, or None if absent.

    The header holds either a number of seconds or an HTTP date.
    """
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def check_status_code(response: requests.Response) -> None:
    """Raise an HFAPIError subclass if the API returned an error status code."""
    if response.status_code == 401:
        raise HFAuthError(
            f"Hugging Face token is invalid. Status code: {response.status_code}",
            response.status_code,
        )
    if response.status_code == 403:
        raise HFAuthError(
            f"Hugging Face token is not allowed. Status code: {response.status_code}",
            response.status_code,
        )
    if response.status_code == 429:
        raise HFTransientError(
            f"Too many requests, the rate limit is reached. Status code: {response.status_code}",
            status_code=response.status_code,
            retry_after=parse_retry_after(response),
        )
    if response.status_code == 503:
        raise HFTransientError(
            f"The model is loading or unavailable, retry in a moment. Status code: {response.status_code}",
            status_code=response.status_code,
            retry_after=parse_retry_after(response),
        )
    if response.status_code in TRANSIENT_STATUS_CODES:
        raise HFTransientError(
            f"Temporary server error. Status code: {response.status_code}, response: {response.text[:200]}",
            status_code=response.status_code,
            retry_after=parse_retry_after(response),
        )
    if response.status_code != 200:
        raise HFAPIError(
            f"Unexpected error occurred. Status code: {response.status_code}, response: {response.text[:200]}",
            response.status_code,
        )
