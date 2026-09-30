import os
from pathlib import Path

import requests

WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
SEGMENTATION_URL = (
    "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
)


class HFAPIError(Exception):
    """Raised when the Hugging Face token or API request is unusable."""


def get_hf_token() -> str:
    """Return the Hugging Face token from the environment variables."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise HFAPIError("Hugging Face token is not set in the environment variables.")
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
        raise HFAPIError("The request timed out") from e
    except requests.RequestException as e:
        raise HFAPIError(f"Error occurred while making the request: {e}") from e

    check_status_code(response)
    return response.json()


def segment_image(
    token: str, image_path: str | Path, timeout: float = 30
) -> list[dict]:
    """
    Send a PNG image to the segmentation model and return its predictions.

    Args:
        token (str): Hugging Face token.
        image_path (str | Path): Path to the PNG image to segment.
        timeout (float): Maximum time to wait for the API, in seconds.

    Returns:
        list[dict]: One dictionary per detected class, with a 'label' key
            (class name) and a 'mask' key (base64-encoded PNG mask).

    Raises:
        HFAPIError: If the request fails, times out, returns an error
            status code or a response that is not a list of predictions.
    """
    image_bytes = Path(image_path).read_bytes()
    try:
        response = requests.post(
            SEGMENTATION_URL,
            headers={"Content-Type": "image/png", **build_auth_headers(token)},
            data=image_bytes,
            timeout=timeout,
        )
    except requests.Timeout as e:
        raise HFAPIError("The request timed out") from e
    except requests.RequestException as e:
        raise HFAPIError(f"Error occurred while making the request: {e}") from e

    check_status_code(response)
    try:
        predictions = response.json()
    except requests.JSONDecodeError as e:
        raise HFAPIError(
            f"The API response is not valid JSON: {response.text[:200]}"
        ) from e
    if not isinstance(predictions, list):
        raise HFAPIError(f"Unexpected API response: {predictions}")
    return predictions


def check_status_code(response: requests.Response) -> None:
    """Raise an HFAPIError if the Hugging Face API returned an error status code."""
    if response.status_code == 401:
        raise HFAPIError(
            f"Hugging Face token is invalid. Status code: {response.status_code}"
        )
    if response.status_code == 403:
        raise HFAPIError(
            f"Hugging Face token is not allowed. Status code: {response.status_code}"
        )
    if response.status_code == 429:
        raise HFAPIError(
            f"Too many requests, the rate limit is reached. Status code: {response.status_code}"
        )
    if response.status_code == 503:
        raise HFAPIError(
            f"The model is loading or unavailable, retry in a moment. Status code: {response.status_code}"
        )
    if response.status_code != 200:
        raise HFAPIError(
            f"Unexpected error occurred. Status code: {response.status_code}, response: {response.text[:200]}"
        )
