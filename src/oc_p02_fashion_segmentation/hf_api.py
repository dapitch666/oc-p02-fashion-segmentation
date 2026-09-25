import os

import requests

WHOAMI_URL = "https://huggingface.co/api/whoami-v2"


class HFAPIError(Exception):
    """Raised when the Hugging Face token or API request is unusable."""


def get_hf_tokens() -> str:
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

    if response.status_code == 401:
        raise HFAPIError(
            f"Hugging Face token is invalid. Status code: {response.status_code}"
        )
    if response.status_code == 403:
        raise HFAPIError(
            f"Hugging Face token is not allowed. Status code: {response.status_code}"
        )
    if response.status_code != 200:
        raise HFAPIError(
            f"Unexpected error occurred. Status code: {response.status_code}"
        )
    return response.json()
