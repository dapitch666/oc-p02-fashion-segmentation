import os

import requests

WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"


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

def query(token: str, filename: str):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers={"Content-Type": "image/jpeg", **build_auth_headers(token)}, data=data)
    return response.json()