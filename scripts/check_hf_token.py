import sys

from dotenv import load_dotenv

from oc_p02_fashion_segmentation.hf_api import HFAPIError, get_hf_token, whoami


def main() -> None:
    # Local development: populate the environment from .env (no-op if the file
    # is absent, and real environment variables always take precedence).
    load_dotenv()

    try:
        user = whoami(get_hf_token())
    except HFAPIError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"Welcome, {user['name']}!")


if __name__ == "__main__":
    main()
