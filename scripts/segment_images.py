"""
Segment every image of the input directory with the Hugging Face API and
save the predicted masks to the output directory.

Images whose mask already exists are skipped, unless --all is given.
Temporary API errors are retried with an exponential backoff, or after the
delay given by the Retry-After header. Authentication errors stop the script.

One line per processed image is appended to a CSV log file.
"""

import argparse
import csv
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from oc_p02_fashion_segmentation.hf_api import (
    HFAPIError,
    HFAuthError,
    HFCreditsError,
    HFTransientError,
    SegmentationResult,
    get_hf_token,
    request_segmentation,
)
from oc_p02_fashion_segmentation.masks import (
    create_masks,
    get_image_dimensions,
    mask_path_for,
    save_mask,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "input" / "IMG"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "output"
DEFAULT_LOG_FILE = PROJECT_DIR / "data" / "segmentation_log.csv"

LOG_FIELDS = [
    "run_id",
    "image",
    "status",
    "retries",
    "duration_s",
    "api_time_s",
    "size_bytes",
    "width",
    "height",
    "response_bytes",
    "mask_bytes",
    "timestamp",
    "request_id",
    "attempt_errors",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every image, even those whose mask already exists.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries after a temporary error (default: 3).",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=2.0,
        help="Delay before the first retry in seconds, doubled at each retry "
        "(default: 2). Ignored when the API sends a Retry-After header.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout of each API request in seconds (default: 30).",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Downscale images to fit in 512 x 512 before sending them. "
        "Reduces the payload, but may lose details.",
    )
    parser.add_argument(
        "--jpeg",
        action="store_true",
        help="Convert images to JPEG before sending them. Smaller payload, but "
        "lowers the IoU more than --resize.",
    )
    return parser.parse_args()


def image_number(image_path: Path) -> int:
    """Return the number in the file name, to sort image_2 before image_10."""
    match = re.search(r"\d+", image_path.stem)
    return int(match.group()) if match else -1


def describe_error(error: HFAPIError) -> str:
    """Return a short label for a failed attempt: status code or exception name."""
    if error.status_code is not None:
        return str(error.status_code)
    cause = error.__cause__
    return type(cause if cause is not None else error).__name__


def segment_with_retries(
    token: str,
    image_path: Path,
    max_retries: int,
    base_delay: float,
    timeout: float,
    resize: bool = False,
    to_jpeg: bool = False,
) -> tuple[SegmentationResult, list[str]]:
    """
    Call request_segmentation, retrying on HFTransientError.

    Returns:
        tuple[SegmentationResult, list[str]]: The result of the successful
            attempt and the labels of the failed attempts before it.

    Raises:
        HFAPIError: If the last retry still fails, or on a non temporary
            error, without retrying. Its `retries` and `attempt_errors`
            attributes hold the number of retries and the failed attempts.
    """
    attempt_errors = []
    for retries in range(max_retries + 1):
        try:
            result = request_segmentation(
                token, image_path, timeout=timeout, resize=resize, to_jpeg=to_jpeg
            )
            return result, attempt_errors
        except HFAPIError as e:
            attempt_errors.append(describe_error(e))
            if not isinstance(e, HFTransientError) or retries == max_retries:
                e.retries = retries
                e.attempt_errors = attempt_errors
                raise
            delay = (
                e.retry_after if e.retry_after is not None else base_delay * 2**retries
            )
            print(f"  {e} -> retry {retries + 1}/{max_retries} in {delay:.1f} s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def process_image(
    token: str,
    image_path: Path,
    output_dir: Path,
    max_retries: int,
    base_delay: float,
    timeout: float,
    run_id: str,
    resize: bool = False,
    to_jpeg: bool = False,
) -> tuple[dict, Exception | None]:
    """Segment one image, save its mask and return the log row and the error."""
    width, height = get_image_dimensions(image_path)
    row = {
        "run_id": run_id,
        "image": image_path.name,
        "size_bytes": image_path.stat().st_size,
        "width": width,
        "height": height,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    error = None
    attempt_errors = []
    start = time.perf_counter()
    try:
        result, attempt_errors = segment_with_retries(
            token, image_path, max_retries, base_delay, timeout, resize, to_jpeg
        )
        row["api_time_s"] = round(result.api_time_s, 3)
        row["request_id"] = result.request_id
        row["response_bytes"] = result.response_bytes

        mask_path = mask_path_for(image_path, output_dir)
        mask_array = create_masks(result.predictions, width, height)
        save_mask(mask_array, mask_path)
        row["mask_bytes"] = mask_path.stat().st_size
        row["status"] = "success"
    except (HFAPIError, OSError, ValueError, KeyError) as e:
        error = e
        attempt_errors = getattr(e, "attempt_errors", attempt_errors)
        row["status"] = "failure"
        row["error"] = f"{type(e).__name__}: {e}"
    row["duration_s"] = round(time.perf_counter() - start, 3)
    row["retries"] = getattr(error, "retries", len(attempt_errors))
    row["attempt_errors"] = ";".join(attempt_errors)
    return row, error


def main() -> None:
    load_dotenv()
    args = parse_args()

    try:
        token = get_hf_token()
    except HFAuthError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    images = sorted(args.input_dir.glob("*.png"), key=image_number)
    if not args.all:
        images = [p for p in images if not mask_path_for(p, args.output_dir).exists()]
    print(f"{len(images)} image(s) to process")
    if not images:
        return

    # Groups the rows of this run in the log, and sorts in chronological order.
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.log_file.exists()
    successes = failures = 0
    with args.log_file.open("a", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()

        for i, image_path in enumerate(images, start=1):
            print(f"[{i}/{len(images)}] {image_path.name}")
            row, error = process_image(
                token,
                image_path,
                args.output_dir,
                args.max_retries,
                args.base_delay,
                args.timeout,
                run_id,
                args.resize,
                args.jpeg,
            )
            writer.writerow(row)
            log.flush()  # Keep the log up to date if the script is interrupted.

            if isinstance(error, HFAuthError):
                # Every following request would fail the same way.
                print(f"Authentication error, stopping: {error}", file=sys.stderr)
                sys.exit(1)
            if isinstance(error, HFCreditsError):
                print(f"No remaining credits, stopping: {error}", file=sys.stderr)
                sys.exit(1)

            if row["status"] == "success":
                successes += 1
            else:
                failures += 1
                print(f"  failed: {row['error']}", file=sys.stderr)

    print(
        f"Done: {successes} success(es), {failures} failure(s). "
        f"Run {run_id}, log: {args.log_file}"
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
