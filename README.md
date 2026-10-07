# Fashion Segmentation

Clothing segmentation on fashion images using the SegFormer model via the Hugging Face Inference API.

## Context

This project is part of the OpenClassrooms **AI Engineer** training program.

**ModeTrends**, a (fictional) digital marketing agency specialized in fashion,
is launching *Fashion Trend Intelligence*: a system that analyzes clothing
trends on social media to give brands early insight into emerging styles.

The first building block is **clothing segmentation**: identifying and isolating
each garment in a photo. This repository evaluates whether a pre-trained model,
[SegFormer B3 Clothes](https://huggingface.co/sayeed99/segformer_b3_clothes),
queried through the Hugging Face Inference API, is accurate enough for this task.
It also estimates the cost of running it at scale (500,000 images over 30 days).

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- A Hugging Face API token, provided through the `HF_TOKEN` environment variable

## Setup

```sh
uv sync
```

then create a `.env` file in the root of the project (or copy the `.env.example` file) with the following content:

```env
HF_TOKEN=<your_hugging_face_api_token>
```

The scripts load this file with `python-dotenv`. The library code in `src/` only reads the environment, so on a server or in a container you can skip the `.env` file and set `HF_TOKEN` as a regular environment variable (it takes precedence over the file).

Check that the token works:

```sh
uv run python scripts/check_hf_token.py
```

## Data

The test set was provided by OpenClassrooms with the project brief (archive
`top_influenceurs_2024`): 50 images of 400 x 600 pixels, each with a
hand-annotated reference mask. The images are not redistributed in this
repository (`data/` is git-ignored). To run the code, extract the archive into
`data/input/`, so that you get:

```text
data/
├── input/
│   ├── IMG/    # images: image_0.png, image_1.png, ...
│   └── Mask/   # reference masks: mask_0.png, mask_1.png, ...
├── output/                 # predicted masks, created by the batch script
└── segmentation_log.csv    # one line per processed image
```

Each pixel of a mask holds the ID of its class (18 classes, from `Background`
to `Scarf`, see `constants.py`).

## Batch segmentation

Segment all the images of `data/input/IMG` and save the masks to `data/output`:

```sh
uv run python scripts/segment_images.py
```

Images whose mask already exists are skipped; add `--all` to process them again.
Temporary errors (timeout, rate limit, 5xx) are retried up to 3 times, waiting
2, 4 then 8 seconds, or the delay from the `Retry-After` header. An
authentication error stops the script.

Each processed image adds a line to `data/segmentation_log.csv`. Run with
`--help` for all the options.

| Option | Description |
| --- | --- |
| `--input-dir`, `--output-dir`, `--log-file` | Change the default paths. |
| `--max-retries`, `--base-delay` | Number of retries (default 3) and delay before the first one in seconds (default 2, doubled at each retry). |
| `--timeout` | Timeout of each API request in seconds (default 30). |
| `--resize` | Downscale images to fit in 512 x 512 before sending them. Reduces the payload, but may lose details. |
| `--jpeg` | Convert images to JPEG before sending them. Smaller payload, but lowers the IoU more than `--resize`. |

## Notebooks

Run them in this order. The kernel is the `.venv` created by `uv sync`
(`ipykernel` is a dev dependency).

| Notebook | Content |
| --- | --- |
| `01_data_exploration` | Look at the dataset: image-mask pairs, class values, annotation errors (pairs 1, 26 and 29 have faulty masks). |
| `02_segmentation_api` | Segment one image with the API and compare the prediction with the reference mask. |
| `03_iou_metric` | IoU by class and mean IoU for one image, then for the whole dataset with two aggregation methods. Needs the masks in `data/output/`, produced by the batch script. |
| `04_resize_and_jpeg` | Effect of `--resize` and `--jpeg` on payload size, IoU and API time, and check that the model is deterministic. Can re-run the batch script, which needs `HF_TOKEN`. |

## Project structure

```text
src/oc_p02_fashion_segmentation/
├── constants.py       # class names, IDs and clothing classes
├── hf_api.py          # Hugging Face API client: token, requests, retries, errors
├── masks.py           # decoding of the API response, saving of the masks
├── metrics.py         # intersection, union and IoU
└── visualization.py   # image, mask, overlay and comparison plots
scripts/
├── check_hf_token.py  # check that HF_TOKEN is valid
└── segment_images.py  # batch segmentation
notebooks/             # see above
```

## Development

Install the pre-commit hooks once after `uv sync`:

```sh
uv run pre-commit install
```

They run [ruff](https://docs.astral.sh/ruff/) (lint and format, notebooks
included) and [nbstripout](https://github.com/kynan/nbstripout), which removes
the outputs of the notebooks: they are committed without outputs, run them to
see the results.
