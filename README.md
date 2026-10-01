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

