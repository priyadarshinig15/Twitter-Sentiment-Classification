#!/usr/bin/env python3
"""Convert reviewed TweetClaw exports into the project training CSV shape."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = ("id", "entity", "sentiment", "text")

TEXT_FIELDS = (
    "text",
    "tweet",
    "content",
    "full_text",
    "body",
    "message",
)
ID_FIELDS = ("id", "tweet_id", "tweetId", "post_id", "postId")
ENTITY_FIELDS = (
    "entity",
    "keyword",
    "query",
    "topic",
    "author_username",
    "username",
    "screen_name",
    "user",
)
SENTIMENT_FIELDS = (
    "sentiment",
    "label",
    "sentiment_label",
    "classification",
    "target",
)

SENTIMENT_MAP = {
    "positive": "Positive",
    "pos": "Positive",
    "1": "Positive",
    "4": "Positive",
    "negative": "Negative",
    "neg": "Negative",
    "0": "Negative",
    "neutral": "Neutral",
    "neu": "Neutral",
    "2": "Neutral",
}


def as_text(value: object) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def merged_record(record: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(record)
    for nested_key in ("tweet", "post", "data"):
        nested = record.get(nested_key)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                merged.setdefault(str(key), value)
    return merged


def first_text(record: Mapping[str, Any], fields: Iterable[str]) -> str | None:
    for field in fields:
        value = as_text(record.get(field))
        if value is not None:
            return value
    return None


def normalize_sentiment(value: object) -> str | None:
    text = as_text(value)
    if text is None:
        return None
    return SENTIMENT_MAP.get(text.lower())


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSONL: {error}"
                ) from error
            if isinstance(row, Mapping):
                yield dict(row)
        return

    if isinstance(parsed, list):
        for row in parsed:
            if isinstance(row, Mapping):
                yield dict(row)
        return

    if isinstance(parsed, Mapping):
        for key in ("records", "tweets", "items", "results", "data"):
            rows = parsed.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, Mapping):
                        yield dict(row)
                return
        yield dict(parsed)


def iter_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        yield from iter_csv_records(path)
    else:
        yield from iter_json_records(path)


def convert_rows(path: Path, default_entity: str) -> Iterator[dict[str, str]]:
    fallback_id = 1
    for record in iter_records(path):
        merged = merged_record(record)
        text = first_text(merged, TEXT_FIELDS)
        sentiment = normalize_sentiment(first_text(merged, SENTIMENT_FIELDS))
        if text is None or sentiment is None:
            continue

        row_id = first_text(merged, ID_FIELDS) or str(fallback_id)
        entity = first_text(merged, ENTITY_FIELDS) or default_entity
        fallback_id += 1
        yield {
            "id": row_id,
            "entity": entity,
            "sentiment": sentiment,
            "text": text,
        }


def write_training_csv(input_path: Path, output_path: Path, default_entity: str) -> int:
    rows = list(convert_rows(input_path, default_entity))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert reviewed TweetClaw exports into id, entity, sentiment, text CSV rows.",
    )
    parser.add_argument("input", type=Path, help="TweetClaw JSON, JSONL, NDJSON, or CSV export")
    parser.add_argument("output", type=Path, help="Training CSV output path")
    parser.add_argument(
        "--default-entity",
        default="TweetClaw",
        help="Entity value when the export does not include a query or author",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_training_csv(args.input, args.output, args.default_entity)
    print(f"Wrote {count} labeled rows to {args.output}")


if __name__ == "__main__":
    main()
