"""Load and write GRAND-ROUNDS-style physician grading data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import GradedItem


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        for key in ("data", "records", "items", "rows"):
            if key in data and isinstance(data[key], list):
                return list(data[key])
        return [dict(data)]
    if isinstance(data, list):
        return [dict(record) for record in data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(dict(json.loads(line)))
    return records


def _records_from_csv(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def _records_from_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _records_from_jsonl(path)
    if suffix == ".json":
        return _records_from_json(path)
    if suffix == ".csv":
        return _records_from_csv(path)
    if suffix in {".parquet", ".pq"}:
        return _records_from_parquet(path)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def load_items(
    source: str | Path | Sequence[Mapping[str, Any]] | Sequence[GradedItem],
    *,
    benchmark: str | None = None,
    limit: int | None = None,
) -> list[GradedItem]:
    """Load records from JSON, JSONL, CSV, Parquet, or in-memory records."""
    if isinstance(source, (str, Path)):
        records: list[Any] = read_records(source)
    else:
        records = list(source)

    items: list[GradedItem] = []
    seen_ids: dict[str, int] = {}
    for order, record in enumerate(records):
        item = record if isinstance(record, GradedItem) else GradedItem.from_record(record, order)
        if benchmark is not None and item.benchmark != benchmark:
            continue
        count = seen_ids.get(item.item_id, 0)
        seen_ids[item.item_id] = count + 1
        if count:
            item.item_id = f"{item.item_id}#{count + 1}"
        items.append(item)
        if limit is not None and len(items) >= limit:
            break
    return items


def download_grand_rounds(
    destination: str | Path = "data/grand_rounds",
    *,
    filename: str = "combined_dataset.json",
    repo_id: str = "tbuckley/GRAND-ROUNDS",
    token: str | None = None,
) -> Path:
    """Download the gated GRAND-ROUNDS release and return the local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch GRAND-ROUNDS: "
            "pip install 'phys-tester[hub]'"
        ) from exc

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(destination),
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download GRAND-ROUNDS. The dataset is gated: accept the "
            "terms at https://huggingface.co/datasets/tbuckley/GRAND-ROUNDS and "
            "authenticate with `hf auth login` or set HF_TOKEN."
        ) from exc
    return Path(path)


def load_grand_rounds(
    destination: str | Path = "data/grand_rounds",
    *,
    filename: str = "combined_dataset.json",
    repo_id: str = "tbuckley/GRAND-ROUNDS",
    token: str | None = None,
    benchmark: str | None = None,
    limit: int | None = None,
) -> list[GradedItem]:
    """Download the gated GRAND-ROUNDS release and load it."""
    path = download_grand_rounds(
        destination,
        filename=filename,
        repo_id=repo_id,
        token=token,
    )
    return load_items(path, benchmark=benchmark, limit=limit)


def write_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False, default=_json_default)
        handle.write("\n")
    return path


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default))
            handle.write("\n")
    return path


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
