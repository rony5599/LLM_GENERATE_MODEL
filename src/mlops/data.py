from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Record:
    text: str
    class_group: str
    stream: str
    subject: str
    source_path: str


def iter_text_files(data_dir: Path) -> Iterable[Path]:
    for path in data_dir.rglob("*.txt"):
        if path.is_file():
            yield path


def record_from_path(path: Path, root: Path) -> Record:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) < 3:
        raise ValueError(f"Expected <group>/<stream>/<file>.txt structure, got: {relative}")
    class_group, stream, filename = parts[0], parts[1], parts[-1]
    subject = filename.replace(".txt", "")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")
    return Record(text=text, class_group=class_group, stream=stream, subject=subject, source_path=str(relative))


def build_dataframe(data_dir: str) -> pd.DataFrame:
    root = Path(data_dir)
    rows = [record_from_path(path, root).__dict__ for path in iter_text_files(root)]
    if not rows:
        raise ValueError(f"No .txt files found under {data_dir}")
    df = pd.DataFrame(rows)
    # Multi-level label compressed to one class for modeling
    df["label"] = df["class_group"] + "__" + df["stream"] + "__" + df["subject"]
    return df
