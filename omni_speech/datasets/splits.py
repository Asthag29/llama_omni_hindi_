"""Fixed train/validation/test split of the Hindi instruct data.

The held-out ids live in ``data/splits/{validation,test}_ids.txt``; every other
sample is train. They are produced by
``python -m omni_speech.datasets.processing.split_hindi_instruct``.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

SPLITS = ("train", "validation", "test")
SOURCES = ("anudesh", "flan_v2", "hh-rlhf", "lm_sys")

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HEX32_RE = re.compile(r"[0-9a-fA-F]{32}")


def load_id_list(path) -> set[str]:
    """Read one id per line; blank lines are ignored."""
    with Path(path).open(encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def partition_by_ids(
    samples: Iterable[Mapping],
    validation_ids: Iterable[str],
    test_ids: Iterable[str],
) -> Dict[str, List]:
    """Route each sample to validation/test by ``sample["id"]``; the rest is train.

    Input order is preserved within each split.
    """
    validation_ids = set(validation_ids)
    test_ids = set(test_ids)
    overlap = validation_ids & test_ids
    if overlap:
        example = sorted(overlap)[:5]
        raise ValueError(
            f"validation and test id sets overlap ({len(overlap)} ids, e.g. {example})"
        )

    parts: Dict[str, List] = {name: [] for name in SPLITS}
    for sample in samples:
        sample_id = sample["id"]
        if sample_id in validation_ids:
            parts["validation"].append(sample)
        elif sample_id in test_ids:
            parts["test"].append(sample)
        else:
            parts["train"].append(sample)
    return parts


def source_of(sample_id: str) -> str:
    """Infer the upstream source of a sample from its id format."""
    sample_id = str(sample_id)
    if sample_id.startswith("flan_v2-"):
        return "flan_v2"
    if sample_id.startswith("hh-rlhf-"):
        return "hh-rlhf"
    if _UUID_RE.fullmatch(sample_id):
        return "anudesh"
    if _HEX32_RE.fullmatch(sample_id):
        return "lm_sys"
    raise ValueError(f"Unrecognised sample id format: {sample_id!r}")


def stratified_split_ids(
    ids: Iterable[str],
    fractions=(0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split ids into train/validation/test within each source.

    Independent of input order: ids are de-duplicated, grouped by
    ``source_of``, sorted, and shuffled with a fresh ``random.Random(seed)``
    per source (sources in sorted name order). Validation and test get
    ``round(fraction * n)`` ids each; train gets the remainder.
    """
    _, val_fraction, test_fraction = fractions
    by_source: Dict[str, List[str]] = {}
    for sample_id in set(ids):
        by_source.setdefault(source_of(sample_id), []).append(sample_id)

    result: Dict[str, List[str]] = {name: [] for name in SPLITS}
    for source in sorted(by_source):
        group = sorted(by_source[source])
        random.Random(seed).shuffle(group)
        n_val = round(val_fraction * len(group))
        n_test = round(test_fraction * len(group))
        result["validation"].extend(group[:n_val])
        result["test"].extend(group[n_val : n_val + n_test])
        result["train"].extend(group[n_val + n_test :])
    return {name: sorted(values) for name, values in result.items()}


def format_split_table(ids_by_split: Mapping[str, Iterable[str]]) -> str:
    """Rows per source x split, with each source's percentage in each split."""
    counts = {name: {} for name in SPLITS}
    for split in SPLITS:
        for sample_id in ids_by_split.get(split, ()):
            try:
                source = source_of(sample_id)
            except ValueError:
                source = "unknown"
            counts[split][source] = counts[split].get(source, 0) + 1

    sources = [s for s in SOURCES if any(s in counts[sp] for sp in SPLITS)]
    if any("unknown" in counts[sp] for sp in SPLITS):
        sources.append("unknown")

    header = f"{'source':<9}" + "".join(f"{sp:>21}" for sp in SPLITS) + f"{'total':>9}"
    lines = [header, "-" * len(header)]
    for source in sources + ["total"]:
        if source == "total":
            row = [sum(counts[sp].values()) for sp in SPLITS]
        else:
            row = [counts[sp].get(source, 0) for sp in SPLITS]
        total = sum(row)
        cells = "".join(
            f"{n:>12} ({100 * n / total if total else 0:5.1f}%)" for n in row
        )
        lines.append(f"{source:<9}{cells}{total:>9}")
    return "\n".join(lines)
