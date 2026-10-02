"""Small JSON helpers shared by training and data-processing scripts."""

import json
from pathlib import Path
from typing import Union


def load_json_array_maybe_prefixed(path: Union[str, Path]):
    """Load the JSON array in ``path``, ignoring any text before the first ``[`` or after the last ``]``."""
    text = Path(path).read_text(encoding="utf-8")
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not locate a JSON array in {path}")
    return json.loads(text[start : end + 1])
