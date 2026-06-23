"""Robust CSV loading, shared by every node.

`low_memory=False` reads the file in one pass instead of chunks, which avoids a
pandas bug where chunked reads of a mixed-type/ragged column crash in
`_concatenate_chunks` (IndexError). A second pass with the Python engine and
`on_bad_lines="skip"` salvages genuinely ragged uploads.
"""
from __future__ import annotations

import pandas as pd


def load_csv(path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        kwargs.pop("low_memory", None)  # not supported by the python engine
        return pd.read_csv(path, engine="python", on_bad_lines="skip", **kwargs)
