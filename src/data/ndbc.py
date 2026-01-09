from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

import pandas as pd
import requests


@dataclass
class NDBCStation:
    station_id: str  # e.g., '46042'
    lat: Optional[float] = None
    lon: Optional[float] = None


def fetch_ndbc_realtime(station_id: str) -> pd.DataFrame:
    """
    Fetch NDBC realtime observations for a station.
    Source: https://www.ndbc.noaa.gov/data/realtime2/{station}.txt

    Returns pandas DataFrame with timestamps and key columns if present:
    - WVHT (m): Significant wave height
    - DPD (s): Dominant wave period
    - MWD (deg): Mean wave direction
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text

    # The file is whitespace-separated with header lines starting with '#'
    # The header line with column names starts with '#YY  MM DD ...'
    header_lines = [line for line in text.splitlines() if line.startswith("#")]
    if not header_lines:
        raise ValueError("Unexpected NDBC format: missing header")
    # Last header line is the column names
    col_line = header_lines[-1].lstrip("#").strip()
    cols = [c for c in col_line.split()]

    # Build a text buffer without comment lines
    data_lines = [line for line in text.splitlines() if not line.startswith("#") and line.strip()]
    buf = StringIO("\n".join(data_lines))
    df = pd.read_csv(buf, delim_whitespace=True, names=cols, na_values=["MM"], engine="python")

    # Construct timestamp (UTC)
    # Columns: 'YY', 'MM', 'DD', 'hh', 'mm'
    def to_ts(row) -> datetime:
        yy = int(row["YY"]) + (2000 if int(row["YY"]) < 70 else 1900)
        return datetime(yy, int(row["MM"]), int(row["DD"]), int(row["hh"]), int(row["mm"]), tzinfo=timezone.utc)

    df["time"] = df.apply(to_ts, axis=1)
    df = df.sort_values("time").reset_index(drop=True)
    keep_cols = [c for c in df.columns if c in {"time", "WVHT", "DPD", "MWD", "APD"}]
    return df[keep_cols]
