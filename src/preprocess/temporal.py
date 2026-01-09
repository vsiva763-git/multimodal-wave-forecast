from __future__ import annotations

from typing import Sequence

import xarray as xr


def align_time(datasets: Sequence[xr.Dataset], freq: str = "1H", method: str = "nearest") -> list[xr.Dataset]:
    ref = datasets[0]
    if "time" not in ref.coords:
        return list(datasets)
    timeline = ref["time"].to_index().to_series().asfreq(freq).index
    out: list[xr.Dataset] = []
    for ds in datasets:
        if "time" not in ds.coords:
            out.append(ds)
        else:
            out.append(ds.interp(time=timeline, method=method))
    return out
