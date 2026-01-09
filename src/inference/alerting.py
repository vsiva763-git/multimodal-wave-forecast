from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


def trigger_http_alert(webhook_url: str, payload: Dict[str, Any], timeout: int = 10) -> Optional[int]:
    try:
        r = requests.post(webhook_url, json=payload, timeout=timeout)
        return r.status_code
    except Exception:  # noqa: BLE001
        return None


def evaluate_and_alert(forecast: Dict[str, Any], threshold_m: float, webhook_url: Optional[str] = None) -> Dict[str, Any]:
    """
    forecast: { 'station_id': str, 'lead_hours': [1..H], 'swh': [values in m] }
    """
    station_id = forecast.get("station_id", "unknown")
    swh = forecast.get("swh", [])
    lead_hours = forecast.get("lead_hours", list(range(1, len(swh) + 1)))
    exceed = [int(v >= threshold_m) for v in swh]
    event = {
        "station_id": station_id,
        "threshold_m": threshold_m,
        "lead_hours": lead_hours,
        "swh": swh,
        "exceed": exceed,
    }
    if webhook_url:
        trigger_http_alert(webhook_url, event)
    return event
