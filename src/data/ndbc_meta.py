from __future__ import annotations

from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import requests


def get_station_latlon(station_id: str) -> Optional[Tuple[float, float]]:
    """
    Parse NDBC active stations XML and return (lat, lon) for station.
    Source: https://www.ndbc.noaa.gov/activestations.xml
    """
    url = "https://www.ndbc.noaa.gov/activestations.xml"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    for st in root.iter("station"):
        sid = st.attrib.get("id")
        if sid == station_id:
            lat = float(st.attrib.get("lat"))
            lon = float(st.attrib.get("lon"))
            return (lat, lon)
    return None


def get_stations_in_bbox(bbox: Tuple[float, float, float, float]) -> List[Tuple[str, float, float]]:
    """
    Get all NDBC stations within a bounding box.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        List of (station_id, lat, lon) tuples
    """
    url = "https://www.ndbc.noaa.gov/activestations.xml"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    
    min_lon, min_lat, max_lon, max_lat = bbox
    stations = []
    
    for st in root.iter("station"):
        sid = st.attrib.get("id")
        try:
            lat = float(st.attrib.get("lat"))
            lon = float(st.attrib.get("lon"))
            
            # Handle longitude wrap-around for regions crossing 180°
            if min_lon > max_lon:  # crosses dateline
                lon_match = lon >= min_lon or lon <= max_lon
            else:
                lon_match = min_lon <= lon <= max_lon
            
            if lon_match and min_lat <= lat <= max_lat:
                stations.append((sid, lat, lon))
        except (TypeError, ValueError):
            continue
    
    return stations
