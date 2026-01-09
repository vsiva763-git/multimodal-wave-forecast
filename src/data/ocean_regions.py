from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class OceanRegion:
    name: str
    bbox: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    description: str = ""


# Major ocean basins with approximate bounding boxes
OCEAN_REGIONS: Dict[str, OceanRegion] = {
    "north_pacific": OceanRegion(
        name="North Pacific",
        bbox=(-180, 0, -100, 60),
        description="North Pacific Ocean including US West Coast"
    ),
    "south_pacific": OceanRegion(
        name="South Pacific",
        bbox=(140, -60, -70, 0),
        description="South Pacific Ocean"
    ),
    "north_atlantic": OceanRegion(
        name="North Atlantic",
        bbox=(-80, 0, 0, 65),
        description="North Atlantic Ocean including US East Coast"
    ),
    "south_atlantic": OceanRegion(
        name="South Atlantic",
        bbox=(-60, -60, 20, 0),
        description="South Atlantic Ocean"
    ),
    "indian_ocean": OceanRegion(
        name="Indian Ocean",
        bbox=(20, -60, 120, 30),
        description="Indian Ocean"
    ),
    "arctic": OceanRegion(
        name="Arctic Ocean",
        bbox=(-180, 65, 180, 90),
        description="Arctic Ocean"
    ),
    "southern": OceanRegion(
        name="Southern Ocean",
        bbox=(-180, -90, 180, -60),
        description="Southern Ocean (Antarctic)"
    ),
    # Regional seas and sub-regions
    "caribbean": OceanRegion(
        name="Caribbean Sea",
        bbox=(-90, 8, -60, 28),
        description="Caribbean Sea and Gulf of Mexico"
    ),
    "mediterranean": OceanRegion(
        name="Mediterranean Sea",
        bbox=(-6, 30, 37, 46),
        description="Mediterranean Sea"
    ),
    "gulf_of_mexico": OceanRegion(
        name="Gulf of Mexico",
        bbox=(-98, 18, -80, 31),
        description="Gulf of Mexico"
    ),
    "bering_sea": OceanRegion(
        name="Bering Sea",
        bbox=(160, 51, -160, 66),
        description="Bering Sea between Alaska and Russia"
    ),
    "arabian_sea": OceanRegion(
        name="Arabian Sea",
        bbox=(50, 0, 80, 30),
        description="Arabian Sea (northwestern Indian Ocean)"
    ),
    "south_china_sea": OceanRegion(
        name="South China Sea",
        bbox=(99, -5, 121, 25),
        description="South China Sea"
    ),
    # US Coast regions
    "us_west_coast": OceanRegion(
        name="US West Coast",
        bbox=(-130, 30, -115, 50),
        description="US West Coast from California to Washington"
    ),
    "us_east_coast": OceanRegion(
        name="US East Coast",
        bbox=(-80, 25, -65, 45),
        description="US East Coast from Florida to Maine"
    ),
    "hawaii": OceanRegion(
        name="Hawaii",
        bbox=(-162, 18, -154, 23),
        description="Hawaiian Islands region"
    ),
    "alaska": OceanRegion(
        name="Alaska",
        bbox=(-170, 51, -130, 71),
        description="Alaska coastal waters"
    ),
}


def get_ocean_region(ocean_name: str) -> Optional[OceanRegion]:
    """Get ocean region by name (case-insensitive, underscore/space flexible)."""
    normalized = ocean_name.lower().replace(" ", "_").replace("-", "_")
    return OCEAN_REGIONS.get(normalized)


def list_ocean_regions() -> List[str]:
    """Return list of available ocean region names."""
    return sorted(OCEAN_REGIONS.keys())


def find_region_for_point(lat: float, lon: float) -> List[str]:
    """Find all ocean regions that contain a given lat/lon point."""
    containing = []
    for key, region in OCEAN_REGIONS.items():
        min_lon, min_lat, max_lon, max_lat = region.bbox
        # Handle longitude wrap-around for regions crossing 180°
        if min_lon > max_lon:  # crosses dateline
            lon_match = lon >= min_lon or lon <= max_lon
        else:
            lon_match = min_lon <= lon <= max_lon
        
        if lon_match and min_lat <= lat <= max_lat:
            containing.append(key)
    return containing
