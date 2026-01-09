#!/usr/bin/env python3
"""Demo script showing ocean-based region selection."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.ocean_regions import get_ocean_region, list_ocean_regions, find_region_for_point

print("=" * 60)
print("Available Ocean Regions:")
print("=" * 60)
for key in list_ocean_regions():
    region = get_ocean_region(key)
    print(f"{key:20s} | {region.name:25s} | {region.bbox}")

print("\n" + "=" * 60)
print("Example: Find regions for specific coordinates")
print("=" * 60)

# Test coordinates
test_points = [
    ("San Francisco", 37.77, -122.42),
    ("Honolulu", 21.31, -157.86),
    ("Miami", 25.76, -80.19),
    ("Seattle", 47.61, -122.33),
]

for name, lat, lon in test_points:
    regions = find_region_for_point(lat, lon)
    print(f"{name:15s} ({lat:6.2f}, {lon:7.2f}) -> {', '.join(regions)}")
