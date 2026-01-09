Multimodal Wave Forecast (WW3 + GFS + NDBC)

Overview
- Goal: Predict dangerous ocean waves 1–6 hours ahead and trigger IoT-style alerts using free, near-real-time data: NOAA WW3 (waves), GFS (winds/pressure), and NDBC buoys for live validation. Optional: CMEMS NRT waves and Sentinel-3/6 altimetry.
- Stack: Python, xarray/cfgrib for GRIB/NetCDF, PyTorch Lightning for training. Run on GitHub Codespaces (CPU for prep) and a GPU VM for training.

Status
- This repo ships a runnable scaffold with a synthetic dataset, a multimodal model (CNN + Transformer + LSTM), and an alerting example. Hook in real data via the data fetchers and preprocessing stubs.

Project Structure
- src/data: WW3/GFS/NDBC helpers and utilities
- src/preprocess: spatial cropping and temporal alignment
- src/model: CNN+Transformer+LSTM architecture and Lightning DataModule
- src/inference: alerting helper
- scripts: CLI scripts to download, prepare, train, and predict
- configs/default.yaml: region, buoys, horizons, thresholds

Environment Setup
- Python: 3.10+
- System libs (for GRIB via cfgrib):

```bash
sudo apt update
sudo apt install -y eccodes libeccodes0 libeccodes-dev
```

- Python packages:

```bash
pip install -r requirements.txt
```

Quickstart (Synthetic Data)
- Train a small model (CPU-friendly):

```bash
python scripts/train.py --epochs 3 --batch 16 --time_steps 12 --horizon 6 --patch 9
```

- Run a demo prediction + alert check (prints event JSON):

```bash
python scripts/predict_and_alert.py --threshold 4.0 --station 46042
```

Fetching Near-Real-Time Data (NOMADS)
- WW3: See product root: https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod/
  - Choose the date directory (e.g., multi_1.YYYYMMDD) and grid (e.g., glo_30m) and forecast files (e.g., f000..f006).
- GFS: See product root: https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/
  - Choose date/hour (e.g., gfs.YYYYMMDD/HH) and select GRIB2 files for desired fields.
- Use the downloader stub and pass explicit filenames you want to fetch:

```bash
python scripts/download_data.py \
  --dst data/raw/ww3 \
  --type ww3 \
  --base https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod \
  --dir multi_1.20260109 \
  --file multi_1.glo_30m.f000.grib2 --file multi_1.glo_30m.f003.grib2 --file multi_1.glo_30m.f006.grib2
```

NRT Buoy Patches (WW3 + GFS)
- Build buoy-centered patches for a station with latest NOMADS runs (requires eccodes/cfgrib):

```bash
python scripts/prepare_nrt_buoy_patches.py --station 46042 --time_steps 12 --horizon 6 --patch 9
```

- Or use ocean region names to auto-select buoys:

```bash
# List available ocean regions
python scripts/prepare_nrt_buoy_patches.py --list-oceans

# Use a region (automatically finds stations in that ocean)
python scripts/prepare_nrt_buoy_patches.py --ocean us_west_coast --time_steps 12 --horizon 6 --patch 9
python scripts/prepare_nrt_buoy_patches.py --ocean north_atlantic --time_steps 12 --horizon 6 --patch 9
python scripts/prepare_nrt_buoy_patches.py --ocean hawaii --time_steps 12 --horizon 6 --patch 9
```

- Output: data/processed/nrt_46042.npz with arrays `ww3` [N,T,Cw,H,W], `gfs` [N,T,Cg,H,W], `target` [N,H].

Preparing Training Samples (extend from synthetic)
- Replace synthetic generation with real data:
  1) Open WW3 and GFS GRIB2 with cfgrib (see src/data/ww3.py and src/data/gfs.py)
  2) Spatially crop patches around buoy coordinates (src/preprocess/spatial.py)
  3) Align time to hourly cadence (src/preprocess/temporal.py)
  4) Build sequences: past T -> future H targets (e.g., SWH)
  5) Save npz files for the DataModule

Realtime Validation (NDBC)
- Pull buoy observations for stations (e.g., 46042):

```python
from src.data.ndbc import fetch_ndbc_realtime
df = fetch_ndbc_realtime("46042")
print(df.tail())  # columns include time, WVHT (m), DPD (s), MWD (deg)
```

IoT-Style Alerts
- The inference example triggers an HTTP POST if a threshold is exceeded. Provide a webhook URL to receive alerts:

```bash
python scripts/predict_and_alert.py --threshold 4.0 --webhook https://example.com/hook --station 46042
```

Training on GPU
- For a real training run, move to a GPU VM (student credits) and increase dataset size and epochs.
- Example (adjust per environment):

```bash
python scripts/train.py --epochs 20 --batch 64 --time_steps 24 --horizon 6 --patch 9
```

Next Steps
- Wire real WW3/GFS ingestion to replace synthetic data
- Add CMEMS wave fields and Sentinel-3/6 altimetry as optional modalities
- Implement regridding if mixing different grids (e.g., xESMF)
- Add metrics vs NDBC (RMSE/MAE) and basic plots

Web Interface
- Launch the interactive web UI for easy forecasting:

```bash
pip install Flask>=3.0 flask-cors>=4.0
python web/app.py
```

- Open http://localhost:5000 to access:
  - 🗺️ Interactive map with buoy selection
  - 📊 Real-time forecast charts
  - 🚨 Visual alert system
  - 🌊 Ocean region-based predictions

See [WEB_DEPLOYMENT.md](WEB_DEPLOYMENT.md) for full deployment guide.
# multimodal-wave-forecast