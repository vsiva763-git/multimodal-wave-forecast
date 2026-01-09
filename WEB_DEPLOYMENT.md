# Web UI Deployment Guide

## Quick Start

1. **Install dependencies** (if not already done):
```bash
pip install Flask>=3.0 flask-cors>=4.0
```

2. **Start the web server**:
```bash
python web/app.py
```

3. **Open browser**:
Navigate to http://localhost:5000

## Advanced Options

### Custom Port
```bash
python web/app.py --port 8080
```

### Load Trained Model
```bash
python web/app.py --checkpoint lightning_logs/version_0/checkpoints/epoch=9.ckpt
```

### Production Deployment
```bash
# Install production WSGI server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web.app:app
```

### Docker Deployment
```bash
# Build image
docker build -t wave-forecast-web .

# Run container
docker run -p 5000:5000 wave-forecast-web
```

## API Endpoints

- `GET /` - Main web UI
- `GET /api/health` - Health check
- `GET /api/ocean-regions` - List all ocean regions
- `GET /api/stations?ocean=<region>` - Get stations in region
- `GET /api/station/<id>` - Get station info
- `POST /api/predict` - Single station forecast
- `POST /api/predict-ocean` - Regional forecast

## Features

- 🗺️ **Interactive Map**: Leaflet-based map with buoy locations
- 📊 **Chart Visualization**: Chart.js for forecast plots
- 🚨 **Alert System**: Visual alerts when thresholds exceeded
- 🌊 **Ocean Regions**: Pre-defined regions for easy selection
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- ⚡ **Real-time Predictions**: Generate forecasts on demand

## Customization

### Change Alert Threshold
Adjust the threshold slider in the UI (default: 4.0m)

### Add More Regions
Edit `src/data/ocean_regions.py` to add custom regions

### Styling
Modify `web/static/css/style.css` for custom themes

## Troubleshooting

**Port already in use:**
```bash
python web/app.py --port 8080
```

**API connection fails:**
Check that Flask server is running and no firewall blocking

**Map not loading:**
Ensure internet connection (loads OpenStreetMap tiles)

**Model predictions identical:**
Currently using synthetic data for demo. Train a model and load checkpoint to get real predictions.
