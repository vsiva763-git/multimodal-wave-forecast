#!/usr/bin/env python3
"""
Flask API backend for wave forecast web application.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.ocean_regions import OCEAN_REGIONS, get_ocean_region, list_ocean_regions
from src.data.ndbc_meta import get_station_latlon, get_stations_in_bbox
from src.model.multimodal_model import MultiModalWaveForecaster, ModelConfig
from src.inference.alerting import evaluate_and_alert

app = Flask(__name__, static_folder='../web/static', template_folder='../web/templates')
CORS(app)

# Global model instance (load on startup)
MODEL = None
MODEL_CONFIG = None


def load_model(checkpoint_path: str | None = None):
    """Load the trained model."""
    global MODEL, MODEL_CONFIG
    MODEL_CONFIG = ModelConfig()
    MODEL = MultiModalWaveForecaster(MODEL_CONFIG)
    if checkpoint_path and Path(checkpoint_path).exists():
        MODEL = MODEL.load_from_checkpoint(checkpoint_path)
    MODEL.eval()
    print(f"Model loaded: {checkpoint_path or 'untrained'}")


@app.route('/')
def index():
    """Serve main HTML page."""
    return send_from_directory('../web/templates', 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None
    })


@app.route('/api/ocean-regions', methods=['GET'])
def get_ocean_regions():
    """Get all ocean regions."""
    regions = []
    for key in list_ocean_regions():
        region = get_ocean_region(key)
        regions.append({
            'id': key,
            'name': region.name,
            'bbox': region.bbox,
            'description': region.description
        })
    return jsonify({'regions': regions})


@app.route('/api/stations', methods=['GET'])
def get_stations():
    """Get stations for a region or all stations."""
    ocean = request.args.get('ocean')
    
    if ocean:
        region = get_ocean_region(ocean)
        if not region:
            return jsonify({'error': 'Invalid ocean region'}), 400
        try:
            stations = get_stations_in_bbox(region.bbox)
            return jsonify({
                'stations': [
                    {'id': s[0], 'lat': s[1], 'lon': s[2]}
                    for s in stations[:50]  # Limit to 50 for performance
                ],
                'total': len(stations)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'ocean parameter required'}), 400


@app.route('/api/station/<station_id>', methods=['GET'])
def get_station_info(station_id: str):
    """Get station location."""
    try:
        latlon = get_station_latlon(station_id)
        if not latlon:
            return jsonify({'error': 'Station not found'}), 404
        return jsonify({
            'id': station_id,
            'lat': latlon[0],
            'lon': latlon[1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run wave forecast prediction."""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    station_id = data.get('station_id', 'unknown')
    threshold = float(data.get('threshold', 4.0))
    
    # Generate synthetic input for demo (in production, fetch real WW3/GFS data)
    cfg = MODEL_CONFIG
    ww3 = torch.randn(1, cfg.time_steps, cfg.ww3_channels, cfg.patch_size, cfg.patch_size)
    gfs = torch.randn(1, cfg.time_steps, cfg.gfs_channels, cfg.patch_size, cfg.patch_size)
    
    with torch.no_grad():
        y_hat = MODEL({'ww3_patch': ww3, 'gfs_patch': gfs}).squeeze(0).cpu().numpy()
    
    # Convert to list for JSON
    forecast_values = y_hat.tolist()
    lead_hours = list(range(1, cfg.horizon + 1))
    
    # Evaluate alerts
    alert_event = evaluate_and_alert(
        {'station_id': station_id, 'lead_hours': lead_hours, 'swh': forecast_values},
        threshold_m=threshold
    )
    
    return jsonify({
        'station_id': station_id,
        'forecast': {
            'lead_hours': lead_hours,
            'swh_m': forecast_values,
            'threshold_m': threshold
        },
        'alert': alert_event
    })


@app.route('/api/predict-ocean', methods=['POST'])
def predict_ocean():
    """Run predictions for multiple stations in an ocean region."""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    ocean = data.get('ocean')
    threshold = float(data.get('threshold', 4.0))
    max_stations = int(data.get('max_stations', 5))
    
    region = get_ocean_region(ocean)
    if not region:
        return jsonify({'error': 'Invalid ocean region'}), 400
    
    try:
        stations = get_stations_in_bbox(region.bbox)[:max_stations]
    except Exception as e:
        return jsonify({'error': f'Failed to get stations: {str(e)}'}), 500
    
    predictions = []
    cfg = MODEL_CONFIG
    
    for station_id, lat, lon in stations:
        # Generate synthetic input
        ww3 = torch.randn(1, cfg.time_steps, cfg.ww3_channels, cfg.patch_size, cfg.patch_size)
        gfs = torch.randn(1, cfg.time_steps, cfg.gfs_channels, cfg.patch_size, cfg.patch_size)
        
        with torch.no_grad():
            y_hat = MODEL({'ww3_patch': ww3, 'gfs_patch': gfs}).squeeze(0).cpu().numpy()
        
        forecast_values = y_hat.tolist()
        lead_hours = list(range(1, cfg.horizon + 1))
        
        alert_event = evaluate_and_alert(
            {'station_id': station_id, 'lead_hours': lead_hours, 'swh': forecast_values},
            threshold_m=threshold
        )
        
        predictions.append({
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'forecast': forecast_values,
            'alert': any(alert_event['exceed'])
        })
    
    return jsonify({
        'ocean': ocean,
        'region_name': region.name,
        'threshold_m': threshold,
        'predictions': predictions,
        'lead_hours': list(range(1, cfg.horizon + 1))
    })


def main():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0', help='Host to bind')
    ap.add_argument('--port', type=int, default=5000, help='Port to bind')
    ap.add_argument('--checkpoint', default=None, help='Model checkpoint path')
    ap.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = ap.parse_args()
    
    # Load model
    load_model(args.checkpoint)
    
    # Start server
    print(f"Starting Wave Forecast API on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
