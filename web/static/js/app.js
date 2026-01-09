// Main JavaScript for Wave Forecast UI
const API_BASE = window.location.origin;

// Global state
let map = null;
let markers = [];
let currentStations = [];
let forecastChart = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    loadOceanRegions();
    setupEventListeners();
    checkAPIHealth();
});

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showError('Unable to connect to API server');
    }
}

// Initialize Leaflet map
function initMap() {
    map = L.map('map').setView([25, -80], 4);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);
}

// Load ocean regions
async function loadOceanRegions() {
    try {
        const response = await fetch(`${API_BASE}/api/ocean-regions`);
        const data = await response.json();
        
        const select = document.getElementById('ocean-select');
        data.regions.forEach(region => {
            const option = document.createElement('option');
            option.value = region.id;
            option.textContent = `${region.name} - ${region.description}`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load ocean regions:', error);
        showError('Failed to load ocean regions');
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('ocean-select').addEventListener('change', onOceanChange);
    document.getElementById('station-select').addEventListener('change', onStationChange);
    document.getElementById('predict-btn').addEventListener('click', runPrediction);
    document.getElementById('predict-region-btn').addEventListener('click', runRegionPrediction);
}

// Handle ocean region selection
async function onOceanChange(e) {
    const oceanId = e.target.value;
    const stationSelect = document.getElementById('station-select');
    const predictBtn = document.getElementById('predict-btn');
    const predictRegionBtn = document.getElementById('predict-region-btn');
    
    if (!oceanId) {
        stationSelect.disabled = true;
        stationSelect.innerHTML = '<option value="">Select a region first...</option>';
        predictBtn.disabled = true;
        predictRegionBtn.disabled = true;
        clearMarkers();
        return;
    }
    
    // Enable region prediction
    predictRegionBtn.disabled = false;
    
    // Load stations
    try {
        const response = await fetch(`${API_BASE}/api/stations?ocean=${oceanId}`);
        const data = await response.json();
        
        currentStations = data.stations;
        
        stationSelect.disabled = false;
        stationSelect.innerHTML = '<option value="">Select a buoy station...</option>';
        
        data.stations.forEach(station => {
            const option = document.createElement('option');
            option.value = station.id;
            option.textContent = `${station.id} (${station.lat.toFixed(2)}, ${station.lon.toFixed(2)})`;
            option.dataset.lat = station.lat;
            option.dataset.lon = station.lon;
            stationSelect.appendChild(option);
        });
        
        // Display stations on map
        displayStations(data.stations);
        
        // Show total count
        if (data.total > data.stations.length) {
            const option = document.createElement('option');
            option.disabled = true;
            option.textContent = `... and ${data.total - data.stations.length} more`;
            stationSelect.appendChild(option);
        }
        
    } catch (error) {
        console.error('Failed to load stations:', error);
        showError('Failed to load stations');
    }
}

// Handle station selection
function onStationChange(e) {
    const stationId = e.target.value;
    const predictBtn = document.getElementById('predict-btn');
    
    if (!stationId) {
        predictBtn.disabled = true;
        return;
    }
    
    predictBtn.disabled = false;
    
    // Highlight selected station on map
    const option = e.target.selectedOptions[0];
    const lat = parseFloat(option.dataset.lat);
    const lon = parseFloat(option.dataset.lon);
    map.setView([lat, lon], 8);
}

// Display stations on map
function displayStations(stations) {
    clearMarkers();
    
    if (stations.length === 0) return;
    
    const bounds = [];
    
    stations.forEach(station => {
        const marker = L.circleMarker([station.lat, station.lon], {
            radius: 6,
            fillColor: '#3498db',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }).addTo(map);
        
        marker.bindPopup(`
            <b>Station ${station.id}</b><br>
            Lat: ${station.lat.toFixed(4)}<br>
            Lon: ${station.lon.toFixed(4)}
        `);
        
        markers.push(marker);
        bounds.push([station.lat, station.lon]);
    });
    
    if (bounds.length > 0) {
        map.fitBounds(bounds, { padding: [50, 50] });
    }
}

// Clear map markers
function clearMarkers() {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
}

// Run prediction for single station
async function runPrediction() {
    const stationId = document.getElementById('station-select').value;
    const threshold = parseFloat(document.getElementById('threshold').value);
    const btn = document.getElementById('predict-btn');
    
    if (!stationId) return;
    
    // Show loading
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Generating...';
    
    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ station_id: stationId, threshold })
        });
        
        const data = await response.json();
        displayForecast(data);
        
        // Hide region forecast if showing
        document.getElementById('region-forecast-section').style.display = 'none';
        
    } catch (error) {
        console.error('Prediction failed:', error);
        showError('Prediction failed: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Forecast';
    }
}

// Run prediction for entire region
async function runRegionPrediction() {
    const oceanId = document.getElementById('ocean-select').value;
    const threshold = parseFloat(document.getElementById('threshold').value);
    const btn = document.getElementById('predict-region-btn');
    
    if (!oceanId) return;
    
    // Show loading
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Forecasting...';
    
    try {
        const response = await fetch(`${API_BASE}/api/predict-ocean`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ocean: oceanId, threshold, max_stations: 10 })
        });
        
        const data = await response.json();
        displayRegionForecast(data);
        
        // Hide single station forecast
        document.getElementById('forecast-section').style.display = 'none';
        
    } catch (error) {
        console.error('Region prediction failed:', error);
        showError('Region prediction failed: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Forecast Entire Region';
    }
}

// Display forecast results
function displayForecast(data) {
    const section = document.getElementById('forecast-section');
    const stationInfo = document.getElementById('station-info');
    const alertBanner = document.getElementById('alert-banner');
    
    section.style.display = 'block';
    
    // Station info
    stationInfo.innerHTML = `
        <h3>Station: ${data.station_id}</h3>
        <p>Forecast generated for next ${data.forecast.lead_hours.length} hours</p>
        <p><strong>Alert Threshold:</strong> ${data.forecast.threshold_m}m significant wave height</p>
    `;
    
    // Alert banner
    const hasAlert = data.alert.exceed.some(e => e === 1);
    if (hasAlert) {
        alertBanner.style.display = 'block';
        alertBanner.className = 'alert-banner';
        const maxSwh = Math.max(...data.forecast.swh_m);
        alertBanner.innerHTML = `
            🚨 ALERT: Wave height exceeds threshold! 
            Maximum predicted: ${maxSwh.toFixed(2)}m at +${data.alert.lead_hours[data.alert.swh.indexOf(maxSwh)]}h
        `;
    } else {
        alertBanner.style.display = 'block';
        alertBanner.className = 'alert-banner success';
        alertBanner.textContent = '✓ Normal conditions - No alerts';
    }
    
    // Create chart
    createForecastChart(data.forecast);
    
    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth' });
}

// Create forecast chart
function createForecastChart(forecast) {
    const ctx = document.getElementById('forecast-chart').getContext('2d');
    
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: forecast.lead_hours.map(h => `+${h}h`),
            datasets: [{
                label: 'Significant Wave Height (m)',
                data: forecast.swh_m,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 3,
                pointRadius: 5,
                pointBackgroundColor: '#3498db',
                tension: 0.4
            }, {
                label: 'Alert Threshold',
                data: Array(forecast.lead_hours.length).fill(forecast.threshold_m),
                borderColor: '#e74c3c',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Wave Height Forecast',
                    font: { size: 16 }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Wave Height (meters)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Lead Time (hours)'
                    }
                }
            }
        }
    });
}

// Display region forecast
function displayRegionForecast(data) {
    const section = document.getElementById('region-forecast-section');
    const regionInfo = document.getElementById('region-info');
    const stationsGrid = document.getElementById('region-stations');
    
    section.style.display = 'block';
    
    // Region info
    const alertCount = data.predictions.filter(p => p.alert).length;
    regionInfo.innerHTML = `
        <h3>${data.region_name}</h3>
        <p><strong>${data.predictions.length}</strong> stations forecasted | 
           <strong>${alertCount}</strong> alert(s) | 
           Threshold: <strong>${data.threshold_m}m</strong></p>
    `;
    
    // Station cards
    stationsGrid.innerHTML = '';
    data.predictions.forEach(pred => {
        const maxSwh = Math.max(...pred.forecast);
        const card = document.createElement('div');
        card.className = `station-card ${pred.alert ? 'alert' : ''}`;
        card.innerHTML = `
            <h3>${pred.station_id}</h3>
            <div class="coords">${pred.lat.toFixed(2)}, ${pred.lon.toFixed(2)}</div>
            <div class="forecast-preview">
                <strong>Max SWH:</strong> ${maxSwh.toFixed(2)}m<br>
                ${pred.alert ? '🚨 <strong>ALERT</strong>' : '✓ Normal'}
            </div>
        `;
        stationsGrid.appendChild(card);
    });
    
    // Update map markers
    updateMapWithAlerts(data.predictions);
    
    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth' });
}

// Update map with alert colors
function updateMapWithAlerts(predictions) {
    clearMarkers();
    
    const bounds = [];
    
    predictions.forEach(pred => {
        const color = pred.alert ? '#e74c3c' : '#2ecc71';
        
        const marker = L.circleMarker([pred.lat, pred.lon], {
            radius: 8,
            fillColor: color,
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9
        }).addTo(map);
        
        const maxSwh = Math.max(...pred.forecast);
        marker.bindPopup(`
            <b>Station ${pred.station_id}</b><br>
            Max SWH: ${maxSwh.toFixed(2)}m<br>
            ${pred.alert ? '🚨 <strong>ALERT</strong>' : '✓ Normal'}
        `);
        
        markers.push(marker);
        bounds.push([pred.lat, pred.lon]);
    });
    
    if (bounds.length > 0) {
        map.fitBounds(bounds, { padding: [50, 50] });
    }
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}
