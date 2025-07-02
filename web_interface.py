
from flask import Flask, render_template_string, jsonify
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Last-Mile Delivery Optimization Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .results-panel { background: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .map-container { width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 10px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { background: white; padding: 15px; border-radius: 5px; text-align: center; }
        .route-list { background: white; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚁 Last-Mile Delivery Optimization Results</h1>
            <h2>Walmart Drone vs Human Delivery System</h2>
        </div>
        
        <div class="results-panel">
            <h3>Optimization Summary</h3>
            <div class="stats" id="stats">
                <!-- Stats will be loaded here -->
            </div>
            
            <div class="route-list" id="routes">
                <!-- Routes will be loaded here -->
            </div>
        </div>
        
        <div class="map-container">
            <iframe src="/map" width="100%" height="100%" frameborder="0"></iframe>
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
            <button onclick="location.reload()" style="padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">
                🔄 Refresh Results
            </button>
            <button onclick="runOptimization()" style="padding: 10px 20px; font-size: 16px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: 10px;">
                ▶️ Run New Optimization
            </button>
        </div>
    </div>

    <script>
        async function loadResults() {
            try {
                const response = await fetch('/api/results');
                const data = await response.json();
                
                // Update stats
                const statsDiv = document.getElementById('stats');
                statsDiv.innerHTML = `
                    <div class="stat-box">
                        <h4>Total Vehicles</h4>
                        <p>${data.summary.total_vehicles}</p>
                    </div>
                    <div class="stat-box">
                        <h4>Drone Routes</h4>
                        <p>${data.summary.drone_routes}</p>
                    </div>
                    <div class="stat-box">
                        <h4>Driver Routes</h4>
                        <p>${data.summary.driver_routes}</p>
                    </div>
                    <div class="stat-box">
                        <h4>Total Weight</h4>
                        <p>${data.summary.total_weight.toFixed(1)} kg</p>
                    </div>
                    <div class="stat-box">
                        <h4>Efficiency Score</h4>
                        <p>${data.summary.efficiency_score.toFixed(2)}</p>
                    </div>
                `;
                
                // Update routes
                const routesDiv = document.getElementById('routes');
                let routesHtml = '<h4>Route Details:</h4>';
                data.routes.forEach(route => {
                    const vehicleIcon = route.vehicle_type === 'drone' ? '🚁' : '🚗';
                    routesHtml += `
                        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                            <strong>${vehicleIcon} ${route.vehicle_id}</strong> - 
                            Customers: ${route.customers.join(', ')} - 
                            Weight: ${route.total_weight.toFixed(1)}kg
                        </div>
                    `;
                });
                routesDiv.innerHTML = routesHtml;
                
            } catch (error) {
                console.error('Error loading results:', error);
                document.getElementById('stats').innerHTML = '<p>Error loading results. Please run optimization first.</p>';
            }
        }
        
        async function runOptimization() {
            const button = event.target;
            button.disabled = true;
            button.innerHTML = '⏳ Running...';
            
            try {
                const response = await fetch('/api/run-optimization', { method: 'POST' });
                const result = await response.text();
                
                if (response.ok) {
                    alert('Optimization completed successfully!');
                    location.reload();
                } else {
                    alert('Optimization failed: ' + result);
                }
            } catch (error) {
                alert('Error running optimization: ' + error.message);
            }
            
            button.disabled = false;
            button.innerHTML = '▶️ Run New Optimization';
        }
        
        // Load results when page loads
        loadResults();
    </script>
</body>
</html>
    ''')

@app.route('/map')
def map_view():
    # Serve the generated map
    if os.path.exists('optimization_map.html'):
        with open('optimization_map.html', 'r') as f:
            return f.read()
    else:
        return '<h3>No map available. Please run optimization first.</h3>'

@app.route('/api/results')
def api_results():
    try:
        if os.path.exists('optimization_results.json'):
            with open('optimization_results.json', 'r') as f:
                return json.load(f)
        else:
            return {'error': 'No results available'}, 404
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/run-optimization', methods=['POST'])
def api_run_optimization():
    try:
        import subprocess
        result = subprocess.run(['python', 'main.py'], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return 'Optimization completed successfully!'
        else:
            return f'Optimization failed: {result.stderr}', 500
    except subprocess.TimeoutExpired:
        return 'Optimization timed out', 500
    except Exception as e:
        return f'Error: {str(e)}', 500

if __name__ == '__main__':
    print("🌐 Starting web interface...")
    print("📍 Open your browser and go to the URL shown below to view results")
    app.run(host='0.0.0.0', port=5000, debug=True)
