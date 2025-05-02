from flask import Flask, request, jsonify
import subprocess
import os
import signal
import json

app = Flask(__name__)
# Dictionary to store running processes
processes = {}

@app.route('/')
def home():
    with open('index.html') as f:
        return f.read()

# Endpoint for the script that takes multiple inputs
@app.route('/run-sync-data', methods=['POST'])
def run_sync_data():
    try:
        # Get inputs from the request
        data = request.get_json()
        cycle = data.get('cycle', '')
        max_range = data.get('max_range', '')
        step = data.get('step', '')
        freq = data.get('freq', '')

        # Pass the inputs to the script
        result = subprocess.run(
            ['python', 'sync_data/sync_data.py', cycle, max_range, step, freq],
            capture_output=True,
            text=True
        )
        return result.stdout or "Script executed"
    except Exception as e:
        return str(e), 500
    
# Endpoint for the script that takes multiple inputs
@app.route('/run-sin-sync-data', methods=['POST'])
def run_sin_sync_data():
    try:
        # Get inputs from the request
        data = request.get_json()
        max_range = data.get('max_range', '')
        pts = data.get('pts', '')
        freq = data.get('freq', '')

        # Pass the inputs to the script
        result = subprocess.run(
            ['python', 'sync_data/sync_data.py', max_range, pts, freq],
            capture_output=True,
            text=True
        )
        return result.stdout or "Script executed"
    except Exception as e:
        return str(e), 500

# Endpoint for the script without inputs
# @app.route('/run-video-cap', methods=['POST'])
# def run_video_cap():
#     try:
#         # Run the script without inputs
#         result = subprocess.run(
#             ['python', 'aruco1.py'],
#             capture_output=True,
#             text=True
#         )
#         return result.stdout or "Script executed"
#     except Exception as e:
#         return str(e), 500

@app.route('/start/<script_name>', methods=['POST'])
def start_script(script_name):
    """Start a specific Python script."""
    if script_name in processes:
        return jsonify({"message": f"{script_name} is already running"}), 400

    process = subprocess.Popen(["python", script_name], preexec_fn=os.setsid)
    processes[script_name] = process
    return jsonify({"message": f"Started {script_name}", "pid": process.pid})


# @app.route('/stop/<script_name>', methods=['POST'])
# def stop_script(script_name):
#     """Stop the specific Python script."""
#     process = processes.get(script_name)
#     if not process:
#         return jsonify({"message": f"No such script running: {script_name}"}), 400

#     os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Kill the process group
#     del processes[script_name]
#     process = processes.get(script_name)
#     return jsonify({"message": f"Stopped {script_name}"})

if __name__ == '__main__':
    app.run(debug=True)
