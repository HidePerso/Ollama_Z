from flask import Flask, render_template, request, jsonify, Response
import subprocess
import json
import re
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Store active pull operations
pull_progress = {}
pull_threads = {}

OLLAMA_URL = "http://localhost:11434"

def run_ollama_command(command):
    """Run an ollama command and return the output"""
    try:
        result = subprocess.run(
            f"ollama {command}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": "Command timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e)
        }

def parse_ps_output(output):
    """Parse the output of ollama ps command"""
    lines = output.strip().split('\n')
    if len(lines) < 2:
        return []
    
    models = []
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 6:
            models.append({
                "name": parts[0],
                "id": parts[1],
                "size": f"{parts[2]} {parts[3]}",
                "processor": parts[4],
                "until": " ".join(parts[5:])
            })
    return models

def parse_list_output(output):
    """Parse the output of ollama list command"""
    lines = output.strip().split('\n')
    if len(lines) < 2:
        return []
    
    models = []
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 3:
            models.append({
                "name": parts[0],
                "id": parts[1],
                "size": f"{parts[2]} {parts[3] if len(parts) > 3 else ''}".strip(),
                "modified": " ".join(parts[4:]) if len(parts) > 4 else ""
            })
    return models

def pull_model_with_progress(model_name):
    """Pull a model and track progress"""
    pull_progress[model_name] = {
        "status": "starting",
        "progress": 0,
        "message": "Initializing download..."
    }
    
    try:
        process = subprocess.Popen(
            f"ollama pull {model_name}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Parse progress from ollama output
            if "pulling" in line.lower():
                pull_progress[model_name]["message"] = line
                pull_progress[model_name]["status"] = "pulling"
                
                # Try to extract percentage
                percentage_match = re.search(r'(\d+)%', line)
                if percentage_match:
                    pull_progress[model_name]["progress"] = int(percentage_match.group(1))
            elif "success" in line.lower():
                pull_progress[model_name]["status"] = "complete"
                pull_progress[model_name]["progress"] = 100
                pull_progress[model_name]["message"] = "Download complete!"
            elif "error" in line.lower() or "failed" in line.lower():
                pull_progress[model_name]["status"] = "error"
                pull_progress[model_name]["message"] = line
        
        process.wait()
        
        if process.returncode == 0 and pull_progress[model_name]["status"] != "error":
            pull_progress[model_name]["status"] = "complete"
            pull_progress[model_name]["progress"] = 100
            pull_progress[model_name]["message"] = f"Successfully pulled {model_name}"
        elif pull_progress[model_name]["status"] != "error":
            pull_progress[model_name]["status"] = "error"
            pull_progress[model_name]["message"] = "Download failed"
            
    except Exception as e:
        pull_progress[model_name]["status"] = "error"
        pull_progress[model_name]["message"] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ps')
def get_running_models():
    """Get list of running models"""
    result = run_ollama_command("ps")
    if result["success"]:
        models = parse_ps_output(result["output"])
        return jsonify({"success": True, "models": models})
    return jsonify({"success": False, "error": result["error"]})

@app.route('/api/list')
def get_model_list():
    """Get list of all models"""
    result = run_ollama_command("list")
    if result["success"]:
        models = parse_list_output(result["output"])
        return jsonify({"success": True, "models": models})
    return jsonify({"success": False, "error": result["error"]})

@app.route('/api/pull', methods=['POST'])
def pull_model():
    """Start pulling a model"""
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"success": False, "error": "Model name required"})
    
    if model_name in pull_threads and pull_threads[model_name].is_alive():
        return jsonify({"success": False, "error": "Model is already being pulled"})
    
    # Start pull in background thread
    thread = threading.Thread(target=pull_model_with_progress, args=(model_name,))
    thread.start()
    pull_threads[model_name] = thread
    
    return jsonify({"success": True, "message": f"Started pulling {model_name}"})

@app.route('/api/pull/progress/<model_name>')
def get_pull_progress(model_name):
    """Get progress of a model pull"""
    if model_name in pull_progress:
        return jsonify(pull_progress[model_name])
    return jsonify({"status": "not_found"})

@app.route('/api/stop', methods=['POST'])
def stop_model():
    """Stop a running model"""
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"success": False, "error": "Model name required"})
    
    result = run_ollama_command(f"stop {model_name}")
    return jsonify(result)

@app.route('/api/rm', methods=['POST'])
def remove_model():
    """Remove a model"""
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"success": False, "error": "Model name required"})
    
    result = run_ollama_command(f"rm {model_name}")
    return jsonify(result)

@app.route('/api/cp', methods=['POST'])
def copy_model():
    """Copy a model"""
    data = request.json
    source = data.get('source')
    destination = data.get('destination')
    
    if not source or not destination:
        return jsonify({"success": False, "error": "Source and destination required"})
    
    result = run_ollama_command(f"cp {source} {destination}")
    return jsonify(result)

@app.route('/api/rename', methods=['POST'])
def rename_model():
    """Rename a model (copy then remove)"""
    data = request.json
    source = data.get('source')
    destination = data.get('destination')
    
    if not source or not destination:
        return jsonify({"success": False, "error": "Source and destination required"})
    
    # Copy first
    copy_result = run_ollama_command(f"cp {source} {destination}")
    if not copy_result["success"]:
        return jsonify(copy_result)
    
    # Then remove original
    rm_result = run_ollama_command(f"rm {source}")
    return jsonify(rm_result)

@app.route('/api/show', methods=['POST'])
def show_model():
    """Show information about a model"""
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"success": False, "error": "Model name required"})
    
    result = run_ollama_command(f"show {model_name}")
    return jsonify(result)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with a model"""
    data = request.json
    model = data.get('model')
    message = data.get('message')
    
    if not model or not message:
        return jsonify({"success": False, "error": "Model and message required"})
    
    try:
        # Use ollama run command for chat
        result = subprocess.run(
            f'ollama run {model} "{message}"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        return jsonify({
            "success": result.returncode == 0,
            "response": result.stdout,
            "error": result.stderr
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "response": "",
            "error": "Chat request timed out"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "response": "",
            "error": str(e)
        })

@app.route('/api/serve', methods=['POST'])
def serve_ollama():
    """Start ollama serve (if not already running)"""
    result = run_ollama_command("serve")
    return jsonify(result)

if __name__ == '__main__':
    print("Starting Ollama Web Interface on http://0.0.0.0:11435")
    app.run(host='0.0.0.0', port=11435, debug=False)
