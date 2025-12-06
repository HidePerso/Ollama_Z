from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import subprocess
import json
import re
from datetime import datetime
import threading
import time
import requests
import signal
import os
import psutil

# Try to import GPUtil for GPU monitoring (optional, primarily for Windows)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

app = Flask(__name__)

# Store active pull operations with process info for cancellation
pull_progress = {}
pull_threads = {}
pull_processes = {}  # Store subprocess objects for cancellation

# Store active chat generation
active_generation = {"stop": False}

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

def get_gpu_stats():
    """Get GPU statistics - works on both Linux (nvidia-smi) and Windows (GPUtil)"""
    stats = {
        'gpu_percent': 0,
        'vram_percent': 0,
        'vram_used': 0,
        'vram_total': 0,
        'gpu_available': False,
        'gpu_name': 'N/A'
    }
    
    # Try nvidia-smi first (Linux/Windows with NVIDIA)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,memory.free', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Parse nvidia-smi output: index, name, utilization.gpu, memory.used, memory.total, memory.free
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 5:
                stats['gpu_name'] = parts[1]
                stats['gpu_percent'] = float(parts[2])
                stats['vram_used'] = round(float(parts[3]) / 1024, 2)  # Convert MB to GB
                stats['vram_total'] = round(float(parts[4]) / 1024, 2)  # Convert MB to GB
                stats['vram_percent'] = round((float(parts[3]) / float(parts[4])) * 100, 1) if float(parts[4]) > 0 else 0
                stats['gpu_available'] = True
                return stats
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass  # nvidia-smi not available or failed, try GPUtil
    
    # Fallback to GPUtil (primarily for Windows or non-NVIDIA)
    if GPUTIL_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                stats['gpu_percent'] = round(gpu.load * 100, 1)
                stats['vram_percent'] = round(gpu.memoryUtil * 100, 1)
                stats['vram_used'] = round(gpu.memoryUsed / 1024, 2)  # GB
                stats['vram_total'] = round(gpu.memoryTotal / 1024, 2)  # GB
                stats['gpu_name'] = gpu.name
                stats['gpu_available'] = True
        except Exception:
            pass  # GPUtil failed, return default values
    
    return stats

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
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Store process for potential cancellation
        pull_processes[model_name] = process
        
        for line in iter(process.stdout.readline, ''):
            # Check if cancelled
            if model_name not in pull_progress or pull_progress[model_name]["status"] == "cancelled":
                process.terminate()
                break
                
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Always update with latest line for visibility
            print(f"[{model_name}] {line}")  # Debug output
            
            # Parse different types of ollama output
            if "success" in line.lower():
                pull_progress[model_name]["status"] = "complete"
                pull_progress[model_name]["progress"] = 100
                pull_progress[model_name]["message"] = "Download complete!"
            elif "error" in line.lower() or "failed" in line.lower():
                pull_progress[model_name]["status"] = "error"
                pull_progress[model_name]["message"] = line
                # Remove from global state after delay
                time.sleep(5)
                if model_name in pull_progress:
                    del pull_progress[model_name]
            else:
                # Extract download progress info
                pull_progress[model_name]["status"] = "pulling"
                
                # Try to extract size and speed: "943 MB/8.6 GB  3.2 MB/s"
                size_match = re.search(r'([\d.]+\s*[KMG]B)\s*/\s*([\d.]+\s*[KMG]B)', line)
                speed_match = re.search(r'([\d.]+\s*[KMG]B/s)', line)
                percent_match = re.search(r'(\d+)%', line)
                
                if size_match:
                    downloaded = size_match.group(1).strip()
                    total = size_match.group(2).strip()
                    speed = speed_match.group(1).strip() if speed_match else ""
                    
                    # Calculate percentage from sizes
                    try:
                        def parse_size(size_str):
                            size_str = size_str.upper()
                            value = float(re.search(r'[\d.]+', size_str).group())
                            if 'GB' in size_str:
                                return value * 1024
                            elif 'KB' in size_str:
                                return value / 1024
                            else:  # MB
                                return value
                        
                        downloaded_mb = parse_size(downloaded)
                        total_mb = parse_size(total)
                        percentage = int((downloaded_mb / total_mb) * 100) if total_mb > 0 else 0
                        pull_progress[model_name]["progress"] = percentage
                    except:
                        pull_progress[model_name]["progress"] = 0
                    
                    # Format message
                    msg_parts = [f"{downloaded}/{total}"]
                    if speed:
                        msg_parts.append(speed)
                    pull_progress[model_name]["message"] = " | ".join(msg_parts)
                elif percent_match:
                    # If we have a percentage but no size info
                    pull_progress[model_name]["progress"] = int(percent_match.group(1))
                    pull_progress[model_name]["message"] = line
                else:
                    # Show raw line for other status messages
                    pull_progress[model_name]["message"] = line
                    if "manifest" in line.lower():
                        pull_progress[model_name]["progress"] = 5
                    elif "verify" in line.lower():
                        pull_progress[model_name]["progress"] = 95
            

        
        process.wait()
        
        # Clean up process reference
        if model_name in pull_processes:
            del pull_processes[model_name]
        
        if pull_progress[model_name]["status"] == "cancelled":
            pull_progress[model_name]["message"] = "Download cancelled"
            # Remove from global state after delay
            time.sleep(3)
            if model_name in pull_progress:
                del pull_progress[model_name]
        elif process.returncode == 0 and pull_progress[model_name]["status"] != "error":
            pull_progress[model_name]["status"] = "complete"
            pull_progress[model_name]["progress"] = 100
            pull_progress[model_name]["message"] = f"Successfully pulled {model_name}"
            # Remove from global state after delay
            time.sleep(5)
            if model_name in pull_progress:
                del pull_progress[model_name]
        elif pull_progress[model_name]["status"] not in ["error", "cancelled"]:
            pull_progress[model_name]["status"] = "error"
            pull_progress[model_name]["message"] = "Download failed"
            # Remove from global state after delay
            time.sleep(10)
            if model_name in pull_progress:
                del pull_progress[model_name]
            
    except Exception as e:
        pull_progress[model_name]["status"] = "error"
        pull_progress[model_name]["message"] = str(e)
        if model_name in pull_processes:
            del pull_processes[model_name]
        # Remove from global state after delay
        time.sleep(10)
        if model_name in pull_progress:
            del pull_progress[model_name]

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
    
    # Start pull in background thread (allows multiple concurrent downloads)
    thread = threading.Thread(target=pull_model_with_progress, args=(model_name,), daemon=True)
    thread.start()
    pull_threads[model_name] = thread
    
    return jsonify({"success": True, "message": f"Started pulling {model_name}"})

@app.route('/api/pull/progress/<model_name>')
def get_pull_progress(model_name):
    """Get progress of a model pull"""
    if model_name in pull_progress:
        return jsonify(pull_progress[model_name])
    return jsonify({"status": "not_found"})

@app.route('/api/pull/all')
def get_all_pull_progress():
    """Get progress of all active downloads"""
    return jsonify(pull_progress)

@app.route('/api/pull/cancel', methods=['POST'])
def cancel_pull():
    """Cancel a model download"""
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"success": False, "error": "Model name required"})
    
    if model_name in pull_progress:
        pull_progress[model_name]["status"] = "cancelled"
        
        # Try to terminate the process
        if model_name in pull_processes:
            try:
                process = pull_processes[model_name]
                if os.name == 'nt':
                    # Windows
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    time.sleep(0.5)
                    process.terminate()
                else:
                    # Unix-like
                    process.terminate()
                    time.sleep(0.5)
                    if process.poll() is None:
                        process.kill()
            except Exception as e:
                print(f"Error terminating process: {e}")
        
        return jsonify({"success": True, "message": f"Cancelled download of {model_name}"})
    
    return jsonify({"success": False, "error": "Model download not found"})

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
    """Chat with a model using streaming"""
    data = request.json
    model = data.get('model')
    message = data.get('message')
    context = data.get('context', [])  # Get conversation history
    
    if not model or not message:
        return jsonify({"success": False, "error": "Model and message required"})
    
    def generate():
        active_generation["stop"] = False
        try:
            # Build full prompt with conversation history
            full_prompt = ""
            for msg in context:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    full_prompt += f"User: {content}\n"
                elif role == 'assistant':
                    full_prompt += f"Assistant: {content}\n"
            full_prompt += f"User: {message}\nAssistant:"
            
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": True
                },
                stream=True,
                timeout=300
            )
            
            for line in response.iter_lines():
                if active_generation["stop"]:
                    yield f"data: {json.dumps({'done': True, 'stopped': True})}\n\n"
                    break
                    
                if line:
                    try:
                        chunk = json.loads(line)
                        yield f"data: {json.dumps(chunk)}\n\n"
                        
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/chat/stop', methods=['POST'])
def stop_chat():
    """Stop the active chat generation"""
    active_generation["stop"] = True
    return jsonify({"success": True, "message": "Generation stopped"})

@app.route('/api/serve', methods=['POST'])
def serve_ollama():
    """Start ollama serve (if not already running)"""
    result = run_ollama_command("serve")
    return jsonify(result)

@app.route('/api/kill', methods=['POST'])
def kill_ollama():
    """Kill all Ollama processes"""
    try:
        if os.name == 'nt':
            # Windows
            subprocess.run('taskkill /F /IM ollama.exe', shell=True, capture_output=True)
            subprocess.run('taskkill /F /IM ollama_llama_server.exe', shell=True, capture_output=True)
        else:
            # Unix-like
            subprocess.run('pkill -9 ollama', shell=True, capture_output=True)
        
        time.sleep(1)
        return jsonify({"success": True, "message": "Ollama processes killed"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/start', methods=['POST'])
def start_ollama():
    """Start Ollama serve in background"""
    try:
        if os.name == 'nt':
            # Windows - start in background
            subprocess.Popen(
                'start /B ollama serve',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            # Unix-like
            subprocess.Popen(
                'ollama serve &',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp
            )
        
        time.sleep(2)
        return jsonify({"success": True, "message": "Ollama started"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/system/stats', methods=['GET'])
def get_system_stats():
    """Get system resource usage statistics"""
    try:
        stats = {}
        
        # CPU Usage
        stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        
        # RAM Usage
        ram = psutil.virtual_memory()
        stats['ram_percent'] = ram.percent
        stats['ram_used'] = round(ram.used / (1024**3), 2)  # GB
        stats['ram_total'] = round(ram.total / (1024**3), 2)  # GB
        
        # GPU Usage (cross-platform)
        gpu_stats = get_gpu_stats()
        stats.update(gpu_stats)
        
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print("Starting Ollama Web Interface on http://0.0.0.0:11435")
    app.run(host='0.0.0.0', port=11435, debug=False)
