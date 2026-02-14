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
import os
import psutil
# import webbrowser

# Try to import GPUtil for GPU monitoring (optional, primarily for Windows)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

app = Flask(__name__)
SETTINGS_FILE = "pipeline_settings.json"

# Store active downloads
pull_progress = {}
pull_threads = {}
pull_processes = {}  # Store subprocess objects for cancellation

# Lock for Ollama CLI commands to prevent concurrent launches on Windows
ollama_lock = threading.Lock()

OLLAMA_URL = "http://localhost:11434"

# -----------------------------
# Pipeline System Prompts
# -----------------------------

TRANSLATOR_SYSTEM = """You are a literal translator.

Translate the user text into English.

Rules:
- Preserve all explicit details.
- Preserve numbers, age, nationality, colors, object placement, and spatial relations.
- Do not summarize.
- Do not interpret.
- Do not embellish.
- Keep meaning exact.
- Output only the translation in English.
"""

EXTRACTOR_SYSTEM = """You are a factual extractor.

Convert the user description into atomic factual statements.

This is a LOSSLESS transformation.

Rules:
- Do not remove any explicit detail from the input.
- Do not generalize (e.g., do not replace "35-year-old German woman" with "adult woman").
- Do not interpret or infer missing details.
- Do not add any new objects, textures, brands, or micro-details.
- Preserve numbers, age, nationality, colors, object placement, and spatial relations.
- One fact per line.
- No headings, no blocks, no commentary.
- Output only the list of facts.
- Output language: English only.
"""

STRUCTURER_SYSTEM = """You are a structural scene organizer for zimage / Flux-based models.

You will receive:
- Original description (English)
- Extracted atomic facts (English)

Your task:
- Organize facts into structured blocks.
- LOSSLESS: every explicit detail from the input must appear in the output.
- Do not add anything new.
- Do not change age, nationality, colors, counts, placements.
- Preserve spatial relationships and perspective constraints.
- Use short atomic statements.
- Output language: English only.
- Output only the structured prompt.

Output format (omit irrelevant sections):

[Perspective]
Viewer position and viewing constraints

[Subject]
All characters with age, nationality

[Body]
Body position and anatomy

[Face]
Facial orientation and visibility

[Hair]
Hairstyle details

[Clothing]
All garments and exposure state

[Pose]
Body position and orientation

[Environment]
Room condition and objects

[Lighting]
Light source, direction, shadows

[Camera]
Framing and spatial compression

[Style]
Rendering style (only if explicitly present or requested by the user)

[Details]
Only details explicitly present in input (no invented micro-details)

[Negative]
Explicitly disallowed additions (no extra people, no extra objects, no text/logos)
"""

VALIDATOR_SYSTEM = """You are a strict validator and fixer for a zimage structured prompt.

You will receive:
- Original description (English)
- Extracted atomic facts (English)
- Candidate structured prompt (English)

Your task:
1) Check that every explicit detail from the original description and facts is present in the candidate structured prompt.
2) Check that the candidate does not add anything not present in the original/facts.
3) If anything is missing or added or generalized, rewrite the structured prompt to fix it.

Rules:
- Output ONLY the corrected structured prompt (no explanations).
- Keep the same block format as provided.
- Preserve all explicit details. Do not add new information.
- Do not generalize numbers, ages, nationalities, colors, or counts.
- Output language: English only.
"""

def run_ollama_command(command):
    """Run an ollama command and return the output"""
    with ollama_lock:
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

# -----------------------------
# Pipeline Helper Functions
# -----------------------------

def normalize_facts(text: str) -> str:
    """Normalize facts list by removing bullets and numbering"""
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # Remove bullets and numbering
        s = re.sub(r'^\s*[-*•]\s+', '', s)
        s = re.sub(r'^\s*\d+[.)]\s+', '', s)
        lines.append(s)
    return '\n'.join(lines).strip()

def has_cyrillic(text: str, threshold: float = 0.3) -> bool:
    """Check if text has more than threshold% of Cyrillic characters"""
    if not text:
        return False
    
    # Count Cyrillic characters
    cyrillic_count = len(re.findall(r'[\u0400-\u04FF]', text))
    # Count total letters (excluding spaces, punctuation, etc.)
    letter_count = len(re.findall(r'[\w]', text, re.UNICODE))
    
    if letter_count == 0:
        return False
    
    cyrillic_ratio = cyrillic_count / letter_count
    return cyrillic_ratio > threshold

def ollama_generate_sync(model: str, prompt: str, temperature: float = 0.1, timeout: int = 300) -> str:
    """Synchronous Ollama generation with configurable timeout"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.85,
                    "repeat_penalty": 1.15,
                    "num_ctx": 4096
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return (data.get('response') or '').strip()
    except requests.exceptions.Timeout:
        raise Exception(f"Request to model '{model}' timed out after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling model '{model}': {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error with model '{model}': {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')


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

@app.route('/api/settings/save', methods=['POST'])
def save_settings():
    """Save pipeline settings to a JSON file"""
    try:
        data = request.json
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f)
        return jsonify({"success": True, "message": "Settings saved"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/settings/load', methods=['GET'])
def load_settings():
    """Load pipeline settings from JSON file"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
            return jsonify({"success": True, "settings": data})
        return jsonify({"success": False, "error": "No settings found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/compile', methods=['POST'])
def compile_prompt():
    """Compile prompt through the pipeline: translate → extract → structure → validate"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        mode = data.get('mode', '2-stage')
        translator_model = data.get('translator_model', '')
        extractor_model = data.get('extractor_model', '')
        structurer_model = data.get('structurer_model', '')
        validator_model = data.get('validator_model', '')
        
        if not text:
            return jsonify({"success": False, "error": "Text is required", "stage": "input"})
        
        if not extractor_model or not structurer_model:
            return jsonify({"success": False, "error": "Extractor and Structurer models are required", "stage": "input"})
        
        # Stage 0: Translation (if needed)
        input_en = text
        if has_cyrillic(text, threshold=0.3):
            if not translator_model:
                return jsonify({"success": False, "error": "Translator model is required for Russian text", "stage": "translation"})
            
            try:
                translator_prompt = f"{TRANSLATOR_SYSTEM}\n\nUSER TEXT:\n{text}\n"
                input_en = ollama_generate_sync(translator_model, translator_prompt, temperature=0.05, timeout=300)
            except Exception as e:
                return jsonify({"success": False, "error": str(e), "stage": "translation"})
        
        # Stage 1: Extract facts
        try:
            extractor_prompt = f"{EXTRACTOR_SYSTEM}\n\nUSER DESCRIPTION (English):\n{input_en}\n"
            facts_raw = ollama_generate_sync(extractor_model, extractor_prompt, temperature=0.10, timeout=300)
            facts = normalize_facts(facts_raw)
        except Exception as e:
            return jsonify({"success": False, "error": str(e), "stage": "extraction", "input_en": input_en})
        
        # Stage 2: Structure
        structured = ""
        try:
            structurer_prompt = (
                f"{STRUCTURER_SYSTEM}\n\n"
                f"ORIGINAL DESCRIPTION (English):\n{input_en}\n\n"
                f"ATOMIC FACTS (one per line, English):\n{facts}\n\n"
                f"OUTPUT ONLY THE STRUCTURED PROMPT (English):\n"
            )
            structured = ollama_generate_sync(structurer_model, structurer_prompt, temperature=0.15, timeout=300)
        except Exception as e:
            return jsonify({"success": False, "error": str(e), "stage": "structuring", "input_en": input_en, "facts": facts})
        
        # Stage 3: Validate (optional)
        if mode == '3-stage':
            if not validator_model:
                return jsonify({"success": False, "error": "Validator model is required for 3-stage mode", "stage": "validation"})
            
            try:
                validator_prompt = (
                    f"{VALIDATOR_SYSTEM}\n\n"
                    f"ORIGINAL DESCRIPTION (English):\n{input_en}\n\n"
                    f"ATOMIC FACTS (English):\n{facts}\n\n"
                    f"CANDIDATE STRUCTURED PROMPT (English):\n{structured}\n\n"
                    f"OUTPUT ONLY THE CORRECTED STRUCTURED PROMPT (English):\n"
                )
                structured = ollama_generate_sync(validator_model, validator_prompt, temperature=0.12, timeout=300)
            except Exception as e:
                return jsonify({"success": False, "error": str(e), "stage": "validation", "input_en": input_en, "facts": facts, "structured": structured})
        
        return jsonify({
            "success": True,
            "input_en": input_en,
            "facts": facts,
            "structured": structured
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "stage": "unknown"})

@app.after_request
def add_header(response):
    """Add headers to both force latest IE rendering engine or 
    to disable caching."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    print("Starting Ollama Web Interface on http://localhost:11435")
    
    # Auto-open browser disabled for Pinokio internal WebUI
    # if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    #     threading.Timer(1.5, lambda: webbrowser.open('http://localhost:11435')).start()
        
    app.run(host='0.0.0.0', port=11435, debug=False)
