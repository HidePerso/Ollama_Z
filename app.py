import gradio as gr
import subprocess
import json
import html as html_lib
import re
from datetime import datetime
import threading
import time
import requests
import signal
import os
import psutil
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Try to import GPUtil for GPU monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Try to import torch for better GPU info
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# -----------------------------
# Configuration & Constants
# -----------------------------
APP_DIR = Path(__file__).parent
SETTINGS_FILE = APP_DIR / "pipeline_settings.json"
PIPELINE_PRESETS_FILE = APP_DIR / "pipeline_presets.json"
OLLAMA_URL = "http://localhost:11434"

# Lock for Ollama CLI commands
ollama_lock = threading.Lock()

# Track active pull process for cancellation
current_pull_process = None
is_pull_cancelled = False

# -----------------------------
# Template Manager
# -----------------------------
TEMPLATES_FILE = APP_DIR / "templates.json"
DEFAULT_TEMPLATES = {
    "translator": "Translate usage text into English. Output only translation.",
    "extractor": "Extract atomic facts from description. LOSSLESS. One fact per line.",
    "structurer": "Organize facts into structured blocks [Perspective], [Subject], etc. LOSSLESS.",
    "validator": "Strict validator. Fix missing/added details in structured prompt."
}

UNDEFINED_ROLE = "-- undefined --"

class TemplateManager:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._cache = {}
        if not self.file_path.exists():
            self._cache = DEFAULT_TEMPLATES.copy()
            self.save(self._cache)
        else:
            self.load()

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except:
            self._cache = DEFAULT_TEMPLATES.copy()
        return self._cache

    def save(self, data):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._cache = data.copy()
            return True
        except:
            return False

    def get(self, key):
        return self._cache.get(key, "")

    def set(self, key, value):
        self._cache[key] = value
        self.save(self._cache)
        
    def delete(self, key):
        if key in self._cache:
            del self._cache[key]
            self.save(self._cache)
            return True
        return False

    def keys(self):
        return list(self._cache.keys())

template_manager = TemplateManager(TEMPLATES_FILE)

# -----------------------------
# Pipeline Preset Manager
# -----------------------------
DEFAULT_PIPELINE_PRESET = {
    "Default": {
        "translator": "translator",
        "extractor": "extractor",
        "structurer": "structurer",
        "validator": "validator"
    }
}

class PipelinePresetManager:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._cache = {}
        if not self.file_path.exists():
            self._cache = DEFAULT_PIPELINE_PRESET.copy()
            self.save(self._cache)
        else:
            self.load()

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except:
            self._cache = DEFAULT_PIPELINE_PRESET.copy()
        return self._cache

    def save(self, data):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._cache = data.copy()
            return True
        except:
            return False

    def get(self, key):
        return self._cache.get(key, {})

    def set(self, key, value):
        self._cache[key] = value
        self.save(self._cache)
        
    def delete(self, key):
        if key in self._cache:
            del self._cache[key]
            self.save(self._cache)
            return True
        return False

    def keys(self):
        return list(self._cache.keys())

pipeline_preset_manager = PipelinePresetManager(PIPELINE_PRESETS_FILE)

# -----------------------------
# Settings Manager
# -----------------------------
class SettingsManager:
    def __init__(self, settings_file: Path):
        self.settings_file = Path(settings_file)
        self._cache = {}
        self._loaded = False
    
    def load(self) -> Dict[str, Any]:
        if self._loaded:
            return self._cache.copy()
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                    self._loaded = True
            except:
                self._cache = {}
        else:
            self._cache = {}
        self._loaded = True
        return self._cache.copy()
    
    def save(self, settings: Dict[str, Any]):
        with open(self.settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        self._cache = settings.copy()

    def get(self, key, default=None):
        return self.load().get(key, default)

    def set(self, key, value):
        settings = self.load()
        settings[key] = value
        self.save(settings)

settings_manager = SettingsManager(SETTINGS_FILE)

# -----------------------------
# Ollama Core Logic
# -----------------------------
def run_ollama_command(command):
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
        except Exception as e:
            return {"success": False, "error": str(e)}

def run_ollama_command_args(args, timeout=30):
    with ollama_lock:
        try:
            result = subprocess.run(
                ["ollama", *args],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

def parse_list_output(output):
    lines = output.strip().split('\n')
    if len(lines) < 2:
        return []
    models = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            models.append({
                "name": parts[0],
                "id": parts[1],
                "size": f"{parts[2]} {parts[3] if len(parts) > 3 else ''}".strip()
            })
    return models

def get_gpu_stats():
    stats = {
        'gpu_percent': 0, 'vram_percent': 0, 'vram_used': 0,
        'vram_total': 0, 'gpu_available': False, 'gpu_name': 'N/A'
    }
    # Check with nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.split(',')]
            if len(parts) >= 4:
                stats['gpu_name'] = parts[0]
                stats['gpu_percent'] = float(parts[1])
                stats['vram_used'] = round(float(parts[2]) / 1024, 2)
                stats['vram_total'] = round(float(parts[3]) / 1024, 2)
                stats['vram_percent'] = round((float(parts[2]) / float(parts[3])) * 100, 1)
                stats['gpu_available'] = True
                return stats
    except: pass
    
    if GPUTIL_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats.update({
                    'gpu_percent': round(gpu.load * 100, 1),
                    'vram_percent': round(gpu.memoryUtil * 100, 1),
                    'vram_used': round(gpu.memoryUsed / 1024, 2),
                    'vram_total': round(gpu.memoryTotal / 1024, 2),
                    'gpu_name': gpu.name,
                    'gpu_available': True
                })
        except: pass
    return stats

# -----------------------------
# Pipeline Helpers
# -----------------------------
TRANSLATOR_SYSTEM = "Translate usage text into English. Output only translation."
EXTRACTOR_SYSTEM = "Extract atomic facts from description. LOSSLESS. One fact per line."
STRUCTURER_SYSTEM = "Organize facts into structured blocks [Perspective], [Subject], etc. LOSSLESS."
VALIDATOR_SYSTEM = "Strict validator. Fix missing/added details in structured prompt."

def normalize_facts(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s: continue
        s = re.sub(r'^\s*[-*•]\s+', '', s)
        s = re.sub(r'^\s*\d+[.)]\s+', '', s)
        lines.append(s)
    return '\n'.join(lines).strip()

def has_cyrillic(text: str) -> bool:
    if not text: return False
    cyrillic_count = len(re.findall(r'[\u0400-\u04FF]', text))
    letter_count = len(re.findall(r'[\w]', text, re.UNICODE))
    return (cyrillic_count / letter_count > 0.3) if letter_count > 0 else False

def ollama_generate_sync(model: str, prompt: str, temperature: float = 0.1) -> str:
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096}
        }, timeout=300)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        raise Exception(f"Model {model} error: {str(e)}")

# -----------------------------
# Gradio UI Components
# -----------------------------

# =============================================================================
# Theme Configuration
# =============================================================================

# Built-in Gradio themes
BUILTIN_THEMES = {
    "Default": gr.themes.Default(),
    "Soft": gr.themes.Soft(),
    "Monochrome": gr.themes.Monochrome(),
    "Glass": gr.themes.Glass(),
    "Base": gr.themes.Base(),
    "Ocean": gr.themes.Ocean(),
    "Origin": gr.themes.Origin(),
    "Citrus": gr.themes.Citrus(),
}

# Community themes from Hugging Face Spaces
COMMUNITY_THEMES = {
    "Miku": "NoCrypt/miku",
    "Interstellar": "Nymbo/Interstellar",
    "xkcd": "gstaff/xkcd",
}

def get_theme_from_settings():
    theme_name = settings_manager.get("ui_theme", "Default")
    # Check built-in themes first
    if theme_name in BUILTIN_THEMES:
        return BUILTIN_THEMES[theme_name]
    # Check community themes
    if theme_name in COMMUNITY_THEMES:
        return COMMUNITY_THEMES[theme_name]
    return gr.themes.Default()

CSS = """
/* Z-Fusion Accent Monitors */
.monitor-box textarea {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    font-size: 0.85em !important;
    line-height: 1.6 !important;
    padding: 12px !important;
    border-radius: 8px !important;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(255, 255, 255, 0.05) 100%) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    resize: none !important;
    box-shadow: none !important;
    border: none !important;
    border-radius: 0 !important;
}
.gpu-monitor textarea { border-left: 4px solid #667eea !important; }
.cpu-monitor textarea { border-left: 4px solid #f5576c !important; }
.ram-monitor textarea { border-left: 4px solid #34d399 !important; }

.gradio-container { width: 100% !important; max-width: 100% !important; margin: 0 !important; padding: 0 !important; }

/* Custom Accordion and Cards */
.pipeline-card { border: 1px solid rgba(0,0,0,0.1) !important; border-radius: 12px !important; }
.status-badge { padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
.status-badge.accent { background: rgba(102, 126, 234, 0.1); color: #667eea; border: 1px solid rgba(102, 126, 234, 0.2); }

/* Model Item Styles */
.model-item { padding: 12px; border-bottom: 1px solid rgba(0,0,0,0.05); }
.model-item:last-child { border-bottom: none; }
.model-name { font-weight: 600; color: var(--body-text-color); }
.model-meta { font-size: 0.8em; color: gray; font-family: monospace; }
.model-actions { margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; }
.model-action-btn {
  border: 1px solid rgba(0, 0, 0, 0.15);
  background: transparent;
  border-radius: 8px;
  height: 22px;
  min-height: 22px;
  padding: 0 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
  font-size: 0.8em;
  cursor: pointer;
  pointer-events: auto;
}
.model-action-btn.delete {
  border-color: rgba(220, 38, 38, 0.35);
  color: #dc2626;
}
.model-action-proxy {
  display: none !important;
}
.model-info-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.45);
  z-index: 10000;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
.model-info-modal-overlay.open {
  display: flex;
}
.model-info-modal {
  width: min(920px, 100%);
  max-height: 85vh;
  background: var(--body-background-fill, #fff);
  color: var(--body-text-color, #111827);
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 12px;
  box-shadow: 0 22px 48px rgba(15, 23, 42, 0.25);
  display: flex;
  flex-direction: column;
}
.model-info-modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.25);
}
.model-info-modal-title {
  margin: 0;
  font-size: 1rem;
}
.model-info-modal-close {
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: transparent;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
  padding: 4px 10px;
}
.model-info-modal-body {
  margin: 0;
  padding: 16px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.84rem;
  line-height: 1.45;
}

"""

ACTION_BUTTONS_JS = """
<script>
if (!window.handleInstalledModelAction) {
  function queryWithShadow(root, selector) {
    if (!root || !selector) return null;
    if (typeof root.querySelector === 'function') {
      var direct = root.querySelector(selector);
      if (direct) return direct;
    }
    if (typeof root.querySelectorAll === 'function') {
      var nodes = root.querySelectorAll('*');
      for (var i = 0; i < nodes.length; i++) {
        var shadow = nodes[i] && nodes[i].shadowRoot;
        if (!shadow) continue;
        var nested = queryWithShadow(shadow, selector);
        if (nested) return nested;
      }
    }
    return null;
  }

  function findInRoots(selector) {
    var direct = queryWithShadow(document, selector);
    if (direct) return direct;

    var apps = document.querySelectorAll('gradio-app');
    for (var i = 0; i < apps.length; i++) {
      var root = apps[i] && apps[i].shadowRoot;
      if (!root) continue;
      var found = queryWithShadow(root, selector);
      if (found) return found;
    }
    return null;
  }

  function getActionButtonFromEvent(event) {
    if (event && typeof event.composedPath === 'function') {
      var path = event.composedPath();
      for (var i = 0; i < path.length; i++) {
        var node = path[i];
        if (!node || !node.classList || !node.dataset) continue;
        if (node.classList.contains('model-action-btn') && node.dataset.action && node.dataset.model) {
          return node;
        }
      }
    }
    if (event && event.target && typeof event.target.closest === 'function') {
      return event.target.closest('.model-action-btn[data-action][data-model]');
    }
    return null;
  }

  window.handleInstalledModelAction = function(action, model) {
    var payloadInput = findInRoots('#model_action_payload textarea, #model_action_payload input');
    var submitBtn = findInRoots('#model_action_submit button, button#model_action_submit, #model_action_submit');
    if (!payloadInput || !submitBtn) return;

    var payload = JSON.stringify({ action: action, model: model });
    var proto = payloadInput.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    var descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
    var valueSetter = descriptor && descriptor.set ? descriptor.set : null;

    if (valueSetter) {
      valueSetter.call(payloadInput, payload);
    } else {
      payloadInput.value = payload;
    }
    payloadInput.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
    payloadInput.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
    submitBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true }));
  };
}

if (!window.ensureModelInfoPopup) {
  window.ensureModelInfoPopup = function() {
    var existing = document.getElementById('model_info_popup_overlay');
    if (existing) return existing;

    var overlay = document.createElement('div');
    overlay.id = 'model_info_popup_overlay';
    overlay.className = 'model-info-modal-overlay';
    overlay.innerHTML = [
      '<div class="model-info-modal" role="dialog" aria-modal="true" aria-labelledby="model_info_popup_title">',
      '  <div class="model-info-modal-header">',
      '    <h3 id="model_info_popup_title" class="model-info-modal-title">Model Info</h3>',
      '    <button type="button" class="model-info-modal-close" data-close-model-info-popup>Close</button>',
      '  </div>',
      '  <pre id="model_info_popup_body" class="model-info-modal-body"></pre>',
      '</div>'
    ].join('');

    overlay.addEventListener('click', function(evt) {
      if (evt.target === overlay) overlay.classList.remove('open');
    });

    document.addEventListener('keydown', function(evt) {
      if (evt.key === 'Escape') overlay.classList.remove('open');
    });

    document.body.appendChild(overlay);
    return overlay;
  };

  window.openModelInfoPopup = function(modelName, infoText) {
    var overlay = window.ensureModelInfoPopup();
    if (!overlay) return;
    var titleNode = overlay.querySelector('#model_info_popup_title');
    var bodyNode = overlay.querySelector('#model_info_popup_body');
    var closeNode = overlay.querySelector('[data-close-model-info-popup]');
    if (titleNode) titleNode.textContent = modelName ? ('Model Info: ' + modelName) : 'Model Info';
    if (bodyNode) bodyNode.textContent = infoText || '(No output)';
    if (closeNode && !closeNode.__boundClose) {
      closeNode.addEventListener('click', function() {
        overlay.classList.remove('open');
      });
      closeNode.__boundClose = true;
    }
    overlay.classList.add('open');
  };
}

if (!window.__bindModelInfoPopupSource) {
  window.__readModelInfoPopupPayload = function() {
    var source = findInRoots('#model_info_output textarea, #model_info_output input');
    if (!source) return "";

    // Gradio may recreate this node after updates; rebind listeners when that happens.
    if (window.__modelInfoPopupSourceNode !== source) {
      window.__modelInfoPopupSourceNode = source;
      if (!source.__modelInfoPopupBound) {
        var onPayloadEvent = function() {
          window.__checkModelInfoPopupPayload();
        };
        source.addEventListener('input', onPayloadEvent);
        source.addEventListener('change', onPayloadEvent);
        source.__modelInfoPopupBound = true;
      }
    }
    return (source.value || source.textContent || '').trim();
  };

  window.__checkModelInfoPopupPayload = function() {
    var raw = window.__readModelInfoPopupPayload();
    if (!raw || raw === window.__lastModelInfoPopupPayload) return;
    window.__lastModelInfoPopupPayload = raw;
    try {
      var payload = JSON.parse(raw);
      window.openModelInfoPopup(payload.model || '', payload.info || '');
    } catch (_err) {}
  };
}

if (!window.__modelInfoPopupSourceWatchdog) {
  window.__modelInfoPopupSourceWatchdog = window.setInterval(function() {
    window.__checkModelInfoPopupPayload();
  }, 350);
}

if (!window.__installedModelActionBound) {
  document.addEventListener('click', function(event) {
    var button = getActionButtonFromEvent(event);
    if (!button) return;
    event.preventDefault();
    var action = button.dataset.action;
    var model = button.dataset.model || '';
    if (action === 'delete') {
      var confirmed = window.confirm('Delete model "' + model + '" from Ollama?');
      if (confirmed) {
        window.handleInstalledModelAction('delete_confirmed', model);
      }
      return;
    }
    window.handleInstalledModelAction(action, model);
  });
  window.__installedModelActionBound = true;
}
</script>
"""

def get_system_monitor_info():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    gpu = get_gpu_stats()
    
    gpu_text = f"╔═ {gpu['gpu_name']}\n║ VRAM: {gpu['vram_used']:.1f}/{gpu['vram_total']:.1f}GB\n╚═ Load: {gpu['gpu_percent']}%"
    cpu_text = f"╔═ CPU: {psutil.cpu_count()} Threads\n╚═ Usage: {cpu}%"
    ram_text = f"╔═ RAM: {ram.used/(1024**3):.1f}/{ram.total/(1024**3):.1f}GB\n╚═ Usage: {ram.percent}%"
    return gpu_text, cpu_text, ram_text

def list_models():
    res = run_ollama_command("list")
    if res["success"]:
        models = [m["name"] for m in parse_list_output(res["output"])]
        return models
    return []

def update_model_dropdowns():
    models = list_models()
    return [gr.update(choices=models) for _ in range(4)]

def get_default_model(models, keyword, saved_val=None):
    if saved_val and saved_val in models:
        return saved_val
    for m in models:
        if keyword.lower() in m.lower():
            return m
    return None

def parse_role_config(role_key):
    if not role_key or role_key == UNDEFINED_ROLE:
        return None, None
        
    content = template_manager.get(role_key)
    if not content:
        # Fallback to defaults if key exists in defaults
        content = DEFAULT_TEMPLATES.get(role_key, "")
    
    # Simple parsing of Modelfile-like content
    model_match = re.search(r'^FROM\s+([^\s]+)', content, re.MULTILINE | re.IGNORECASE)
    model = model_match.group(1) if model_match else "qwen2.5:latest" # Default fallback
    
    # Extract System Prompt
    system_match = re.search(r'SYSTEM\s+"""(.*?)"""', content, re.DOTALL | re.IGNORECASE)
    if not system_match:
        system_match = re.search(r'SYSTEM\s+(.*)', content, re.IGNORECASE)
        
    system = system_match.group(1).strip() if system_match else content
    
    # Clean system prompt (remove potential escape issues if needed)
    return model, system

def compile_process(text, preset_name, progress=gr.Progress()):
    if not text: return "Error: Input text required", "", ""
    try:
        progress(0.1, desc="Checking language...")
        input_en = text
        
        # Get roles from preset
        preset = pipeline_preset_manager.get(preset_name)
        trans_role = preset.get("translator")
        extr_role = preset.get("extractor")
        stru_role = preset.get("structurer")
        vali_role = preset.get("validator")

        # Stage 0: Translator
        t_model, t_sys = parse_role_config(trans_role)
        if has_cyrillic(text):
            if not t_model: return "Error: Translator required for Cyrillic (role undefined)", "", ""
            
        if t_model:
            progress(0.2, desc=f"Translating via {trans_role} ({t_model})...")
            input_en = ollama_generate_sync(t_model, f"{t_sys}\n\nUSER TEXT:\n{text}", 0.05)
        else:
            # If no translator, use original text as input_en (already set)
            pass
        
        # Stage 1: Extractor
        e_model, e_sys = parse_role_config(extr_role)
        if not e_model: return "Error: Extractor role undefined", "", ""
        progress(0.4, desc=f"Extracting facts via {extr_role} ({e_model})...")
        facts = normalize_facts(ollama_generate_sync(e_model, f"{e_sys}\n\nINPUT:\n{input_en}", 0.1))
        
        # Stage 2: Structurer
        s_model, s_sys = parse_role_config(stru_role)
        if not s_model: return "Error: Structurer role undefined", "", ""
        progress(0.7, desc=f"Structuring via {stru_role} ({s_model})...")
        stru_p = (f"{s_sys}\n\nDESCRIPTION:\n{input_en}\n\nFACTS:\n{facts}\n\nOUTPUT:")
        structured = ollama_generate_sync(s_model, stru_p, 0.15)
        
        # Stage 3: Validator (Optional)
        v_model, v_sys = parse_role_config(vali_role)
        # Stage 3: Validator (Optional)
        v_model, v_sys = parse_role_config(vali_role)
        if v_model:
            progress(0.9, desc=f"Validating via {vali_role} ({v_model})...")
            vali_p = f"{v_sys}\n\nORIGINAL:\n{input_en}\n\nFACTS:\n{facts}\n\nCANDIDATE:\n{structured}\n\nFIXED:"
            structured = ollama_generate_sync(v_model, vali_p, 0.1)
            
        progress(1.0, desc="Complete!")
        return structured
    except Exception as e:
        return f"Error: {str(e)}"

def parse_size(size_str):
    try:
        size_str = size_str.upper()
        # Find first number
        match = re.search(r'([\d.]+)', size_str)
        if not match: return 0.0
        val = float(match.group(1))
        
        if 'GB' in size_str: return val * 1024
        elif 'KB' in size_str: return val / 1024
        return val # MB
    except: return 0.0

def pull_model_handler(model_name, progress=gr.Progress()):
    global current_pull_process, is_pull_cancelled

    def pull_ui_state(is_pulling, status_text=""):
        return (
            gr.update(visible=not is_pulling),
            gr.update(visible=is_pulling),
            gr.update(visible=not is_pulling),
            gr.update(value=status_text)
        )
    
    # 1. Update UI to show 'Cancel' button immediately
    yield pull_ui_state(True, "Connecting to Ollama...")
    
    if not model_name: 
        gr.Warning("Please enter a model name")
        yield pull_ui_state(False, "Please enter a model name")
        return

    try:
        is_pull_cancelled = False
        model_name = re.sub(r'^ollama\s+(run|pull)\s+', '', model_name, flags=re.IGNORECASE).strip()
        
        # Diagnostic: Check if ollama exists
        ollama_path = shutil.which("ollama")
        print(f"DEBUG: 'ollama' executable path: {ollama_path}")
        if not ollama_path:
            err_msg = "Ollama executable not found in PATH. Please ensure Ollama is installed and accessible."
            gr.Warning(err_msg)
            yield pull_ui_state(False, f"Error: {err_msg}")
            return

        print(f"DEBUG: Starting pull for '{model_name}'")
        progress(0, desc=f"Connecting to Ollama...")
        yield pull_ui_state(True, f"Connecting to Ollama ({model_name})...")
        
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace',
            creationflags=creationflags
        )
        
        current_pull_process = process
        time.sleep(0.5) # Give it a moment to start
        
        last_line = ""
        for line in iter(process.stdout.readline, ''):
            if is_pull_cancelled:
                break
            
            line = line.strip()
            if not line: continue
            last_line = line
            
            print(f"OLLAMA: {line}")
            
            # Parse progress
            pct = None
            desc = line
            
            pct_match = re.search(r'(\d+)%', line)
            size_match = re.search(r'([\d.]+\s*[KMG]B)\s*/\s*([\d.]+\s*[KMG]B)', line, re.IGNORECASE)
            
            if size_match:
                done_mb = parse_size(size_match.group(1))
                total_mb = parse_size(size_match.group(2))
                if total_mb > 0:
                    pct = done_mb / total_mb
                    desc = f"Downloading: {size_match.group(1)} / {size_match.group(2)}"
            elif pct_match:
                pct = float(pct_match.group(1)) / 100
            
            progress(0 if pct is None else pct, desc=desc)
            # Yield to keep UI state consistent
            yield pull_ui_state(True, desc)

        process.wait()
        print(f"DEBUG: Process exit code: {process.returncode}")
        
        if process.returncode == 0:
            progress(1.0, desc="Download complete")
            gr.Info(f"Successfully pulled {model_name}")
            yield pull_ui_state(False, f"Successfully pulled {model_name}")
        elif not is_pull_cancelled:
            err_msg = last_line if last_line else f"Exit code: {process.returncode}"
            gr.Warning(f"Failed to pull {model_name}: {err_msg}")
            # Keep the error message on the progress bar for a moment
            progress(1.0, desc=f"Error: {err_msg}")
            yield pull_ui_state(False, f"Failed to pull {model_name}: {err_msg}")
        else:
            yield pull_ui_state(False, "Pull cancelled")
        
    except Exception as e:
        print(f"DEBUG EXCEPTION: {e}")
        err_msg = f"Pull Error: {str(e)}"
        gr.Warning(err_msg)
        yield pull_ui_state(False, err_msg)
    finally:
        current_pull_process = None
        is_pull_cancelled = False
        print("DEBUG: Pull handler finished, resetting UI")
        yield pull_ui_state(False)

def cancel_pull_handler():
    global current_pull_process, is_pull_cancelled
    if current_pull_process:
        is_pull_cancelled = True
        try:
            if os.name == 'nt':
                # Send CTRL_BREAK to process group on Windows
                current_pull_process.send_signal(signal.CTRL_BREAK_EVENT)
                current_pull_process.terminate()
            else:
                current_pull_process.terminate()
            gr.Info("Pull cancelled")
        except Exception as e:
            pass # Process might already be dead
        current_pull_process = None
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="Pull cancelled")
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value="")
    )

def get_models_html():
    res = run_ollama_command("list")
    if res["success"]:
        models = parse_list_output(res["output"])
        if not models:
            return "<p>No models installed</p>"
        models_html = '<div class="models-container">'
        for m in models:
            safe_name = html_lib.escape(m["name"])
            safe_id = html_lib.escape(m["id"])
            safe_size = html_lib.escape(m["size"])
            safe_model_attr = html_lib.escape(m["name"], quote=True)
            models_html += f"""
            <div class="model-item">
                <div class="model-name">{safe_name}</div>
                <div class="model-meta">ID: {safe_id} | Size: {safe_size}</div>
                <div class="model-actions">
                    <button type="button" class="model-action-btn" data-action="info" data-model="{safe_model_attr}">Info</button>
                    <button type="button" class="model-action-btn delete" data-action="delete" data-model="{safe_model_attr}">Delete</button>
                </div>
            </div>
            """
        models_html += '</div>'
        return models_html
    return "<p>Error loading models</p>"

def refresh_models_panel():
    return (
        get_models_html(),
        gr.update(value=""),
        gr.update(value="", visible=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False),
    )

def handle_model_action(payload):
    if not payload:
        return (
            get_models_html(),
            gr.update(value="No model action provided."),
            gr.update(value="", visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
        )
    try:
        data = json.loads(payload)
    except Exception:
        return (
            get_models_html(),
            gr.update(value="Invalid model action payload."),
            gr.update(value="", visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
        )

    action = data.get("action")
    model_name = (data.get("model") or "").strip()
    if not action or not model_name:
        return (
            get_models_html(),
            gr.update(value="Missing action or model name."),
            gr.update(value="", visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
        )

    if action == "info":
        res = run_ollama_command_args(["show", model_name], timeout=60)
        if res["success"]:
            info_text = (res.get("output") or "").strip() or "(No output)"
            info_payload = json.dumps({"model": model_name, "info": info_text}, ensure_ascii=False)
            return (
                get_models_html(),
                gr.update(value=f"Loaded info for `{model_name}`"),
                gr.update(value=info_payload, visible=True),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(visible=False),
            )
        err = (res.get("error") or "Unknown error").strip()
        return (
            get_models_html(),
            gr.update(value=f"Failed to load info for `{model_name}`: `{err}`"),
            gr.update(value="", visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
        )

    if action == "delete":
        return (
            get_models_html(),
            gr.update(value=f"Delete requested for `{model_name}`. Confirm below."),
            gr.update(value="", visible=True),
            gr.update(value=model_name),
            gr.update(value=f"Are you sure you want to delete `{model_name}` from Ollama?"),
            gr.update(visible=True),
        )

    if action == "delete_confirmed":
        return confirm_model_delete(model_name)

    return (
        get_models_html(),
        gr.update(value=f"Unsupported action: `{action}`"),
        gr.update(value="", visible=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False),
    )

def cancel_model_delete():
    return (
        gr.update(value="Delete canceled."),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False),
    )

def confirm_model_delete(model_name):
    model_name = (model_name or "").strip()
    if not model_name:
        return (
            get_models_html(),
            gr.update(value="No model selected for deletion."),
            gr.update(value="", visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
        )
    res = run_ollama_command_args(["rm", model_name], timeout=90)
    if res["success"]:
        return (
            get_models_html(),
            gr.update(value=f"Deleted model `{model_name}`"),
            gr.update(value="", visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
        )
    err = (res.get("error") or "Unknown error").strip()
    return (
        get_models_html(),
        gr.update(value=f"Failed to delete `{model_name}`: `{err}`"),
        gr.update(value="", visible=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False),
    )

print("--- Building Gradio UI ---")

with gr.Blocks() as demo:
    
    with gr.Tabs() as main_tabs:
        with gr.Tab("📝 Prompt Helper"):
            with gr.Row():
                with gr.Column(scale=1, min_width=0):
                    gpu_mon = gr.Textbox(show_label=False, container=False, elem_classes=["monitor-box", "gpu-monitor"], lines=3, interactive=False)
                with gr.Column(scale=1, min_width=0):
                    cpu_mon = gr.Textbox(show_label=False, container=False, elem_classes=["monitor-box", "cpu-monitor"], lines=3, interactive=False)
                with gr.Column(scale=1, min_width=0):
                    ram_mon = gr.Textbox(show_label=False, container=False, elem_classes=["monitor-box", "ram-monitor"], lines=3, interactive=False)
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.TextArea(label="User Description", placeholder="Deep detailed scene description...", lines=10)
                    mode_sel = gr.Dropdown(choices=pipeline_preset_manager.keys(), label="Pipeline Preset", value="Default")
                    compile_btn = gr.Button("🚀 Compile Pipeline", variant="primary")
                
                with gr.Column(scale=1):
                    final_out = gr.TextArea(label="Final Structured Prompt", lines=15)
                    with gr.Row():
                        copy_btn = gr.Button("📋 Copy Result")
                        clear_btn = gr.Button("🗑️ Clear")
                    


        with gr.Tab("⚙️ Pipeline Settings"):
            gr.Markdown("### Configure models for each stage")
            current_presets = template_manager.keys()
            
            # Pipeline Preset Controls
            with gr.Row() as pipeline_preset_select_row:
                pipeline_preset = gr.Dropdown(choices=pipeline_preset_manager.keys(), label="Pipeline Presets", value="Default", scale=3)
                with gr.Column(scale=1):
                    add_pipeline_preset_btn = gr.Button("➕ Add New Preset")
                    delete_pipeline_preset_btn = gr.Button("🗑️ Delete Preset", variant="stop")

            # Add New Pipeline Preset UI
            with gr.Row(visible=False) as new_pipeline_preset_row:
                new_pipeline_preset_name = gr.Textbox(label="New Preset Name", placeholder="e.g., config-v2", scale=3)
                with gr.Column(scale=1):
                    save_new_pipeline_preset_btn = gr.Button("✅ Confirm Add")
                    cancel_add_pipeline_preset_btn = gr.Button("❌ Cancel")

            # Delete Pipeline Preset UI
            with gr.Row(visible=False) as delete_pipeline_preset_confirm_row:
                gr.Markdown("Are you sure you want to delete this preset?")
                confirm_delete_pipeline_preset_btn = gr.Button("✅ Yes, Delete", variant="stop")
                cancel_delete_pipeline_preset_btn = gr.Button("❌ Cancel")

            # Main Pipeline Drops
            current_role_choices = [UNDEFINED_ROLE] + current_presets
            
            def resolve_role_value(val):
                return val if val else UNDEFINED_ROLE

            with gr.Group():
                with gr.Row():
                    t_role = gr.Dropdown(label="Translator Role", choices=current_role_choices, value=resolve_role_value(settings_manager.get("translator_role", "translator")))
                    e_role = gr.Dropdown(label="Extractor Role", choices=current_role_choices, value=resolve_role_value(settings_manager.get("extractor_role", "extractor")))
                with gr.Row():
                    s_role = gr.Dropdown(label="Structurer Role", choices=current_role_choices, value=resolve_role_value(settings_manager.get("structurer_role", "structurer")))
                    v_role = gr.Dropdown(label="Validator Role", choices=current_role_choices, value=resolve_role_value(settings_manager.get("validator_role", "validator")))
            
            with gr.Row():
                # save_set_btn removed
                save_pipeline_preset_btn = gr.Button("💾 Save Preset", variant="primary")
                refresh_roles_btn = gr.Button("🔄 Refresh Roles")
            
            # pipeline_status removed

        with gr.Tab("⚙️ Pipeline Models"):
            gr.Markdown("### Define Pipeline Models")
            
            with gr.Row() as preset_select_row:
                role_preset = gr.Dropdown(choices=template_manager.keys(), label="Role Preset", value="translator", allow_custom_value=True, scale=3)
                with gr.Column(scale=1):
                    add_preset_btn = gr.Button("➕ Add New Preset")
                    delete_preset_btn = gr.Button("🗑️ Delete Preset", variant="stop")
            
            with gr.Row(visible=False) as delete_confirm_row:
                gr.Markdown("Are you sure you want to delete this preset?")
                confirm_delete_btn = gr.Button("✅ Yes, Delete", variant="stop")
                cancel_delete_btn = gr.Button("❌ Cancel")

            with gr.Row(visible=False) as new_preset_row:
                new_preset_name = gr.Textbox(label="New Preset Name", placeholder="e.g., reviewer", scale=3)
                with gr.Column(scale=1):
                    save_new_preset_btn = gr.Button("✅ Confirm Add")
                    cancel_add_btn = gr.Button("❌ Cancel")
                
            template_content = gr.TextArea(label="System Template", lines=10, value=template_manager.get("translator"))
            with gr.Row():
                save_template_btn = gr.Button("Save role preset", variant="primary", scale=1, min_width=90)
                generate_model_btn = gr.Button("Generate Pipeline Model", scale=1, min_width=140)
            generate_model_status = gr.Markdown("---")

        with gr.Tab("🦙 Ollama models"):

            with gr.Column():
                with gr.Row():
                    pull_name = gr.Textbox(
                        label="Pull Model (ollama pull)",
                        placeholder="Enter model name (e.g., llama2, mistral, qwen3:8b)",
                        scale=3
                    )
                    pull_btn = gr.Button("⬇️ Pull", scale=1, min_width=120)
                    cancel_pull_btn = gr.Button("❌ Cancel", scale=1, min_width=120, visible=False, variant="stop")
                pull_help = gr.HTML("""
                <div style="margin-top: 4px; margin-bottom: 8px; line-height: 1.5;">
                    <p style="margin: 0 0 6px 0;">
                        Enter an Ollama model name (for example: <code>llama2</code>, <code>mistral</code>, <code>qwen3:8b</code>) or a full pull/run command.
                    </p>
                    <a href="https://ollama.com/library" target="_blank" style="margin-right: 12px; text-decoration: none;">Ollama Library</a>
                    <a href="https://huggingface.co/models?library=gguf&sort=trending" target="_blank" style="text-decoration: none;">GGUF Models</a>
                </div>
                """)
                pull_status = gr.Markdown("---")
                
                refresh_models_btn = gr.Button("🔄 Refresh List")
                models_html = gr.HTML(get_models_html())
                model_action_status = gr.Markdown("---")
                model_info_output = gr.TextArea(
                    label="Model Info",
                    lines=12,
                    interactive=False,
                    visible=True,
                    elem_id="model_info_output",
                    elem_classes=["model-action-proxy"]
                )
                model_action_payload = gr.Textbox(visible=True, elem_id="model_action_payload", elem_classes=["model-action-proxy"])
                model_action_submit = gr.Button("Model Action Trigger", visible=True, elem_id="model_action_submit", elem_classes=["model-action-proxy"])
                model_delete_pending = gr.Textbox(visible=False)
                with gr.Row(visible=False) as model_delete_confirm_row:
                    model_delete_confirm_text = gr.Markdown("")
                    confirm_model_delete_btn = gr.Button("✅ Confirm Delete", variant="stop")
                    cancel_model_delete_btn = gr.Button("❌ Cancel")

        with gr.Tab("🛠️ App Settings"):
            all_themes = list(BUILTIN_THEMES.keys()) + list(COMMUNITY_THEMES.keys())
            theme_sel = gr.Dropdown(label="UI Theme", choices=all_themes, value=settings_manager.get("ui_theme", "Default"))
            save_theme_btn = gr.Button("💾 Save Theme (Requires Restart)", variant="primary")
            gr.Markdown("Theme application via Gradio `theme` parameter (requires restart/reload for non-dynamic themes).")

    # Initial loading
    timer = gr.Timer(2.0)
    timer.tick(get_system_monitor_info, outputs=[gpu_mon, cpu_mon, ram_mon])

    # Events
    compile_btn.click(compile_process, [input_text, mode_sel], [final_out])
    
    def save_theme(theme_name):
        settings_manager.set("ui_theme", theme_name)
        gr.Info(f"Theme saved as {theme_name}. Please restart the app.")
        
    save_theme_btn.click(save_theme, theme_sel, None)
    
    def refresh_roles():
        keys = [UNDEFINED_ROLE] + template_manager.keys()
        return [gr.update(choices=keys) for _ in range(4)]
        
    refresh_roles_btn.click(refresh_roles, None, [t_role, e_role, s_role, v_role])
    
    # Pipeline Preset Events
    def on_pipeline_preset_change(name):
        preset = pipeline_preset_manager.get(name)
        def get_val(key, default):
            v = preset.get(key, default)
            return v if v else UNDEFINED_ROLE
            
        return [
            gr.update(value=get_val("translator", "translator")),
            gr.update(value=get_val("extractor", "extractor")),
            gr.update(value=get_val("structurer", "structurer")),
            gr.update(value=get_val("validator", "validator"))
        ]

    pipeline_preset.change(on_pipeline_preset_change, pipeline_preset, [t_role, e_role, s_role, v_role])

    def show_add_pipeline_preset():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    add_pipeline_preset_btn.click(show_add_pipeline_preset, None, [new_pipeline_preset_row, pipeline_preset_select_row, save_pipeline_preset_btn])

    def cancel_add_pipeline_preset():
        return gr.update(visible=False), "", gr.update(visible=True), gr.update(visible=True)

    cancel_add_pipeline_preset_btn.click(cancel_add_pipeline_preset, None, [new_pipeline_preset_row, new_pipeline_preset_name, pipeline_preset_select_row, save_pipeline_preset_btn])

    def add_new_pipeline_preset(name, tm, em, sm, vm):
        if not name: return gr.update(), gr.update(choices=pipeline_preset_manager.keys()), gr.update(), gr.update(), gr.update()
        
        def clean_val(v): return v if v != UNDEFINED_ROLE else ""
        
        data = {
            "translator": clean_val(tm), "extractor": clean_val(em),
            "structurer": clean_val(sm), "validator": clean_val(vm)
        }
        pipeline_preset_manager.set(name, data)
        updated_choices = pipeline_preset_manager.keys()
        gr.Info(f"Pipeline Preset '{name}' added!")
        return gr.update(visible=False), gr.update(choices=updated_choices, value=name), gr.update(visible=True), gr.update(visible=True), gr.update(choices=updated_choices)

    save_new_pipeline_preset_btn.click(add_new_pipeline_preset, [new_pipeline_preset_name, t_role, e_role, s_role, v_role], [new_pipeline_preset_row, pipeline_preset, pipeline_preset_select_row, save_pipeline_preset_btn, mode_sel])

    def show_delete_pipeline_preset():
        return gr.update(visible=True)

    delete_pipeline_preset_btn.click(show_delete_pipeline_preset, None, delete_pipeline_preset_confirm_row)

    def cancel_delete_pipeline_preset():
        return gr.update(visible=False)

    cancel_delete_pipeline_preset_btn.click(cancel_delete_pipeline_preset, None, delete_pipeline_preset_confirm_row)

    def confirm_delete_pipeline_preset(name):
        if name == "Default":
            gr.Warning("Cannot delete Default preset")
            return gr.update(visible=False), gr.update(choices=pipeline_preset_manager.keys()), gr.update()
        
        if pipeline_preset_manager.delete(name):
            gr.Info(f"Pipeline Preset '{name}' deleted!")
            keys = pipeline_preset_manager.keys()
            new_val = keys[0] if keys else None
            return gr.update(visible=False), gr.update(choices=keys, value=new_val), gr.update(choices=keys, value="Default")
        return gr.update(visible=False), gr.update(choices=pipeline_preset_manager.keys()), gr.update()

    confirm_delete_pipeline_preset_btn.click(confirm_delete_pipeline_preset, pipeline_preset, [delete_pipeline_preset_confirm_row, pipeline_preset, mode_sel])

    def save_current_pipeline_preset(name, tm, em, sm, vm):
        if not name: return
        
        def clean_val(v): return v if v != UNDEFINED_ROLE else ""
        
        data = {
            "translator": clean_val(tm), "extractor": clean_val(em),
            "structurer": clean_val(sm), "validator": clean_val(vm)
        }
        pipeline_preset_manager.set(name, data)
        gr.Info(f"Pipeline Preset '{name}' updated!")

    save_pipeline_preset_btn.click(save_current_pipeline_preset, [pipeline_preset, t_role, e_role, s_role, v_role], None)
    
    # Models Tab Events
    refresh_models_btn.click(
        refresh_models_panel,
        None,
        [models_html, model_action_status, model_info_output, model_delete_pending, model_delete_confirm_text, model_delete_confirm_row]
    )
    model_action_submit.click(
        handle_model_action,
        model_action_payload,
        [models_html, model_action_status, model_info_output, model_delete_pending, model_delete_confirm_text, model_delete_confirm_row]
    )
    
    # Wired to handler that yields button updates
    pull_event = pull_btn.click(
        pull_model_handler,
        pull_name,
        [pull_btn, cancel_pull_btn, pull_help, pull_status],
        show_progress="full"
    )
    # Refresh list after pull is done (generator finishes)
    pull_event.then(
        refresh_models_panel,
        None,
        [models_html, model_action_status, model_info_output, model_delete_pending, model_delete_confirm_text, model_delete_confirm_row]
    )
    
    cancel_pull_btn.click(cancel_pull_handler, None, [pull_btn, cancel_pull_btn, pull_help, pull_status])
    confirm_model_delete_btn.click(
        confirm_model_delete,
        model_delete_pending,
        [models_html, model_action_status, model_info_output, model_delete_pending, model_delete_confirm_text, model_delete_confirm_row]
    )
    cancel_model_delete_btn.click(
        cancel_model_delete,
        None,
        [model_action_status, model_delete_pending, model_delete_confirm_text, model_delete_confirm_row]
    )
    
    def on_role_change(role):
        return template_manager.get(role)
        
    role_preset.change(on_role_change, role_preset, template_content)
    
    def show_add_preset():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        
    add_preset_btn.click(show_add_preset, None, [new_preset_row, preset_select_row, save_template_btn])
    
    def cancel_add():
        return gr.update(visible=False), "", gr.update(visible=True), gr.update(visible=True)
        
    cancel_add_btn.click(cancel_add, None, [new_preset_row, new_preset_name, preset_select_row, save_template_btn])
    
    def add_new_preset(name, content):
        if not name: return gr.update(), gr.update(choices=template_manager.keys()), gr.update(), gr.update()
        template_manager.set(name, content)
        gr.Info(f"Preset '{name}' added successfully!")
        return gr.update(visible=False), gr.update(choices=template_manager.keys(), value=name), gr.update(visible=True), gr.update(visible=True)
        
    save_new_preset_btn.click(add_new_preset, [new_preset_name, template_content], [new_preset_row, role_preset, preset_select_row, save_template_btn])
    
    def delete_preset_ui():
        return gr.update(visible=True)
        
    delete_preset_btn.click(delete_preset_ui, None, delete_confirm_row)
    
    def cancel_delete():
        return gr.update(visible=False)
        
    cancel_delete_btn.click(cancel_delete, None, delete_confirm_row)
    
    def confirm_delete(role):
        if role in DEFAULT_TEMPLATES:
            gr.Warning(f"Cannot delete default preset '{role}'")
            return gr.update(visible=False), gr.update(choices=template_manager.keys())
            
        if template_manager.delete(role):
            gr.Info(f"Preset '{role}' deleted successfully!")
            new_keys = template_manager.keys()
            new_val = new_keys[0] if new_keys else None
            return gr.update(visible=False), gr.update(choices=new_keys, value=new_val)
        return gr.update(visible=False), gr.update(choices=template_manager.keys())

    confirm_delete_btn.click(confirm_delete, role_preset, [delete_confirm_row, role_preset])

    def save_template(role, content):
        if not role: return
        template_manager.set(role, content)
        gr.Info(f"Template for '{role}' saved!")

    def generate_model(role, content, progress=gr.Progress()):
        def generate_ui_state(is_generating, status_text=""):
            return (
                gr.update(interactive=not is_generating),
                gr.update(value=status_text)
            )

        yield generate_ui_state(True, "Connecting to Ollama create API...")

        if not role:
            gr.Warning("Select a role preset first.")
            yield generate_ui_state(False, "Select a role preset first.")
            return
        if not content or not content.strip():
            gr.Warning("System template is empty.")
            yield generate_ui_state(False, "System template is empty.")
            return

        # Persist editor content to preset, then always build from preset source.
        template_manager.set(role, content)
        preset_content = template_manager.get(role)
        if not preset_content or not preset_content.strip():
            gr.Warning(f"Preset '{role}' is empty.")
            yield generate_ui_state(False, f"Preset '{role}' is empty.")
            return

        lines = preset_content.splitlines()
        first_non_empty = next((ln.strip() for ln in lines if ln.strip()), "")
        from_match = re.match(r"^FROM\s+([^\s#]+)", first_non_empty, flags=re.IGNORECASE)
        if not from_match:
            gr.Warning("First non-empty line must be `FROM <model>` (for example: `FROM qwen3:8b`).")
            yield generate_ui_state(False, "First non-empty line must be `FROM <model>`.")
            return

        source_model = from_match.group(1).strip()
        source_base = source_model.split("/")[-1].split(":", 1)[0]
        safe_source_base = re.sub(r"[^a-zA-Z0-9._-]+", "-", source_base).strip("-").lower()
        safe_role = re.sub(r"[^a-zA-Z0-9._-]+", "-", role.strip()).strip("-").lower()
        if not safe_source_base:
            gr.Warning("Could not derive base model name from the `FROM` line.")
            yield generate_ui_state(False, "Could not derive base model name from the `FROM` line.")
            return
        if not safe_role:
            safe_role = "role"

        model_name = f"{safe_source_base}-{safe_role}"
        modelfile_path = APP_DIR / f"Modelfile.{safe_role}"
        modelfile_path.write_text(preset_content.rstrip() + "\n", encoding="utf-8")

        progress(0, desc=f"Creating {model_name}...")
        yield generate_ui_state(True, f"Creating `{model_name}`...")

        try:
            request_json = {
                "name": model_name,
                "from": source_model,
                "modelfile": preset_content.rstrip() + "\n"
            }

            with ollama_lock:
                with requests.post(
                    f"{OLLAMA_URL}/api/create",
                    json=request_json,
                    stream=True,
                    timeout=(10, 1800)
                ) as response:
                    response.raise_for_status()
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        try:
                            event = json.loads(raw_line)
                        except Exception:
                            continue

                        if event.get("error"):
                            raise RuntimeError(str(event["error"]))

                        status = str(event.get("status") or "").strip() or "Creating model..."
                        completed = event.get("completed")
                        total = event.get("total")
                        pct = None
                        if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
                            pct = max(0.0, min(1.0, float(completed) / float(total)))
                            status = f"{status} ({int(completed)}/{int(total)})"

                        progress(0 if pct is None else pct, desc=status)
                        last_status = status
                        yield generate_ui_state(True, status)

            progress(1.0, desc=f"Model {model_name} created")
            gr.Info(f"Model '{model_name}' generated successfully from {modelfile_path.name}.")
            yield generate_ui_state(False, f"Successfully generated `{model_name}`")
            return

        except requests.HTTPError as e:
            err_text = ""
            try:
                resp = e.response
                if resp is not None:
                    try:
                        err_json = resp.json()
                        err_text = str(err_json.get("error") or err_json).strip()
                    except Exception:
                        err_text = (resp.text or "").strip()
            except Exception:
                err_text = ""
            err = err_text or str(e).strip() or "Unknown error"
            gr.Warning(f"Failed to generate model '{model_name}': {err}")
            progress(1.0, desc=f"Error: {err}")
            yield generate_ui_state(False, f"Failed to generate `{model_name}`: {err}")
            return

        except Exception as e:
            err = str(e).strip() or "Unknown error"
            gr.Warning(f"Failed to generate model '{model_name}': {err}")
            progress(1.0, desc=f"Error: {err}")
            yield generate_ui_state(False, f"Failed to generate `{model_name}`: {err}")
            return
        
    save_template_btn.click(save_template, [role_preset, template_content], None)
    generate_model_btn.click(
        generate_model,
        [role_preset, template_content],
        [generate_model_btn, generate_model_status],
        show_progress="full"
    )
    
    copy_btn.click(fn=None, inputs=final_out, js="(x) => { navigator.clipboard.writeText(x); alert('Copied to clipboard!'); }")
    clear_btn.click(lambda: ["", ""], None, [input_text, final_out])

if __name__ == "__main__":
    print("--- Launching Demo on Port 11436 ---")
    demo.queue()
    demo.launch(
        server_name="127.0.0.1", 
        server_port=11436, 
        css=CSS,
        theme=get_theme_from_settings(),
        head=ACTION_BUTTONS_JS
    )
