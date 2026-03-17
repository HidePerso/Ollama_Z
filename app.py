import gradio as gr
import subprocess
import json
import html as html_lib
import re
import base64
import io
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
import tempfile

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
VLM_PRESETS_FILE = APP_DIR / "vlm_presets.json"
OLLAMA_URL = "http://localhost:11434"
VISION_MODEL_NOT_AVAILABLE = "-- not available --"

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
# VLM Preset Manager
# -----------------------------
ULTRA_DETAIL_PROMPT = """Describe the image as thoroughly and objectively as possible, like a professional visual analyst.
Do not invent details that are not clearly visible.
Structure the response strictly according to the sections below.

1. Overall Scene:
   - What is happening?
   - Type of image (photograph, illustration, 3D render, anime, painting, etc.)
   - General mood and atmosphere.

2. Composition:
   - Camera angle (top-down, low-angle, frontal, etc.)
   - Framing (close-up, medium shot, full body, wide shot, etc.)
   - Placement of key subjects within the frame.
   - Depth structure (foreground, midground, background).

3. Main Subjects or Characters:
   - Number of subjects.
   - Apparent gender, approximate age, build (if visible).
   - Pose, gestures, facial expression.
   - Clothing, accessories, visible textures of materials.
   - Distinguishing features (tattoos, logos, jewelry, markings, etc.).

4. Environment and Background Details:
   - Interior or exterior setting.
   - Architecture, furniture, natural elements, objects.
   - Small visible details that stand out.

5. Color and Lighting:
   - Dominant colors.
   - Contrast level.
   - Type of lighting (natural, studio, neon, cinematic, etc.).
   - Direction of light.
   - Shadows and reflections.

6. Textures and Materials:
   - Skin, fabric, metal, glass, wood, etc.
   - Level of surface detail (pores, brush strokes, grain, noise, gloss, etc.).

7. Style and Technical Characteristics:
   - Artistic style.
   - Realism vs stylization.
   - Signs of AI generation or post-processing (if noticeable).
   - Approximate image quality and resolution impression.

8. Additional Observations:
   - Visual emphasis or focal points.
   - Symbolism or narrative hints (only if clearly implied).
   - Any unusual or notable elements.

Write in an informative, precise manner without unnecessary artistic embellishment.
Limit the response to approximately 400-600 words."""

DEFAULT_VLM_PRESETS = {
    "Fast": (
        "Describe only visible content in this image for image generation. "
        "Use up to 4 concise sentences. Focus on main subject, composition, "
        "lighting, and dominant colors. Do not invent details."
    ),
    "Detailed": (
        "Describe only visible content in this image for image generation. "
        "Use up to 8 concise sentences. Include subject details, framing, "
        "background elements, lighting, colors, textures, and style cues. "
        "Do not invent details."
    ),
    "Ultra Detailed": ULTRA_DETAIL_PROMPT,
    "Ultra detail NSFW": (
        ULTRA_DETAIL_PROMPT
        + "\n\nAdditional focus for NSFW/adult images:\n"
        + "- Describe visible adult anatomy and intimate details explicitly and objectively.\n"
        + "- Include pose, body proportions, skin texture, clothing state, and interaction context if visible.\n"
        + "- Keep factual tone and do not invent hidden details."
    )
}
VLM_PRESET_CHOICES = ["Fast", "Detailed", "Ultra Detailed", "Ultra detail NSFW"]


class VlmPresetManager:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._cache = {}
        if not self.file_path.exists():
            self._cache = DEFAULT_VLM_PRESETS.copy()
            self.save(self._cache)
        else:
            self.load()

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except:
            self._cache = DEFAULT_VLM_PRESETS.copy()
        for key, value in DEFAULT_VLM_PRESETS.items():
            if key not in self._cache:
                self._cache[key] = value
        return self._cache

    def save(self, data):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._cache = data.copy()
            return True
        except:
            return False

    def get(self, key):
        return self._cache.get(key, "")

    def set(self, key, value):
        self._cache[key] = value
        self.save(self._cache)

    def keys(self):
        ordered = VLM_PRESET_CHOICES
        return [k for k in ordered if k in self._cache]


vlm_preset_manager = VlmPresetManager(VLM_PRESETS_FILE)

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


def extract_png_metadata(image_path: str) -> dict:
    result = {
        "prompt_text": "",
        "negative_prompt": "",
        "params": {},
        "error": None,
    }

    def _clean_resource_name(value: Any) -> str:
        text = str(value or "").strip().replace("\\", "/")
        if not text:
            return ""
        base = os.path.basename(text)
        stem, _ = os.path.splitext(base)
        return stem or base

    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            raw = value
            encodings = []
            # EXIF UserComment often starts with an 8-byte charset prefix:
            # ASCII\x00\x00\x00 / UNICODE\x00 / JIS\x00...
            if len(raw) >= 8:
                head = raw[:8]
                if head.startswith(b"ASCII\x00\x00\x00"):
                    encodings = ["utf-8", "latin-1"]
                    raw = raw[8:]
                elif head.startswith(b"UNICODE\x00"):
                    raw = raw[8:]
                    even_zeros = sum(1 for b in raw[0::2][:32] if b == 0)
                    odd_zeros = sum(1 for b in raw[1::2][:32] if b == 0)
                    # Civitai JPEGs may use either UTF-16BE or UTF-16LE in EXIF UserComment.
                    encodings = ["utf-16be", "utf-16le", "utf-16"] if even_zeros > odd_zeros else ["utf-16le", "utf-16be", "utf-16"]
                elif head.startswith(b"JIS\x00\x00\x00\x00\x00"):
                    encodings = ["shift_jis", "utf-8", "latin-1"]
                    raw = raw[8:]
            if not encodings:
                encodings = ["utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1"]
            for enc in encodings:
                try:
                    return raw.decode(enc, errors="ignore").strip("\x00")
                except Exception:
                    continue
            return ""
        text = str(value)
        # Some tools save EXIF comments as text starting with ASCII marker.
        if text.startswith("ASCII\x00\x00\x00"):
            text = text[8:]
        return text

    def _looks_like_prompt_json(text: str) -> bool:
        sample = (text or "").strip()
        if not sample:
            return False
        hint_tokens = (
            '"prompt"', '"positive"', '"negative"', '"negative_prompt"', '"negativeprompt"',
            '"sui_image_params"', '"sui_models"', '"class_type"', '"sampler"', '"steps"', '"seed"',
        )
        return any(token in sample for token in hint_tokens)

    def _extract_json_substrings(text: str):
        sample = text or ""
        if not sample:
            return
        stack = []
        start = None
        in_string = False
        escape = False
        for idx, ch in enumerate(sample):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                if not stack:
                    start = idx
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or ch != stack[-1]:
                    stack = []
                    start = None
                    continue
                stack.pop()
                if not stack and start is not None:
                    candidate = sample[start:idx + 1].strip()
                    if len(candidate) >= 16 and _looks_like_prompt_json(candidate):
                        yield candidate
                    start = None

    def _try_parse_json_candidates(text: str) -> bool:
        parsed = False
        for candidate in _extract_json_substrings(text):
            parsed = _fill_from_comfy_prompt(candidate) or parsed
            if result["prompt_text"] and result["params"]:
                break
        return parsed

    def _fill_from_generic_json(data: Any) -> bool:
        found = False

        def _set_if_longer(key: str, value: Any):
            nonlocal found
            if not isinstance(value, str):
                return
            val = value.strip()
            if not val:
                return
            if len(val) > len(result[key]):
                result[key] = val
                found = True

        if isinstance(data, dict):
            sui_params = data.get("sui_image_params")
            if isinstance(sui_params, dict):
                _set_if_longer("prompt_text", sui_params.get("prompt"))
                _set_if_longer("negative_prompt", sui_params.get("negativeprompt"))
                _set_if_longer("negative_prompt", sui_params.get("negative_prompt"))
                swarm_key_map = {
                    "steps": "steps",
                    "cfgscale": "cfg",
                    "cfg_scale": "cfg",
                    "sampler": "sampler",
                    "scheduler": "scheduler",
                    "seed": "seed",
                    "width": "width",
                    "height": "height",
                    "model": "model",
                    "vae": "vae",
                    "clipstopatlayer": "clip_skip",
                    "refinercontrolpercentage": "denoise",
                    "refinermodel": "hires_checkpoint",
                    "refinervae": "hires_vae",
                    "refinersteps": "hires_steps",
                    "refinercfgscale": "hires_cfg",
                    "refinersampler": "hires_sampler",
                    "refinerscheduler": "hires_scheduler",
                    "refinerupscale": "hires_upscale",
                    "refinerupscalemethod": "hires_upscaler",
                    "swarm_version": "version",
                }
                for src_key, dst_key in swarm_key_map.items():
                    if src_key in sui_params and sui_params.get(src_key) not in (None, ""):
                        result["params"][dst_key] = sui_params.get(src_key)
                        found = True

            sui_models = data.get("sui_models")
            if isinstance(sui_models, list):
                hash_chunks = []
                for item in sui_models:
                    if not isinstance(item, dict):
                        continue
                    param_name = str(item.get("param") or "").strip()
                    model_name = item.get("name")
                    model_hash = item.get("hash")
                    if param_name == "model" and model_name:
                        result["params"]["model"] = model_name
                        found = True
                    elif param_name == "vae" and model_name:
                        result["params"]["vae"] = model_name
                        found = True
                    elif param_name == "refinermodel" and model_name:
                        result["params"]["hires_checkpoint"] = model_name
                        found = True
                    elif param_name == "refinervae" and model_name:
                        result["params"]["hires_vae"] = model_name
                        found = True
                    if model_hash:
                        label = param_name or _clean_resource_name(model_name) or "model"
                        hash_chunks.append(f"{label}: {model_hash}")
                        if param_name == "model":
                            result["params"]["model_hash"] = model_hash
                            found = True
                        elif param_name == "vae":
                            result["params"]["vae_hash"] = model_hash
                            found = True
                if hash_chunks:
                    result["params"]["hashes"] = ", ".join(hash_chunks)
                    found = True

            _set_if_longer("prompt_text", data.get("prompt"))
            _set_if_longer("prompt_text", data.get("positive"))
            _set_if_longer("prompt_text", data.get("positive_prompt"))
            _set_if_longer("negative_prompt", data.get("negative"))
            _set_if_longer("negative_prompt", data.get("negative_prompt"))

            key_map = {
                "steps": "steps",
                "cfg": "cfg",
                "cfg_scale": "cfg",
                "sampler": "sampler",
                "sampler_name": "sampler",
                "scheduler": "scheduler",
                "seed": "seed",
                "width": "width",
                "height": "height",
                "model": "model",
                "model_hash": "model_hash",
                "clip_skip": "clip_skip",
                "denoise": "denoise",
                "denoising_strength": "denoise",
                "vae": "vae",
                "vae_hash": "vae_hash",
                "version": "version",
                "civitai_resources": "civitai_resources",
            }
            for src_key, dst_key in key_map.items():
                if src_key in data and data.get(src_key) not in (None, ""):
                    result["params"][dst_key] = data.get(src_key)
                    found = True

            hashes_obj = data.get("hashes")
            if isinstance(hashes_obj, dict) and hashes_obj:
                chunks = []
                for k, v in hashes_obj.items():
                    ktxt = str(k).strip()
                    vtxt = str(v).strip()
                    if not vtxt:
                        continue
                    chunks.append(f"{ktxt}: {vtxt}" if ktxt else vtxt)
                if chunks:
                    result["params"]["hashes"] = ", ".join(chunks)
                    found = True

            for v in data.values():
                if isinstance(v, (dict, list)):
                    found = _fill_from_generic_json(v) or found

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    found = _fill_from_generic_json(item) or found

        return found

    def _fill_from_comfy_prompt(raw_prompt: str) -> bool:
        try:
            prompt_data = json.loads(raw_prompt)
        except json.JSONDecodeError:
            return False

        if not isinstance(prompt_data, dict):
            return _fill_from_generic_json(prompt_data)

        found = False
        lora_resources = []

        def _get_prompt_node(node_ref: Any) -> Optional[dict]:
            if not isinstance(node_ref, (list, tuple)) or not node_ref:
                return None
            node_key = str(node_ref[0])
            node = prompt_data.get(node_key)
            return node if isinstance(node, dict) else None

        def _resolve_prompt_ref(node_ref: Any) -> Any:
            node = _get_prompt_node(node_ref)
            if not node:
                return node_ref
            output_index = node_ref[1] if isinstance(node_ref, (list, tuple)) and len(node_ref) > 1 else 0
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                return node_ref
            class_type_l = str(node.get("class_type", "")).lower()
            if "mxslider2d" in class_type_l:
                if output_index == 0:
                    return inputs.get("Xi", inputs.get("Xf"))
                if output_index == 1:
                    return inputs.get("Yi", inputs.get("Yf"))
            if output_index == 0:
                for key in ("width", "value", "Xi", "Xf"):
                    if key in inputs and inputs.get(key) not in (None, ""):
                        return inputs.get(key)
            if output_index == 1:
                for key in ("height", "value", "Yi", "Yf"):
                    if key in inputs and inputs.get(key) not in (None, ""):
                        return inputs.get(key)
            return node_ref

        def _remember_lora(name_value: Any, weight_value: Any):
            nonlocal found
            clean_name = _clean_resource_name(name_value)
            if not clean_name:
                return
            weight = weight_value if weight_value not in (None, "") else 1
            try:
                weight = float(weight)
            except Exception:
                pass
            entry = {
                "type": "lora",
                "weight": weight,
                "modelName": clean_name,
            }
            if entry not in lora_resources:
                lora_resources.append(entry)
                found = True

        for _, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            inputs = node_data.get("inputs", {})
            class_type = str(node_data.get("class_type", ""))
            class_type_l = class_type.lower()

            if isinstance(inputs, dict):
                for text_key in ("text", "prompt", "positive", "positive_prompt", "clip_l", "t5xxl"):
                    text_val = inputs.get(text_key)
                    if isinstance(text_val, str) and len(text_val) > len(result["prompt_text"]):
                        result["prompt_text"] = text_val
                        found = True
                for text_key in ("negative", "negative_prompt", "uc"):
                    text_val = inputs.get(text_key)
                    if isinstance(text_val, str) and len(text_val) > len(result["negative_prompt"]):
                        result["negative_prompt"] = text_val
                        found = True

                if "ksampler" in class_type_l or "sampler" in class_type_l:
                    if "seed" in inputs:
                        result["params"]["seed"] = inputs["seed"]
                    if "steps" in inputs:
                        result["params"]["steps"] = inputs["steps"]
                    if "cfg" in inputs:
                        result["params"]["cfg"] = inputs["cfg"]
                    if "sampler_name" in inputs:
                        result["params"]["sampler"] = inputs["sampler_name"]
                    if "scheduler" in inputs:
                        result["params"]["scheduler"] = inputs["scheduler"]
                if "width" in inputs and "height" in inputs:
                    result["params"]["width"] = _resolve_prompt_ref(inputs["width"])
                    result["params"]["height"] = _resolve_prompt_ref(inputs["height"])
                if "vaeloader" in class_type_l and "vae_name" in inputs:
                    result["params"]["vae"] = inputs["vae_name"]
                if "checkpointloader" in class_type_l and "ckpt_name" in inputs:
                    result["params"]["model"] = inputs["ckpt_name"]
                if "unetloader" in class_type_l and "unet_name" in inputs:
                    result["params"]["model"] = inputs["unet_name"]
                if "loraloader" in class_type_l and "lora_name" in inputs:
                    weight = inputs.get("strength_model")
                    if weight in (None, ""):
                        weight = inputs.get("strength_clip")
                    _remember_lora(inputs.get("lora_name"), weight)
                if "power lora loader" in class_type_l:
                    for key, value in inputs.items():
                        if not str(key).lower().startswith("lora_") or not isinstance(value, dict):
                            continue
                        if not value.get("on"):
                            continue
                        _remember_lora(value.get("lora"), value.get("strength"))
                if "modelsamplingsd3" in class_type_l and "model" in inputs and not result["params"].get("model"):
                    result["params"]["model"] = inputs["model"]

        if lora_resources:
            result["params"]["civitai_resources"] = lora_resources

        if not found and not result["params"]:
            return _fill_from_generic_json(prompt_data)
        return found or bool(result["params"])

    def _fill_from_a1111_parameters(raw_text: str) -> bool:
        text = (raw_text or "").strip()
        if not text:
            return False
        if text.startswith("{") or text.startswith("["):
            return False

        found = False
        parts = [p.strip() for p in re.split(r"\nNegative prompt:", text, maxsplit=1, flags=re.IGNORECASE)]
        if parts and parts[0]:
            if len(parts[0]) > len(result["prompt_text"]):
                result["prompt_text"] = parts[0]
            found = True

        lines_raw = [line.rstrip() for line in text.splitlines()]
        lines = [line.strip() for line in lines_raw if line.strip()]
        for idx, raw_line in enumerate(lines_raw):
            line = raw_line.strip()
            if not line.lower().startswith("negative prompt:"):
                continue
            neg = line.split(":", 1)[1].strip() if ":" in line else ""
            extra_lines = []
            for next_line in lines_raw[idx + 1:]:
                next_clean = next_line.strip()
                if not next_clean:
                    continue
                if re.search(r"\bsteps\s*:", next_clean, flags=re.IGNORECASE) and "," in next_clean:
                    break
                extra_lines.append(next_clean)
            if extra_lines:
                neg = "\n".join(([neg] if neg else []) + extra_lines)
            if neg and len(neg) > len(result["negative_prompt"]):
                result["negative_prompt"] = neg
                found = True
            break

        kv_line = ""
        if lines:
            last_line = lines[-1]
            if "," in last_line and ":" in last_line:
                kv_line = last_line
            if len(lines) >= 2 and lines[-2].lower().startswith("negative prompt:"):
                kv_line = lines[-1]
        if not kv_line and ":" in text and "," in text:
            kv_line = text.splitlines()[-1]

        if kv_line:
            civitai_match = re.search(r"Civitai resources\s*:\s*(\[[\s\S]*\])\s*$", kv_line, flags=re.IGNORECASE)
            if civitai_match:
                civitai_raw = civitai_match.group(1).strip()
                if civitai_raw:
                    try:
                        result["params"]["civitai_resources"] = json.loads(civitai_raw)
                    except Exception:
                        result["params"]["civitai_resources"] = civitai_raw
                    found = True
                kv_line = kv_line[:civitai_match.start()].rstrip().rstrip(",").strip()
            hashes_match = re.search(
                r"Hashes\s*:\s*(.+?)(?=,\s+(?:ENSD|Face restoration|Postprocess upscaler|Postprocess upscale by|Hires upscaler|Hires steps|Hires upscale|Hires checkpoint|Hires VAE|Hires CFG scale|Hires sampler|Hires scheduler|RNG|Version|Civitai resources)\s*:|$)",
                kv_line,
                flags=re.IGNORECASE,
            )
            if hashes_match:
                hashes_raw = hashes_match.group(1).strip()
                if hashes_raw:
                    result["params"]["hashes"] = hashes_raw
                    found = True
                    for hash_chunk in [c.strip() for c in hashes_raw.split(",") if c.strip() and ":" in c]:
                        hash_key, hash_val = [p.strip() for p in hash_chunk.split(":", 1)]
                        hash_key_l = hash_key.lower()
                        if hash_key_l == "model" and hash_val:
                            result["params"]["model_hash"] = hash_val
                        elif hash_key_l == "vae" and hash_val:
                            result["params"]["vae_hash"] = hash_val
                kv_line = (kv_line[:hashes_match.start()] + kv_line[hashes_match.end():]).strip().strip(",").strip()
            for chunk in [c.strip() for c in kv_line.split(",") if c.strip()]:
                if ":" not in chunk:
                    continue
                key, val = [p.strip() for p in chunk.split(":", 1)]
                key_l = key.lower()
                if key_l == "steps":
                    result["params"]["steps"] = val
                    found = True
                elif key_l in ("cfg scale", "cfg"):
                    result["params"]["cfg"] = val
                    found = True
                elif key_l == "sampler":
                    result["params"]["sampler"] = val
                    found = True
                elif key_l == "scheduler":
                    result["params"]["scheduler"] = val
                    found = True
                elif key_l == "seed":
                    result["params"]["seed"] = val
                    found = True
                elif key_l == "model":
                    result["params"]["model"] = val
                    found = True
                elif key_l == "model hash":
                    result["params"]["model_hash"] = val
                    found = True
                elif key_l == "clip skip":
                    result["params"]["clip_skip"] = val
                    found = True
                elif key_l == "denoising strength":
                    result["params"]["denoise"] = val
                    found = True
                elif key_l == "vae":
                    result["params"]["vae"] = val
                    found = True
                elif key_l == "vae hash":
                    result["params"]["vae_hash"] = val
                    found = True
                elif key_l == "hashes":
                    result["params"]["hashes"] = val
                    found = True
                elif key_l == "ensd":
                    result["params"]["ensd"] = val
                    found = True
                elif key_l == "version":
                    result["params"]["version"] = val
                    found = True
                elif key_l == "face restoration":
                    result["params"]["face_restoration"] = val
                    found = True
                elif key_l == "postprocess upscaler":
                    result["params"]["postprocess_upscaler"] = val
                    found = True
                elif key_l == "postprocess upscale by":
                    result["params"]["postprocess_upscale_by"] = val
                    found = True
                elif key_l == "hires upscale":
                    result["params"]["hires_upscale"] = val
                    found = True
                elif key_l == "hires upscaler":
                    result["params"]["hires_upscaler"] = val
                    found = True
                elif key_l == "hires steps":
                    result["params"]["hires_steps"] = val
                    found = True
                elif key_l == "hires checkpoint":
                    result["params"]["hires_checkpoint"] = val
                    found = True
                elif key_l == "hires vae":
                    result["params"]["hires_vae"] = val
                    found = True
                elif key_l == "hires cfg scale":
                    result["params"]["hires_cfg"] = val
                    found = True
                elif key_l == "hires sampler":
                    result["params"]["hires_sampler"] = val
                    found = True
                elif key_l == "hires scheduler":
                    result["params"]["hires_scheduler"] = val
                    found = True
                elif key_l == "rng":
                    result["params"]["rng"] = val
                    found = True
                elif key_l == "size" and "x" in val.lower():
                    dims = [d.strip() for d in re.split(r"[xX]", val, maxsplit=1)]
                    if len(dims) == 2:
                        result["params"]["width"] = dims[0]
                        result["params"]["height"] = dims[1]
                        found = True
        return found

    try:
        from PIL import Image
        from PIL.ExifTags import Base
    except Exception as e:
        result["error"] = f"Pillow import error: {e}"
        return result

    try:
        with Image.open(image_path) as img:
            info_items = getattr(img, "info", {}) or {}
            text_meta = {}
            for k, v in info_items.items():
                text_val = _to_text(v)
                if text_val:
                    text_meta[str(k)] = text_val

            parsed_any = False
            raw_prompt = text_meta.get("prompt")
            if raw_prompt:
                parsed_any = _fill_from_comfy_prompt(raw_prompt) or parsed_any

            for key in ("parameters", "comment", "Comment", "Description", "description", "UserComment"):
                raw_val = text_meta.get(key, "")
                if not raw_val:
                    continue
                parsed_any = _fill_from_comfy_prompt(raw_val) or parsed_any
                parsed_any = _fill_from_a1111_parameters(raw_val) or parsed_any
                if not (result["prompt_text"] and result["params"]):
                    parsed_any = _try_parse_json_candidates(raw_val) or parsed_any
                if result["prompt_text"] and result["params"]:
                    break

            if hasattr(img, "getexif"):
                try:
                    exif = img.getexif()
                except Exception:
                    exif = None
                if exif:
                    for tag_id in (0x9286, 0x010E):  # UserComment, ImageDescription
                        if tag_id not in exif:
                            continue
                        raw_exif = _to_text(exif.get(tag_id))
                        if not raw_exif:
                            continue
                        parsed_any = _fill_from_comfy_prompt(raw_exif) or parsed_any
                        parsed_any = _fill_from_a1111_parameters(raw_exif) or parsed_any
                        if not (result["prompt_text"] and result["params"]):
                            parsed_any = _try_parse_json_candidates(raw_exif) or parsed_any
                    try:
                        nested_exif = exif.get_ifd(Base.ExifOffset)
                    except Exception:
                        nested_exif = None
                    if isinstance(nested_exif, dict):
                        for tag_id in (0x9286, 0x010E):  # UserComment, ImageDescription
                            raw_nested = _to_text(nested_exif.get(tag_id))
                            if not raw_nested:
                                continue
                            parsed_any = _fill_from_comfy_prompt(raw_nested) or parsed_any
                            parsed_any = _fill_from_a1111_parameters(raw_nested) or parsed_any
                            if not (result["prompt_text"] and result["params"]):
                                parsed_any = _try_parse_json_candidates(raw_nested) or parsed_any

            if not (result["prompt_text"] and result["params"]):
                raw_bytes = b""
                try:
                    raw_bytes = Path(image_path).read_bytes()
                except Exception:
                    raw_bytes = b""
                if raw_bytes:
                    for enc in ("utf-8", "utf-16le", "utf-16be", "latin-1"):
                        try:
                            decoded = raw_bytes.decode(enc, errors="ignore")
                        except Exception:
                            continue
                        if not _looks_like_prompt_json(decoded):
                            continue
                        parsed_any = _try_parse_json_candidates(decoded) or parsed_any
                        if result["prompt_text"] and result["params"]:
                            break

            if not result["prompt_text"]:
                for key in ("prompt", "Prompt", "Description", "description"):
                    val = (text_meta.get(key) or "").strip()
                    if val and not val.startswith("{") and not val.startswith("["):
                        result["prompt_text"] = val
                        parsed_any = True
                        break

            if not result["prompt_text"] and not result["params"]:
                if parsed_any:
                    result["error"] = "Prompt text not found in metadata"
                else:
                    result["error"] = "No supported prompt metadata found"

    except Exception as e:
        result["error"] = f"Error reading image: {str(e)}"

    return result


def format_metadata_display(metadata: dict) -> str:
    if metadata.get("error"):
        return f"⚠️ {metadata['error']}"

    lines = []
    if metadata.get("prompt_text"):
        lines.append(f"📝 Prompt:\n{metadata['prompt_text']}\n")
    if metadata.get("negative_prompt"):
        lines.append(f"🚫 Negative prompt:\n{metadata['negative_prompt']}\n")

    params = metadata.get("params", {})
    if params:
        lines.append("⚙️ Settings:")
        if "seed" in params:
            lines.append(f"  Seed: {params['seed']}")
        if "steps" in params:
            lines.append(f"  Steps: {params['steps']}")
        if "cfg" in params:
            lines.append(f"  CFG: {params['cfg']}")
        if "sampler" in params:
            lines.append(f"  Sampler: {params['sampler']}")
        if "scheduler" in params:
            lines.append(f"  Scheduler: {params['scheduler']}")
        if "width" in params and "height" in params:
            lines.append(f"  Size: {params['width']}x{params['height']}")

    if not lines:
        return "No generation parameters found in metadata"
    return "\n".join(lines)

def build_civitai_parameters_text(metadata: dict) -> str:
    prompt_text = (metadata.get("prompt_text") or "").strip()
    negative_prompt = (metadata.get("negative_prompt") or "").strip()
    params = metadata.get("params", {}) or {}

    lines = []
    if prompt_text:
        lines.append(prompt_text)
    # Keep explicit negative-prompt line for maximum A1111/Civitai compatibility.
    lines.append(f"Negative prompt: {negative_prompt}")

    kv_parts = []

    def add_pair(label: str, value: Any):
        if value is None:
            return
        value_text = str(value).strip()
        if not value_text:
            return
        kv_parts.append(f"{label}: {value_text}")

    add_pair("Steps", params.get("steps"))
    add_pair("Sampler", params.get("sampler"))
    add_pair("Scheduler", params.get("scheduler"))
    add_pair("CFG scale", params.get("cfg"))
    add_pair("Seed", params.get("seed"))
    if params.get("width") and params.get("height"):
        kv_parts.append(f"Size: {params['width']}x{params['height']}")
    add_pair("Model", params.get("model"))
    add_pair("Model hash", params.get("model_hash"))
    add_pair("Clip skip", params.get("clip_skip"))
    add_pair("Denoising strength", params.get("denoise"))
    add_pair("VAE", params.get("vae"))
    add_pair("VAE hash", params.get("vae_hash"))
    add_pair("Hashes", params.get("hashes"))
    add_pair("ENSD", params.get("ensd"))
    add_pair("Face restoration", params.get("face_restoration"))
    add_pair("Postprocess upscaler", params.get("postprocess_upscaler"))
    add_pair("Postprocess upscale by", params.get("postprocess_upscale_by"))
    add_pair("Hires upscaler", params.get("hires_upscaler"))
    add_pair("Hires steps", params.get("hires_steps"))
    add_pair("Hires upscale", params.get("hires_upscale"))
    add_pair("Hires checkpoint", params.get("hires_checkpoint"))
    add_pair("Hires VAE", params.get("hires_vae"))
    add_pair("Hires CFG scale", params.get("hires_cfg"))
    add_pair("Hires sampler", params.get("hires_sampler"))
    add_pair("Hires scheduler", params.get("hires_scheduler"))
    add_pair("RNG", params.get("rng"))
    add_pair("Version", params.get("version"))
    civitai_resources = params.get("civitai_resources")
    if civitai_resources:
        if not isinstance(civitai_resources, str):
            civitai_resources = json.dumps(civitai_resources, ensure_ascii=False, separators=(",", ":"))
        add_pair("Civitai resources", civitai_resources)

    if kv_parts:
        lines.append(", ".join(kv_parts))
    return "\n".join(lines).strip()

def rewrite_png_metadata_for_civitai(image_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"success": False, "error": None, "metadata": None}
    if not image_path:
        out["error"] = "Please upload a PNG image first."
        return out

    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
    except Exception as e:
        out["error"] = f"Pillow import error: {e}"
        return out

    try:
        metadata = extract_png_metadata(image_path)
        prompt_text = (metadata.get("prompt_text") or "").strip()

        with Image.open(image_path) as img:
            if str(img.format).upper() != "PNG":
                out["error"] = "Only PNG images are supported."
                return out

            # Ensure canonical Civitai fields are present even if source metadata is partial.
            params = metadata.get("params", {}) or {}
            if not params.get("width") or not params.get("height"):
                params["width"], params["height"] = img.size
                metadata["params"] = params

            civitai_parameters = build_civitai_parameters_text(metadata)
            if not civitai_parameters:
                out["error"] = "Cannot build Civitai metadata from this image."
                return out

            source_info = dict(getattr(img, "info", {}) or {})
            pnginfo = PngInfo()

            def _to_text(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, bytes):
                    for enc in ("utf-8", "utf-16", "utf-16le", "latin-1"):
                        try:
                            return value.decode(enc, errors="ignore").strip("\x00")
                        except Exception:
                            continue
                    return ""
                return str(value)

            for key, value in source_info.items():
                key_text = str(key)
                if key_text.lower() in {"parameters", "comment", "description"}:
                    continue
                text_value = _to_text(value).strip()
                if not text_value:
                    continue
                pnginfo.add_text(key_text, text_value)

            # Keep A1111/Civitai-compatible primary field and add common fallbacks
            # used by metadata editors that parse generic comment/description tags.
            pnginfo.add_text("parameters", civitai_parameters)
            pnginfo.add_text("Comment", civitai_parameters)
            if prompt_text:
                pnginfo.add_text("Description", prompt_text)

            save_kwargs: Dict[str, Any] = {"pnginfo": pnginfo}
            for preserve_key in ("icc_profile", "exif", "dpi", "gamma", "transparency"):
                if preserve_key in source_info:
                    save_kwargs[preserve_key] = source_info[preserve_key]

            img.save(image_path, format="PNG", **save_kwargs)

        refreshed = extract_png_metadata(image_path)
        out["success"] = True
        out["metadata"] = refreshed
        return out
    except Exception as e:
        out["error"] = f"Failed to rewrite metadata: {e}"
        return out

def create_fixed_metadata_download_copy(image_path: str) -> Optional[str]:
    if not image_path:
        return None
    try:
        src = Path(image_path)
        if not src.exists():
            return None
        suffix = src.suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
        shutil.copy2(src, tmp_path)
        return str(tmp_path)
    except Exception:
        return None


def get_model_capabilities(model_name: str) -> set:
    capabilities_out = set()
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"model": model_name},
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        capabilities = data.get("capabilities")
        if capabilities is None:
            capabilities = data.get("details", {}).get("capabilities", [])
        if isinstance(capabilities, str):
            capabilities = [capabilities]
        if isinstance(capabilities, list):
            capabilities_lower = {str(c).strip().lower() for c in capabilities}
            if {"vision", "image", "multimodal"} & capabilities_lower:
                capabilities_out.add("vision")
            if {"completion", "generate", "chat", "text"} & capabilities_lower:
                capabilities_out.add("text")

        details = data.get("details", {})
        family_blob = " ".join([
            str(details.get("family", "")),
            str(details.get("families", "")),
            str(details.get("parent_model", "")),
        ]).lower()
        if any(k in family_blob for k in ("llava", "vision", "vl", "moondream", "minicpm", "qwen2.5vl", "qwen2vl")):
            capabilities_out.add("vision")
    except Exception:
        pass

    name_lower = model_name.lower()
    if any(
        k in name_lower
        for k in ("llava", "bakllava", "moondream", "minicpm-v", "qwen2.5vl", "qwen2vl", "vision", "-vl")
    ):
        capabilities_out.add("vision")
    if "embed" not in name_lower:
        capabilities_out.add("text")

    return capabilities_out


def supports_vision_model(model_name: str) -> bool:
    return "vision" in get_model_capabilities(model_name)


def get_vision_model_choices():
    models = list_models()
    vision_models = [model for model in models if "vision" in get_model_capabilities(model)]
    if not vision_models:
        return [VISION_MODEL_NOT_AVAILABLE]
    return vision_models


def ollama_describe_image_stream(model: str, image_path: str, prompt: str, detail_mode: str = "Fast"):
    from PIL import Image

    mode = (detail_mode or "Fast").strip().lower()
    is_ultra = mode.startswith("ultra")
    is_detailed = mode.startswith("detailed")
    mode_label = "Ultra Detail" if is_ultra else ("Detailed" if is_detailed else "Fast")

    prompt = " ".join((prompt or "").split())
    prompt_limit = 700 if is_ultra else (420 if is_detailed else 220)
    if len(prompt) > prompt_limit:
        prompt = prompt[:prompt_limit]

    # Keep image long side at 1024px before sending to Ollama.
    # Retry by adjusting generation budget only, inside requested ranges.
    if is_ultra:
        attempts = [
            {"max_side": 1024, "num_predict": 420, "num_ctx": 6144, "timeout": 180, "prompt": prompt},
            {
                "max_side": 896,
                "num_predict": 360,
                "num_ctx": 5120,
                "timeout": 210,
                "prompt": "Describe only visible content in this image in up to 12 concise sentences."
            },
            {
                "max_side": 768,
                "num_predict": 300,
                "num_ctx": 4096,
                "timeout": 240,
                "prompt": "Describe only visible content in this image in up to 12 concise sentences."
            },
            {
                "max_side": 512,
                "num_predict": 240,
                "num_ctx": 3072,
                "timeout": 270,
                "prompt": "Describe only visible content in this image in up to 12 concise sentences."
            },
        ]
    elif is_detailed:
        attempts = [
            {"max_side": 1024, "num_predict": 300, "num_ctx": 4096, "timeout": 120, "prompt": prompt},
            {
                "max_side": 896,
                "num_predict": 260,
                "num_ctx": 3584,
                "timeout": 140,
                "prompt": "Describe only visible content in this image in up to 8 concise sentences."
            },
            {
                "max_side": 768,
                "num_predict": 220,
                "num_ctx": 3072,
                "timeout": 160,
                "prompt": "Describe only visible content in this image in up to 8 concise sentences."
            },
            {
                "max_side": 512,
                "num_predict": 180,
                "num_ctx": 2040,
                "timeout": 180,
                "prompt": "Describe only visible content in this image in up to 8 concise sentences."
            },
        ]
    else:
        attempts = [
            {"max_side": 1024, "num_predict": 180, "num_ctx": 3072, "timeout": 45, "prompt": prompt},
            {
                "max_side": 896,
                "num_predict": 170,
                "num_ctx": 2560,
                "timeout": 55,
                "prompt": "Describe only visible content in this image in up to 4 concise sentences."
            },
            {
                "max_side": 768,
                "num_predict": 160,
                "num_ctx": 2304,
                "timeout": 65,
                "prompt": "Describe only visible content in this image in up to 4 concise sentences."
            },
            {
                "max_side": 512,
                "num_predict": 150,
                "num_ctx": 2040,
                "timeout": 80,
                "prompt": "Describe only visible content in this image in up to 4 concise sentences."
            },
        ]

    for idx, attempt in enumerate(attempts):
        yield None, f"Connecting to Ollama ({mode_label})... attempt {idx + 1}/{len(attempts)}"

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            if max(width, height) > attempt["max_side"]:
                scale = attempt["max_side"] / float(max(width, height))
                img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        request_json = {
            "model": model,
            "prompt": attempt["prompt"],
            "images": [image_b64],
            "stream": False,
            "options": {
                "num_predict": attempt["num_predict"],
                "temperature": 0.2,
                "num_ctx": attempt["num_ctx"]
            }
        }
        response = None
        chunks = []
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=request_json,
                stream=True,
                timeout=(15, attempt["timeout"])
            )
            response.raise_for_status()
            last_emit = 0.0
            yield None, "Model loaded. Generating..."

            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue

                if obj.get("error"):
                    raise Exception(str(obj.get("error")))

                piece = obj.get("response") or ""
                if piece:
                    chunks.append(piece)
                    now = time.time()
                    if now - last_emit >= 0.3:
                        partial = "".join(chunks).strip()
                        yield partial, f"Generating... {len(partial)} chars"
                        last_emit = now

                if obj.get("done"):
                    break

            final_text = "".join(chunks).strip()
            if not final_text:
                raise Exception(f"Empty response from model `{model}`.")
            yield final_text, f"Description generated via `{model}`."
            return
        except requests.ReadTimeout:
            if chunks:
                partial = "".join(chunks).strip()
                if partial:
                    yield partial, "Timed out, returning partial description."
                    return
            if idx < len(attempts) - 1:
                yield None, "No response yet. Retrying with lighter settings..."
                continue
            raise Exception(
                "Ollama stream timed out after retries. "
                "The selected vision model is too slow or stuck on this image."
            )
        except requests.ConnectionError as e:
            if idx < len(attempts) - 1:
                yield None, f"Connection issue: {str(e)}. Retrying..."
                continue
            raise Exception(f"Ollama connection failed: {str(e)}")
        except requests.Timeout:
            if idx < len(attempts) - 1:
                yield None, "Request timed out. Retrying with lighter settings..."
                continue
            raise Exception("Ollama request timed out.")
        except requests.HTTPError as e:
            detail = ""
            if e.response is not None:
                try:
                    detail = (e.response.json().get("error") or "").strip()
                except Exception:
                    detail = (e.response.text or "").strip()
            status = e.response.status_code if e.response is not None else "unknown"
            detail_lower = detail.lower()
            runner_crash = (
                "model runner has unexpectedly stopped" in detail_lower
                or "forcibly closed" in detail_lower
                or "assertion failed" in detail_lower
            )
            if runner_crash and idx < len(attempts) - 1:
                yield None, "Vision runner crashed on current settings. Retrying with safer image size..."
                continue
            if detail:
                raise Exception(f"Ollama API {status}: {detail}")
            raise Exception(f"Ollama API {status}: {str(e)}")
        except requests.RequestException as e:
            if idx < len(attempts) - 1:
                yield None, f"Request error: {str(e)}. Retrying..."
                continue
            raise Exception(f"Ollama request failed: {str(e)}")
        finally:
            try:
                response.close()
            except Exception:
                pass

    raise Exception(
        "Ollama request timed out after retries. "
        f"The selected vision model is too slow or stuck on this image (mode: {mode_label})."
    )

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

def ollama_generate_sync(model: str, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
    try:
        request_json = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            request_json["options"] = options
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=request_json,
            timeout=300
        )
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
.fix-meta-btn button {
  min-height: 28px !important;
  height: 28px !important;
  padding: 0 10px !important;
  font-size: 0.8rem !important;
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

def parse_parameters(content: str):
    """Return list of (key, value) pairs for PARAMETER lines at tail of template."""
    params = []
    if not content:
        return params
    for line in content.splitlines():
        match = re.match(r'^\s*PARAMETER\s+(\S+)\s+(.+)$', line, flags=re.IGNORECASE)
        if not match:
            continue
        key = match.group(1).strip()
        val_raw = match.group(2).strip()
        # Try numeric conversion; keep string on failure
        try:
            if re.match(r'^[+-]?\d+$', val_raw):
                val = int(val_raw)
            else:
                val = float(val_raw.replace(",", "."))
        except Exception:
            val = val_raw
        params.append((key, val))
    return params

def build_generated_model_name(source_model: str, role: str) -> str:
    source_base = source_model.split("/")[-1].split(":", 1)[0]
    safe_source_base = re.sub(r"[^a-zA-Z0-9._-]+", "-", source_base).strip("-").lower()
    safe_role = re.sub(r"[^a-zA-Z0-9._-]+", "-", (role or "").strip()).strip("-").lower()
    if not safe_source_base:
        safe_source_base = "model"
    if not safe_role:
        safe_role = "role"
    return f"{safe_source_base}-{safe_role}"

def parse_role_config(role_key):
    """
    Returns (source_model, system_prompt, params_list).
    - source_model: string from FROM <model> or default qwen2.5:latest
    - system_prompt: text after SYSTEM or whole content
    - params_list: list of (key, value) pairs from PARAMETER lines
    """
    if not role_key or role_key == UNDEFINED_ROLE:
        return None, None, []
        
    content = template_manager.get(role_key)
    if not content:
        # Fallback to defaults if key exists in defaults
        content = DEFAULT_TEMPLATES.get(role_key, "")
    
    model_match = re.search(r'^\s*FROM\s+([^\s]+)', content, re.MULTILINE | re.IGNORECASE)
    source_model = model_match.group(1).strip() if model_match else None
    effective_model = source_model or "qwen2.5:latest"
    
    system_match = re.search(r'SYSTEM\s+"""(.*?)"""', content, re.DOTALL | re.IGNORECASE)
    if not system_match:
        system_match = re.search(r'SYSTEM\s+(.*)', content, re.IGNORECASE)
    system = system_match.group(1).strip() if system_match else content.strip()
    
    params = parse_parameters(content)
    
    return effective_model, system, params

def params_to_options(params, fallback_temp: Optional[float] = None) -> Dict[str, Any]:
    opts = {}
    for key, val in params:
        opts[key] = val
    lower_keys = {k.lower() for k in opts.keys()}
    if "temperature" not in lower_keys and fallback_temp is not None:
        opts["temperature"] = fallback_temp
    return opts

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

        # Stage 0: Translator (optional)
        t_model, t_sys, t_params = parse_role_config(trans_role)
        if t_model:
            progress(0.2, desc=f"Translating via {trans_role} ({t_model})...")
            translator_prompt = f"{t_sys}\n\nUSER TEXT:\n{text}" if t_sys else text
            t_options = params_to_options(t_params, fallback_temp=0.05)
            input_en = ollama_generate_sync(t_model, translator_prompt, t_options)
        
        # Stage 1: Extractor (optional)
        e_model, e_sys, e_params = parse_role_config(extr_role)
        if e_model:
            progress(0.4, desc=f"Extracting facts via {extr_role} ({e_model})...")
            extractor_prompt = f"{e_sys}\n\nINPUT:\n{input_en}" if e_sys else input_en
            e_options = params_to_options(e_params, fallback_temp=0.1)
            facts = normalize_facts(ollama_generate_sync(e_model, extractor_prompt, e_options))
        else:
            facts = normalize_facts(input_en)
        
        # Stage 2: Structurer (optional)
        s_model, s_sys, s_params = parse_role_config(stru_role)
        if s_model:
            progress(0.7, desc=f"Structuring via {stru_role} ({s_model})...")
            stru_p = f"{s_sys}\n\nDESCRIPTION:\n{input_en}\n\nFACTS:\n{facts}\n\nOUTPUT:" if s_sys else f"{facts}\n\nOUTPUT:"
            s_options = params_to_options(s_params, fallback_temp=0.15)
            structured = ollama_generate_sync(s_model, stru_p, s_options)
        else:
            structured = facts
        
        # Stage 3: Validator (optional)
        v_model, v_sys, v_params = parse_role_config(vali_role)
        if v_model:
            progress(0.9, desc=f"Validating via {vali_role} ({v_model})...")
            vali_p = f"{v_sys}\n\nFACTS:\n{facts}\n\nCANDIDATE:\n{structured}\n\nFIXED:" if v_sys else f"FACTS:\n{facts}\n\nCANDIDATE:\n{structured}\n\nFIXED:"
            v_options = params_to_options(v_params, fallback_temp=0.1)
            structured = ollama_generate_sync(v_model, vali_p, v_options)
            
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
            capabilities = get_model_capabilities(m["name"])
            capability_bits = []
            if "text" in capabilities:
                capability_bits.append("📝 text")
            if "vision" in capabilities:
                capability_bits.append("🖼️ vision")
            capability_suffix = f" | {' | '.join(capability_bits)}" if capability_bits else ""
            models_html += f"""
            <div class="model-item">
                <div class="model-name">{safe_name}</div>
                <div class="model-meta">ID: {safe_id} | Size: {safe_size}{capability_suffix}</div>
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
    vision_model_choices = get_vision_model_choices()
    return (
        get_models_html(),
        gr.update(value=""),
        gr.update(value="", visible=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False),
        gr.update(choices=vision_model_choices, value=vision_model_choices[0]),
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
vlm_preset_choices = vlm_preset_manager.keys()
default_vlm_preset = settings_manager.get("vlm_preset", "Fast")
if default_vlm_preset not in vlm_preset_choices:
    default_vlm_preset = vlm_preset_choices[0] if vlm_preset_choices else "Fast"

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
                    with gr.Row():
                        copy_description_btn = gr.Button("Copy description")
                        clear_description_btn = gr.Button("Clear")
                    mode_sel = gr.Dropdown(choices=pipeline_preset_manager.keys(), label="Pipeline Preset", value="Default")
                    compile_btn = gr.Button("🚀 Compile prompt", variant="primary")
                    with gr.Accordion("Get PNG info", open=False):
                        gr.Markdown("*Upload PNG to extract prompt metadata*")
                        png_meta_image = gr.Image(label="PNG image", type="filepath", height=220)
                        png_meta_output = gr.Textbox(
                            label="PNG info",
                            lines=8,
                            interactive=False
                        )
                        fixed_meta_file = gr.File(
                            label="Fixed PNG download",
                            interactive=False,
                            visible=False
                        )
                        with gr.Row():
                            fix_meta_btn = gr.Button(
                                "Fix meta",
                                variant="secondary",
                                min_width=90,
                                elem_classes=["fix-meta-btn"]
                            )
                    with gr.Accordion("Create description from image", open=False):
                        gr.Markdown("*Upload image and click Get description*")
                        desc_image = gr.Image(label="Image", type="filepath", height=220)
                        vision_model_choices = get_vision_model_choices()
                        desc_model_sel = gr.Dropdown(
                            label="Vision model",
                            choices=vision_model_choices,
                            value=vision_model_choices[0],
                            allow_custom_value=False
                        )
                        desc_detail_mode = gr.Dropdown(
                            label="Description mode",
                            choices=VLM_PRESET_CHOICES,
                            value=default_vlm_preset
                        )
                        desc_btn = gr.Button("Get description", variant="primary")
                        desc_status = gr.Markdown("")
                
                with gr.Column(scale=1):
                    final_out = gr.TextArea(label="Final Structured Prompt", lines=15)
                    with gr.Row():
                        copy_btn = gr.Button("📋 Copy Result")
                        clear_btn = gr.Button("🗑️ Clear")
                    


        with gr.Tab("⚙️ Pipeline Presets"):
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

        with gr.Tab("⚙️ Pipeline Roles"):
            gr.Markdown("### Define Pipeline Roles")
            
            with gr.Row() as preset_select_row:
                role_preset = gr.Dropdown(choices=template_manager.keys(), label="Role Preset", value="translator", allow_custom_value=True, scale=3)
                with gr.Column(scale=1):
                    add_preset_btn = gr.Button("➕ Add New Role")
                    delete_preset_btn = gr.Button("🗑️ Delete Role", variant="stop")
            
            with gr.Row(visible=False) as delete_confirm_row:
                gr.Markdown("Are you sure you want to delete this role?")
                confirm_delete_btn = gr.Button("✅ Yes, Delete", variant="stop")
                cancel_delete_btn = gr.Button("❌ Cancel")

            with gr.Row(visible=False) as new_preset_row:
                new_preset_name = gr.Textbox(label="New Role Name", placeholder="e.g., reviewer", scale=3)
                with gr.Column(scale=1):
                    save_new_preset_btn = gr.Button("✅ Confirm Add")
                    cancel_add_btn = gr.Button("❌ Cancel")
                
            template_content = gr.TextArea(label="System Template", lines=10, value=template_manager.get("translator"))
            save_template_btn = gr.Button("Save role preset", variant="primary", scale=1, min_width=90)

        with gr.Tab("👁️ VLM Presets"):
            gr.Markdown("### Define Ollama instructions for image description")
            vlm_preset_sel = gr.Dropdown(
                choices=VLM_PRESET_CHOICES,
                label="VLM Preset",
                value=default_vlm_preset,
                allow_custom_value=False
            )
            vlm_instruction = gr.TextArea(
                label="Instruction",
                lines=16,
                value=vlm_preset_manager.get(default_vlm_preset)
            )
            save_vlm_preset_btn = gr.Button("💾 Save VLM Preset", variant="primary")

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

    def on_png_info_image_change(image_path):
        if not image_path:
            return "", gr.update(), gr.update(value=None, visible=False)
        metadata = extract_png_metadata(image_path)
        display = format_metadata_display(metadata)
        prompt_text = (metadata.get("prompt_text") or "").strip()
        if prompt_text:
            return display, prompt_text, gr.update(value=None, visible=False)
        return display, gr.update(), gr.update(value=None, visible=False)

    def on_fix_meta_click(image_path):
        result = rewrite_png_metadata_for_civitai(image_path)
        if not result.get("success"):
            err_msg = result.get('error') or 'Unable to update metadata'
            gr.Warning(f"Fix meta failed: {err_msg}")
            return f"⚠️ {err_msg}", gr.update(), gr.update(), gr.update(value=None, visible=False)

        metadata = result.get("metadata") or extract_png_metadata(image_path)
        display = format_metadata_display(metadata)
        download_copy = create_fixed_metadata_download_copy(image_path)
        gr.Info("Fix meta completed successfully.")
        out_text = f"✅ Civitai metadata updated.\n\n{display}"
        prompt_text = (metadata.get("prompt_text") or "").strip()
        file_update = gr.update(value=download_copy, visible=bool(download_copy))
        if prompt_text:
            return out_text, prompt_text, gr.update(value=image_path), file_update
        return out_text, gr.update(), gr.update(value=image_path), file_update

    png_meta_image.change(
        fn=on_png_info_image_change,
        inputs=[png_meta_image],
        outputs=[png_meta_output, input_text, fixed_meta_file]
    )
    fix_meta_btn.click(
        fn=on_fix_meta_click,
        inputs=[png_meta_image],
        outputs=[png_meta_output, input_text, png_meta_image, fixed_meta_file]
    )

    def on_get_description_click(image_path, current_text, selected_model, detail_mode):
        if not image_path:
            yield gr.update(), "Please upload an image first."
            return

        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
            if width < 28 or height < 28:
                yield gr.update(), f"Image is too small ({width}x{height}). Use at least 28x28."
                return
        except Exception:
            pass

        model = (selected_model or "").strip()
        if not model or model == VISION_MODEL_NOT_AVAILABLE:
            yield gr.update(), "Vision model not available in Ollama."
            return

        user_context = (current_text or "").strip()
        mode = (detail_mode or "Fast").strip().lower()
        is_ultra = mode.startswith("ultra")
        is_detailed = mode.startswith("detailed")
        context_limit = 700 if is_ultra else (420 if is_detailed else 220)
        sentence_limit = 12 if is_ultra else (8 if is_detailed else 4)
        preset_key = detail_mode if detail_mode in VLM_PRESET_CHOICES else "Fast"
        prompt = (vlm_preset_manager.get(preset_key) or "").strip()
        if not prompt:
            prompt = DEFAULT_VLM_PRESETS.get(preset_key, DEFAULT_VLM_PRESETS["Fast"])
        if user_context:
            if len(user_context) > context_limit:
                user_context = user_context[:context_limit]
            prompt = f"{prompt}\n\nOptional user context: {user_context}"
        try:
            for partial_text, status in ollama_describe_image_stream(model, image_path, prompt, detail_mode):
                if partial_text is None:
                    yield gr.update(), status
                else:
                    yield partial_text, status
        except Exception as e:
            err = str(e)
            if "model runner has unexpectedly stopped" in err.lower():
                alternatives = [
                    m for m in get_vision_model_choices()
                    if m != VISION_MODEL_NOT_AVAILABLE and m != model
                ]
                suggestion = f" Try `{alternatives[0]}`." if alternatives else ""
                yield gr.update(), (
                    f"Failed: `{err}` Ollama vision runner crashed for `{model}`.{suggestion} "
                    "You may also need to restart Ollama."
                )
                return
            yield gr.update(), f"Failed: `{err}`"
            return

    desc_btn.click(
        fn=on_get_description_click,
        inputs=[desc_image, input_text, desc_model_sel, desc_detail_mode],
        outputs=[input_text, desc_status]
    )

    def on_vlm_preset_change(name):
        preset = name if name in VLM_PRESET_CHOICES else "Fast"
        settings_manager.set("vlm_preset", preset)
        return vlm_preset_manager.get(preset)

    vlm_preset_sel.change(
        on_vlm_preset_change,
        vlm_preset_sel,
        [vlm_instruction]
    )

    def on_desc_mode_change(name):
        preset = name if name in VLM_PRESET_CHOICES else "Fast"
        settings_manager.set("vlm_preset", preset)
        return gr.update(value=preset), vlm_preset_manager.get(preset)

    desc_detail_mode.change(
        on_desc_mode_change,
        desc_detail_mode,
        [vlm_preset_sel, vlm_instruction]
    )

    def save_vlm_preset(name, content):
        preset = name if name in VLM_PRESET_CHOICES else "Fast"
        vlm_preset_manager.set(preset, (content or "").strip())
        settings_manager.set("vlm_preset", preset)
        gr.Info(f"VLM Preset '{preset}' saved!")

    save_vlm_preset_btn.click(save_vlm_preset, [vlm_preset_sel, vlm_instruction], None)
    
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
        [models_html, model_action_status, model_info_output, model_delete_pending, model_delete_confirm_text, model_delete_confirm_row, desc_model_sel]
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
        [models_html, model_action_status, model_info_output, model_delete_pending, model_delete_confirm_text, model_delete_confirm_row, desc_model_sel]
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

    save_template_btn.click(save_template, [role_preset, template_content], None)

    
    def notify_copied(_):
        gr.Info("Copied to clipboard")
        return None

    copy_btn.click(
        fn=notify_copied,
        inputs=final_out,
        outputs=None,
        js="(x) => { navigator.clipboard.writeText(x); }"
    )
    def notify_description_copied(_):
        gr.Info("Description copied to clipboard")
        return None

    copy_description_btn.click(
        fn=notify_description_copied,
        inputs=input_text,
        outputs=None,
        js="(x) => { navigator.clipboard.writeText(x || ''); }"
    )
    clear_description_btn.click(lambda: "", None, [input_text])
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


