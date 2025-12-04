# Ollama Web Interface - AI Agent Instructions

## Architecture Overview

This is a Flask-based web UI for managing Ollama (local LLM runtime). The app acts as a bridge between a browser client and the Ollama CLI/API running on `localhost:11434`.

**Key Components:**
- `app.py` - Flask backend with CLI subprocess management and SSE streaming
- `templates/index.html` - Single-page dark-themed UI with vanilla JavaScript
- Pinokio integration files (`*.json`) - Deployment scripts for the Pinokio platform

**Data Flow:**
1. Browser → Flask endpoints → Ollama CLI subprocess or HTTP API → Response back to browser
2. Long-running operations (downloads, chat) use background threads with shared global state
3. Frontend polls `/api/pull/all` every second for real-time download progress across all users

## Critical Patterns

### Background Process Management
Downloads run in daemon threads with process objects stored in `pull_processes` dict for cancellation. Always use:
```python
process = subprocess.Popen(
    f"ollama pull {model_name}",
    shell=True,
    encoding='utf-8',
    errors='replace',  # CRITICAL: Windows Unicode handling
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
)
pull_processes[model_name] = process  # Store for cancellation
```

### Streaming Chat (SSE)
Chat uses Server-Sent Events with Ollama's `/api/generate` streaming endpoint:
- Frontend reads stream with ReadableStream decoder
- Backend yields `data: {json}\n\n` formatted chunks
- `active_generation["stop"]` flag enables mid-stream cancellation

### Progress Parsing
Ollama CLI outputs progress like: `pulling d2a932bcf54a:   9% ▕████▏  70 MB/806 MB  3.6 MB/s`
- Extract sizes with regex: `r'([\d.]+\s*[KMG]B)/([\d.]+\s*[KMG]B)'`
- Calculate percentage by converting to common unit (MB)
- Always use `errors='replace'` in subprocess to handle Unicode block characters on Windows

### Shared State for Multi-User Downloads
`pull_progress` dict is global - all users see same download state. Items auto-delete after completion:
- Complete: 5 seconds
- Cancelled: 3 seconds  
- Error: 5 seconds + notification

## Development Commands

**Start dev server:**
```bash
python app.py  # Runs on 0.0.0.0:11435
```

**Via Pinokio:** 
- Install: Runs `install.json` (creates venv, installs deps)
- Start: Runs `start.json` (launches app, opens browser to localhost:11435)

**No test suite exists** - test manually in browser

## UI Conventions

- **Dark theme** with glassmorphism: `rgba(30, 30, 46, 0.9)` backgrounds, `#818cf8` purple accents
- **Mobile-first responsive**: Uses `clamp()` for text, `min(100%, 350px)` for grid columns
- **No frameworks** - vanilla JS with inline event handlers (`onclick="funcName()"`)
- **Monospace for CLI output** - Shows raw Ollama text in progress displays
- **Modal overlays** - Created dynamically with inline styles for model info display

## Integration Points

- **Ollama CLI**: All operations except chat use `subprocess.run("ollama <cmd>", shell=True)`
- **Ollama HTTP API**: Chat uses POST to `http://localhost:11434/api/generate` with `stream: true`
- **No authentication** - Designed for localhost-only use
- **Port 11435** hardcoded - Ollama runs on 11434, this app on 11435

## File-Specific Notes

**app.py:**
- `run_ollama_command()` - Wrapper for CLI with 30s timeout
- `pull_model_with_progress()` - Long-running thread function, reads CLI line-by-line
- Always clean up `pull_processes[model]` after completion to prevent memory leaks

**templates/index.html:**
- Single 1100+ line file (no build step)
- Functions follow pattern: `functionName()` without jQuery/React
- Uses `loadAllDownloads()` polling instead of WebSockets for simplicity
- CSS in `<style>` tag with mobile breakpoints at 640px

## Common Pitfalls

1. **Windows encoding** - Always set `encoding='utf-8', errors='replace'` in Popen or get `'charmap' codec` errors
2. **Process cleanup** - Delete from `pull_processes` dict after termination or processes accumulate
3. **Case-sensitive model names** - Ollama is case-sensitive; UI enforces lowercase comparisons for rename validation
4. **Daemon threads** - Use `daemon=True` for download threads so they don't block app shutdown
5. **Progress calculation** - Convert KB→MB→GB before percentage calc or get wrong values

## Extensions Points

To add new Ollama operations:
1. Add Flask route in `app.py` with `@app.route('/api/<operation>')`
2. Call `run_ollama_command("<cmd>")` or use subprocess directly
3. Add JavaScript function in `index.html` 
4. Add UI button/trigger calling the function

Model action buttons use this pattern:
```html
<button onclick="operationName('${modelName}')">Label</button>
```
