# Ollama Web Interface

A modern web interface for managing and interacting with Ollama models on localhost.

## Features

### 🎨 Modern Dark Theme UI
- Sleek glassmorphism design with gradient accents
- Fully responsive - optimized for desktop, tablet, and mobile
- Touch-friendly controls with proper hit targets

### 🔄 Model Management
- **Monitor Running Models** - View all running models with resource usage
- **Start/Kill Ollama** - Control the Ollama service directly from the UI
- **One-Click Run** - Load any model directly into chat interface

### ⬇️ Smart Downloads
- **Concurrent Downloads** - Download multiple models simultaneously
- **Real-Time Progress** - Live progress bars with speed and size info (e.g., "943 MB/8.6 GB | 3.2 MB/s")
- **Cancel Anytime** - Stop downloads mid-process
- **Persistent Progress** - Downloads continue in background, visible to all users
- **Auto-Cleanup** - Completed/failed downloads automatically removed

### 💬 Streaming Chat Interface
- **Real-Time Streaming** - See responses token-by-token as they generate
- **Stop Generation** - Cancel responses mid-generation with pulsing stop button
- **Model Switching** - Quickly switch between installed models
- **Clean History** - Smooth chat bubbles with user/assistant styling

### 🛠️ Per-Model Actions
- **Run** - Load model into chat interface
- **Info** - View detailed model information in modal
- **Copy** - Duplicate models with new names
- **Remove** - Delete models with confirmation

## Installation

### Via Pinokio (Recommended)

1. Clone or place this repository in your Pinokio API directory: `C:\pinokio\api\Ollama_Web.git\`
2. Open Pinokio and find "Ollama Web Interface" in your apps
3. Click "Install" to set up the environment
4. Click "Start" to launch the web interface

### Manual Installation

1. Ensure Python 3.8+ is installed
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Start the Server

**Via Pinokio:** Click the "Start" button in Pinokio

**Manually:**
```bash
python app.py
```

The web interface will be available at: `http://localhost:11435`

### Prerequisites

- Ollama must be installed and running on `localhost:11434`
- Download Ollama from: https://ollama.ai

### Available Ollama Commands

The interface supports all major Ollama operations:

- **serve** - Start ollama (if not already running)
- **pull** - Download models from registry with progress tracking
- **list** - View all installed models
- **ps** - Monitor running models with resource info
- **run** - Chat with models through the interface
- **stop** - Stop running models
- **cp** - Copy models
- **rm** - Remove models
- **show** - Display detailed model information

## Project Structure

```
Ollama_Web.git/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Web interface
├── requirements.txt    # Python dependencies
├── pinokio.js         # Pinokio configuration
├── install.json       # Pinokio install script
├── start.json         # Pinokio start script
├── stop.json          # Pinokio stop script
├── update.json        # Pinokio update script
└── reset.json         # Pinokio reset script
```

## API Endpoints

### Model Management
- `GET /api/ps` - List running models with resource info
- `GET /api/list` - List all installed models
- `POST /api/stop` - Stop a specific running model
- `POST /api/rm` - Remove a model
- `POST /api/cp` - Copy a model to new name
- `POST /api/show` - Get detailed model information

### Download Operations
- `POST /api/pull` - Start downloading a model
- `GET /api/pull/all` - Get all active download progress (real-time)
- `GET /api/pull/progress/<model>` - Get specific model download progress
- `POST /api/pull/cancel` - Cancel an active download

### Chat Operations
- `POST /api/chat` - Stream chat responses (Server-Sent Events)
- `POST /api/chat/stop` - Stop active generation

### Ollama Control
- `POST /api/start` - Start Ollama service
- `POST /api/kill` - Kill all Ollama processes
- `POST /api/serve` - Start Ollama serve (deprecated, use /api/start)

## Technical Details

### Architecture
- **Backend**: Flask with threading for concurrent operations
- **Frontend**: Vanilla JavaScript (no build step required)
- **Streaming**: Server-Sent Events (SSE) for chat responses
- **Progress Tracking**: Global shared state with auto-cleanup
- **Process Management**: Direct subprocess control for Ollama CLI

### Configuration

The server runs on `0.0.0.0:11435` by default and connects to Ollama at `localhost:11434`.

To change these settings, edit `app.py`:

```python
OLLAMA_URL = "http://localhost:11434"
app.run(host='0.0.0.0', port=11435)
```

### Browser Compatibility
- Modern browsers with ES6+ support
- ReadableStream API for streaming responses
- CSS Grid and Flexbox for responsive layout

## License

MIT

## Credits

Built for use with [Ollama](https://ollama.ai) and [Pinokio](https://pinokio.computer)
