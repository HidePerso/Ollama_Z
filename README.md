# Ollama Web Interface

A modern web interface for managing and interacting with Ollama models on localhost.

## Features

- 🔄 **Monitor Running Models** - View all running models with resource usage and auto-unload timers
- ⬇️ **Pull Models** - Download models with real-time progress bars
- 📦 **Manage Models** - List all available models with size and modification info
- 💬 **Chat Interface** - Test models with a simple chat interface
- 🛠️ **Model Operations**:
  - Copy models
  - Rename models (copy + remove)
  - Remove models
  - Show detailed model information
  - Stop running models

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

- `GET /api/ps` - List running models
- `GET /api/list` - List all models
- `POST /api/pull` - Pull a model
- `GET /api/pull/progress/<model>` - Get pull progress
- `POST /api/stop` - Stop a running model
- `POST /api/rm` - Remove a model
- `POST /api/cp` - Copy a model
- `POST /api/rename` - Rename a model
- `POST /api/show` - Show model info
- `POST /api/chat` - Chat with a model

## Configuration

The server runs on `0.0.0.0:11435` by default and connects to Ollama at `localhost:11434`.

To change these settings, edit `app.py`:

```python
OLLAMA_URL = "http://localhost:11434"
app.run(host='0.0.0.0', port=11435)
```

## License

MIT

## Credits

Built for use with [Ollama](https://ollama.ai) and [Pinokio](https://pinokio.computer)
