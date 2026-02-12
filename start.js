module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "venv",
        env: {
          "PYTHONUNBUFFERED": "1"
        },
        message: "{{platform==='win32' ? '.\\\\venv\\\\Scripts\\\\python.exe' : './venv/bin/python'}} app.py",
        on: [{
          "event": "/http:\\/\\/[a-zA-Z0-9.:]+/",
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    }
  ]
}
