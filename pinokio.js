module.exports = {
  title: "Ollama Web Interface",
  description: "A web interface for managing and interacting with Ollama models",
  icon: "icon.png",
  menu: async (kernel) => {
    let installed = await kernel.exists(__dirname, "venv")
    let running = kernel.running(__dirname, "start.json")
    
    if (running) {
      return [{
        default: true,
        icon: "fa-solid fa-spin fa-circle-notch",
        text: "Running",
        href: "start.json",
      }, {
        icon: "fa-solid fa-power-off",
        text: "Stop",
        href: "stop.json",
      }, {
        icon: "fa-solid fa-plug",
        text: "Update",
        href: "update.json",
      }, {
        icon: "fa-regular fa-circle-xmark",
        text: "Reset",
        href: "reset.json",
      }]
    } else if (installed) {
      return [{
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.json",
      }, {
        icon: "fa-solid fa-plug",
        text: "Update",
        href: "update.json",
      }, {
        icon: "fa-solid fa-download",
        text: "Install",
        href: "install.json",
      }, {
        icon: "fa-regular fa-circle-xmark",
        text: "Reset",
        href: "reset.json",
      }]
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-download",
        text: "Install",
        href: "install.json",
      }]
    }
  }
}
