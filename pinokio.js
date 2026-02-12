module.exports = {
  title: "Ollama Zimage prompt helper",
  description: "A web interface for managing and interacting with Ollama models",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("venv")
    let running = {
      install: info.running("install.json"),
      start: info.running("start.js"),
      update: info.running("update.json"),
      reset: info.running("reset.json")
    }

    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-spin fa-circle-notch",
        text: "Installing",
        href: "install.json",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        } else {
          return [{
            default: true,
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: "fa-solid fa-spin fa-circle-notch",
          text: "Updating",
          href: "update.json",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: "fa-solid fa-spin fa-circle-notch",
          text: "Resetting",
          href: "reset.json",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
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
          confirm: "Are you sure you wish to reset the app?"
        }]
      }
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
