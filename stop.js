module.exports = {
    run: [
        {
            method: "script.stop",
            params: {
                uri: "start.js"
            }
        },
        {
            method: "web.open",
            params: {
                uri: "{{cwd}}",
                target: "_top"
            }
        }
    ]
}
