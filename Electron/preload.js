const { contextBridge, ipcRenderer } = require('electron')
const fs   = require('fs')
const path = require('path')

// Read session token written by the Python backend at startup.
// Exposed to the renderer so WebSocket connections can authenticate.
function readSessionToken() {
    try {
        const tokenPath = path.join(__dirname, '../Backend/models/.session_token')
        if (fs.existsSync(tokenPath)) {
            return fs.readFileSync(tokenPath, 'utf-8').trim()
        }
    } catch (_) {}
    return ''
}

contextBridge.exposeInMainWorld('mayaElectron', {
    getSessionToken: () => readSessionToken(),
    captureScreen: async () => {
        try {
            return await ipcRenderer.invoke('capture-screen')
        } catch (e) {
            console.error('[preload] screen capture error:', e)
            return null
        }
    },

    saveRecording: async (buffer, filename) => {
        try {
            // Strip path separators to prevent path traversal attacks
            const safeName = String(filename).replace(/[/\\]/g, '_')
            return await ipcRenderer.invoke('save-recording', buffer, safeName)
        } catch (e) {
            console.error('[preload] save-recording error:', e)
            return null
        }
    },

    shutdown: () => ipcRenderer.invoke('shutdown'),

    videoAuthPopup: (url) => ipcRenderer.invoke('video-auth-popup', url),
    onVideoAuthComplete: (cb) => ipcRenderer.on('video-auth-complete', cb),
})
