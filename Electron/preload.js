const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('mayaElectron', {
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
            return await ipcRenderer.invoke('save-recording', buffer, filename)
        } catch (e) {
            console.error('[preload] save-recording error:', e)
            return null
        }
    },

    shutdown: () => ipcRenderer.invoke('shutdown'),
})
