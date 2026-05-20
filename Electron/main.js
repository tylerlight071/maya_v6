const { app, BrowserWindow, globalShortcut, ipcMain, desktopCapturer, session } = require('electron');
const path = require('path');
const fs   = require('fs');
const { spawn } = require('child_process');

let mainWindow;
let backend;

function createWindow() {
    const isDev = !app.isPackaged;

    mainWindow = new BrowserWindow({
        frame: false,
        show: false,
        fullscreen: true,
        thickFrame: false,
        backgroundColor: '#0d0a18',
        icon: path.join(__dirname, 'assets/icons/1024x1024.png'),
        webPreferences: {
            contextIsolation: true,
            nodeIntegration: false,
            webSecurity: false,   // allows file:// image URLs from local filesystem
            webviewTag: true,     // enables <webview> element for the browser widget
            preload: path.join(__dirname, 'preload.js'),
        },
    });

    mainWindow.once('ready-to-show', () => mainWindow.show());

    // Grant camera and mic permissions so getUserMedia() works in the renderer.
    try {
        const MEDIA = ['media', 'camera', 'microphone', 'display-capture', 'screen'];
        session.defaultSession.setPermissionRequestHandler((_wc, perm, cb) => cb(MEDIA.includes(perm)));
        if (typeof session.defaultSession.setPermissionCheckHandler === 'function') {
            session.defaultSession.setPermissionCheckHandler((_wc, perm) => MEDIA.includes(perm));
        }
    } catch (e) {
        console.error('[main] permission handler error:', e);
    }

    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
    } else {
        mainWindow.loadFile(path.join(__dirname, '../UI/dist/index.html'));
    }

    globalShortcut.register('Escape', () => {
        mainWindow.setFullScreen(!mainWindow.isFullScreen());
    });

    globalShortcut.register('CommandOrControl+Shift+I', () => {
        mainWindow.webContents.toggleDevTools();
    });

    // Screen capture — must live in main process (desktopCapturer restricted in renderer since Electron 20)
    ipcMain.handle('capture-screen', async () => {
        try {
            const sources = await desktopCapturer.getSources({
                types: ['screen'],
                thumbnailSize: { width: 1920, height: 1080 },
            });
            if (sources.length > 0) {
                return sources[0].thumbnail.toDataURL('image/jpeg');
            }
            return null;
        } catch (e) {
            console.error('[main] screen capture error:', e);
            return null;
        }
    });

    // Graceful shutdown triggered by MAYA's shutdown_system tool
    ipcMain.handle('shutdown', () => {
        backend?.kill();
        app.quit();
    });

    // Dev Log recording — save WebM blob to Videos/Dev Logs/
    ipcMain.handle('save-recording', async (_event, data, filename) => {
        try {
            const devLogsDir = path.join(app.getPath('videos'), 'Dev Logs');
            if (!fs.existsSync(devLogsDir)) fs.mkdirSync(devLogsDir, { recursive: true });
            const fullPath = path.join(devLogsDir, filename);
            // data arrives as a Uint8Array or Buffer via Electron IPC structured clone
            const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
            fs.writeFileSync(fullPath, buf);
            console.log('[main] dev log saved:', fullPath, `(${(buf.length / 1024 / 1024).toFixed(1)} MB)`);
            return fullPath;
        } catch (e) {
            console.error('[main] save-recording error:', e);
            return null;
        }
    });

    // Spawn unified Python backend
    const backendDir = isDev
        ? path.join(__dirname, '../Backend')
        : path.join(process.resourcesPath, 'Backend');

    const pythonExe = isDev
        ? path.join(backendDir, '.venv/Scripts/python.exe')
        : 'python';

    backend = spawn(pythonExe, ['server.py'], {
        cwd: backendDir,
        stdio: 'pipe',
    });
    backend.stdout.on('data', d => console.log('[maya]', d.toString().trim()));
    backend.stderr.on('data', d => console.error('[maya]', d.toString().trim()));
    backend.on('exit', code => {
        console.log(`[maya] backend exited (${code})`);
        // If the backend exits cleanly (code 0), close Electron too
        if (code === 0) app.quit();
    });
}

app.whenReady().then(() => {
    // Ensure Dev Logs folder exists before any recording could be triggered
    try {
        const devLogsDir = path.join(app.getPath('videos'), 'Dev Logs');
        if (!fs.existsSync(devLogsDir)) fs.mkdirSync(devLogsDir, { recursive: true });
        console.log('[main] Dev Logs folder ready:', devLogsDir);
    } catch (e) {
        console.error('[main] could not create Dev Logs folder:', e);
    }
    createWindow();
});

app.on('will-quit', () => {
    globalShortcut.unregisterAll();
    backend?.kill();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});
