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
        mainWindow?.close();
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
        if (code === 0 || code === null) {
            mainWindow?.close();
            app.quit();
        }
    });
}

// ── Video auth popup ──────────────────────────────────────────────────────────
// Google blocks sign-in in webviews. We intercept the auth URL and open it in
// a proper BrowserWindow using the same partition — cookies are shared, so
// after signing in the video webview picks up the session automatically.
ipcMain.handle('video-auth-popup', (_event, url) => {
    const popup = new BrowserWindow({
        width:  500,
        height: 680,
        parent: mainWindow,
        modal:  false,
        frame:  true,
        title:  'Sign in to Google',
        webPreferences: {
            partition:        'persist:maya-video',   // shares cookies with video webview
            contextIsolation: true,
            nodeIntegration:  false,
        },
    });
    popup.setMenuBarVisibility(false);
    popup.loadURL(url);
    popup.once('closed', () => {
        // Notify renderer so the video webview can reload with the new session
        mainWindow?.webContents?.send('video-auth-complete');
    });
});

// ── YouTube cookie import ──────────────────────────────────────────────────────
// Parses a Netscape-format cookies.txt and imports into the video session so
// YouTube recognises the account without any sign-in flow.
async function importYoutubeCookies(videoSession) {
    const cookieFile = path.join(__dirname, '../Backend/models/youtube_cookies.txt');
    if (!fs.existsSync(cookieFile)) return;
    const lines = fs.readFileSync(cookieFile, 'utf-8').split('\n');
    let imported = 0;
    for (const line of lines) {
        if (!line.trim() || line.startsWith('#')) continue;
        const parts = line.split('\t');
        if (parts.length < 7) continue;
        const [domain, , cookiePath, secureStr, expiryStr, name, ...valueParts] = parts;
        const value  = valueParts.join('\t').trim();
        const secure = secureStr.trim() === 'TRUE';
        const expiry = parseInt(expiryStr.trim(), 10);
        try {
            await videoSession.cookies.set({
                url:            `https://www.youtube.com`,
                domain:         domain.trim(),
                name:           name.trim(),
                value,
                path:           cookiePath.trim(),
                secure,
                httpOnly:       false,
                expirationDate: expiry > 0 ? expiry : undefined,
            });
            imported++;
        } catch (_) { /* skip cookies that fail */ }
    }
    console.log(`[main] imported ${imported} YouTube cookies into video session`);
}

app.whenReady().then(() => {
    // Configure browser session — Chrome UA + pre-accept common news site consent cookies
    try {
        const browserSession = session.fromPartition('persist:maya-browser');
        const CHROME_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
                          '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';
        browserSession.setUserAgent(CHROME_UA);
        // BBC News GDPR consent — prevents cookie banner blocking article content
        const bbcConsent = [
            { name: 'ckns_policy',   value: '111',  domain: '.bbc.co.uk' },
            { name: 'ckns_explicit', value: '1',    domain: '.bbc.co.uk' },
            { name: 'ckns_policy',   value: '111',  domain: '.bbc.com' },
            { name: 'ckns_explicit', value: '1',    domain: '.bbc.com' },
        ];
        for (const c of bbcConsent) {
            browserSession.cookies.set({
                url: `https://www.bbc.co.uk`, ...c, path: '/', secure: true,
            }).catch(() => {});
        }
        console.log('[main] browser session ready');
    } catch (e) {
        console.error('[main] browser session setup error:', e);
    }

    // Configure video session — UA + pre-import YouTube cookies
    try {
        const videoSession = session.fromPartition('persist:maya-video');
        videoSession.setUserAgent(
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
            '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        );
        importYoutubeCookies(videoSession).catch(e =>
            console.error('[main] cookie import error:', e)
        );
    } catch (e) {
        console.error('[main] video session setup error:', e);
    }
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
