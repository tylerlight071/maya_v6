const { app, BrowserWindow, globalShortcut, ipcMain, desktopCapturer, session, protocol } = require('electron');
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
            webSecurity: true,    // re-enabled; file:// images served via custom protocol
            webviewTag: true,     // enables <webview> element for the browser widget
            preload: path.join(__dirname, 'preload.js'),
        },
    });

    mainWindow.once('ready-to-show', () => mainWindow.show());

    // Grant camera/mic/screen ONLY to our own renderer (the MAYA UI), never to
    // arbitrary websites loaded inside the <webview> browser widget. Without the
    // origin check, any page you browse to could silently turn on your camera/mic.
    try {
        const MEDIA = ['media', 'camera', 'microphone', 'display-capture', 'screen'];
        const isTrustedOrigin = (url) =>
            typeof url === 'string' &&
            (url.startsWith('http://localhost:5173') || url.startsWith('file://'));
        session.defaultSession.setPermissionRequestHandler((wc, perm, cb) =>
            cb(isTrustedOrigin(wc?.getURL?.()) && MEDIA.includes(perm)));
        if (typeof session.defaultSession.setPermissionCheckHandler === 'function') {
            session.defaultSession.setPermissionCheckHandler((wc, perm, origin) =>
                (isTrustedOrigin(origin) || isTrustedOrigin(wc?.getURL?.())) && MEDIA.includes(perm));
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

    // DevTools only in development — don't ship a debug backdoor in packaged builds.
    if (isDev) {
        globalShortcut.register('CommandOrControl+Shift+I', () => {
            mainWindow.webContents.toggleDevTools();
        });
    }

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

    // Graceful shutdown — tell backend to flush memory/session, then force-kill after 5s
    ipcMain.handle('shutdown', async () => {
        console.log('[main] shutdown requested — waiting for backend to flush...');
        const forceKill = setTimeout(() => {
            backend?.kill();
            mainWindow?.close();
            app.quit();
        }, 5000);
        try {
            // Give backend a chance to save session logs before exiting
            if (backend && backend.exitCode === null) {
                await new Promise(resolve => {
                    backend.once('exit', resolve);
                    backend.kill('SIGTERM');
                });
            }
        } catch (_) {}
        clearTimeout(forceKill);
        mainWindow?.close();
        app.quit();
    });

    // Dev Log recording — save WebM blob to Videos/Dev Logs/
    ipcMain.handle('save-recording', async (_event, data, filename) => {
        try {
            const devLogsDir = path.join(app.getPath('videos'), 'Dev Logs');
            if (!fs.existsSync(devLogsDir)) fs.mkdirSync(devLogsDir, { recursive: true });
            // Sanitise filename — no path traversal
            const safeName = path.basename(String(filename)).replace(/[/\\]/g, '_');
            const fullPath = path.join(devLogsDir, safeName);
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

    if (!fs.existsSync(backendDir)) {
        console.error('[main] Backend directory not found:', backendDir);
    }

    backend = spawn(pythonExe, ['server.py'], {
        cwd: backendDir,
        stdio: 'pipe',
    });
    backend.stdout.on('data', d => process.stdout.write(d));
    backend.stderr.on('data', d => process.stderr.write(d));
    backend.on('exit', code => {
        console.log(`[maya] backend exited (${code})`);
        if (code === 0 || code === null) {
            mainWindow?.close();
            app.quit();
        }
    });
}

// ── Custom file protocol — serves Pictures/Images without webSecurity: false ──
function registerImageProtocol() {
    protocol.registerFileProtocol('maya-images', (request, callback) => {
        const filePath = decodeURIComponent(request.url.replace('maya-images://', ''));
        // Restrict to the Pictures/Images folder only
        const picturesDir = path.join(app.getPath('pictures'), 'Images');
        const resolved    = path.resolve(picturesDir, filePath);
        const rel = path.relative(picturesDir, resolved);
        if (rel.startsWith('..') || path.isAbsolute(rel)) {
            callback({ error: -10 });  // BLOCKED — path traversal attempt
            return;
        }
        callback({ path: resolved });
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

// ── Cookie import helper ───────────────────────────────────────────────────────
// Parses a Netscape-format cookies.txt and imports into the given session.
async function importCookiesFromFile(cookieFile, targetSession, defaultUrl) {
    if (!fs.existsSync(cookieFile)) return 0;
    const lines = fs.readFileSync(cookieFile, 'utf-8').split('\n');
    let imported = 0;
    for (const rawLine of lines) {
        // Netscape format marks HttpOnly cookies with a "#HttpOnly_" domain prefix.
        // Detect and strip it BEFORE the comment skip, so these aren't dropped.
        let line = rawLine;
        let httpOnly = false;
        if (line.startsWith('#HttpOnly_')) {
            httpOnly = true;
            line = line.slice('#HttpOnly_'.length);
        }
        if (!line.trim() || line.startsWith('#')) continue;
        const parts = line.split('\t');
        if (parts.length < 7) continue;
        const [domain, , cookiePath, secureStr, expiryStr, name, ...valueParts] = parts;
        const value  = valueParts.join('\t').trim();
        const secure = secureStr.trim() === 'TRUE';
        const expiry = parseInt(expiryStr.trim(), 10);
        const cleanDomain = domain.trim().replace(/^\./, '');
        const cookieUrl   = defaultUrl || `https://${cleanDomain}`;
        try {
            await targetSession.cookies.set({
                url:            cookieUrl,
                domain:         domain.trim(),
                name:           name.trim(),
                value,
                path:           cookiePath.trim(),
                secure,
                httpOnly,
                expirationDate: expiry > 0 ? expiry : undefined,
            });
            imported++;
        } catch (_) { /* skip cookies that fail */ }
    }
    return imported;
}

async function importYoutubeCookies(videoSession) {
    const cookieFile = path.join(__dirname, '../Backend/models/youtube_cookies.txt');
    const n = await importCookiesFromFile(cookieFile, videoSession, 'https://www.youtube.com');
    console.log(`[main] imported ${n} YouTube cookies into video session`);
}

app.whenReady().then(() => {
    registerImageProtocol();

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

        // Import any site-specific cookie files found in Backend/models/
        const cookieSites = [
            { file: 'pornhub_cookies.txt', url: 'https://www.pornhub.com', label: 'Pornhub' },
        ];
        for (const site of cookieSites) {
            const cookieFile = path.join(__dirname, '../Backend/models/', site.file);
            importCookiesFromFile(cookieFile, browserSession, site.url)
                .then(n => { if (n > 0) console.log(`[main] imported ${n} ${site.label} cookies into browser session`); })
                .catch(() => {});
        }
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
    // Send SIGTERM to allow Python to flush logs before SIGKILL
    if (backend && backend.exitCode === null) {
        backend.kill('SIGTERM');
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});
