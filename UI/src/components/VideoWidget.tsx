import { useRef, useEffect, useState, useCallback } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import styles from './VideoWidget.module.css'

// Reuse Electron webview declaration from BrowserWidget
declare global {
  namespace JSX {
    interface IntrinsicElements {
      webview: React.DetailedHTMLProps<
        React.HTMLAttributes<HTMLElement> & {
          src?: string; partition?: string; useragent?: string
          allowpopups?: string; disablewebsecurity?: string
        },
        HTMLElement
      >
    }
  }
}

const CHROME_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
const DEFAULT_W = 640
const DEFAULT_H = 420

interface Props {
  visible: boolean
  url?:    string
  isAway?: boolean
}

function normaliseUrl(raw: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return ''
  // YouTube short links → full embed not needed, just load the watch URL
  if (/^https?:\/\//.test(trimmed)) return trimmed
  if (/^www\./.test(trimmed)) return `https://${trimmed}`
  // Plain search query → YouTube search
  if (!trimmed.includes('.')) return `https://www.youtube.com/results?search_query=${encodeURIComponent(trimmed)}`
  return `https://${trimmed}`
}

export default function VideoWidget({ visible, url = '', isAway = false }: Props) {
  const panelRef   = useRef<HTMLDivElement>(null)
  const webviewRef = useRef<HTMLElement>(null)
  const resizeRef  = useRef<{ startX: number; startY: number; startW: number; startH: number } | null>(null)

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-video')

  const [inputUrl,    setInputUrl]    = useState('')
  const [currentUrl,  setCurrentUrl]  = useState('')
  const [loading,     setLoading]     = useState(false)
  const [size,        setSize]        = useState({ w: DEFAULT_W, h: DEFAULT_H })

  // Open to default position (centre-screen)
  useEffect(() => {
    if (visible) initTopLeft(Math.round(window.innerWidth / 2 - DEFAULT_W / 2), Math.round(window.innerHeight / 2 - DEFAULT_H / 2))
  }, [visible, initTopLeft])

  // Load URL when prop changes (e.g. from Maya tool call)
  useEffect(() => {
    if (url && url !== currentUrl) {
      const norm = normaliseUrl(url)
      setCurrentUrl(norm)
      setInputUrl(url)
    }
  }, [url])

  // Wire up webview events
  useEffect(() => {
    const wv = webviewRef.current
    if (!wv) return
    const maya = (window as any).mayaElectron as any

    const onStart  = () => setLoading(true)
    const onFinish = () => { setLoading(false); setCurrentUrl((wv as any).getURL?.() ?? currentUrl) }

    // Intercept Google auth navigation — open in a real BrowserWindow so Google allows sign-in
    const onWillNavigate = (e: any) => {
      const url: string = e.url ?? ''
      if (url.includes('accounts.google.com') || url.includes('google.com/signin')) {
        e.preventDefault?.()
        maya?.videoAuthPopup?.(url)
      }
    }

    // After sign-in popup closes, reload the webview to pick up the new session
    const cleanup = maya?.onVideoAuthComplete?.(() => {
      const wv2 = webviewRef.current as any
      wv2?.reload?.()
    })

    wv.addEventListener('did-start-loading',  onStart)
    wv.addEventListener('did-stop-loading',   onFinish)
    wv.addEventListener('did-fail-load',      onFinish)
    wv.addEventListener('will-navigate',      onWillNavigate)
    return () => {
      wv.removeEventListener('did-start-loading',  onStart)
      wv.removeEventListener('did-stop-loading',   onFinish)
      wv.removeEventListener('did-fail-load',      onFinish)
      wv.removeEventListener('will-navigate',      onWillNavigate)
      cleanup?.()
    }
  }, [visible])

  function navigate(raw: string) {
    const norm = normaliseUrl(raw)
    if (!norm) return
    setCurrentUrl(norm)
    setInputUrl(raw)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') navigate(inputUrl)
  }

  // Custom resize — drag the bottom-right handle
  const onResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    e.preventDefault()
    resizeRef.current = {
      startX: e.clientX, startY: e.clientY,
      startW: panelRef.current?.offsetWidth  ?? DEFAULT_W,
      startH: panelRef.current?.offsetHeight ?? DEFAULT_H,
    }
    function onMove(ev: MouseEvent) {
      if (!resizeRef.current) return
      const { startX, startY, startW, startH } = resizeRef.current
      setSize({
        w: Math.max(360, startW + ev.clientX - startX),
        h: Math.max(260, startH + ev.clientY - startY),
      })
    }
    function onUp() {
      resizeRef.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup',   onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup',   onUp)
  }, [])

  const panelBaseStyle = {
    ...panelStyle,
    width:  size.w,
    height: size.h,
    ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none' as const, transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
  }

  return (
    <div
      ref={panelRef}
      className={`${styles.panel} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={panelBaseStyle}
    >
      {/* Header */}
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <span className={styles.title}>VIDEO</span>
        {loading && <span className={styles.spinner}>⟳</span>}
      </div>

      {/* URL bar */}
      <div className={styles.urlBar} onMouseDown={e => e.stopPropagation()}>
        <input
          className={styles.urlInput}
          value={inputUrl}
          onChange={e => setInputUrl(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="YouTube URL, any video URL, or search..."
          spellCheck={false}
        />
        <button className={styles.goBtn} onClick={() => navigate(inputUrl)}>▶</button>
        {currentUrl && (
          <button className={styles.goBtn} title="Reload" onClick={() => {
            const wv = webviewRef.current as any
            wv?.reload?.()
          }}>↺</button>
        )}
      </div>

      {/* Webview */}
      <div className={styles.viewContainer}>
        {currentUrl ? (
          <webview
            ref={webviewRef as any}
            src={currentUrl}
            className={styles.webview}
            useragent={CHROME_UA}
            allowpopups="true"
            partition="persist:maya-video"
            style={{ width: '100%', height: '100%' }}
          />
        ) : (
          <div className={styles.empty}>
            <span className={styles.emptyIcon}>▶</span>
            <p>Paste a YouTube link or any video URL above</p>
            <p className={styles.emptyHint}>or say "Maya, play [video]"</p>
          </div>
        )}
      </div>

      {/* Resize handle */}
      <div className={styles.resizeHandle} onMouseDown={onResizeMouseDown} />
    </div>
  )
}
