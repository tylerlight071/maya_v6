import { useRef, useEffect, useState, useCallback } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import styles from './BrowserWidget.module.css'

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

const CHROME_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'

const DEFAULT_W = 920
const DEFAULT_H = 660

interface BrowserWidgetProps {
  visible:  boolean
  url:      string
  title:    string
  isAway?:  boolean
  onClose?: () => void
}

function extractDomain(url: string): string {
  try { return new URL(url).hostname.replace(/^www\./, '') } catch { return url }
}

function getServiceIcon(url: string): string {
  if (url.includes('google.com/maps')) return '🗺'
  if (url.includes('youtube.com'))     return '▶'
  if (url.includes('wikipedia.org'))   return '◎'
  if (url.includes('github.com'))      return '◈'
  if (url.includes('localhost'))       return '⬡'
  if (url.includes('imdb.com'))        return '★'
  return '◻'
}

export default function BrowserWidget({ visible, url, title, isAway = false, onClose }: BrowserWidgetProps) {
  const panelRef   = useRef<HTMLDivElement>(null)
  const webviewRef = useRef<HTMLElement>(null)
  const resizeRef  = useRef<{ startX: number; startY: number; startW: number; startH: number } | null>(null)

  const [loading,    setLoading]    = useState(false)
  const [pageTitle,  setPageTitle]  = useState('')
  const [currentUrl, setCurrentUrl] = useState('')
  const [canGoBack,  setCanGoBack]  = useState(false)
  const [size,       setSize]       = useState({ w: DEFAULT_W, h: DEFAULT_H })

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-browser')

  useEffect(() => {
    if (visible) initTopLeft(Math.round(window.innerWidth / 2 - DEFAULT_W / 2), 65)
  }, [visible, initTopLeft])

  useEffect(() => {
    const wv = webviewRef.current
    if (!wv) return

    const onStart  = () => setLoading(true)
    const onStop   = () => setLoading(false)
    const onFinish = () => {
      setLoading(false)
      setCurrentUrl((wv as any).getURL?.() ?? url)
      setCanGoBack((wv as any).canGoBack?.() ?? false)
      // Trigger resize + scroll so IntersectionObserver-based lazy loaders fire
      setTimeout(() => {
        ;(wv as any).executeJavaScript?.(
          'window.dispatchEvent(new Event("resize"));' +
          'window.scrollTo(0,1);window.scrollTo(0,0);'
        ).catch(() => {})
      }, 300)
    }
    const onTitle = (e: any) => setPageTitle(e.title ?? '')
    const onNav   = (e: any) => setCurrentUrl(e.url ?? '')

    wv.addEventListener('did-start-loading',   onStart)
    wv.addEventListener('did-finish-load',      onFinish)
    wv.addEventListener('did-stop-loading',     onStop)
    wv.addEventListener('page-title-updated',   onTitle)
    wv.addEventListener('did-navigate',         onNav)
    wv.addEventListener('did-navigate-in-page', onNav)
    return () => {
      wv.removeEventListener('did-start-loading',   onStart)
      wv.removeEventListener('did-finish-load',      onFinish)
      wv.removeEventListener('did-stop-loading',     onStop)
      wv.removeEventListener('page-title-updated',   onTitle)
      wv.removeEventListener('did-navigate',         onNav)
      wv.removeEventListener('did-navigate-in-page', onNav)
    }
  }, [url])

  useEffect(() => {
    if (!url || !webviewRef.current) return
    setLoading(true)
    setPageTitle('')
    setCurrentUrl(url)
  }, [url])

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
        w: Math.max(480, startW + ev.clientX - startX),
        h: Math.max(320, startH + ev.clientY - startY),
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

  const displayTitle  = title || pageTitle || extractDomain(url)
  const displayDomain = extractDomain(currentUrl || url)
  const icon          = getServiceIcon(url)

  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{
        ...panelStyle,
        width: size.w,
        height: size.h,
        ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
    >
      {/* Header */}
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <div className={styles.headerLeft}>
          <span className={`${styles.dot} ${loading ? styles.dotLoading : styles.dotReady}`} />
          <span className={styles.icon}>{icon}</span>
          <span className={styles.title} title={displayTitle}>{displayTitle}</span>
        </div>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <div className={styles.headerRight}>
          {canGoBack && (
            <button
              className={styles.navBtn}
              onMouseDown={e => e.stopPropagation()}
              onClick={() => (webviewRef.current as any)?.goBack?.()}
              title="Back"
            >←</button>
          )}
          <span className={styles.domain} title={currentUrl}>{displayDomain}</span>
          <button
            className={styles.closeBtn}
            onMouseDown={e => e.stopPropagation()}
            onClick={onClose}
            title="Close"
          >✕</button>
        </div>
      </div>

      {/* Loading bar */}
      {loading && <div className={styles.loadBar}><div className={styles.loadFill} /></div>}

      {/* Webview */}
      <div className={styles.viewportWrap}>
        {url ? (
          // @ts-ignore
          <webview
            ref={webviewRef}
            src={url}
            useragent={CHROME_UA}
            allowpopups="true"
            partition="persist:maya-browser"
            className={styles.webview}
            style={{ width: '100%', height: '100%' }}
          />
        ) : (
          <div className={styles.idle}>
            <span className={styles.idleIcon}>◻</span>
            <span className={styles.idleText}>BROWSER READY</span>
            <span className={styles.idleHint}>MAYA will navigate here when needed</span>
          </div>
        )}
        <div className={`${styles.corner} ${styles.cornerTL}`} />
        <div className={`${styles.corner} ${styles.cornerTR}`} />
        <div className={`${styles.corner} ${styles.cornerBL}`} />
        <div className={`${styles.corner} ${styles.cornerBR}`} />
      </div>

      {/* Resize handle */}
      <div className={styles.resizeHandle} onMouseDown={onResizeMouseDown} />
    </div>
  )
}
