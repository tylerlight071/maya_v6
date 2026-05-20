import { useRef, useEffect, useState } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import styles from './BrowserWidget.module.css'

// Electron webview element — not in standard DOM types
declare global {
  namespace JSX {
    interface IntrinsicElements {
      webview: React.DetailedHTMLProps<
        React.HTMLAttributes<HTMLElement> & {
          src?: string
          partition?: string
          useragent?: string
          allowpopups?: string
          disablewebsecurity?: string
        },
        HTMLElement
      >
    }
  }
}

// Realistic Chrome UA so sites don't block Electron
const CHROME_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'

interface BrowserWidgetProps {
  visible:  boolean
  url:      string
  title:    string
  isAway?:  boolean
}

function extractDomain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

function getServiceIcon(url: string): string {
  if (url.includes('google.com/maps'))   return '🗺'
  if (url.includes('youtube.com'))       return '▶'
  if (url.includes('wikipedia.org'))     return '◎'
  if (url.includes('github.com'))        return '◈'
  if (url.includes('localhost'))         return '⬡'
  if (url.includes('imdb.com'))          return '★'
  return '◻'
}

export default function BrowserWidget({ visible, url, title, isAway = false }: BrowserWidgetProps) {
  const panelRef   = useRef<HTMLDivElement>(null)
  const webviewRef = useRef<HTMLElement>(null)

  const [loading,    setLoading]    = useState(false)
  const [pageTitle,  setPageTitle]  = useState('')
  const [currentUrl, setCurrentUrl] = useState('')
  const [canGoBack,  setCanGoBack]  = useState(false)

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-browser')

  useEffect(() => {
    if (visible) initTopLeft(Math.round(window.innerWidth / 2 - 460), 65)
  }, [visible, initTopLeft])

  // Wire up webview events once it's mounted
  useEffect(() => {
    const wv = webviewRef.current
    if (!wv) return

    const onStart  = () => setLoading(true)
    const onFinish = () => {
      setLoading(false)
      setCurrentUrl((wv as any).getURL?.() ?? url)
      setCanGoBack((wv as any).canGoBack?.() ?? false)
    }
    const onTitle  = (e: any) => setPageTitle(e.title ?? '')
    const onNav    = (e: any) => setCurrentUrl(e.url ?? '')

    wv.addEventListener('did-start-loading',    onStart)
    wv.addEventListener('did-finish-load',       onFinish)
    wv.addEventListener('did-stop-loading',      () => setLoading(false))
    wv.addEventListener('page-title-updated',    onTitle)
    wv.addEventListener('did-navigate',          onNav)
    wv.addEventListener('did-navigate-in-page',  onNav)

    return () => {
      wv.removeEventListener('did-start-loading',   onStart)
      wv.removeEventListener('did-finish-load',      onFinish)
      wv.removeEventListener('did-stop-loading',     () => setLoading(false))
      wv.removeEventListener('page-title-updated',   onTitle)
      wv.removeEventListener('did-navigate',         onNav)
      wv.removeEventListener('did-navigate-in-page', onNav)
    }
  }, [url])

  // Navigate when url prop changes
  useEffect(() => {
    if (!url || !webviewRef.current) return
    setLoading(true)
    setPageTitle('')
    setCurrentUrl(url)
  }, [url])

  const displayTitle  = title || pageTitle || extractDomain(url)
  const displayDomain = extractDomain(currentUrl || url)
  const icon          = getServiceIcon(url)

  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{
        ...panelStyle,
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
        </div>
      </div>

      {/* Loading bar */}
      {loading && <div className={styles.loadBar}><div className={styles.loadFill} /></div>}

      {/* Webview */}
      <div className={styles.viewportWrap}>
        {url ? (
          // @ts-ignore — Electron webview, valid in Electron renderer with webviewTag:true
          <webview
            ref={webviewRef}
            src={url}
            useragent={CHROME_UA}
            partition="persist:maya-browser"
            className={styles.webview}
          />
        ) : (
          <div className={styles.idle}>
            <span className={styles.idleIcon}>◻</span>
            <span className={styles.idleText}>BROWSER READY</span>
            <span className={styles.idleHint}>MAYA will navigate here when needed</span>
          </div>
        )}

        {/* Corner brackets */}
        <div className={`${styles.corner} ${styles.cornerTL}`} />
        <div className={`${styles.corner} ${styles.cornerTR}`} />
        <div className={`${styles.corner} ${styles.cornerBL}`} />
        <div className={`${styles.corner} ${styles.cornerBR}`} />
      </div>
    </div>
  )
}
