import { useRef, useEffect, useState, useCallback } from 'react'
import { IMAGES_WS, wsUrl } from '../constants'
import { useDraggable } from '../hooks/useDraggable'
import styles from './ImagesWidget.module.css'

interface ImageEntry {
  filename:   string
  thumb_b64:  string
  width:      number
  height:     number
  mtime:      number
  size_bytes: number
}

interface ImagesWidgetProps {
  visible:           boolean
  isAway?:           boolean
  initialFilename?:  string | null
  onInitialHandled?: () => void
}

const IMAGES_BASE = 'maya-images://'

function fmtBytes(n: number): string {
  if (n < 1024)       return `${n} B`
  if (n < 1048576)    return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1048576).toFixed(1)} MB`
}

export default function ImagesWidget({ visible, isAway = false, initialFilename, onInitialHandled }: ImagesWidgetProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)

  const [images,           setImages]           = useState<ImageEntry[]>([])
  const [loading,          setLoading]          = useState(true)
  const [wsOk,             setWsOk]             = useState(false)
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null)
  const [imgError,         setImgError]         = useState(false)
  const [error,            setError]            = useState<string | null>(null)

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-images')

  useEffect(() => {
    if (visible) initTopLeft(window.innerWidth - 540, 65)
  }, [visible, initTopLeft])

  // ── WebSocket connection ─────────────────────────────────────────────────
  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000

    function connect() {
      const ws = new WebSocket(wsUrl(IMAGES_WS))
      ws.onopen  = () => { setWsOk(true); backoff = 2000 }
      ws.onerror = () => ws.close()
      ws.onclose = () => {
        setWsOk(false)
        backoff = Math.min(backoff * 1.5, 30_000)
        reconnect = setTimeout(connect, backoff)
      }
      ws.onmessage = e => {
        let msg: Record<string, unknown>
        try { msg = JSON.parse(e.data as string) } catch { return }
        switch (msg.type as string) {
          case 'images_catalog':
            setImages(msg.images as ImageEntry[])
            setLoading(false)
            setError(null)
            break
          case 'images_loading':
            setLoading(true)
            break
          case 'images_error':
            setError(msg.message as string || 'Could not load images.')
            setLoading(false)
            break
        }
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible])

  // ── Navigate to image when MAYA uses show_image tool ──────────────────────
  useEffect(() => {
    if (!initialFilename || images.length === 0) return
    const lower   = initialFilename.toLowerCase()
    const matched = images.find(img => img.filename.toLowerCase().includes(lower))
    if (matched) {
      setSelectedFilename(matched.filename)
      setImgError(false)
    }
    onInitialHandled?.()
  }, [initialFilename, images])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Keyboard navigation in lightbox ─────────────────────────────────────
  useEffect(() => {
    if (selectedFilename === null) return
    function onKey(e: KeyboardEvent) {
      const idx = images.findIndex(img => img.filename === selectedFilename)
      if (idx === -1) return
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault()
        if (idx < images.length - 1) { setSelectedFilename(images[idx + 1].filename); setImgError(false) }
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault()
        if (idx > 0) { setSelectedFilename(images[idx - 1].filename); setImgError(false) }
      } else if (e.key === 'Escape') {
        setSelectedFilename(null)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedFilename, images])

  // ── After catalog refresh, if selected image was removed → back to grid ──
  useEffect(() => {
    if (selectedFilename && images.length > 0) {
      const still = images.some(img => img.filename === selectedFilename)
      if (!still) setSelectedFilename(null)
    }
  }, [images, selectedFilename])

  const nav = useCallback((dir: 1 | -1) => {
    const idx = images.findIndex(img => img.filename === selectedFilename)
    if (idx === -1) return
    const next = idx + dir
    if (next >= 0 && next < images.length) {
      setSelectedFilename(images[next].filename)
      setImgError(false)
    }
  }, [images, selectedFilename])

  const selectedIdx = selectedFilename ? images.findIndex(img => img.filename === selectedFilename) : -1
  const selectedImg = selectedIdx >= 0 ? images[selectedIdx] : null

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{
        ...panelStyle,
        ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
    >
      {/* Header — changes content between grid and lightbox */}
      {selectedImg ? (
        <div className={styles.lightboxHeader} onMouseDown={onHeaderMouseDown}>
          <button
            className={styles.backBtn}
            onMouseDown={e => e.stopPropagation()}
            onClick={() => setSelectedFilename(null)}
          >← BACK</button>
          <span className={styles.lightboxTitle}>{selectedImg.filename}</span>
          <span className={styles.counter}>{selectedIdx + 1} / {images.length}</span>
          <span className={styles.dragHint}>⠿ ⠿</span>
        </div>
      ) : (
        <div className={styles.header} onMouseDown={onHeaderMouseDown}>
          <div className={styles.headerLeft}>
            <span className={`${styles.dot} ${wsOk ? styles.dotLive : ''}`} />
            <span className={styles.title}>IMAGES</span>
          </div>
          <span className={styles.dragHint}>⠿ ⠿</span>
          <span className={styles.count}>{loading ? '—' : `${images.length} PHOTOS`}</span>
        </div>
      )}

      {/* Body */}
      {loading && !error && (
        <div className={styles.loading}>
          <span className={styles.loadingText}>LOADING GALLERY</span>
          <span className={styles.loadingDots}>· · ·</span>
        </div>
      )}

      {error && (
        <div className={styles.error}>{error}</div>
      )}

      {!loading && !error && !selectedImg && (
        <div className={styles.grid}>
          {images.length === 0 ? (
            <span className={styles.empty}>No images found</span>
          ) : images.map(img => (
            <img
              key={img.filename}
              src={img.thumb_b64}
              className={styles.thumb}
              title={img.filename}
              alt={img.filename}
              onClick={() => { setSelectedFilename(img.filename); setImgError(false) }}
              onMouseDown={e => e.stopPropagation()}
            />
          ))}
        </div>
      )}

      {!loading && !error && selectedImg && (
        <div className={styles.lightbox}>
          {imgError ? (
            <div className={styles.imgError}>FILE NOT FOUND</div>
          ) : (
            <img
              src={`${IMAGES_BASE}${encodeURIComponent(selectedImg.filename)}`}
              className={styles.fullImage}
              alt={selectedImg.filename}
              onError={() => setImgError(true)}
              onMouseDown={e => e.stopPropagation()}
            />
          )}

          <div className={styles.navRow}>
            <button
              className={styles.navBtn}
              onMouseDown={e => e.stopPropagation()}
              onClick={() => nav(-1)}
              disabled={selectedIdx <= 0}
            >◀ PREV</button>
            <button
              className={styles.navBtn}
              onMouseDown={e => e.stopPropagation()}
              onClick={() => nav(1)}
              disabled={selectedIdx >= images.length - 1}
            >NEXT ▶</button>
          </div>

          <div className={styles.metaRow}>
            {selectedImg.width > 0 && (
              <span className={styles.metaItem}>{selectedImg.width} × {selectedImg.height}</span>
            )}
            <span className={styles.metaItem}>{fmtBytes(selectedImg.size_bytes)}</span>
            <span className={styles.metaItem}>
              {new Date(selectedImg.mtime).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
