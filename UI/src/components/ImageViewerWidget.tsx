import { useRef, useState, useEffect } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import styles from './ImageViewerWidget.module.css'

interface Props {
  filename:    string
  viewerIndex: number   // 0-based, used only for initial position cascade
  onClose:     () => void
  isAway?:     boolean
}

export default function ImageViewerWidget({ filename, viewerIndex, onClose, isAway = false }: Props) {
  const panelRef  = useRef<HTMLDivElement>(null)
  const [imgError, setImgError] = useState(false)
  const [size,     setSize]     = useState({ w: 520, h: 560 })

  // No storageKey — viewers are ephemeral; position is not persisted across sessions
  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef)

  useEffect(() => {
    const cascade = viewerIndex * 32
    initTopLeft(
      Math.min(window.innerWidth  - 540, 120 + cascade),
      Math.min(window.innerHeight - 580, 80  + cascade),
    )
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Resize handle ────────────────────────────────────────────────────────────
  const resizeOrigin = useRef({ mx: 0, my: 0, w: 0, h: 0 })

  function onResizeMouseDown(e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    resizeOrigin.current = { mx: e.clientX, my: e.clientY, w: size.w, h: size.h }

    function onMove(e: MouseEvent) {
      setSize({
        w: Math.max(260, resizeOrigin.current.w + e.clientX - resizeOrigin.current.mx),
        h: Math.max(220, resizeOrigin.current.h + e.clientY - resizeOrigin.current.my),
      })
    }
    function onUp() {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup',  onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup',  onUp)
  }

  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${dragging ? styles.dragging : ''}`}
      style={{
        ...panelStyle,
        width:  size.w,
        height: size.h,
        ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
    >
      {/* Header */}
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <span className={styles.filename}>{filename}</span>
        <button
          className={styles.closeBtn}
          onClick={onClose}
          onMouseDown={e => e.stopPropagation()}
          title="Close"
        >✕</button>
      </div>

      {/* Image body */}
      <div className={styles.body}>
        {imgError ? (
          <div className={styles.error}>FILE NOT FOUND</div>
        ) : (
          <img
            src={`maya-images://${filename}`}
            className={styles.image}
            alt={filename}
            onError={() => setImgError(true)}
            onMouseDown={e => e.stopPropagation()}
          />
        )}
      </div>

      {/* Resize handle — bottom-right corner */}
      <div className={styles.resizeHandle} onMouseDown={onResizeMouseDown} />
    </div>
  )
}
