import { useRef, useEffect, useState, useCallback } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import styles from './WebcamWidget.module.css'

const VISION_WS       = 'ws://127.0.0.1:8766'
const FRAME_INTERVAL  = 2000   // ms between frames sent to vision server
const AWAY_DEBOUNCE   = 5000   // ms of no-face before marking as away

interface VisionState {
  present:             boolean
  recognized:          'Tyler' | 'unknown' | null
  confidence:          number
  emotion:             string
  emotion_display:     string
  emotion_confidence:  number
}

const EMOTION_COLOUR: Record<string, string> = {
  happy:       'rgba(80, 200, 120, 0.85)',
  surprised:   'rgba(255, 200, 60, 0.85)',
  neutral:     'rgba(140, 80, 255, 0.45)',
  low:         'rgba(80, 120, 255, 0.80)',
  frustrated:  'rgba(255, 80, 80, 0.85)',
  anxious:     'rgba(255, 140, 50, 0.85)',
  uncomfortable:'rgba(180, 100, 80, 0.80)',
}

interface WebcamWidgetProps {
  visible:          boolean
  devLogMode?:      boolean
  onPresenceChange?: (present: boolean) => void
}

export default function WebcamWidget({ visible, devLogMode = false, onPresenceChange }: WebcamWidgetProps) {
  const panelRef   = useRef<HTMLDivElement>(null)
  const videoRef   = useRef<HTMLVideoElement>(null)
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const streamRef  = useRef<MediaStream | null>(null)
  const wsRef      = useRef<WebSocket | null>(null)
  const timerRef   = useRef<ReturnType<typeof setInterval> | null>(null)
  const awayTimer  = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastPres   = useRef<boolean | null>(null)

  const [camActive, setCamActive] = useState(false)
  const [hasError,  setHasError]  = useState(false)
  const [vision, setVision] = useState<VisionState>({
    present: false, recognized: null, confidence: 0,
    emotion: 'neutral', emotion_display: 'neutral', emotion_confidence: 0,
  })

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-webcam')

  useEffect(() => {
    if (visible) initTopLeft(window.innerWidth - 304, 65)
  }, [visible, initTopLeft])

  // Camera stream
  useEffect(() => {
    if (!visible) return
    setHasError(false)
    setCamActive(false)
    const t = setTimeout(() => {
      navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
        .then(stream => {
          streamRef.current = stream
          if (videoRef.current) {
            videoRef.current.srcObject = stream
            setCamActive(true)
          }
        })
        .catch(() => setHasError(true))
    }, 150)
    return () => {
      clearTimeout(t)
      streamRef.current?.getTracks().forEach(t => t.stop())
      streamRef.current = null
      setCamActive(false)
    }
  }, [visible])

  // Vision WebSocket with auto-reconnect
  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000  // start at 2s, cap at 30s

    function connect() {
      const ws = new WebSocket(VISION_WS)
      ws.onerror = () => ws.close()
      ws.onopen  = () => { backoff = 2000 }  // reset backoff on successful connect
      ws.onmessage = e => {
        try {
          const msg = JSON.parse(e.data)
          if (msg.type !== 'vision_update') return
          setVision({
            present:            msg.present,
            recognized:         msg.recognized,
            confidence:         msg.confidence ?? 0,
            emotion:            (msg.emotion as string)         ?? 'neutral',
            emotion_display:    (msg.emotion_display as string) ?? 'neutral',
            emotion_confidence: (msg.emotion_confidence as number) ?? 0,
          })
          const present = msg.present === true
          if (present) {
            if (awayTimer.current) { clearTimeout(awayTimer.current); awayTimer.current = null }
            if (lastPres.current !== true) { lastPres.current = true; onPresenceChange?.(true) }
          } else if (!awayTimer.current && lastPres.current !== false) {
            awayTimer.current = setTimeout(() => {
              awayTimer.current = null; lastPres.current = false; onPresenceChange?.(false)
            }, AWAY_DEBOUNCE)
          }
        } catch { /* ignore parse errors */ }
      }
      ws.onclose = () => {
        backoff = Math.min(backoff * 1.5, 30_000)
        reconnect = setTimeout(connect, backoff)
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible, onPresenceChange])

  // Capture frame and send to vision server
  const sendFrame = useCallback(() => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return
    if (!videoRef.current || !canvasRef.current) return
    const v = videoRef.current, c = canvasRef.current
    c.width = v.videoWidth || 320; c.height = v.videoHeight || 240
    c.getContext('2d')?.drawImage(v, 0, 0)
    const b64 = c.toDataURL('image/jpeg', 0.6).split(',')[1]
    if (!b64) return
    try {
      wsRef.current.send(JSON.stringify({ type: 'vision_frame', image: b64 }))
    } catch { /* socket closed between readyState check and send */ }
  }, [])

  useEffect(() => {
    // Pause vision frame sending during dev log (face recognition off)
    if (!camActive || devLogMode) return
    timerRef.current = setInterval(sendFrame, FRAME_INTERVAL)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [camActive, sendFrame, devLogMode])

  const statusLabel = !vision.present ? 'AWAY'
    : vision.recognized === 'Tyler'   ? 'TYLER'
    : 'UNIDENTIFIED'

  const statusClass = !vision.present ? styles.statusAway
    : vision.recognized === 'Tyler'   ? styles.statusTyler
    : styles.statusUnknown

  // In dev log mode: animate to center, expand to 640 px, no transition when dragging
  const devLogStyle = devLogMode ? {
    left:       `${Math.round(window.innerWidth  / 2 - 320)}px`,
    top:        `${Math.round(window.innerHeight / 2 - 240)}px`,
    bottom:     'auto',
    transform:  'none',
    width:      '640px',
    transition: 'left 0.85s cubic-bezier(0.16,1,0.3,1), top 0.85s cubic-bezier(0.16,1,0.3,1), width 0.85s cubic-bezier(0.16,1,0.3,1)',
  } : {
    transition: dragging ? 'none' : 'left 0.85s cubic-bezier(0.16,1,0.3,1), top 0.85s cubic-bezier(0.16,1,0.3,1)',
  }

  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{ ...panelStyle, ...devLogStyle }}
    >
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <div className={styles.headerLeft}>
          <span className={`${styles.liveDot} ${camActive ? styles.active : ''}`} />
          <span className={styles.title}>CAMERA</span>
        </div>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <span className={`${styles.liveFeedLabel} ${devLogMode ? styles.devLogLabel : ''}`}>
          {devLogMode ? 'DEV LOG' : 'LIVE FEED'}
        </span>
      </div>

      <div className={styles.viewport}>
        {hasError ? (
          <div className={styles.noSignal}>
            <span className={styles.noSignalText}>NO SIGNAL</span>
          </div>
        ) : (
          <video ref={videoRef} autoPlay muted playsInline className={styles.video} />
        )}

        {camActive && (
          <div className={`${styles.visionOverlay} ${statusClass}`}>
            <span className={styles.visionLabel}>{statusLabel}</span>
            {vision.recognized === 'Tyler' && vision.confidence > 0 && (
              <div className={styles.confidenceBar}>
                <div className={styles.confidenceFill} style={{ width: `${Math.round(vision.confidence * 100)}%` }} />
              </div>
            )}
          </div>
        )}

        {/* Emotion badge — only shown when Tyler is recognised and emotion is notable */}
        {camActive && vision.recognized === 'Tyler' && vision.emotion !== 'neutral' && vision.emotion_confidence >= 0.35 && (
          <div
            className={styles.emotionBadge}
            style={{ color: EMOTION_COLOUR[vision.emotion_display] ?? 'rgba(185,155,255,0.8)' }}
            title={`${vision.emotion_display} (${Math.round(vision.emotion_confidence * 100)}%)`}
          >
            {vision.emotion_display.toUpperCase()}
          </div>
        )}

        <div className={styles.scanlines} />
        <div className={styles.cornerTL} />
        <div className={styles.cornerTR} />
        <div className={styles.cornerBL} />
        <div className={styles.cornerBR} />
      </div>
    </div>
  )
}
