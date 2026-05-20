import { useRef, useEffect, useState, useCallback } from 'react'
import { IMAGEGEN_WS } from '../constants'
import { useDraggable } from '../hooks/useDraggable'
import styles from './ImageGenWidget.module.css'

interface GenStatus {
  status:          'idle' | 'enhancing' | 'generating' | 'done' | 'error'
  step:            number
  total:           number
  filename:        string | null
  error:           string | null
  enhanced_prompt?: string
}

interface ImageGenWidgetProps {
  visible:  boolean
  isAway?:  boolean
}

const IMAGES_BASE = 'file:///C:/Users/tyler/Pictures/Images/'

const STATUS_LABEL: Record<GenStatus['status'], string> = {
  idle:       'READY',
  enhancing:  'ENHANCING PROMPT',
  generating: 'GENERATING',
  done:       'COMPLETE',
  error:      'ERROR',
}

export default function ImageGenWidget({ visible, isAway = false }: ImageGenWidgetProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef    = useRef<number>(0)
  const inputRef  = useRef<HTMLTextAreaElement>(null)

  const [status,        setStatus]        = useState<GenStatus>({
    status: 'idle', step: 0, total: 0, filename: null, error: null,
  })
  const [prompt,         setPrompt]        = useState('')
  const [originalPrompt, setOriginalPrompt] = useState('')   // saved before enhance
  const [wsOk,           setWsOk]          = useState(false)
  const [imgError,       setImgError]      = useState(false)
  const [wasEnhanced,    setWasEnhanced]   = useState(false)

  const statusRef  = useRef(status)
  const pollRef    = useRef<ReturnType<typeof setInterval> | null>(null)
  useEffect(() => { statusRef.current = status }, [status])

  // When generating, poll the backend every 5 s so we catch completion even
  // if the broadcast "done" event was missed (e.g. WebSocket timing)
  useEffect(() => {
    if (status.status === 'generating' || status.status === 'enhancing') {
      pollRef.current = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'poll_status' }))
        }
      }, 5000)
    } else {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    }
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null } }
  }, [status.status])

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-imagegen')

  useEffect(() => {
    if (visible) initTopLeft(Math.max(0, window.innerWidth / 2 - 270), 65)
  }, [visible, initTopLeft])

  // ── Idle / generating canvas animation ────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let t = 0
    function draw() {
      const s = statusRef.current.status
      const W = canvas!.width, H = canvas!.height
      ctx!.clearRect(0, 0, W, H)

      if (s === 'idle') {
        drawIdleScanner(ctx!, W, H, t)
      } else if (s === 'enhancing') {
        drawEnhancing(ctx!, W, H, t)
      } else if (s === 'generating') {
        drawGenerating(ctx!, W, H, t, statusRef.current.step, statusRef.current.total)
      }

      t += 0.02
      rafRef.current = requestAnimationFrame(draw)
    }
    rafRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafRef.current)
  }, [])

  // ── WebSocket ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000

    function connect() {
      const ws = new WebSocket(IMAGEGEN_WS)
      ws.onopen  = () => { setWsOk(true); backoff = 2000 }
      ws.onerror = () => ws.close()
      ws.onclose = () => {
        setWsOk(false)
        backoff = Math.min(backoff * 1.5, 30_000)
        reconnect = setTimeout(connect, backoff)
      }
      ws.onmessage = e => {
        try {
          const msg = JSON.parse(e.data as string)
          if (msg.type === 'enhanced') {
            setPrompt(msg.result)       // overwrite the text field with the optimised prompt
            setWasEnhanced(true)
            return
          }
          if (msg.status) {
            setStatus(msg as GenStatus)
            if (msg.status === 'done') setImgError(false)
          }
        } catch { /* ignore */ }
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible])

  // ── Send generation request ────────────────────────────────────────────────
  const generate = useCallback(() => {
    const text = prompt.trim()
    if (!text) return
    if (status.status === 'enhancing' || status.status === 'generating') return
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setStatus({ status: 'error', step: 0, total: 0, filename: null,
        error: 'Not connected to image server — restart MAYA and check the backend started on port 8770.' })
      return
    }
    wsRef.current.send(JSON.stringify({ type: 'generate', prompt: text }))
    setWasEnhanced(false)
  }, [prompt, status.status])

  const enhanceOnly = useCallback(() => {
    const text = prompt.trim()
    if (!text || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    setOriginalPrompt(text)   // save so user can revert
    wsRef.current.send(JSON.stringify({ type: 'enhance_only', prompt: text }))
  }, [prompt])

  const pct = status.total > 0 ? (status.step / status.total) * 100 : 0
  const isActive = status.status === 'enhancing' || status.status === 'generating'
  const showCanvas = status.status !== 'done'
  const showImage  = status.status === 'done' && status.filename

  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{
        ...panelStyle,
        ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
    >
      {/* ── Header ── */}
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <div className={styles.headerLeft}>
          <span className={`${styles.dot} ${wsOk ? styles.dotLive : ''} ${isActive ? styles.dotPulsing : ''}`} />
          <span className={styles.title}>IMAGE GEN</span>
        </div>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <span className={`${styles.statusBadge} ${styles['status_' + status.status]}`}>
          {STATUS_LABEL[status.status]}
        </span>
      </div>

      {/* ── Main canvas / image area ── */}
      <div className={styles.mainArea}>
        {showImage && !imgError ? (
          <img
            src={`${IMAGES_BASE}${encodeURIComponent(status.filename!)}`}
            className={styles.generatedImage}
            alt="Generated"
            onError={() => setImgError(true)}
            onMouseDown={e => e.stopPropagation()}
          />
        ) : showImage && imgError ? (
          <div className={styles.imgError}>FILE NOT FOUND</div>
        ) : (
          <canvas ref={canvasRef} width={520} height={390} className={styles.canvas} />
        )}

        {/* Scan-line overlay */}
        <div className={styles.scanlines} />

        {/* Corner brackets */}
        <div className={`${styles.corner} ${styles.cornerTL}`} />
        <div className={`${styles.corner} ${styles.cornerTR}`} />
        <div className={`${styles.corner} ${styles.cornerBL}`} />
        <div className={`${styles.corner} ${styles.cornerBR}`} />
      </div>

      {/* ── Progress bar (during generation) ── */}
      {(status.status === 'generating' || status.status === 'enhancing') && (
        <div className={styles.progressSection}>
          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{ width: status.status === 'enhancing' ? '100%' : `${pct}%` }}
            />
          </div>
          <span className={styles.progressLabel}>
            {status.status === 'enhancing'
              ? 'ENHANCING PROMPT...'
              : `SAMPLING  ${status.step} / ${status.total}`}
          </span>
        </div>
      )}

      {/* Enhanced indicator — shows when prompt was rewritten by enhance */}
      {wasEnhanced && (
        <div className={styles.enhancedPill}>
          <span>◈ PROMPT ENHANCED</span>
          <button
            className={styles.revertBtn}
            onMouseDown={e => e.stopPropagation()}
            onClick={() => { setPrompt(originalPrompt); setWasEnhanced(false) }}
          >↩ REVERT</button>
        </div>
      )}

      {/* ── Error state ── */}
      {status.status === 'error' && status.error && (
        <div className={styles.errorBlock}>{status.error}</div>
      )}

      {/* ── Done stats ── */}
      {status.status === 'done' && (
        <div className={styles.doneBar}>
          <span className={styles.doneLabel}>✓ GENERATION COMPLETE</span>
        </div>
      )}

      {/* ── Input area ── */}
      <div className={styles.inputSection}>
        <textarea
          ref={inputRef}
          className={styles.promptInput}
          placeholder="Describe the scene in natural language — MAYA converts it to the right format..."
          value={prompt}
          rows={2}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generate() }
          }}
          onMouseDown={e => e.stopPropagation()}
          disabled={isActive}
        />
        <div className={styles.btnRow}>
          <button
            className={styles.enhanceBtn}
            onMouseDown={e => e.stopPropagation()}
            onClick={enhanceOnly}
            disabled={isActive || !prompt.trim()}
            title="Preview the enhanced prompt without generating"
          >
            ◈ ENHANCE
          </button>
          <button
            className={`${styles.generateBtn} ${isActive ? styles.generateBtnActive : ''}`}
            onMouseDown={e => e.stopPropagation()}
            onClick={generate}
            disabled={isActive || !prompt.trim()}
          >
            {isActive ? '⟳ GENERATING...' : '▶ GENERATE'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Canvas animation functions ─────────────────────────────────────────────────

function drawIdleScanner(ctx: CanvasRenderingContext2D, W: number, H: number, t: number) {
  const cx = W / 2, cy = H / 2
  const maxR = Math.min(W, H) * 0.38

  // Dark background
  ctx.fillStyle = 'rgba(7, 3, 18, 0.6)'
  ctx.fillRect(0, 0, W, H)

  // Rotating concentric rings
  for (let i = 3; i >= 0; i--) {
    const r = maxR * (0.3 + i * 0.23)
    const alpha = 0.06 + i * 0.04
    ctx.beginPath()
    ctx.arc(cx, cy, r, 0, Math.PI * 2)
    ctx.strokeStyle = `rgba(140, 80, 255, ${alpha})`
    ctx.lineWidth = 1
    ctx.stroke()
  }

  // Radar sweep
  const sweep = (t * 0.8) % (Math.PI * 2)
  const grad = ctx.createConicalGradient
    ? null  // not standard, use manual
    : null
  // Manual sweep arc
  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(sweep)
  const sweepGrad = ctx.createLinearGradient(0, 0, maxR, 0)
  sweepGrad.addColorStop(0,   'rgba(140, 80, 255, 0.35)')
  sweepGrad.addColorStop(0.7, 'rgba(140, 80, 255, 0.08)')
  sweepGrad.addColorStop(1,   'rgba(140, 80, 255, 0)')
  ctx.beginPath()
  ctx.moveTo(0, 0)
  ctx.arc(0, 0, maxR * 0.95, -0.18, 0)
  ctx.fillStyle = sweepGrad
  ctx.fill()
  ctx.restore()

  // Dashed outer ring rotating slowly
  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(t * 0.3)
  ctx.setLineDash([6, 10])
  ctx.beginPath()
  ctx.arc(0, 0, maxR, 0, Math.PI * 2)
  ctx.strokeStyle = 'rgba(165, 100, 235, 0.25)'
  ctx.lineWidth = 1.5
  ctx.stroke()
  ctx.setLineDash([])
  ctx.restore()

  // Counter-rotating inner dashed ring
  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(-t * 0.5)
  ctx.setLineDash([3, 8])
  ctx.beginPath()
  ctx.arc(0, 0, maxR * 0.55, 0, Math.PI * 2)
  ctx.strokeStyle = 'rgba(140, 80, 255, 0.2)'
  ctx.lineWidth = 1
  ctx.stroke()
  ctx.setLineDash([])
  ctx.restore()

  // Centre glow
  const cGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR * 0.18)
  cGlow.addColorStop(0,   'rgba(165, 100, 235, 0.22)')
  cGlow.addColorStop(0.5, 'rgba(140, 80, 255, 0.06)')
  cGlow.addColorStop(1,   'rgba(0,0,0,0)')
  ctx.fillStyle = cGlow
  ctx.beginPath()
  ctx.arc(cx, cy, maxR * 0.18, 0, Math.PI * 2)
  ctx.fill()

  // READY label
  ctx.font = '700 10px Orbitron, monospace'
  ctx.letterSpacing = '3px'
  ctx.fillStyle = 'rgba(185, 140, 255, 0.45)'
  ctx.textAlign = 'center'
  ctx.fillText('AWAITING PROMPT', cx, cy + maxR * 0.62)
}

function drawEnhancing(ctx: CanvasRenderingContext2D, W: number, H: number, t: number) {
  ctx.fillStyle = 'rgba(7, 3, 18, 0.7)'
  ctx.fillRect(0, 0, W, H)

  const cx = W / 2, cy = H / 2
  // Pulsing rings
  for (let i = 0; i < 4; i++) {
    const phase = (t + i * 0.6) % (Math.PI * 2)
    const r = 60 + i * 28 + Math.sin(phase) * 8
    const alpha = 0.08 + Math.sin(phase + Math.PI) * 0.05
    ctx.beginPath()
    ctx.arc(cx, cy, r, 0, Math.PI * 2)
    ctx.strokeStyle = `rgba(100, 200, 255, ${alpha})`
    ctx.lineWidth = 1.5
    ctx.stroke()
  }

  // Orbiting dots
  for (let i = 0; i < 6; i++) {
    const angle = t * 1.4 + (i / 6) * Math.PI * 2
    const r = 70
    const x = cx + Math.cos(angle) * r
    const y = cy + Math.sin(angle) * r
    ctx.beginPath()
    ctx.arc(x, y, 2.5, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(100, 180, 255, 0.7)'
    ctx.fill()
  }

  ctx.font = '700 10px Orbitron, monospace'
  ctx.fillStyle = 'rgba(120, 190, 255, 0.6)'
  ctx.textAlign = 'center'
  ctx.fillText('ENHANCING PROMPT', cx, cy + 120)
}

function drawGenerating(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  t: number,
  step: number,
  total: number,
) {
  ctx.fillStyle = 'rgba(7, 3, 18, 0.75)'
  ctx.fillRect(0, 0, W, H)

  const cx = W / 2, cy = H / 2
  const barCount = 18
  const barW = 8
  const gap   = 6
  const totalW = barCount * (barW + gap)
  const startX = cx - totalW / 2

  // Waveform bars
  for (let i = 0; i < barCount; i++) {
    const phase = t * 3 + i * 0.55
    const h = 20 + Math.abs(Math.sin(phase) * 80)
    const x = startX + i * (barW + gap)
    const alpha = 0.4 + Math.abs(Math.sin(phase)) * 0.5

    const barGrad = ctx.createLinearGradient(x, cy - h / 2, x, cy + h / 2)
    barGrad.addColorStop(0,   `rgba(165, 100, 235, ${alpha * 0.4})`)
    barGrad.addColorStop(0.5, `rgba(165, 100, 235, ${alpha})`)
    barGrad.addColorStop(1,   `rgba(165, 100, 235, ${alpha * 0.4})`)

    ctx.fillStyle = barGrad
    ctx.beginPath()
    ctx.roundRect(x, cy - h / 2, barW, h, 3)
    ctx.fill()
  }

  // Step counter
  if (total > 0) {
    const pct = step / total
    const label = `${step} / ${total}`
    ctx.font = '700 22px Orbitron, monospace'
    ctx.fillStyle = 'rgba(215, 185, 255, 0.9)'
    ctx.textAlign = 'center'
    ctx.fillText(label, cx, cy + 130)

    // Mini arc progress
    const arcR = 55
    ctx.beginPath()
    ctx.arc(cx, cy - 30, arcR, -Math.PI / 2, -Math.PI / 2 + pct * Math.PI * 2)
    ctx.strokeStyle = 'rgba(165, 100, 235, 0.7)'
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.stroke()

    ctx.beginPath()
    ctx.arc(cx, cy - 30, arcR, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(140, 80, 255, 0.12)'
    ctx.lineWidth = 3
    ctx.lineCap = 'butt'
    ctx.stroke()
  }

  ctx.font = '700 9px Orbitron, monospace'
  ctx.fillStyle = 'rgba(165, 100, 235, 0.5)'
  ctx.textAlign = 'center'
  ctx.fillText('AI SAMPLING', cx, cy + 155)
}
