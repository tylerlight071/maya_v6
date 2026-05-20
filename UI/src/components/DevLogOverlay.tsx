import { useRef, useEffect, useState, useCallback } from 'react'
import styles from './DevLogOverlay.module.css'

interface DevLogOverlayProps {
  active:     boolean
  showScreen: boolean
  onEnd:      () => void
}

type Phase = 'idle' | 'starting' | 'recording' | 'stopping'

function fmtTime(s: number): string {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = s % 60
  if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`
  return `${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`
}

function dateStamp(): string {
  const d = new Date()
  const pad = (n: number) => String(n).padStart(2,'0')
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ` +
         `${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`
}

// Pick the first supported mimeType
function bestMime(): string {
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
    'video/mp4',
  ]
  return candidates.find(m => MediaRecorder.isTypeSupported(m)) ?? ''
}

export default function DevLogOverlay({ active, showScreen, onEnd }: DevLogOverlayProps) {
  const [phase,   setPhase]   = useState<Phase>('idle')
  const [elapsed, setElapsed] = useState(0)
  const [saved,   setSaved]   = useState<string | null>(null)

  const recorderRef  = useRef<MediaRecorder | null>(null)
  const chunksRef    = useRef<Blob[]>([])
  const timerRef     = useRef<ReturnType<typeof setInterval> | null>(null)
  const rafRef       = useRef<number>(0)
  const canvasRef    = useRef<HTMLCanvasElement | null>(null)
  const webcamRef    = useRef<HTMLVideoElement | null>(null)
  const screenRef    = useRef<HTMLVideoElement | null>(null)
  const streamsRef   = useRef<MediaStream[]>([])

  // ── Cleanup helper ─────────────────────────────────────────────────────────
  const cleanup = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    cancelAnimationFrame(rafRef.current)
    streamsRef.current.forEach(s => s.getTracks().forEach(t => t.stop()))
    streamsRef.current = []
    if (canvasRef.current) { canvasRef.current = null }
    webcamRef.current = null
    screenRef.current = null
  }, [])

  // ── Start recording ────────────────────────────────────────────────────────
  const startRecording = useCallback(async (withScreen: boolean) => {
    setPhase('starting')
    setSaved(null)
    chunksRef.current = []

    try {
      // 1. Webcam + mic
      const camStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, frameRate: 30, facingMode: 'user' },
        audio: { echoCancellation: false, noiseSuppression: false },
      })
      streamsRef.current.push(camStream)

      let videoTrack: MediaStreamTrack
      const audioTrack = camStream.getAudioTracks()[0]

      if (withScreen) {
        // 2. Screen capture
        const screenStream = await (navigator.mediaDevices as MediaDevices & {
          getDisplayMedia: (c: object) => Promise<MediaStream>
        }).getDisplayMedia({ video: { frameRate: 30 }, audio: false })
        streamsRef.current.push(screenStream)

        // 3. Canvas compositor: screen full-frame + webcam PIP (bottom-right)
        const canvas  = document.createElement('canvas')
        canvas.width  = 1280
        canvas.height = 720
        canvasRef.current = canvas

        const screenVid = document.createElement('video')
        screenVid.srcObject = screenStream
        screenVid.muted     = true
        await screenVid.play()
        screenRef.current = screenVid

        const camVid = document.createElement('video')
        camVid.srcObject = new MediaStream(camStream.getVideoTracks())
        camVid.muted     = true
        await camVid.play()
        webcamRef.current = camVid

        const ctx = canvas.getContext('2d')!
        const pipW = 320, pipH = 180, pipX = 1280 - pipW - 12, pipY = 720 - pipH - 12

        function drawFrame() {
          ctx.drawImage(screenVid, 0, 0, 1280, 720)
          // PIP border
          ctx.fillStyle = 'rgba(140,80,255,0.6)'
          ctx.fillRect(pipX - 2, pipY - 2, pipW + 4, pipH + 4)
          ctx.drawImage(camVid, pipX, pipY, pipW, pipH)
          rafRef.current = requestAnimationFrame(drawFrame)
        }
        rafRef.current = requestAnimationFrame(drawFrame)

        videoTrack = canvas.captureStream(30).getVideoTracks()[0]
      } else {
        videoTrack = camStream.getVideoTracks()[0]
      }

      // 4. Combine tracks into final stream
      const tracks: MediaStreamTrack[] = [videoTrack]
      if (audioTrack) tracks.push(audioTrack)
      const finalStream = new MediaStream(tracks)

      // 5. MediaRecorder
      const mime     = bestMime()
      const recorder = new MediaRecorder(finalStream, mime ? { mimeType: mime } : undefined)
      recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      recorder.onstop = async () => {
        setPhase('stopping')
        try {
          const ext   = (mime || '').includes('mp4') ? 'mp4' : 'webm'
          const fname = `Dev Log ${dateStamp()}.${ext}`
          const blob  = new Blob(chunksRef.current, { type: mime || 'video/webm' })

          if (blob.size === 0) {
            console.warn('[devlog] empty recording — no data captured')
            setPhase('idle'); setElapsed(0); cleanup(); return
          }

          // Convert to Uint8Array before IPC — more reliable than ArrayBuffer across Electron IPC
          const arrayBuf = await blob.arrayBuffer()
          const uint8    = new Uint8Array(arrayBuf)

          const maya = (window as unknown as Record<string, unknown>).mayaElectron as
            { saveRecording?: (buf: Uint8Array, name: string) => Promise<string | null> } | undefined

          const savedPath = await maya?.saveRecording?.(uint8, fname) ?? null
          console.log('[devlog] saved to:', savedPath ?? '(saveRecording unavailable)')
          setSaved(savedPath ?? fname)
        } catch (e) {
          console.error('[devlog] save error:', e)
        }
        setPhase('idle')
        setElapsed(0)
        cleanup()
      }
      recorder.start(1000)
      recorderRef.current = recorder

      // 6. Timer
      timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000)
      setPhase('recording')

    } catch (e) {
      console.error('[devlog] start error:', e)
      cleanup()
      setPhase('idle')
    }
  }, [cleanup])

  // ── Stop recording ─────────────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    cancelAnimationFrame(rafRef.current)
    if (recorderRef.current?.state !== 'inactive') {
      recorderRef.current?.stop()
    }
    recorderRef.current = null
  }, [])

  // ── React to active / showScreen changes ──────────────────────────────────
  useEffect(() => {
    if (active && phase === 'idle') {
      startRecording(showScreen)
    } else if (!active && (phase === 'recording' || phase === 'starting')) {
      stopRecording()
    }
  }, [active])  // eslint-disable-line react-hooks/exhaustive-deps

  // When showScreen toggles mid-session, restart with new mode
  const prevShowScreen = useRef(showScreen)
  useEffect(() => {
    if (!active || phase !== 'recording') { prevShowScreen.current = showScreen; return }
    if (showScreen !== prevShowScreen.current) {
      prevShowScreen.current = showScreen
      stopRecording()
      // Brief pause then restart with screen
      setTimeout(() => startRecording(showScreen), 400)
    }
  }, [showScreen])  // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount
  useEffect(() => () => { stopRecording(); cleanup() }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  const handleEnd = useCallback(() => {
    stopRecording()
    onEnd()
  }, [stopRecording, onEnd])

  if (!active && phase === 'idle') return null

  return (
    <div className={styles.bar}>
      {/* Recording indicator */}
      <div className={styles.left}>
        <span className={`${styles.recDot} ${phase === 'recording' ? styles.recDotActive : ''}`} />
        <span className={styles.recLabel}>{phase === 'stopping' ? 'SAVING...' : 'REC'}</span>
        {phase === 'recording' && (
          <span className={styles.timer}>{fmtTime(elapsed)}</span>
        )}
      </div>

      <span className={styles.title}>DEV LOG</span>

      <div className={styles.right}>
        {showScreen && phase === 'recording' && (
          <span className={styles.screenBadge}>◈ SCREEN</span>
        )}
        {saved && (
          <span className={styles.savedBadge} title={saved}>✓ SAVED</span>
        )}
        {phase !== 'stopping' && (
          <button
            className={styles.endBtn}
            onClick={handleEnd}
            onMouseDown={e => e.stopPropagation()}
          >
            ■ END LOG
          </button>
        )}
      </div>
    </div>
  )
}
