import { useRef, useEffect, useState, useCallback } from 'react'
import type {} from '../types/electron'
import { TIMING } from '../constants'
import styles from './DevLogOverlay.module.css'

interface DevLogOverlayProps {
  active:     boolean
  showScreen: boolean
  onEnd:      () => void
}

type Phase = 'idle' | 'starting' | 'recording' | 'stopping' | 'error'

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
  const [phase,    setPhase]    = useState<Phase>('idle')
  const [elapsed,  setElapsed]  = useState(0)
  const [saved,    setSaved]    = useState<string | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const recorderRef      = useRef<MediaRecorder | null>(null)
  const chunksRef        = useRef<Blob[]>([])
  const timerRef         = useRef<ReturnType<typeof setInterval> | null>(null)
  const rafRef           = useRef<number>(0)
  const canvasRef        = useRef<HTMLCanvasElement | null>(null)
  const webcamRef        = useRef<HTMLVideoElement | null>(null)
  const screenRef        = useRef<HTMLVideoElement | null>(null)
  const streamsRef       = useRef<MediaStream[]>([])
  const endRequestedRef  = useRef(false)  // true when END LOG button was clicked
  const stopTimeoutRef   = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Keep onEnd stable in a ref so onstop closure always has the latest version
  const onEndRef = useRef(onEnd)
  useEffect(() => { onEndRef.current = onEnd }, [onEnd])

  // ── Cleanup media resources ────────────────────────────────────────────────
  const cleanup = useCallback(() => {
    if (timerRef.current)    { clearInterval(timerRef.current); timerRef.current = null }
    if (stopTimeoutRef.current) { clearTimeout(stopTimeoutRef.current); stopTimeoutRef.current = null }
    cancelAnimationFrame(rafRef.current)
    streamsRef.current.forEach(s => s.getTracks().forEach(t => t.stop()))
    streamsRef.current = []
    canvasRef.current  = null
    webcamRef.current  = null
    screenRef.current  = null
  }, [])

  // ── Start recording ────────────────────────────────────────────────────────
  const startRecording = useCallback(async (withScreen: boolean) => {
    setPhase('starting')
    setSaved(null)
    setErrorMsg(null)
    chunksRef.current = []
    endRequestedRef.current = false

    try {
      const camStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, frameRate: 30, facingMode: 'user' },
        audio: { echoCancellation: false, noiseSuppression: false },
      })
      streamsRef.current.push(camStream)

      let videoTrack: MediaStreamTrack
      const audioTrack = camStream.getAudioTracks()[0]

      if (withScreen) {
        const screenStream = await (navigator.mediaDevices as MediaDevices & {
          getDisplayMedia: (c: object) => Promise<MediaStream>
        }).getDisplayMedia({ video: { frameRate: 30 }, audio: false })
        streamsRef.current.push(screenStream)

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

        const ctx  = canvas.getContext('2d')!
        const pipW = 320, pipH = 180
        const pipX = 1280 - pipW - 12
        const pipY = 720  - pipH - 12

        function drawFrame() {
          ctx.drawImage(screenVid, 0, 0, 1280, 720)
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

      const tracks: MediaStreamTrack[] = [videoTrack]
      if (audioTrack) tracks.push(audioTrack)
      const finalStream = new MediaStream(tracks)

      const mime     = bestMime()
      const recorder = new MediaRecorder(finalStream, mime ? { mimeType: mime } : undefined)

      recorder.ondataavailable = e => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        // Cancel the safety timeout — onstop fired normally
        if (stopTimeoutRef.current) {
          clearTimeout(stopTimeoutRef.current)
          stopTimeoutRef.current = null
        }

        setPhase('stopping')
        let savedPath: string | null = null

        try {
          const ext   = (mime || '').includes('mp4') ? 'mp4' : 'webm'
          const fname = `Dev Log ${dateStamp()}.${ext}`
          const blob  = new Blob(chunksRef.current, { type: mime || 'video/webm' })

          if (blob.size === 0) {
            console.warn('[devlog] empty recording — no data captured')
            setErrorMsg('Nothing was captured — check camera/mic permissions')
            setPhase('error')
            cleanup()
            if (endRequestedRef.current) {
              endRequestedRef.current = false
              setTimeout(() => onEndRef.current(), TIMING.DEVLOG_BADGE_MS + 500)
            }
            return
          }

          const arrayBuf = await blob.arrayBuffer()
          const uint8    = new Uint8Array(arrayBuf)
          const maya = window.mayaElectron
          savedPath = await maya?.saveRecording?.(uint8, fname) ?? null
          console.log('[devlog] saved to:', savedPath ?? '(saveRecording unavailable)')
          setSaved(savedPath ?? fname)
        } catch (e) {
          console.error('[devlog] save error:', e)
          setErrorMsg('Failed to save recording')
          setPhase('error')
        }

        if (savedPath !== null || errorMsg === null) setPhase('idle')
        setElapsed(0)
        cleanup()

        // Call onEnd after showing SAVED badge — this resets active to false
        if (endRequestedRef.current) {
          endRequestedRef.current = false
          setTimeout(() => onEndRef.current(), TIMING.DEVLOG_BADGE_MS)
        }
      }

      recorder.start(1000)
      recorderRef.current = recorder
      timerRef.current    = setInterval(() => setElapsed(s => s + 1), 1000)
      setPhase('recording')

    } catch (e) {
      console.error('[devlog] start error:', e)
      const msg = e instanceof Error ? e.message : String(e)
      setErrorMsg(`Could not start recording: ${msg}`)
      setPhase('error')
      cleanup()
    }
  }, [cleanup]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Stop recording ─────────────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    cancelAnimationFrame(rafRef.current)

    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop()

      // Safety timeout: if onstop hasn't fired within 10s, force-close
      stopTimeoutRef.current = setTimeout(() => {
        console.error('[devlog] onstop timeout — forcing close')
        setPhase('idle')
        setElapsed(0)
        cleanup()
        if (endRequestedRef.current) {
          endRequestedRef.current = false
          onEndRef.current()
        }
      }, TIMING.DEVLOG_STOP_MS)
    }
    recorderRef.current = null
  }, [cleanup])

  // ── React to active changes ────────────────────────────────────────────────
  useEffect(() => {
    if (active && phase === 'idle') {
      startRecording(showScreen)
    } else if (!active && (phase === 'recording' || phase === 'starting')) {
      // Voice command ended dev log — stop without triggering onEnd (already handled by App.tsx)
      stopRecording()
    }
  }, [active]) // eslint-disable-line react-hooks/exhaustive-deps

  // When showScreen toggles mid-session, restart with new mode
  const prevShowScreen = useRef(showScreen)
  useEffect(() => {
    if (!active || phase !== 'recording') { prevShowScreen.current = showScreen; return }
    if (showScreen !== prevShowScreen.current) {
      prevShowScreen.current = showScreen
      // Restart: stop current, wait for onstop to settle, then start new
      const wasEndRequested = endRequestedRef.current
      endRequestedRef.current = false   // prevent onEnd firing during restart
      stopRecording()
      setTimeout(() => {
        endRequestedRef.current = wasEndRequested
        if (active) startRecording(showScreen)
      }, 500)
    }
  }, [showScreen]) // eslint-disable-line react-hooks/exhaustive-deps

  // Full cleanup on unmount
  useEffect(() => () => {
    stopRecording()
    cleanup()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Button handler — marks end requested, stops recording ─────────────────
  const handleEnd = useCallback(() => {
    if (phase === 'error' || phase === 'idle') {
      // No active recording — just close immediately
      onEndRef.current()
      return
    }
    endRequestedRef.current = true
    stopRecording()
    // onEnd() will be called from recorder.onstop after saving
  }, [phase, stopRecording])

  if (!active && (phase === 'idle' || phase === 'error') && !saved) return null

  const isRecording = phase === 'recording'
  const isStopping  = phase === 'stopping'
  const isError     = phase === 'error'

  return (
    <div className={styles.bar}>
      <div className={styles.left}>
        <span className={`${styles.recDot} ${isRecording ? styles.recDotActive : isError ? styles.recDotError : ''}`} />
        <span className={styles.recLabel}>
          {isStopping ? 'SAVING...' : isError ? 'ERROR' : 'REC'}
        </span>
        {isRecording && <span className={styles.timer}>{fmtTime(elapsed)}</span>}
        {isError && errorMsg && <span className={styles.errorMsg}>{errorMsg}</span>}
      </div>

      <span className={styles.title}>DEV LOG</span>

      <div className={styles.right}>
        {showScreen && isRecording && (
          <span className={styles.screenBadge}>◈ SCREEN</span>
        )}
        {saved && (
          <span className={styles.savedBadge} title={saved}>✓ SAVED</span>
        )}
        {!isStopping && (
          <button
            className={styles.endBtn}
            onClick={handleEnd}
            onMouseDown={e => e.stopPropagation()}
          >
            {isError ? '✕ CLOSE' : '■ END LOG'}
          </button>
        )}
      </div>
    </div>
  )
}
