import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import type { Dot, MayaState } from '../types'
import { CANVAS, TIMING } from '../constants'
import { makeDots } from '../canvas/helpers'
import { drawBackground } from '../canvas/drawBackground'
import { drawGrid } from '../canvas/drawGrid'
import { drawTopBracket, drawCornerBrackets } from '../canvas/drawBrackets'
import { drawDots } from '../canvas/drawDots'
import { drawOrb } from '../canvas/drawOrb'
import { drawWaveformRing } from '../canvas/drawWaveform'
import { drawIntro } from '../canvas/drawIntro'

type AnimPhase = 'intro' | 'normal' | 'outro' | 'away' | 'reintro'

export function useMAYACanvas(
  canvasRef:  RefObject<HTMLCanvasElement | null>,
  mayaState:  MayaState,
  presence:   'present' | 'away' = 'present',
): void {
  const dotsRef      = useRef<Dot[]>([])
  const rafRef       = useRef<number>(0)
  const startRef     = useRef<number | null>(null)
  const stateRef     = useRef<MayaState>(mayaState)
  const presenceRef  = useRef<'present' | 'away'>(presence)
  const phaseRef     = useRef<AnimPhase>('intro')
  const phaseStart   = useRef<number>(0)

  useEffect(() => { stateRef.current   = mayaState  })
  useEffect(() => { presenceRef.current = presence  })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    function resize(): void {
      if (!canvas) return
      canvas.width  = window.innerWidth
      canvas.height = window.innerHeight
      dotsRef.current = makeDots(CANVAS.DOT_COUNT)
    }
    resize()
    window.addEventListener('resize', resize)

    function startLoop(): void {
      function frame(ts: number): void {
        if (!canvas || !ctx) return
        if (startRef.current === null) startRef.current = ts
        const t   = ts - startRef.current
        const W   = canvas.width, H = canvas.height
        const cx  = W / 2
        const cy  = H * CANVAS.ORB_Y_FACTOR
        const R   = Math.min(W, H) * CANVAS.RADIUS_FACTOR
        const ms  = stateRef.current
        const prs = presenceRef.current

        // ── Phase transitions ──────────────────────────────────────────────
        let phase = phaseRef.current
        const pe  = t - phaseStart.current  // elapsed in current phase

        if (phase === 'intro'   && t  >= TIMING.INTRO_MS) { phase = 'normal';  phaseRef.current = phase; phaseStart.current = t }
        if (phase === 'reintro' && pe >= TIMING.INTRO_MS) { phase = 'normal';  phaseRef.current = phase; phaseStart.current = t }
        if (phase === 'outro'   && pe >= TIMING.INTRO_MS) { phase = 'away';    phaseRef.current = phase; phaseStart.current = t }
        if (phase === 'normal'  && prs === 'away')         { phase = 'outro';   phaseRef.current = phase; phaseStart.current = t }
        if (phase === 'away'    && prs === 'present')      { phase = 'reintro'; phaseRef.current = phase; phaseStart.current = t }

        const phaseElapsed = t - phaseStart.current

        // ── Background (always) ───────────────────────────────────────────
        ctx.clearRect(0, 0, W, H)
        drawBackground(ctx, W, H, cx, cy)

        // In 'away' phase nothing animates — throttle to 5fps to save CPU/GPU
        if (phase === 'away') {
          rafRef.current = setTimeout(() => {
            rafRef.current = requestAnimationFrame(frame)
          }, 200) as unknown as number
          return
        }

        // ── Grid + top bracket with phase-aware alpha ──────────────────────
        let gridAlpha = 1.0
        if      (phase === 'intro')   gridAlpha = Math.min(1, t / 500)
        else if (phase === 'reintro') gridAlpha = Math.min(1, phaseElapsed / 500)
        else if (phase === 'outro')   gridAlpha = Math.max(0, 1 - phaseElapsed / TIMING.INTRO_MS)
        else if (phase === 'away')    gridAlpha = 0

        ctx.globalAlpha = gridAlpha
        drawGrid(ctx, W, H)
        drawTopBracket(ctx, W, H, cx)
        ctx.globalAlpha = 1

        // ── Main drawing ───────────────────────────────────────────────────
        if (phase === 'intro' || phase === 'reintro') {
          const introT = phase === 'intro' ? t : phaseElapsed
          drawIntro(ctx, cx, cy, R, introT, TIMING.INTRO_MS)

        } else if (phase === 'away') {
          // Silent — just background

        } else if (phase === 'outro') {
          const outroProgress = Math.min(1, phaseElapsed / TIMING.INTRO_MS)
          const fade          = Math.max(0, 1 - outroProgress)
          const reverseT      = TIMING.INTRO_MS * (1 - outroProgress)

          // drawIntro (reversed) handles: base orb, brackets, circle, glow, text
          drawIntro(ctx, cx, cy, R, reverseT, TIMING.INTRO_MS)

          // Fade out ONLY elements that aren't drawn by drawIntro
          if (fade > 0) {
            ctx.globalAlpha = fade
            drawDots(ctx, cx, cy, R, dotsRef.current, t)
            ctx.globalAlpha = 1
            drawOrb(ctx, cx, cy, R, t, ms, fade, true)   // dynamicOnly — no base overlap
            drawWaveformRing(ctx, cx, cy, R, t, ms, fade)
          }

        } else {
          // Normal — fade dynamic elements in over 800ms post-intro
          const postFade = Math.min(1, phaseElapsed / 800)
          drawCornerBrackets(ctx, cx, cy, R)
          drawDots(ctx, cx, cy, R, dotsRef.current, t)
          drawOrb(ctx, cx, cy, R, t, ms, postFade)
          drawWaveformRing(ctx, cx, cy, R, t, ms, postFade)
        }

        rafRef.current = requestAnimationFrame(frame)
      }
      rafRef.current = requestAnimationFrame(frame)
    }

    if (document.fonts?.ready) {
      document.fonts.ready.then(startLoop)
    } else {
      startLoop()
    }

    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(rafRef.current)
      clearTimeout(rafRef.current as unknown as ReturnType<typeof setTimeout>)
    }
  }, [canvasRef])
}
