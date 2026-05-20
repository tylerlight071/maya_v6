import type { MayaState } from '../types'

export function drawWaveformRing(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, R: number, t: number,
  mayaState: MayaState = 'idle',
  postFade = 1.0,
): void {
  if (postFade <= 0) return
  const N = 128

  // Amplitude and colour driven by state
  const maxAmp = mayaState === 'listening' ? R * 0.075
               : mayaState === 'speaking'  ? R * 0.062
               : R * 0.038

  // Speaking: speech-like noise with stronger low-frequency component
  // Listening: higher energy, more uniform
  // Idle/thinking: current subtle behaviour
  const n0 = mayaState === 'speaking' ? 0.50 : 0.38
  const n1 = mayaState === 'speaking' ? 0.30 : 0.32
  const n2 = mayaState === 'speaking' ? 0.20 : 0.30

  const t0 = mayaState === 'listening' ? t * 0.0045 : t * 0.0024
  const t1 = mayaState === 'listening' ? t * 0.0060 : t * 0.0037
  const t2 = mayaState === 'listening' ? t * 0.0025 : t * 0.0014

  // Colour: green tint when listening, brighter purple when speaking
  const [cr, cg, cb] = mayaState === 'listening' ? [120, 220, 160]
                      : mayaState === 'speaking'  ? [210, 140, 255]
                      : [185, 125, 255]

  for (let i = 0; i < N; i++) {
    const angle = (i / N) * Math.PI * 2
    const n = Math.sin(angle * 7  + t0) * n0
            + Math.sin(angle * 13 + t1) * n1
            + Math.sin(angle * 3  + t2) * n2
    const amp = Math.max(0, n) * maxAmp
    if (amp < 0.6) continue

    const ir = R + 2.5
    const or = R + 2.5 + amp
    ctx.strokeStyle = `rgba(${cr}, ${cg}, ${cb}, ${(0.22 + Math.max(0, n) * 0.55) * postFade})`
    ctx.lineWidth = 1.1
    ctx.beginPath()
    ctx.moveTo(cx + Math.cos(angle) * ir, cy + Math.sin(angle) * ir)
    ctx.lineTo(cx + Math.cos(angle) * or, cy + Math.sin(angle) * or)
    ctx.stroke()
  }
}
