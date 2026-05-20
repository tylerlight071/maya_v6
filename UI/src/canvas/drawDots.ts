import type { Dot } from '../types'
import { C } from '../constants'

export function drawDots(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, R: number,
  dots: Dot[], t: number,
): void {
  dots.forEach(dot => {
    const a = dot.a + dot.spd * t
    const d = R * dot.distFactor
    const x = cx + Math.cos(a) * d
    const y = cy + Math.sin(a) * d
    const pulse = 0.65 + Math.sin(t * 0.0018 + dot.a * 2) * 0.35
    ctx.fillStyle = C.dot(dot.op * pulse)
    ctx.beginPath(); ctx.arc(x, y, dot.sz, 0, Math.PI * 2); ctx.fill()
  })
}
