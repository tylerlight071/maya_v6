import type { Dot } from '../types'
import { CANVAS } from '../constants'

export function makeDots(count: number = CANVAS.DOT_COUNT): Dot[] {
  return Array.from({ length: count }, () => ({
    a: Math.random() * Math.PI * 2,
    distFactor: 1.3 + Math.random() * 1.5,
    sz: Math.random() < 0.12 ? 1.9 : 0.85,
    spd: (Math.random() - 0.5) * 0.00014,
    op: 0.25 + Math.random() * 0.5,
  }))
}

export function glowArc(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, R: number,
  a0: number, a1: number,
  r: number, g: number, b: number,
  alpha: number, lw: number, blur: number,
): void {
  ctx.save()
  ctx.shadowColor = `rgb(${r},${g},${b})`
  ctx.shadowBlur = blur
  ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`
  ctx.lineWidth = lw
  ctx.beginPath(); ctx.arc(cx, cy, R, a0, a1); ctx.stroke()
  ctx.shadowBlur = blur * 0.25
  ctx.strokeStyle = `rgba(230,210,255,${alpha * 0.5})`
  ctx.lineWidth = lw * 0.3
  ctx.beginPath(); ctx.arc(cx, cy, R, a0, a1); ctx.stroke()
  ctx.restore()
}
