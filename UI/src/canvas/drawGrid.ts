import { C } from '../constants'

export function drawGrid(ctx: CanvasRenderingContext2D, W: number, H: number): void {
  const gs = Math.min(W, H) / 38
  ctx.strokeStyle = C.gridLine
  ctx.lineWidth = 0.55
  for (let x = 0; x < W; x += gs) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
  }
  for (let y = 0; y < H; y += gs) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
  }
  ctx.fillStyle = C.gridDot
  for (let x = 0; x < W; x += gs) {
    for (let y = 0; y < H; y += gs) {
      ctx.beginPath(); ctx.arc(x, y, 0.75, 0, Math.PI * 2); ctx.fill()
    }
  }
}
