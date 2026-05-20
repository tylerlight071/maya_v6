import { C } from '../constants'

export function drawBackground(
  ctx: CanvasRenderingContext2D,
  W: number, H: number,
  cx: number, cy: number,
): void {
  const bg = ctx.createRadialGradient(cx, cy * 0.85, 0, cx, H * 0.5, Math.max(W, H) * 0.85)
  bg.addColorStop(0, C.bgCenter)
  bg.addColorStop(1, C.bgEdge)
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, W, H)
}
