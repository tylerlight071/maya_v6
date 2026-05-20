import { C } from '../constants'

export function drawTopBracket(
  ctx: CanvasRenderingContext2D,
  W: number, H: number, cx: number,
): void {
  const bw = W * 0.09, bh = H * 0.024, notch = W * 0.014, by = 5
  ctx.strokeStyle = C.bracketTop
  ctx.lineWidth = 1.4
  ctx.beginPath()
  ctx.moveTo(cx - bw, by + bh); ctx.lineTo(cx - bw, by); ctx.lineTo(cx - notch, by)
  ctx.moveTo(cx - notch, by); ctx.lineTo(cx - 7, by - 6); ctx.lineTo(cx + 7, by - 6); ctx.lineTo(cx + notch, by)
  ctx.moveTo(cx + notch, by); ctx.lineTo(cx + bw, by); ctx.lineTo(cx + bw, by + bh)
  ctx.stroke()

  const ts = W / 52
  ctx.strokeStyle = C.bracketRuler
  ctx.lineWidth = 0.7
  for (let x = ts; x < W; x += ts) {
    const big = Math.round(x / ts) % 5 === 0
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, big ? 9 : 5); ctx.stroke()
  }
  for (let y = ts; y < H; y += ts) {
    const big = Math.round(y / ts) % 5 === 0
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(big ? 9 : 5, y); ctx.stroke()
  }
}

export function drawCornerBrackets(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, R: number,
): void {
  const bS = R * 1.55, arm = R * 0.2
  ctx.strokeStyle = C.cornerStroke
  ctx.lineWidth = 1.1
  ;([
    [cx - bS, cy - bS,  1,  1],
    [cx + bS, cy - bS, -1,  1],
    [cx - bS, cy + bS,  1, -1],
    [cx + bS, cy + bS, -1, -1],
  ] as [number, number, number, number][]).forEach(([x, y, sx, sy]) => {
    ctx.beginPath()
    ctx.moveTo(x + sx * arm, y); ctx.lineTo(x, y); ctx.lineTo(x, y + sy * arm)
    ctx.stroke()
  })
  ;([
    [cx,      cy - bS, 'h'],
    [cx,      cy + bS, 'h'],
    [cx - bS, cy,      'v'],
    [cx + bS, cy,      'v'],
  ] as [number, number, string][]).forEach(([tx, ty, dir]) => {
    ctx.strokeStyle = C.cornerMid
    ctx.lineWidth = 1
    ctx.beginPath()
    if (dir === 'h') { ctx.moveTo(tx - 5, ty); ctx.lineTo(tx + 5, ty) }
    else             { ctx.moveTo(tx, ty - 5); ctx.lineTo(tx, ty + 5) }
    ctx.stroke()
    ctx.fillStyle = C.cornerDot
    ctx.beginPath(); ctx.arc(tx, ty, 1.2, 0, Math.PI * 2); ctx.fill()
  })
}
