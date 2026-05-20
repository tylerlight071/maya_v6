// Startup intro animation — runs once over TIMING.INTRO_MS milliseconds

function easeOut(t: number): number {
  return 1 - Math.pow(1 - t, 3)
}

function easeInOut(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

function phase(t: number, start: number, end: number): number {
  return Math.max(0, Math.min(1, (t - start) / (end - start)))
}

export function drawIntro(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, R: number,
  elapsed: number,
  totalMs: number,
): void {
  const p = elapsed / totalMs   // 0 → 1 over the full intro

  // ── 1. Corner brackets expand from centre ───────────────────────────────
  const bp = easeOut(phase(p, 0.09, 0.45))
  if (bp > 0) _drawBrackets(ctx, cx, cy, R, bp)

  // ── 2a. Horizontal scan line materialises ────────────────────────────────
  const lp = easeOut(phase(p, 0.35, 0.54))
  if (lp > 0 && p < 0.54) {
    ctx.save()
    ctx.shadowColor = 'rgba(215, 180, 255, 0.8)'
    ctx.shadowBlur = 12
    ctx.strokeStyle = `rgba(228, 205, 255, ${lp * 0.92})`
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(cx - R * lp, cy)
    ctx.lineTo(cx + R * lp, cy)
    ctx.stroke()
    ctx.restore()
  }

  // ── 2b. Scan line morphs into circle ─────────────────────────────────────
  const cp = easeInOut(phase(p, 0.54, 0.80))
  if (cp > 0) {
    // Dark centre fills in only once the ellipse is fully circular (cp > 0.92)
    if (cp > 0.92) {
      const fillP = easeOut((cp - 0.92) / 0.08)
      const bc = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.975)
      bc.addColorStop(0,    `rgba(6, 3, 15, ${fillP})`)
      bc.addColorStop(0.72, `rgba(10, 5, 20, ${fillP})`)
      bc.addColorStop(1,    `rgba(13, 6, 24, ${fillP})`)
      ctx.fillStyle = bc
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.975, 0, Math.PI * 2); ctx.fill()
    }

    // Morphing ellipse: y-axis scales from 0 (line) → 1 (circle)
    ctx.save()
    ctx.shadowColor = 'rgba(215, 180, 255, 0.45)'
    ctx.shadowBlur = 6
    ctx.strokeStyle = 'rgba(228, 205, 255, 0.84)'
    ctx.lineWidth = 1.3
    ctx.beginPath()
    const N = 128
    for (let i = 0; i <= N; i++) {
      const θ = (i / N) * Math.PI * 2
      const x = cx + R * Math.cos(θ)
      const y = cy + R * Math.sin(θ) * cp
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.closePath()
    ctx.stroke()

    // Faint outer halo ring
    ctx.strokeStyle = `rgba(122, 52, 228, ${0.18 * cp})`
    ctx.lineWidth = 0.9
    ctx.shadowBlur = 0
    ctx.beginPath(); ctx.arc(cx, cy, R + 5, 0, Math.PI * 2); ctx.stroke()
    ctx.restore()
  }

  // ── 3. Glow blooms outward once circle is formed ─────────────────────────
  const gp = easeOut(phase(p, 0.72, 0.92))
  if (gp > 0) {
    const g1 = ctx.createRadialGradient(cx, cy, R * 0.6, cx, cy, R * 2.6)
    g1.addColorStop(0,    `rgba(85, 20, 240, ${0.14 * gp})`)
    g1.addColorStop(0.45, `rgba(55, 10, 160, ${0.06 * gp})`)
    g1.addColorStop(1,    'rgba(0,0,0,0)')
    ctx.fillStyle = g1
    ctx.beginPath(); ctx.arc(cx, cy, R * 2.6, 0, Math.PI * 2); ctx.fill()

    const g2 = ctx.createRadialGradient(cx, cy, R * 0.8, cx, cy, R * 1.65)
    g2.addColorStop(0,   `rgba(105, 30, 255, ${0.24 * gp})`)
    g2.addColorStop(0.5, `rgba(68, 14, 215, ${0.1 * gp})`)
    g2.addColorStop(1,   'rgba(8, 0, 55, 0)')
    ctx.fillStyle = g2
    ctx.beginPath(); ctx.arc(cx, cy, R * 1.65, 0, Math.PI * 2); ctx.fill()
  }

  // ── 4. MAYA text — letter by letter ──────────────────────────────────────
  const tp = phase(p, 0.80, 1.0)
  if (tp > 0) {
    const word = 'MAYA'
    const fs = R * 0.225
    ctx.font = `500 ${fs}px 'Orbitron', sans-serif`
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'left'
    const gap = fs * 0.14
    const widths = word.split('').map(ch => ctx.measureText(ch).width)
    const totalW = widths.reduce((s, w) => s + w, 0) + gap * (word.length - 1)
    let startX = cx - totalW / 2

    for (let i = 0; i < word.length; i++) {
      // Each letter staggers in 0.07 apart
      const letterP = easeOut(Math.max(0, Math.min(1, (tp - i * 0.07) / 0.35)))
      if (letterP > 0) {
        ctx.save()
        ctx.shadowColor = `rgba(200, 150, 255, ${0.6 * letterP})`
        ctx.shadowBlur = 16 * letterP
        ctx.fillStyle = `rgba(240, 225, 255, ${0.93 * letterP})`
        ctx.fillText(word[i], startX, cy - (1 - letterP) * 8)
        ctx.restore()
      }
      startX += widths[i] + gap
    }
  }
}

function _drawBrackets(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, R: number,
  progress: number,
): void {
  const bS  = R * 1.55 * progress
  const arm = R * 0.2  * progress
  const a   = progress

  ctx.strokeStyle = `rgba(165, 100, 235, ${0.48 * a})`
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
    ctx.strokeStyle = `rgba(165, 100, 235, ${0.36 * a})`
    ctx.lineWidth = 1
    ctx.beginPath()
    if (dir === 'h') { ctx.moveTo(tx - 5, ty); ctx.lineTo(tx + 5, ty) }
    else             { ctx.moveTo(tx, ty - 5); ctx.lineTo(tx, ty + 5) }
    ctx.stroke()
    ctx.fillStyle = `rgba(165, 100, 235, ${0.4 * a})`
    ctx.beginPath(); ctx.arc(tx, ty, 1.2, 0, Math.PI * 2); ctx.fill()
  })
}
