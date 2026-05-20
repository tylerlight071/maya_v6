import type { MayaState } from '../types'
import { C } from '../constants'
import { glowArc } from './helpers'

interface RingConfig {
  r: number
  speed: number
  dashCount: number
  rgba: [number, number, number, number]
  lw: number
}

function drawInnerRings(ctx: CanvasRenderingContext2D, cx: number, cy: number, R: number, t: number, mayaState: MayaState): void {
  // Speed up rings when active, pulse opacity when speaking
  const speedMult = mayaState === 'thinking'  ? 3.5
                  : mayaState === 'listening' ? 2.0
                  : mayaState === 'speaking'  ? 1.5
                  : 1.0
  const opacityMod = mayaState === 'speaking'
    ? 0.6 + Math.sin(t * 0.004) * 0.4
    : 1.0
  const rings: RingConfig[] = [
    { r: R * 0.68, speed:  0.00032 * speedMult, dashCount: 12, rgba: [150, 90, 255, 0.28 * opacityMod], lw: 0.7 },
    { r: R * 0.50, speed: -0.00052 * speedMult, dashCount:  8, rgba: [180, 120, 255, 0.2  * opacityMod], lw: 0.6 },
    { r: R * 0.33, speed:  0.00078 * speedMult, dashCount:  6, rgba: [130,  70, 230, 0.26 * opacityMod], lw: 0.65 },
  ]
  rings.forEach(({ r, speed, dashCount, rgba, lw }) => {
    const offset = t * speed
    const dashArc = (Math.PI * 2) / dashCount
    ctx.strokeStyle = `rgba(${rgba.join(',')})`
    ctx.lineWidth = lw
    for (let i = 0; i < dashCount; i++) {
      const a0 = offset + i * dashArc
      ctx.beginPath()
      ctx.arc(cx, cy, r, a0, a0 + dashArc * 0.62)
      ctx.stroke()
    }
  })
}

function drawScanSweep(ctx: CanvasRenderingContext2D, cx: number, cy: number, R: number, t: number, speedMult = 1.0): void {
  const tip = (t * 0.00042 * speedMult) % (Math.PI * 2)
  const sweepArc = Math.PI * 0.5
  const innerR = R * 0.94

  for (let i = 0; i < 28; i++) {
    const frac = i / 28
    const a0 = tip - sweepArc * (1 - frac)
    const a1 = tip - sweepArc * (1 - (i + 1) / 28)
    ctx.save()
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.arc(cx, cy, innerR, a0, a1)
    ctx.closePath()
    ctx.fillStyle = `rgba(140, 70, 255, ${0.018 * frac * frac})`
    ctx.fill()
    ctx.restore()
  }

  ctx.save()
  ctx.shadowColor = 'rgba(180, 100, 255, 0.8)'
  ctx.shadowBlur = 10
  ctx.beginPath()
  ctx.moveTo(cx, cy)
  ctx.lineTo(cx + Math.cos(tip) * innerR, cy + Math.sin(tip) * innerR)
  ctx.strokeStyle = 'rgba(195, 140, 255, 0.42)'
  ctx.lineWidth = 0.75
  ctx.stroke()
  ctx.beginPath()
  ctx.arc(cx + Math.cos(tip) * R, cy + Math.sin(tip) * R, 2.2, 0, Math.PI * 2)
  ctx.fillStyle = 'rgba(225, 190, 255, 0.88)'
  ctx.shadowBlur = 6
  ctx.fill()
  ctx.restore()
}

export function drawOrb(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, R: number, t: number,
  mayaState: MayaState = 'idle',
  postFade = 1.0,
  dynamicOnly = false,   // true during outro — skip base elements already drawn by drawIntro
): void {
  const ts = t * 0.001
  // Breathing: faster + wider when listening, very slow when speaking
  const breathSpeed = mayaState === 'listening' ? 0.0022
                    : mayaState === 'speaking'  ? 0.0003
                    : 0.00065
  const breathAmp   = mayaState === 'listening' ? 0.032 : 0.013
  const breath = 1 + Math.sin(t * breathSpeed) * breathAmp

  if (dynamicOnly) {
    // Skip base elements (handled by drawIntro during outro), only draw dynamic ones
    drawInnerRings(ctx, cx, cy, R, t, mayaState)
    if (postFade > 0 && mayaState !== 'listening' && mayaState !== 'speaking') {
      ctx.globalAlpha = postFade
      drawScanSweep(ctx, cx, cy, R, t, mayaState === 'thinking' ? 4.0 : 1.0)
      ctx.globalAlpha = 1
    }
    if (postFade > 0) {
      const a1 = ts * 0.38
      glowArc(ctx, cx, cy, R, a1, a1 + Math.PI * 1.52, 118, 48, 255, 0.72 * postFade, 3.2, 20)
      const a2 = -ts * 0.22
      glowArc(ctx, cx, cy, R, a2, a2 + Math.PI * 0.72, 88, 28, 200, 0.38 * postFade, 1.8, 10)
      glowArc(ctx, cx, cy, R, a1 + Math.PI * 1.52 - 0.22, a1 + Math.PI * 1.52 + 0.07, 212, 175, 255, 0.9 * postFade, 2.5, 8)
    }
    return
  }

  // Diffuse outer glow (breathes)
  const g1 = ctx.createRadialGradient(cx, cy, R * 0.6, cx, cy, R * 2.6 * breath)
  g1.addColorStop(0,    C.outerGlow0)
  g1.addColorStop(0.45, C.outerGlow1)
  g1.addColorStop(1,    'rgba(0,0,0,0)')
  ctx.fillStyle = g1
  ctx.beginPath(); ctx.arc(cx, cy, R * 2.6 * breath, 0, Math.PI * 2); ctx.fill()

  // Tighter ring glow
  const g2 = ctx.createRadialGradient(cx, cy, R * 0.8, cx, cy, R * 1.65)
  g2.addColorStop(0,   C.ringGlow0)
  g2.addColorStop(0.5, C.ringGlow1)
  g2.addColorStop(1,   'rgba(8,0,55,0)')
  ctx.fillStyle = g2
  ctx.beginPath(); ctx.arc(cx, cy, R * 1.65, 0, Math.PI * 2); ctx.fill()

  // Dark center
  const bc = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.975)
  bc.addColorStop(0,    C.centerDark0)
  bc.addColorStop(0.72, C.centerDark1)
  bc.addColorStop(1,    C.centerDark2)
  ctx.fillStyle = bc
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.975, 0, Math.PI * 2); ctx.fill()

  drawInnerRings(ctx, cx, cy, R, t, mayaState)

  // Scan sweep + rotating arcs fade in post-intro
  if (postFade > 0) {
    if (mayaState !== 'listening' && mayaState !== 'speaking') {
      const sweepSpeed = mayaState === 'thinking' ? 4.0 : 1.0
      ctx.globalAlpha = postFade
      drawScanSweep(ctx, cx, cy, R, t, sweepSpeed)
      ctx.globalAlpha = 1
    }

    const a1 = ts * 0.38
    glowArc(ctx, cx, cy, R, a1, a1 + Math.PI * 1.52, 118, 48, 255, 0.72 * postFade, 3.2, 20)

    const a2 = -ts * 0.22
    glowArc(ctx, cx, cy, R, a2, a2 + Math.PI * 0.72, 88, 28, 200, 0.38 * postFade, 1.8, 10)

    // Bright arc tail
    glowArc(ctx, cx, cy, R, a1 + Math.PI * 1.52 - 0.22, a1 + Math.PI * 1.52 + 0.07, 212, 175, 255, 0.9 * postFade, 2.5, 8)
  }

  // Faint outer ring
  ctx.save()
  ctx.strokeStyle = C.faintRing
  ctx.lineWidth = 0.9
  ctx.beginPath(); ctx.arc(cx, cy, R + 5, 0, Math.PI * 2); ctx.stroke()
  ctx.restore()

  // Inner bright ring
  ctx.save()
  ctx.shadowColor = C.innerRingShadow
  ctx.shadowBlur = 6
  ctx.strokeStyle = C.innerRing
  ctx.lineWidth = 1.3
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.985, 0, Math.PI * 2); ctx.stroke()
  ctx.restore()

  // MAYA text — Orbitron, per-character for even visual spacing
  const word = 'MAYA'
  const fs = R * 0.225
  ctx.font = `500 ${fs}px 'Orbitron', sans-serif`
  ctx.textBaseline = 'middle'
  ctx.textAlign = 'left'
  const gap = fs * 0.14
  const widths = word.split('').map(ch => ctx.measureText(ch).width)
  const totalW = widths.reduce((s, w) => s + w, 0) + gap * (word.length - 1)
  let startX = cx - totalW / 2
  ctx.fillStyle = 'rgba(240, 225, 255, 0.93)'
  ctx.save()
  ctx.shadowColor = 'rgba(200, 150, 255, 0.6)'
  ctx.shadowBlur = 16
  for (let i = 0; i < word.length; i++) {
    ctx.fillText(word[i], startX, cy)
    startX += widths[i] + gap
  }
  ctx.restore()
}
