import { useRef, useEffect, useState } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import { SYSTEM_MONITOR_WS } from '../constants'
import styles from './SystemMonitorWidget.module.css'

interface Stats {
  cpu: number; ram_used: number; ram_total: number
  gpu: number; vram_used: number; vram_total: number; gpu_temp: number
  // Second GPU — optional, only present when a second NVIDIA card is detected
  gpu2?: number; vram2_used?: number; vram2_total?: number; gpu2_temp?: number
}

interface Props { visible: boolean; isAway?: boolean }

function pct(used: number, total: number) {
  return total > 0 ? Math.min(100, Math.round(used / total * 100)) : 0
}

function Bar({ label, value, sub }: { label: string; value: number; sub: string }) {
  const colour = value > 90 ? '#ff5555' : value > 70 ? '#ffaa33' : 'rgba(140,80,255,0.85)'
  return (
    <div className={styles.metric}>
      <div className={styles.metricHeader}>
        <span className={styles.metricLabel}>{label}</span>
        <span className={styles.metricSub}>{sub}</span>
      </div>
      <div className={styles.barTrack}>
        <div className={styles.barFill} style={{ width: `${value}%`, background: colour }} />
      </div>
    </div>
  )
}

export default function SystemMonitorWidget({ visible, isAway = false }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)
  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-monitor')

  const [stats, setStats] = useState<Stats | null>(null)
  const [wsOk,  setWsOk]  = useState(false)

  useEffect(() => { if (visible) initTopLeft(window.innerWidth - 300, 65) }, [visible, initTopLeft])

  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000

    function connect() {
      const ws = new WebSocket(SYSTEM_MONITOR_WS)
      ws.onopen    = () => { setWsOk(true); backoff = 2000 }
      ws.onclose   = () => { setWsOk(false); backoff = Math.min(backoff * 1.5, 30_000); reconnect = setTimeout(connect, backoff) }
      ws.onerror   = () => ws.close()
      ws.onmessage = e => {
        try { const m = JSON.parse(e.data); if (m.type === 'monitor_update') setStats(m) } catch { /* */ }
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible])

  const cpuPct  = stats?.cpu  ?? 0
  const ramPct  = stats ? pct(stats.ram_used,  stats.ram_total)  : 0
  const gpuPct  = stats?.gpu  ?? 0
  const vramPct = stats ? pct(stats.vram_used, stats.vram_total) : 0

  return (
    <div
      ref={panelRef}
      className={`${styles.panel} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{ ...panelStyle, ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}) }}
    >
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <span className={`${styles.dot} ${wsOk ? styles.dotLive : ''}`} />
        <span className={styles.title}>SYSTEM</span>
        {stats && <span className={styles.temp}>{stats.gpu_temp}°C{stats.gpu2_temp ? ` / ${stats.gpu2_temp}°C` : ''}</span>}
      </div>

      {stats ? (
        <div className={styles.body}>
          <Bar label="CPU"  value={cpuPct}  sub={`${stats.cpu}%`} />
          <Bar label="RAM"  value={ramPct}  sub={`${stats.ram_used} / ${stats.ram_total} GB`} />
          <Bar label="GPU1" value={gpuPct}  sub={`${stats.gpu}%`} />
          <Bar label="VRAM1" value={vramPct} sub={`${stats.vram_used} / ${stats.vram_total} GB`} />
          {stats.vram2_total ? (
            <>
              <Bar label="GPU2"  value={stats.gpu2  ?? 0} sub={`${stats.gpu2 ?? 0}%`} />
              <Bar label="VRAM2" value={pct(stats.vram2_used ?? 0, stats.vram2_total)}
                   sub={`${stats.vram2_used ?? 0} / ${stats.vram2_total} GB`} />
            </>
          ) : null}
        </div>
      ) : (
        <div className={styles.loading}>connecting...</div>
      )}
    </div>
  )
}
