import { useRef, useEffect, useState } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import { ALARM_WS } from '../constants'
import styles from './AlarmWidget.module.css'

interface Alarm { id: string; label: string; fire_at: string }
interface Props  { visible: boolean; isAway?: boolean; fired?: { id: string; label: string } | null }

export default function AlarmWidget({ visible, isAway = false, fired = null }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)
  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-alarm-v1')

  const [alarms,  setAlarms]  = useState<Alarm[]>([])
  const [wsOk,    setWsOk]    = useState(false)
  const [flash,   setFlash]   = useState(false)

  useEffect(() => { if (visible) initTopLeft(Math.round(window.innerWidth / 2 - 180), 140) }, [visible, initTopLeft])

  // Flash when alarm fires
  useEffect(() => {
    if (!fired) return
    setFlash(true)
    const t = setTimeout(() => setFlash(false), 3000)
    return () => clearTimeout(t)
  }, [fired])

  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000
    function connect() {
      const ws = new WebSocket(ALARM_WS)
      ws.onopen    = () => { setWsOk(true); backoff = 2000 }
      ws.onclose   = () => { setWsOk(false); backoff = Math.min(backoff * 1.5, 30_000); reconnect = setTimeout(connect, backoff) }
      ws.onerror   = () => ws.close()
      ws.onmessage = e => {
        try {
          const m = JSON.parse(e.data)
          if (m.type === 'alarms') setAlarms(m.alarms)
        } catch { /* */ }
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible])

  function cancelAlarm(id: string) {
    wsRef.current?.send(JSON.stringify({ type: 'cancel', id }))
  }

  const away = isAway
  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${dragging ? styles.dragging : ''} ${flash ? styles.flash : ''}`}
      style={{
        ...panelStyle,
        ...(away ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
    >
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <span className={styles.title}>ALARMS</span>
        <span className={`${styles.dot} ${wsOk ? styles.dotOk : styles.dotErr}`} />
      </div>

      <div className={styles.body}>
        {alarms.length === 0 ? (
          <p className={styles.empty}>No active alarms</p>
        ) : (
          alarms.map(a => (
            <div key={a.id} className={styles.alarmRow}>
              <div className={styles.alarmInfo}>
                <span className={styles.alarmTime}>{a.fire_at}</span>
                <span className={styles.alarmLabel}>{a.label}</span>
              </div>
              <button
                className={styles.cancelBtn}
                onClick={() => cancelAlarm(a.id)}
                onMouseDown={e => e.stopPropagation()}
                title="Cancel alarm"
              >✕</button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
