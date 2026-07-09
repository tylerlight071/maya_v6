import { useRef, useEffect, useState } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import { CALENDAR_WS, wsUrl } from '../constants'
import styles from './CalendarWidget.module.css'

interface CalEvent { id: string; title: string; start: string; end: string; location: string }
interface Props     { visible: boolean; isAway?: boolean }

function fmtTime(iso: string) {
  if (!iso || !iso.includes('T')) return ''
  try { return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }) }
  catch { return '' }
}
function fmtDate(iso: string) {
  try { return new Date(iso).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }) }
  catch { return iso }
}
function groupByDate(events: CalEvent[]): Record<string, CalEvent[]> {
  return events.reduce((acc, e) => {
    const k = e.start.split('T')[0]
    acc[k] = acc[k] ?? []
    acc[k].push(e)
    return acc
  }, {} as Record<string, CalEvent[]>)
}

export default function CalendarWidget({ visible, isAway = false }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)
  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-calendar')

  const [today,     setToday]     = useState<CalEvent[]>([])
  const [upcoming,  setUpcoming]  = useState<CalEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [needsAuth, setNeedsAuth] = useState(false)
  const [wsOk,      setWsOk]      = useState(false)

  useEffect(() => { if (visible) initTopLeft(window.innerWidth - 340, 340) }, [visible, initTopLeft])

  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000
    function connect() {
      const ws = new WebSocket(wsUrl(CALENDAR_WS))
      ws.onopen    = () => { setWsOk(true); backoff = 2000 }
      ws.onclose   = () => { setWsOk(false); backoff = Math.min(backoff * 1.5, 30_000); reconnect = setTimeout(connect, backoff) }
      ws.onerror   = () => ws.close()
      ws.onmessage = e => {
        try {
          const m = JSON.parse(e.data)
          if (m.type === 'calendar_update') {
            setToday(m.today ?? []); setUpcoming(m.upcoming ?? [])
            setConnected(true); setNeedsAuth(false)
          } else if (m.type === 'auth_required') {
            setNeedsAuth(true); setConnected(false)
          }
        } catch { /* */ }
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible])

  const todayIds       = new Set(today.map(e => e.id))
  const upcomingGroups = groupByDate(upcoming.filter(e => !todayIds.has(e.id)))

  return (
    <div
      ref={panelRef}
      className={`${styles.panel} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{ ...panelStyle, ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}) }}
    >
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <span className={`${styles.dot} ${wsOk ? styles.dotLive : ''}`} />
        <span className={styles.title}>CALENDAR</span>
      </div>

      {needsAuth && (
        <div className={styles.authPrompt}>
          <p>Say <em>"MAYA, authenticate calendar"</em> to connect Google Calendar.</p>
        </div>
      )}

      {connected && (
        <div className={styles.body}>
          <div className={styles.section}>
            <span className={styles.sectionLabel}>TODAY</span>
            {today.length === 0
              ? <p className={styles.empty}>Nothing scheduled</p>
              : today.map(e => (
                <div key={e.id} className={styles.eventRow}>
                  <span className={styles.eventTime}>{fmtTime(e.start)}</span>
                  <span className={styles.eventTitle}>{e.title}</span>
                </div>
              ))
            }
          </div>

          {Object.entries(upcomingGroups).slice(0, 6).map(([date, evts]) => (
            <div key={date} className={styles.section}>
              <span className={styles.sectionLabel}>{fmtDate(date)}</span>
              {evts.map(e => (
                <div key={e.id} className={styles.eventRow}>
                  <span className={styles.eventTime}>{fmtTime(e.start)}</span>
                  <span className={styles.eventTitle}>{e.title}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
