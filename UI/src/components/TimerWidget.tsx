import { useRef, useEffect, useState } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import { TIMER_WS, wsUrl } from '../constants'
import styles from './TimerWidget.module.css'

interface Timer { name: string; duration: number; remaining: number; state: 'running' | 'paused' | 'done' }
interface Props  { visible: boolean; isAway?: boolean }

function pad(n: number) { return String(Math.floor(Math.max(0, n))).padStart(2, '0') }
function fmt(secs: number) { const m = Math.floor(secs / 60); return `${pad(m)}:${pad(secs % 60)}` }

function TimerCard({ t, onAction }: { t: Timer; onAction: (type: string, name: string) => void }) {
  const circ  = 2 * Math.PI * 22
  const ratio = t.duration > 0 ? Math.max(0, Math.min(1, (t.duration - t.remaining) / t.duration)) : 1
  const dash  = circ * ratio
  const colour = t.state === 'done' ? '#7fff6d' : t.remaining < 60 ? '#ff5555' : t.remaining < 300 ? '#ffaa33' : 'rgba(140,80,255,0.9)'

  return (
    <div className={`${styles.card} ${t.state === 'done' ? styles.done : ''}`}>
      <svg className={styles.ring} viewBox="0 0 48 48">
        <circle cx="24" cy="24" r="22" className={styles.ringTrack} />
        <circle cx="24" cy="24" r="22" className={styles.ringFill}
          style={{ stroke: colour, strokeDasharray: `${dash} ${circ}` }} />
      </svg>
      <div className={styles.cardInner}>
        <span className={styles.timerName}>{t.name}</span>
        <span className={styles.timerTime} style={{ color: colour }}>{fmt(t.remaining)}</span>
        <div className={styles.timerBtns}>
          {t.state === 'running' && <button onMouseDown={e => e.stopPropagation()} onClick={() => onAction('pause',  t.name)}>⏸</button>}
          {t.state === 'paused'  && <button onMouseDown={e => e.stopPropagation()} onClick={() => onAction('resume', t.name)}>▶</button>}
          {t.state !== 'done'    && <button onMouseDown={e => e.stopPropagation()} onClick={() => onAction('cancel', t.name)}>✕</button>}
        </div>
      </div>
    </div>
  )
}

export default function TimerWidget({ visible, isAway = false }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)
  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-timer-v2')

  const [timers,  setTimers]  = useState<Timer[]>([])
  const [wsOk,    setWsOk]    = useState(false)
  const [newName, setNewName] = useState('')
  const [newMins, setNewMins] = useState('')

  useEffect(() => { if (visible) initTopLeft(Math.round(window.innerWidth / 2 - 150), 200) }, [visible, initTopLeft])

  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000
    function connect() {
      const ws = new WebSocket(wsUrl(TIMER_WS))
      ws.onopen    = () => { setWsOk(true); backoff = 2000 }
      ws.onclose   = () => { setWsOk(false); backoff = Math.min(backoff * 1.5, 30_000); reconnect = setTimeout(connect, backoff) }
      ws.onerror   = () => ws.close()
      ws.onmessage = e => {
        try { const m = JSON.parse(e.data); if (m.type === 'timers_update') setTimers(m.timers) } catch { /* */ }
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible])

  function send(msg: object) {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify(msg))
  }

  function addTimer() {
    const name = newName.trim() || 'timer'
    const secs = Math.round(parseFloat(newMins) * 60)
    if (!secs || secs <= 0) return
    send({ type: 'create', name, seconds: secs })
    setNewName(''); setNewMins('')
  }

  const active = timers.filter(t => t.state !== 'done').length

  return (
    <div
      ref={panelRef}
      className={`${styles.panel} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{ ...panelStyle, ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}) }}
    >
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <span className={`${styles.dot} ${wsOk ? styles.dotLive : ''}`} />
        <span className={styles.title}>TIMERS</span>
        {active > 0 && <span className={styles.count}>{active} active</span>}
      </div>

      <div className={styles.body}>
        {timers.length === 0
          ? <p className={styles.empty}>No timers — say "Hey Maya, set a timer"</p>
          : timers.map(t => <TimerCard key={t.name} t={t} onAction={send.bind(null)} />)
        }
      </div>

      <div className={styles.addRow}>
        <input className={styles.addInput} placeholder="name" value={newName}
          onChange={e => setNewName(e.target.value)} onMouseDown={e => e.stopPropagation()} />
        <input className={styles.addInput} placeholder="mins" type="number" value={newMins}
          onChange={e => setNewMins(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') addTimer() }}
          onMouseDown={e => e.stopPropagation()} />
        <button className={styles.addBtn} onClick={addTimer} onMouseDown={e => e.stopPropagation()}>SET</button>
      </div>
    </div>
  )
}
