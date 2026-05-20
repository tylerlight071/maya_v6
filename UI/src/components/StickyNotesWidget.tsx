import { useRef, useEffect, useState } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import { STICKY_NOTES_WS } from '../constants'
import styles from './StickyNotesWidget.module.css'

interface Note { id: string; text: string; created: number; updated: number }
interface Props { visible: boolean; isAway?: boolean }

export default function StickyNotesWidget({ visible, isAway = false }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)
  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-notes')

  const [notes, setNotes] = useState<Note[]>([])
  const [wsOk,  setWsOk]  = useState(false)

  useEffect(() => { if (visible) initTopLeft(20, 420) }, [visible, initTopLeft])

  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000
    function connect() {
      const ws = new WebSocket(STICKY_NOTES_WS)
      ws.onopen    = () => { setWsOk(true); backoff = 2000 }
      ws.onclose   = () => { setWsOk(false); backoff = Math.min(backoff * 1.5, 30_000); reconnect = setTimeout(connect, backoff) }
      ws.onerror   = () => ws.close()
      ws.onmessage = e => {
        try { const m = JSON.parse(e.data); if (m.type === 'notes_data') setNotes(m.notes) } catch { /* */ }
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

  function handleBlur(note: Note, el: HTMLElement) {
    const text = el.textContent ?? ''
    if (text !== note.text) send({ type: 'update_note', id: note.id, text })
  }

  return (
    <div
      ref={panelRef}
      className={`${styles.panel} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{ ...panelStyle, ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}) }}
    >
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <span className={`${styles.dot} ${wsOk ? styles.dotLive : ''}`} />
        <span className={styles.title}>NOTES</span>
        {notes.length > 0 && <span className={styles.count}>{notes.length}</span>}
      </div>

      <div className={styles.body}>
        {notes.length === 0
          ? <p className={styles.empty}>No notes — say "Hey Maya, note that..."</p>
          : notes.map(note => (
            <div key={note.id} className={styles.noteCard}>
              <div
                className={styles.noteText}
                contentEditable
                suppressContentEditableWarning
                onBlur={e => handleBlur(note, e.currentTarget)}
                onMouseDown={e => e.stopPropagation()}
              >{note.text}</div>
              <button
                type="button"
                className={styles.deleteBtn}
                onMouseDown={e => e.stopPropagation()}
                onClick={() => send({ type: 'delete_note', id: note.id })}
                title="Delete"
              >✕</button>
            </div>
          ))
        }
      </div>
    </div>
  )
}
