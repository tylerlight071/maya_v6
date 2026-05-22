import { useState, useEffect, useRef } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import styles from './NewsWidget.module.css'

interface Article {
  title:        string
  url:          string
  summary:      string
  source:       string
  category:     string
  published_ts: number
}

interface Props {
  visible:  boolean
  isAway?:  boolean
  onBrowse?: (url: string, title: string) => void
}

const WS_URL   = 'ws://127.0.0.1:8775'
const CATS     = ['all', 'top', 'world', 'tech'] as const
type  Category = typeof CATS[number]

function timeAgo(ts: number): string {
  const diff = Math.floor((Date.now() / 1000) - ts)
  if (diff < 60)   return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400)return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

export default function NewsWidget({ visible, isAway = false, onBrowse }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)
  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-news')

  const [articles,  setArticles]  = useState<Article[]>([])
  const [category,  setCategory]  = useState<Category>('all')
  const [connected, setConnected] = useState(false)
  const [refreshing,setRefreshing]= useState(false)
  const [tick,      setTick]      = useState(0)

  useEffect(() => {
    if (visible) initTopLeft(window.innerWidth - 420, 65)
  }, [visible, initTopLeft])

  // Tick every 30s to keep time-ago fresh
  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 30_000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 1000

    function connect() {
      const ws = new WebSocket(WS_URL)
      ws.onopen    = () => { setConnected(true); backoff = 1000 }
      ws.onclose   = () => {
        setConnected(false)
        reconnect = setTimeout(connect, backoff)
        backoff   = Math.min(backoff * 1.5, 30_000)
      }
      ws.onerror   = () => ws.close()
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          if (msg.type === 'news_data') {
            setArticles(msg.articles)
            setRefreshing(false)
          }
        } catch { /* ignore */ }
      }
      wsRef.current = ws
    }

    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [])

  function handleRefresh() {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setRefreshing(true)
      wsRef.current.send(JSON.stringify({ type: 'refresh' }))
    }
  }

  const filtered = category === 'all'
    ? articles
    : articles.filter(a => a.category === category)

  return (
    <div
      ref={panelRef}
      className={`${styles.panel} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{
        ...panelStyle,
        ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
    >
      {/* Header */}
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <div className={styles.headerLeft}>
          <span className={`${styles.statusDot} ${connected ? styles.connected : ''}`} />
          <span className={styles.headerTitle}>NEWS</span>
        </div>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <div className={styles.headerRight}>
          <button
            className={`${styles.refreshBtn} ${refreshing ? styles.spinning : ''}`}
            onClick={handleRefresh}
            onMouseDown={e => e.stopPropagation()}
            title="Refresh"
          >↻</button>
        </div>
      </div>

      {/* Category filter */}
      <div className={styles.filters} onMouseDown={e => e.stopPropagation()}>
        {CATS.map(cat => (
          <button
            key={cat}
            className={`${styles.filterBtn} ${category === cat ? styles.active : ''}`}
            onClick={() => setCategory(cat)}
          >
            {cat === 'all' ? 'ALL' : cat === 'top' ? 'TOP' : cat === 'world' ? 'WORLD' : 'TECH'}
          </button>
        ))}
      </div>

      {/* Feed */}
      <div className={styles.feed}>
        {filtered.length === 0 ? (
          <div className={styles.empty}>
            {connected ? 'Fetching headlines…' : 'Connecting…'}
          </div>
        ) : (
          filtered.map((a, i) => (
            <button
              key={a.url + i}
              className={styles.article}
              onMouseDown={e => e.stopPropagation()}
              onClick={() => onBrowse?.(a.url, a.title)}
              title={a.summary || a.title}
            >
              <div className={styles.articleTitle}>{a.title}</div>
              <div className={styles.articleMeta}>
                <span className={styles.source}>{a.source}</span>
                <span className={styles.dot2}>·</span>
                <span className={styles.time}>{timeAgo(a.published_ts)}</span>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  )
}
