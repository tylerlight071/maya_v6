import { useRef, useEffect, useState } from 'react'
import { WEATHER_WS } from '../constants'
import { useDraggable } from '../hooks/useDraggable'
import styles from './WeatherWidget.module.css'

type Tab = 'now' | 'hourly' | 'daily'

interface HourlyItem {
  time: string; temp: number; precip: number
  icon: string; wind: number; is_day: boolean
}

interface DailyItem {
  date: string; desc: string; icon: string
  max: number; min: number; precip: number
  wind: number; sunrise: string; sunset: string
}

interface Weather {
  temp: number; feels_like: number; humidity: number
  wind_speed: number; wind_dir: string; cloud_cover: number
  uv_index: number; visibility_km: number; precipitation: number
  code: number; description: string; icon: string; is_day: boolean
  city: string; country: string
  hourly: HourlyItem[]; daily: DailyItem[]; timestamp: number
}

interface WeatherWidgetProps { visible: boolean; isAway?: boolean }

const fmt12h = (iso: string) => {
  const h = new Date(iso).getHours()
  return h === 0 ? '12am' : h < 12 ? `${h}am` : h === 12 ? '12pm' : `${h - 12}pm`
}

const fmtDay = (iso: string) => {
  const d = new Date(iso)
  const today = new Date()
  if (d.toDateString() === today.toDateString()) return 'Today'
  return d.toLocaleDateString('en-GB', { weekday: 'short' })
}

const fmtTime = (iso: string) => {
  if (!iso) return '—'
  const d = new Date(iso)
  return `${d.getHours()}:${String(d.getMinutes()).padStart(2,'0')}`
}

const uvLabel = (uv: number) => {
  if (uv < 3)  return 'Low'
  if (uv < 6)  return 'Moderate'
  if (uv < 8)  return 'High'
  if (uv < 11) return 'Very High'
  return 'Extreme'
}

export default function WeatherWidget({ visible, isAway = false }: WeatherWidgetProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const wsRef    = useRef<WebSocket | null>(null)

  const [weather,  setWeather]  = useState<Weather | null>(null)
  const [loading,  setLoading]  = useState(true)
  const [tab,      setTab]      = useState<Tab>('now')
  const [wsOk,     setWsOk]     = useState(false)

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-weather')

  useEffect(() => {
    if (visible) initTopLeft(window.innerWidth - 360, 65)
  }, [visible, initTopLeft])

  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>
    let backoff = 2000

    function connect() {
      const ws = new WebSocket(WEATHER_WS)
      ws.onopen  = () => { setWsOk(true); backoff = 2000 }
      ws.onerror = () => ws.close()
      ws.onclose = () => {
        setWsOk(false)
        backoff = Math.min(backoff * 1.5, 30_000)
        reconnect = setTimeout(connect, backoff)
      }
      ws.onmessage = e => {
        try {
          const msg = JSON.parse(e.data)
          if (msg.type === 'weather_update') {
            setWeather(msg as Weather)
            setLoading(false)
          } else if (msg.type === 'weather_loading') {
            setLoading(true)
          }
        } catch { /* ignore */ }
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible])

  const w = weather

  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{
        ...panelStyle,
        ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
    >
      {/* Header */}
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <div className={styles.headerLeft}>
          <span className={`${styles.dot} ${wsOk ? styles.dotLive : ''}`} />
          <span className={styles.title}>WEATHER</span>
        </div>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <span className={styles.location}>
          {w ? `${w.city}${w.country ? `, ${w.country}` : ''}` : '—'}
        </span>
      </div>

      {/* Tabs */}
      <div className={styles.tabs}>
        {(['now', 'hourly', 'daily'] as Tab[]).map(t => (
          <button key={t}
            className={`${styles.tab} ${tab === t ? styles.tabActive : ''}`}
            onMouseDown={e => e.stopPropagation()}
            onClick={() => setTab(t)}
          >
            {t === 'now' ? 'NOW' : t === 'hourly' ? '24H' : '7D'}
          </button>
        ))}
      </div>

      {/* Content */}
      {loading ? (
        <div className={styles.loading}>
          <span className={styles.loadingText}>ACQUIRING WEATHER DATA</span>
          <span className={styles.loadingDots}>· · ·</span>
        </div>
      ) : !w ? (
        <div className={styles.loading}>
          <span className={styles.loadingText}>NO DATA</span>
        </div>
      ) : (
        <>
          {/* ── NOW ── */}
          {tab === 'now' && (
            <div className={styles.nowTab}>
              {/* Hero */}
              <div className={styles.hero}>
                <span className={styles.heroIcon}>{w.icon}</span>
                <div className={styles.heroRight}>
                  <span className={styles.heroTemp}>{w.temp}°</span>
                  <span className={styles.heroDesc}>{w.description}</span>
                  <span className={styles.heroFeels}>Feels like {w.feels_like}°C</span>
                </div>
              </div>

              {/* Stats grid */}
              <div className={styles.statsGrid}>
                <Stat label="HUMIDITY"    value={`${w.humidity}%`} />
                <Stat label="WIND"        value={`${w.wind_speed} mph ${w.wind_dir}`} />
                <Stat label="UV INDEX"    value={`${w.uv_index} — ${uvLabel(w.uv_index)}`} />
                <Stat label="VISIBILITY"  value={`${w.visibility_km} km`} />
                <Stat label="CLOUD COVER" value={`${w.cloud_cover}%`} />
                <Stat label="PRECIP"      value={`${w.precipitation} mm`} />
              </div>

              {/* Today's sunrise/sunset from daily[0] */}
              {w.daily[0] && (
                <div className={styles.sunRow}>
                  <span className={styles.sunItem}>↑ {fmtTime(w.daily[0].sunrise)}</span>
                  <div className={styles.sunBar}>
                    <div className={styles.sunFill} style={{
                      width: (() => {
                        const now = Date.now()
                        const rise = new Date(w.daily[0].sunrise).getTime()
                        const set  = new Date(w.daily[0].sunset).getTime()
                        const pct  = Math.max(0, Math.min(1, (now - rise) / (set - rise)))
                        return `${pct * 100}%`
                      })()
                    }} />
                    <span className={styles.sunDot} />
                  </div>
                  <span className={styles.sunItem}>↓ {fmtTime(w.daily[0].sunset)}</span>
                </div>
              )}
            </div>
          )}

          {/* ── HOURLY ── */}
          {tab === 'hourly' && (
            <div className={styles.hourlyTab}>
              {w.hourly.slice(0, 24).map((h, i) => (
                <div key={i} className={styles.hourRow}>
                  <span className={styles.hourTime}>{i === 0 ? 'Now' : fmt12h(h.time)}</span>
                  <span className={styles.hourIcon}>{h.icon}</span>
                  <span className={styles.hourTemp}>{h.temp}°</span>
                  <div className={styles.hourPrecipBar}>
                    <div className={styles.hourPrecipFill} style={{ width: `${h.precip}%` }} />
                  </div>
                  <span className={styles.hourPrecip}>{h.precip}%</span>
                  <span className={styles.hourWind}>{h.wind} mph</span>
                </div>
              ))}
            </div>
          )}

          {/* ── DAILY ── */}
          {tab === 'daily' && (
            <div className={styles.dailyTab}>
              {w.daily.map((d, i) => (
                <div key={i} className={`${styles.dayRow} ${i === 0 ? styles.dayRowToday : ''}`}>
                  <span className={styles.dayName}>{fmtDay(d.date)}</span>
                  <span className={styles.dayIcon}>{d.icon}</span>
                  <span className={styles.dayDesc}>{d.desc}</span>
                  <span className={styles.dayPrecip}>{d.precip}%</span>
                  <div className={styles.dayTemps}>
                    <span className={styles.dayMax}>{d.max}°</span>
                    <span className={styles.daySep}>/</span>
                    <span className={styles.dayMin}>{d.min}°</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statValue}>{value}</span>
    </div>
  )
}
