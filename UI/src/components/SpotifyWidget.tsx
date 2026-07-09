import { useRef, useEffect, useState, useCallback } from 'react'
import { SPOTIFY_WS, wsUrl } from '../constants'
import { useDraggable } from '../hooks/useDraggable'
import styles from './SpotifyWidget.module.css'

type Tab         = 'playing' | 'queue' | 'playlists' | 'search'
type RepeatState = 'off' | 'track' | 'context'
type WSState     = 'connecting' | 'setup_required' | 'auth_required' | 'ready'

interface Track {
  track: string; artist: string; album: string
  album_art: string | null; duration_ms: number
  progress_ms: number; is_playing: boolean
  shuffle_state: boolean; repeat_state: RepeatState
  volume: number; track_id: string
  is_liked: boolean; is_episode: boolean; timestamp: number
}

interface TrackRow {
  uri: string; track: string; artist: string
  album_art: string | null; duration_ms: number
}

interface Playlist {
  uri: string; name: string; count: number; image: string | null
}

interface SpotifyWidgetProps { visible: boolean; isAway?: boolean }

const fmt = (ms: number) => {
  const s = Math.floor(ms / 1000)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export default function SpotifyWidget({ visible, isAway = false }: SpotifyWidgetProps) {
  const panelRef       = useRef<HTMLDivElement>(null)
  const wsRef          = useRef<WebSocket | null>(null)
  const debRef         = useRef<ReturnType<typeof setTimeout> | null>(null)
  const volDebRef      = useRef<ReturnType<typeof setTimeout> | null>(null)
  const rafRef         = useRef<number>(0)
  const inputRef       = useRef<HTMLInputElement>(null)

  // Direct DOM refs for smooth 60fps progress — avoids React re-renders each frame
  const trackRef       = useRef<Track | null>(null)
  const progFillRef    = useRef<HTMLDivElement>(null)
  const progTimeRef    = useRef<HTMLSpanElement>(null)
  const progBarRef     = useRef<HTMLDivElement>(null)
  const volFillRef     = useRef<HTMLDivElement>(null)
  const volBarRef      = useRef<HTMLDivElement>(null)

  const [wsState,     setWsState]     = useState<WSState>('connecting')
  const [track,       setTrack]       = useState<Track | null>(null)
  const [tab,         setTab]         = useState<Tab>('playing')
  const [queue,       setQueue]       = useState<TrackRow[]>([])
  const [queueLoad,   setQueueLoad]   = useState(false)
  const [playlists,   setPlaylists]   = useState<Playlist[]>([])
  const [plLoad,      setPlLoad]      = useState(false)
  const [recent,      setRecent]      = useState<TrackRow[]>([])
  const [query,       setQuery]       = useState('')
  const [results,     setResults]     = useState<TrackRow[]>([])
  const [searching,   setSearching]   = useState(false)

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-spotify')

  // ── Initial position — hook loads saved pos first, falls back to bottom-centre ─
  useEffect(() => {
    if (visible) initTopLeft(window.innerWidth / 2 - 210, window.innerHeight - 210)
  }, [visible, initTopLeft])

  // ── Keep trackRef in sync with React state ──────────────────────────────────
  useEffect(() => { trackRef.current = track }, [track])

  // ── Single RAF loop — direct DOM mutation, zero React re-renders ─────────────
  // Runs at full 60fps when playing; sleeps at 1fps when paused to save CPU.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | number = 0

    function tick() {
      const t = trackRef.current
      if (t && progFillRef.current && progTimeRef.current) {
        const ms  = t.is_playing
          ? Math.min(t.progress_ms + (Date.now() - t.timestamp), t.duration_ms)
          : t.progress_ms
        const pct = t.duration_ms > 0 ? (ms / t.duration_ms) * 100 : 0
        progFillRef.current.style.width = `${pct}%`
        progTimeRef.current.textContent = fmt(ms)
      }
      if (trackRef.current?.is_playing) {
        timer = requestAnimationFrame(tick)
      } else {
        timer = setTimeout(tick, 1000) as unknown as number  // 1fps when paused
      }
    }

    timer = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(timer as number)
      clearTimeout(timer as ReturnType<typeof setTimeout>)
    }
  }, [])  // runs once — reads from refs, never restarts

  // ── WebSocket ───────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!visible) return
    let reconnect: ReturnType<typeof setTimeout>

    function connect() {
      const ws = new WebSocket(wsUrl(SPOTIFY_WS))
      ws.onopen  = () => ws.send(JSON.stringify({ type: 'get_current' }))
      ws.onerror = () => ws.close()
      ws.onclose = () => { setWsState('connecting'); reconnect = setTimeout(connect, 2000) }
      ws.onmessage = e => {
        let msg: Record<string, unknown>
        try { msg = JSON.parse(e.data as string) } catch { return }

        switch (msg.type as string) {
          case 'setup_required': setWsState('setup_required'); break
          case 'auth_required':  setWsState('auth_required');  break

          case 'track_update':
            setWsState('ready')
            if (!msg.track) {
              setTrack(null)
            } else {
              setTrack({
                track:         msg.track         as string,
                artist:        msg.artist         as string,
                album:         msg.album          as string,
                album_art:     msg.album_art      as string | null,
                duration_ms:   msg.duration_ms    as number,
                progress_ms:   msg.progress_ms    as number,
                is_playing:    msg.is_playing     as boolean,
                shuffle_state: msg.shuffle_state  as boolean,
                repeat_state:  msg.repeat_state   as RepeatState,
                volume:        msg.volume         as number,
                track_id:      msg.track_id       as string,
                is_liked:      Boolean(msg.is_liked),
                is_episode:    Boolean(msg.is_episode),
                timestamp:     (msg.timestamp as number) ?? Date.now(),
              })
            }
            break

          case 'like_update':
            setTrack(t => t ? { ...t, is_liked: msg.is_liked as boolean } : t)
            break
          case 'queue_data':
            setQueue(msg.items as TrackRow[])
            setQueueLoad(false)
            break
          case 'playlists_data':
            setPlaylists(msg.items as Playlist[])
            setPlLoad(false)
            break
          case 'recently_played':
            setRecent(msg.items as TrackRow[])
            break
          case 'search_results':
            setResults(msg.results as TrackRow[])
            setSearching(false)
            break
        }
      }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [visible])

  const send = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify(msg))
  }, [])

  // ── Tab switching ────────────────────────────────────────────────────────────
  function switchTab(t: Tab) {
    setTab(t)
    if (t === 'queue') {
      setQueueLoad(true)
      send({ type: 'get_queue' })
    }
    if (t === 'playlists') {
      setPlLoad(true)
      send({ type: 'get_playlists' })
    }
    if (t === 'search' && !recent.length) send({ type: 'get_recently_played' })
    if (t === 'search') setTimeout(() => inputRef.current?.focus(), 50)
  }

  // ── Search debounce ──────────────────────────────────────────────────────────
  function onQueryChange(e: React.ChangeEvent<HTMLInputElement>) {
    const q = e.target.value
    setQuery(q)
    if (debRef.current) clearTimeout(debRef.current)
    if (!q.trim()) { setResults([]); setSearching(false); return }
    setSearching(true)
    debRef.current = setTimeout(() => send({ type: 'search', query: q }), 320)
  }

  // ── Progress bar — click + drag ──────────────────────────────────────────────
  const draggingProg = useRef(false)

  // Use trackRef so the drag effect doesn't re-register handlers on every Spotify poll
  function seekFromEvent(clientX: number) {
    const t = trackRef.current
    if (!progBarRef.current || !t) return
    const rect = progBarRef.current.getBoundingClientRect()
    const pct  = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const ms   = Math.round(pct * t.duration_ms)
    setTrack(cur => cur ? { ...cur, progress_ms: ms, timestamp: Date.now() } : cur)
    send({ type: 'seek', position_ms: ms })
  }

  function onProgMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (!trackRef.current) return
    e.preventDefault()
    e.stopPropagation()
    draggingProg.current = true
    seekFromEvent(e.clientX)
  }

  useEffect(() => {
    function onMove(e: MouseEvent) { if (draggingProg.current) seekFromEvent(e.clientX) }
    function onUp(e: MouseEvent)   { if (draggingProg.current) { seekFromEvent(e.clientX); draggingProg.current = false } }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup',   onUp)
    return () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp) }
  }, [])  // trackRef kept in sync via separate effect — no need to re-register on track change

  // ── Volume bar — click + drag ────────────────────────────────────────────────
  const draggingVol = useRef(false)
  const localVol    = useRef<number | null>(null)

  function volFromEvent(clientX: number) {
    if (!volBarRef.current) return
    const rect = volBarRef.current.getBoundingClientRect()
    const pct  = Math.round(Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) * 100)
    localVol.current = pct
    if (volFillRef.current) volFillRef.current.style.width = `${pct}%`
    // Debounce the API call while dragging
    if (volDebRef.current) clearTimeout(volDebRef.current)
    volDebRef.current = setTimeout(() => {
      send({ type: 'volume', value: pct })
      setTrack(t => t ? { ...t, volume: pct } : t)
      localVol.current = null
    }, 120)
  }

  function onVolMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault()
    e.stopPropagation()
    draggingVol.current = true
    volFromEvent(e.clientX)
  }

  useEffect(() => {
    function onMove(e: MouseEvent) { if (draggingVol.current) volFromEvent(e.clientX) }
    function onUp(e: MouseEvent)   { if (draggingVol.current) { volFromEvent(e.clientX); draggingVol.current = false } }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup',   onUp)
    return () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp) }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Repeat cycle ─────────────────────────────────────────────────────────────
  function cycleRepeat() {
    if (!track) return
    const next: Record<RepeatState, RepeatState> = { off: 'context', context: 'track', track: 'off' }
    const state = next[track.repeat_state]
    send({ type: 'repeat', state })
    setTrack(t => t ? { ...t, repeat_state: state } : t)
  }

  // ── Toggle play/pause ─────────────────────────────────────────────────────────
  function togglePlay() {
    if (!track) return
    const action = track.is_playing ? 'pause' : 'play'
    setTrack(t => t ? { ...t, is_playing: !t.is_playing, timestamp: Date.now() } : t)
    send({ type: 'control', action })
  }

  const hasTrack = wsState === 'ready' && !!track

  return (
    <div
      ref={panelRef}
      className={`${styles.widget} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''}`}
      style={{
        ...panelStyle,
        ['--album-art' as string]: track?.album_art ? `url(${track.album_art})` : 'none',
        ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
    >
      {track?.album_art && <div className={styles.artBg} />}

      {/* Header */}
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <div className={styles.headerLeft}>
          <span className={styles.spotifyDot} />
          <span className={styles.title}>SPOTIFY</span>
        </div>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <span className={styles.headerStatus}>
          {hasTrack ? (track.is_playing ? 'NOW PLAYING' : 'PAUSED') : wsState === 'connecting' ? 'CONNECTING' : ''}
        </span>
      </div>

      {/* Tabs */}
      <div className={styles.tabs}>
        {(['playing', 'queue', 'playlists', 'search'] as Tab[]).map(t => (
          <button key={t}
            className={`${styles.tab} ${tab === t ? styles.tabActive : ''}`}
            onMouseDown={e => e.stopPropagation()}
            onClick={() => switchTab(t)}
            title={t === 'playing' ? 'Now Playing' : t === 'queue' ? 'Queue' : t === 'playlists' ? 'Playlists' : 'Search'}
          >
            {t === 'playing' ? '♪' : t === 'queue' ? '≡' : t === 'playlists' ? '♫' : '⌕'}
          </button>
        ))}
      </div>

      {/* Status: not ready yet */}
      {wsState !== 'ready' && tab === 'playing' && (
        <div className={styles.statusBlock}>
          {wsState === 'connecting' && <span className={styles.statusText}>CONNECTING...</span>}
          {wsState === 'setup_required' && (
            <span className={styles.statusText}>
              Add credentials to<br />
              <span className={styles.path}>Backend/models/spotify_config.json</span>
            </span>
          )}
          {wsState === 'auth_required' && (
            <>
              <span className={styles.statusText}>Spotify not connected</span>
              <button className={styles.connectBtn}
                onMouseDown={e => e.stopPropagation()}
                onClick={() => send({ type: 'request_auth' })}>
                CONNECT SPOTIFY
              </button>
            </>
          )}
        </div>
      )}

      {/* Now Playing tab */}
      {tab === 'playing' && wsState === 'ready' && (
        <div className={styles.nowPlaying}>
          {!track ? (
            <div className={styles.statusBlock}>
              <span className={styles.statusText}>NOTHING PLAYING</span>
            </div>
          ) : (
            <>
              {/* Track info row */}
              <div className={styles.trackRow}>
                {track.album_art
                  ? <img src={track.album_art} className={styles.artThumb} alt="" />
                  : <div className={styles.artThumbEmpty}>♪</div>
                }
                <div className={styles.trackMeta}>
                  <div className={styles.trackName} title={track.track}>{track.track}</div>
                  <div className={styles.artistName} title={track.artist}>{track.artist}</div>
                  {track.album && <div className={styles.albumName} title={track.album}>{track.album}</div>}
                </div>
                {!track.is_episode && (
                  <button
                    className={`${styles.likeBtn} ${track.is_liked ? styles.liked : ''}`}
                    onMouseDown={e => e.stopPropagation()}
                    onClick={() => send({ type: 'toggle_like', track_id: track.track_id, liked: track.is_liked })}
                    title={track.is_liked ? 'Unlike' : 'Like'}
                  >
                    {track.is_liked ? '♥' : '♡'}
                  </button>
                )}
              </div>

              {/* Progress */}
              <div className={styles.progressSection}>
                <span ref={progTimeRef} className={styles.timeLabel}>{fmt(track.progress_ms)}</span>
                <div
                  ref={progBarRef}
                  className={styles.progressBar}
                  onMouseDown={onProgMouseDown}
                >
                  <div ref={progFillRef} className={styles.progressFill}
                    style={{ width: `${track.duration_ms > 0 ? (track.progress_ms / track.duration_ms) * 100 : 0}%` }}>
                    <div className={styles.progressThumb} />
                  </div>
                </div>
                <span className={styles.timeLabel}>{fmt(track.duration_ms)}</span>
              </div>

              {/* Controls */}
              <div className={styles.controls}>
                <button
                  className={`${styles.ctrl} ${styles.ctrlSm} ${track.shuffle_state ? styles.ctrlOn : ''}`}
                  onMouseDown={e => e.stopPropagation()}
                  onClick={() => { send({ type: 'shuffle', state: !track.shuffle_state }); setTrack(t => t ? { ...t, shuffle_state: !t.shuffle_state } : t) }}
                  title="Shuffle"
                >⇄</button>

                <button className={styles.ctrl} onMouseDown={e => e.stopPropagation()}
                  onClick={() => send({ type: 'control', action: 'prev' })} title="Previous">◀◀</button>

                <button className={`${styles.ctrl} ${styles.ctrlMain}`}
                  onMouseDown={e => e.stopPropagation()} onClick={togglePlay}
                  title={track.is_playing ? 'Pause' : 'Play'}>
                  {track.is_playing ? '■' : '▶'}
                </button>

                <button className={styles.ctrl} onMouseDown={e => e.stopPropagation()}
                  onClick={() => send({ type: 'control', action: 'next' })} title="Next">▶▶</button>

                <button
                  className={`${styles.ctrl} ${styles.ctrlSm} ${track.repeat_state !== 'off' ? styles.ctrlOn : ''}`}
                  onMouseDown={e => e.stopPropagation()} onClick={cycleRepeat}
                  title={`Repeat: ${track.repeat_state}`}
                >{track.repeat_state === 'track' ? '⟳¹' : '⟳'}</button>
              </div>

              {/* Volume */}
              <div className={styles.volumeRow}>
                <span className={styles.volIcon}>◁)</span>
                <div
                  ref={volBarRef}
                  className={styles.volBar}
                  onMouseDown={onVolMouseDown}
                >
                  <div ref={volFillRef} className={styles.volFill}
                    style={{ width: `${localVol.current ?? track.volume}%` }} />
                  <div className={styles.volThumb}
                    style={{ left: `${localVol.current ?? track.volume}%` }} />
                </div>
                <span className={styles.volLabel}>{localVol.current ?? track.volume}</span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Queue tab */}
      {tab === 'queue' && (
        <div className={styles.listTab}>
          {queueLoad ? (
            <div className={styles.loading}><span className={styles.loadingDots}>· · ·</span></div>
          ) : queue.length === 0 ? (
            <div className={styles.emptyMsg}>Queue is empty</div>
          ) : queue.map((item, i) => (
            <TrackItem key={i} item={item}
              onPlay={() => send({ type: 'play_uri', uri: item.uri })}
              onQueue={() => send({ type: 'add_queue', uri: item.uri })}
            />
          ))}
        </div>
      )}

      {/* Playlists tab */}
      {tab === 'playlists' && (
        <div className={styles.listTab}>
          {plLoad ? (
            <div className={styles.loading}><span className={styles.loadingDots}>· · ·</span></div>
          ) : playlists.length === 0 ? (
            <div className={styles.emptyMsg}>No playlists found</div>
          ) : playlists.map((pl, i) => (
            <div key={i} className={styles.playlistRow}
              onClick={() => send({ type: 'play_context', uri: pl.uri })}>
              {pl.image
                ? <img src={pl.image} className={styles.plThumb} alt="" />
                : <div className={styles.plThumbEmpty}>♫</div>
              }
              <div className={styles.plInfo}>
                <div className={styles.plName}>{pl.name}</div>
                <div className={styles.plCount}>{pl.count} tracks</div>
              </div>
              <span className={styles.plPlay}>▶</span>
            </div>
          ))}
        </div>
      )}

      {/* Search tab */}
      {tab === 'search' && (
        <div className={styles.searchTab}>
          <div className={styles.searchRow}>
            <span className={styles.searchIcon}>⌕</span>
            <input ref={inputRef} className={styles.searchInput}
              placeholder="search for a song, artist, or album..."
              value={query}
              onChange={onQueryChange}
              onKeyDown={e => { if (e.key === 'Escape') { setQuery(''); setResults([]) } }}
              onMouseDown={e => e.stopPropagation()}
            />
            {searching && <span className={styles.spinner}>···</span>}
            {query && !searching && (
              <button className={styles.clearBtn} onClick={() => { setQuery(''); setResults([]) }}
                onMouseDown={e => e.stopPropagation()}>✕</button>
            )}
          </div>

          <div className={styles.listTab}>
            {!query && recent.length > 0 && (
              <>
                <div className={styles.sectionLabel}>RECENTLY PLAYED</div>
                {recent.map((item, i) => (
                  <TrackItem key={i} item={item}
                    onPlay={() => { send({ type: 'play_uri', uri: item.uri }); setTab('playing') }}
                    onQueue={() => send({ type: 'add_queue', uri: item.uri })}
                  />
                ))}
              </>
            )}
            {query && !searching && results.length === 0 && (
              <div className={styles.emptyMsg}>No results for "{query}"</div>
            )}
            {query && results.map((item, i) => (
              <TrackItem key={i} item={item}
                onPlay={() => { send({ type: 'play_uri', uri: item.uri }); setTab('playing') }}
                onQueue={() => send({ type: 'add_queue', uri: item.uri })}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Shared track row component ─────────────────────────────────────────────────

function TrackItem({
  item, onPlay, onQueue,
}: {
  item: { track: string; artist: string; album_art: string | null; duration_ms: number }
  onPlay: () => void
  onQueue: () => void
}) {
  return (
    <div className={styles.trackItem}>
      {item.album_art
        ? <img src={item.album_art} className={styles.itemThumb} alt="" />
        : <div className={styles.itemThumbEmpty}>♪</div>
      }
      <div className={styles.itemInfo}>
        <div className={styles.itemTrack}>{item.track}</div>
        <div className={styles.itemArtist}>{item.artist}</div>
      </div>
      <div className={styles.itemActions}>
        <button className={styles.itemBtn} onClick={onQueue}
          onMouseDown={e => e.stopPropagation()} title="Add to queue">+</button>
        <button className={`${styles.itemBtn} ${styles.itemBtnPlay}`} onClick={onPlay}
          onMouseDown={e => e.stopPropagation()} title="Play now">▶</button>
      </div>
    </div>
  )
}
