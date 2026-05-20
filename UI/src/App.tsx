import { useState, useEffect, useRef } from 'react'
import type { MayaState } from './types'
import { TIMING } from './constants'
import MAYACanvas from './components/MAYACanvas'
import HUDOverlay from './components/HUDOverlay'
import ChatPanel from './components/ChatPanel'
import WebcamWidget from './components/WebcamWidget'
import SpotifyWidget from './components/SpotifyWidget'
import WeatherWidget from './components/WeatherWidget'
import ImagesWidget from './components/ImagesWidget'
import MapOverlay, { type MapCommand } from './components/MapOverlay'
import ImageGenWidget from './components/ImageGenWidget'
import DevLogOverlay from './components/DevLogOverlay'
import BrowserWidget from './components/BrowserWidget'
import VideoWidget from './components/VideoWidget'
import SystemMonitorWidget from './components/SystemMonitorWidget'
import TimerWidget from './components/TimerWidget'
import StickyNotesWidget from './components/StickyNotesWidget'
import CalendarWidget from './components/CalendarWidget'

const VISIBILITY_KEY = 'maya-widget-visibility'

function loadSavedVisibility(): Record<string, boolean> {
  try { return JSON.parse(localStorage.getItem(VISIBILITY_KEY) || '{}') }
  catch { return {} }
}

export default function App() {
  const [chatVisible,     setChatVisible]     = useState(false)
  const [webcamVisible,   setWebcamVisible]   = useState(false)
  const [spotifyVisible,  setSpotifyVisible]  = useState(false)
  const [weatherVisible,  setWeatherVisible]  = useState(false)
  const [imagesVisible,   setImagesVisible]   = useState(false)
  const [imageGenVisible, setImageGenVisible] = useState(false)
  const [mapVisible,      setMapVisible]      = useState(false)
  const [mapCommand,      setMapCommand]      = useState<MapCommand | null>(null)
  const [imageFilename,   setImageFilename]   = useState<string | null>(null)
  const [mayaState,       setMayaState]       = useState<MayaState>('idle')
  const [presence,        setPresence]        = useState<'present' | 'away'>('present')

  const [videoVisible,   setVideoVisible]   = useState(false)
  const [videoUrl,       setVideoUrl]       = useState('')
  const [monitorVisible, setMonitorVisible] = useState(false)
  const [timerVisible,   setTimerVisible]   = useState(false)
  const [notesVisible,   setNotesVisible]   = useState(false)
  const [calendarVisible,setCalendarVisible]= useState(false)

  // Browser widget state
  const [browserVisible, setBrowserVisible] = useState(false)
  const [browserUrl,     setBrowserUrl]     = useState('')
  const [browserTitle,   setBrowserTitle]   = useState('')

  // Dev log state
  const [devLogMode,       setDevLogMode]       = useState(false)
  const [devLogShowScreen, setDevLogShowScreen] = useState(false)
  const webcamBeforeLog = useRef(false)

  const visibilityRef = useRef({ chat: false, spotify: false, webcam: false, weather: false, images: false, imagegen: false })
  const wsSendRef     = useRef<((msg: string) => void) | null>(null)

  // Restore persisted visibility after intro
  useEffect(() => {
    const t = setTimeout(() => {
      const saved = loadSavedVisibility()
      setChatVisible(true)  // always start with chat open
      setWebcamVisible(saved.webcam     ?? false)
      setSpotifyVisible(saved.spotify   ?? false)
      setWeatherVisible(saved.weather   ?? false)
      setImagesVisible(saved.images     ?? false)
      setImageGenVisible(saved.imagegen ?? false)
    }, TIMING.CHAT_APPEAR_MS)
    return () => clearTimeout(t)
  }, [])

  // Persist visibility + real-time sync to backend
  useEffect(() => {
    const state = {
      chat: chatVisible, spotify: spotifyVisible, webcam: webcamVisible,
      weather: weatherVisible, images: imagesVisible, imagegen: imageGenVisible,
    }
    visibilityRef.current = state
    try { localStorage.setItem(VISIBILITY_KEY, JSON.stringify(state)) } catch { /* unavailable */ }
    wsSendRef.current?.(JSON.stringify({ type: 'widget_state_sync', state }))
  }, [chatVisible, spotifyVisible, webcamVisible, weatherVisible, imagesVisible, imageGenVisible])

  function handleAction(
    action:  string,
    widget?: string,
    visible?: boolean,
    extra?:  Record<string, unknown>,
  ) {
    // ── Dev log actions ───────────────────────────────────────────────────────
    // Browser actions
    if (action === 'browse') {
      setBrowserUrl(extra?.url as string ?? '')
      setBrowserTitle(extra?.title as string ?? '')
      setBrowserVisible(true)
      return
    }
    if (action === 'hide_browser' || (widget === 'browser' && !visible)) {
      setBrowserVisible(false)
      return
    }
    if (widget === 'browser' && visible) {
      setBrowserVisible(true)
      return
    }

    if (action === 'start_dev_log') {
      webcamBeforeLog.current = webcamVisible
      setWebcamVisible(true)
      setDevLogShowScreen(Boolean(extra?.show_screen))
      setDevLogMode(true)
      return
    }
    if (action === 'end_dev_log') {
      setDevLogMode(false)
      setDevLogShowScreen(false)
      if (!webcamBeforeLog.current) setWebcamVisible(false)
      return
    }
    if (action === 'dev_log_show_screen') {
      setDevLogShowScreen(true)
      return
    }

    // ── Map actions ───────────────────────────────────────────────────────────
    if (action === 'map_show' || action === 'map_goto') {
      setMapVisible(true)
      if (extra?.lat != null && extra?.lng != null) {
        setMapCommand({ id: Date.now(), action: 'fly_to',
          lat: extra.lat as number, lng: extra.lng as number,
          zoom: extra.zoom as number ?? 11, label: extra.label as string ?? '' })
      }
      return
    }
    if (action === 'map_marker') { setMapCommand({ id: Date.now(), action: 'marker', lat: extra?.lat as number, lng: extra?.lng as number, label: extra?.label as string ?? '' }); return }
    if (action === 'map_clear')  { setMapCommand({ id: Date.now(), action: 'clear' }); return }
    if (action === 'map_reset')  { setMapCommand({ id: Date.now(), action: 'reset' }); setTimeout(() => setMapVisible(false), 2600); return }

    // ── Panel show/hide ───────────────────────────────────────────────────────
    if (!widget) return
    const show = visible !== undefined ? visible : action === 'show_widget'

    if (widget === 'chat')           setChatVisible(show)
    if (widget === 'spotify')        setSpotifyVisible(show)
    if (widget === 'webcam')         setWebcamVisible(show)
    if (widget === 'weather')        setWeatherVisible(show)
    if (widget === 'imagegen')       setImageGenVisible(show)
    if (action === 'open_video' || widget === 'video') {
      setVideoUrl(extra?.url as string ?? '')
      setVideoVisible(true)
      return
    }
    if (widget === 'system_monitor') setMonitorVisible(show)
    if (widget === 'timer')          setTimerVisible(show)
    if (widget === 'sticky_notes')   setNotesVisible(show)
    if (widget === 'calendar')       setCalendarVisible(show)
    if (widget === 'images') {
      setImagesVisible(show)
      if (action === 'show_image' && extra?.filename) setImageFilename(extra.filename as string)
    }
  }

  // End dev log from the overlay button — sends a direct signal, NOT through the LLM
  function handleDevLogEnd() {
    setDevLogMode(false)
    setDevLogShowScreen(false)
    if (!webcamBeforeLog.current) setWebcamVisible(false)
    // Direct backend message: bypasses the LLM so _dev_log_mode is always reset
    wsSendRef.current?.(JSON.stringify({ type: 'dev_log_end' }))
  }

  const away = presence === 'away'

  return (
    <>
      <MAYACanvas mayaState={mayaState} presence={presence} />
      <HUDOverlay />

      {/* Dev log bar — sits above everything */}
      <DevLogOverlay
        active={devLogMode}
        showScreen={devLogShowScreen}
        onEnd={handleDevLogEnd}
      />

      {/* Full-screen map */}
      <MapOverlay visible={mapVisible} command={mapCommand} />

      {/* Panels */}
      {/* ChatPanel always mounted — owns the main backend WebSocket */}
      <ChatPanel
        visible={chatVisible}
        isAway={away}
        onStateChange={setMayaState}
        onAction={handleAction}
        visibilityRef={visibilityRef}
        wsSendRef={wsSendRef}
      />
      {/* All other widgets unmount when hidden — no ghost DOM, no dead zones, freed resources */}
      {webcamVisible   && <WebcamWidget visible devLogMode={devLogMode} onPresenceChange={p => setPresence(p ? 'present' : 'away')} />}
      {spotifyVisible  && <SpotifyWidget  visible isAway={away} />}
      {weatherVisible  && <WeatherWidget  visible isAway={away} />}
      {imagesVisible   && <ImagesWidget   visible isAway={away} initialFilename={imageFilename} onInitialHandled={() => setImageFilename(null)} />}
      {imageGenVisible && <ImageGenWidget visible isAway={away} />}
      {browserVisible  && <BrowserWidget  visible url={browserUrl} title={browserTitle} isAway={away} />}
      {videoVisible    && <VideoWidget    visible url={videoUrl} isAway={away} />}
      {monitorVisible  && <SystemMonitorWidget visible isAway={away} />}
      {timerVisible    && <TimerWidget    visible isAway={away} />}
      {notesVisible    && <StickyNotesWidget visible isAway={away} />}
      {calendarVisible && <CalendarWidget visible isAway={away} />}
    </>
  )
}
