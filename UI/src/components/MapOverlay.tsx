import { useRef, useEffect, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import styles from './MapOverlay.module.css'

// CARTO Dark Matter — free, no API key, matches the JARVIS aesthetic exactly
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
const GLOBE_CENTER: [number, number] = [0, 25]
const GLOBE_ZOOM = 1.8

export interface MapCommand {
  id:      number
  action:  'fly_to' | 'marker' | 'clear' | 'reset'
  lat?:    number
  lng?:    number
  zoom?:   number
  label?:  string
}

interface MapOverlayProps {
  visible: boolean
  command: MapCommand | null
}

function fmt(n: number, d = 4): string {
  return n.toFixed(d).padStart(d + 4, ' ')
}

export default function MapOverlay({ visible, command }: MapOverlayProps) {
  const containerRef  = useRef<HTMLDivElement>(null)
  const mapRef        = useRef<maplibregl.Map | null>(null)
  const markersRef    = useRef<maplibregl.Marker[]>([])
  const initialised   = useRef(false)
  const [coords, setCoords] = useState({ lat: 0, lng: 0, zoom: GLOBE_ZOOM })
  const [locationLabel, setLocationLabel] = useState<string | null>(null)

  // ── Init map on first show ─────────────────────────────────────────────────
  const initMap = useCallback(() => {
    if (!containerRef.current || initialised.current) return
    initialised.current = true

    const map = new maplibregl.Map({
      container:  containerRef.current,
      style:      MAP_STYLE,
      center:     GLOBE_CENTER,
      zoom:       GLOBE_ZOOM,
      pitch:      0,
      bearing:    0,
      antialias:  true,
      attributionControl: false,
    })

    // Enable globe projection + MAYA-matched atmosphere once style loads
    map.on('style.load', () => {
      try {
        (map as unknown as { setProjection: (p: unknown) => void })
          .setProjection({ type: 'globe' })
      } catch { /* older MapLibre without globe — mercator fallback */ }

      // Match the space/void colour to MAYA's background and tint the atmosphere purple
      try {
        map.setFog({
          color:             '#0d0a18',   // horizon atmosphere — MAYA background
          'high-color':      '#0d0a18',   // upper atmosphere
          'space-color':     '#0d0a18',   // space behind the globe
          'horizon-blend':   0.06,
          'star-intensity':  0.0,
        } as Parameters<typeof map.setFog>[0])
      } catch { /* setFog not available in this MapLibre version */ }
    })

    map.on('move', () => {
      const c = map.getCenter()
      setCoords({ lat: c.lat, lng: c.lng, zoom: map.getZoom() })
    })

    mapRef.current = map
  }, [])

  useEffect(() => {
    if (visible && !initialised.current) {
      // Small delay so the container has layout dimensions before MapLibre measures it
      const t = setTimeout(initMap, 80)
      return () => clearTimeout(t)
    }
    // Resize when visibility changes so the map fills the container properly
    if (visible && mapRef.current) {
      setTimeout(() => mapRef.current?.resize(), 50)
    }
  }, [visible, initMap])

  // ── Handle map commands from MAYA ──────────────────────────────────────────
  useEffect(() => {
    if (!command || !mapRef.current) return
    const map = mapRef.current

    if (command.action === 'fly_to' && command.lat !== undefined && command.lng !== undefined) {
      const targetZoom  = command.zoom ?? 11
      const currentZoom = map.getZoom()
      const label       = command.label ?? null

      setLocationLabel(null)

      if (currentZoom > 4) {
        // Currently zoomed in — pull back to globe first, then fly to new location
        map.flyTo({ zoom: GLOBE_ZOOM, duration: 2200, essential: true })
        setTimeout(() => {
          map.flyTo({
            center:   [command.lng!, command.lat!],
            zoom:     targetZoom,
            duration: 3800,
            essential: true,
            curve:    1.6,
          })
          if (label) setTimeout(() => { setLocationLabel(label); dropMarker(map, command.lat!, command.lng!, label) }, 4000)
        }, 2600)
      } else {
        // Already at globe level — fly straight in
        map.flyTo({
          center:   [command.lng!, command.lat!],
          zoom:     targetZoom,
          duration: 4000,
          essential: true,
          curve:    1.6,
        })
        if (label) setTimeout(() => { setLocationLabel(label); dropMarker(map, command.lat!, command.lng!, label) }, 4200)
      }
    }

    if (command.action === 'marker' && command.lat !== undefined && command.lng !== undefined) {
      dropMarker(map, command.lat, command.lng, command.label ?? '')
    }

    if (command.action === 'clear') {
      markersRef.current.forEach(m => m.remove())
      markersRef.current = []
      setLocationLabel(null)
    }

    if (command.action === 'reset') {
      markersRef.current.forEach(m => m.remove())
      markersRef.current = []
      setLocationLabel(null)
      map.flyTo({ center: GLOBE_CENTER, zoom: GLOBE_ZOOM, duration: 2500, essential: true })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- command is an object with a stable `id`; only its content changes. Listing full deps would cause double-execution on each command object re-creation.
  }, [command])

  function dropMarker(map: maplibregl.Map, lat: number, lng: number, label: string) {
    const el = document.createElement('div')
    el.className = styles.markerEl

    const marker = new maplibregl.Marker({ element: el })
      .setLngLat([lng, lat])
      .addTo(map)

    markersRef.current.push(marker)

    if (label) {
      const popup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false,
        className:   styles.markerPopup,
        offset:      [0, -12],
      }).setText(label)
      marker.setPopup(popup)
      popup.addTo(map)
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className={`${styles.overlay} ${visible ? styles.visible : ''}`}>
      {/* Top accent line */}
      <div className={styles.topBar} />
      {/* Overlays */}
      <div className={styles.mapTint} />
      <div className={styles.grid} />
      <div className={styles.scanlines} />

      {/* Corner HUD brackets */}
      <div className={`${styles.corner} ${styles.cornerTL}`} />
      <div className={`${styles.corner} ${styles.cornerTR}`} />
      <div className={`${styles.corner} ${styles.cornerBL}`} />
      <div className={`${styles.corner} ${styles.cornerBR}`} />

      {/* Top dot-matrix row */}
      <div className={styles.topDots}>
        {Array.from({ length: 40 }).map((_, i) => <span key={i} className={styles.topDot} />)}
      </div>

      {/* Map fills the rest */}
      <div ref={containerRef} className={styles.map} />

      {/* Bottom-left: live coordinates */}
      <div className={styles.coordPanel}>
        <span className={styles.coordRow}><em>LAT</em>{fmt(coords.lat)}</span>
        <span className={styles.coordRow}><em>LNG</em>{fmt(coords.lng)}</span>
        <span className={styles.coordRow}><em>ALT</em>{fmt(coords.zoom, 1)}</span>
        {locationLabel && <span className={styles.coordLabel}>{locationLabel.toUpperCase()}</span>}
      </div>

      {/* Bottom-right: MAYA text badge */}
      <div className={styles.badge}>
        <span className={styles.badgeText}>MAYA</span>
      </div>
    </div>
  )
}
