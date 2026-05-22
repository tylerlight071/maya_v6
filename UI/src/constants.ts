export const WS_URL = 'ws://127.0.0.1:8765'

export const CANVAS = {
  RADIUS_FACTOR: 0.145,
  ORB_Y_FACTOR: 0.5,
  DOT_COUNT: 65,
} as const

export const TIMING = {
  INTRO_MS:          3200,
  CHAT_APPEAR_MS:    3500,
  WEBCAM_APPEAR_MS:  3700,
  SPOTIFY_APPEAR_MS: 3900,
  MAYA_REPLY_MS:     850,

  // WebSocket reconnect — used by all widget WS connections
  WS_RECONNECT_MS:     2000,   // initial backoff on disconnect
  WS_RECONNECT_MAX_MS: 30_000, // maximum backoff cap

  // DevLog overlay timing
  DEVLOG_BADGE_MS:   1500,   // how long to show ✓ SAVED before closing
  DEVLOG_STOP_MS:    10_000, // safety timeout if onstop never fires

  // Map
  MAP_RESET_HIDE_MS: 2600,   // delay before hiding map after reset
} as const

export const SPOTIFY_WS        = 'ws://127.0.0.1:8767'
export const WEATHER_WS        = 'ws://127.0.0.1:8768'
export const IMAGES_WS         = 'ws://127.0.0.1:8769'
export const IMAGEGEN_WS       = 'ws://127.0.0.1:8770'
export const SYSTEM_MONITOR_WS = 'ws://127.0.0.1:8771'
export const TIMER_WS          = 'ws://127.0.0.1:8772'
export const STICKY_NOTES_WS   = 'ws://127.0.0.1:8773'
export const CALENDAR_WS       = 'ws://127.0.0.1:8774'

export const PANEL = {
  MAX_WIDTH: 580,
  EST_HEIGHT: 380,
  BOTTOM_OFFSET: 16,
} as const

// Purple palette — single source of truth for all canvas colour strings
export const C = {
  gridLine:        'rgba(72, 35, 130, 0.13)',
  gridDot:         'rgba(110, 55, 200, 0.20)',
  bracketTop:      'rgba(205, 170, 255, 0.60)',
  bracketRuler:    'rgba(175, 130, 235, 0.24)',
  cornerStroke:    'rgba(165, 100, 235, 0.48)',
  cornerMid:       'rgba(165, 100, 235, 0.36)',
  cornerDot:       'rgba(165, 100, 235, 0.40)',
  dot:             (op: number) => `rgba(165, 105, 255, ${op})`,
  outerGlow0:      'rgba(85, 20, 240, 0.14)',
  outerGlow1:      'rgba(55, 10, 160, 0.06)',
  ringGlow0:       'rgba(105, 30, 255, 0.24)',
  ringGlow1:       'rgba(68, 14, 215, 0.10)',
  centerDark0:     '#06030f',
  centerDark1:     '#0a0514',
  centerDark2:     '#0d0618',
  faintRing:       'rgba(122, 52, 228, 0.18)',
  innerRing:       'rgba(228, 205, 255, 0.84)',
  innerRingShadow: 'rgba(215, 180, 255, 0.45)',
  waveform:        (a: number) => `rgba(185, 125, 255, ${a})`,
  bgCenter:        '#160c2c',
  bgEdge:          '#0d0a18',
} as const
