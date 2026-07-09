import { useState, useRef, useEffect } from 'react'
import type { MutableRefObject } from 'react'
import type { Message, MayaState } from '../types'
import type {} from '../types/electron'  // augments Window with mayaElectron
import { WS_URL, TIMING, wsUrl } from '../constants'
import { useDraggable } from '../hooks/useDraggable'
import styles from './ChatPanel.module.css'

interface ChatPanelProps {
  visible: boolean
  isAway?: boolean
  onStateChange: (state: MayaState) => void
  onAction?: (action: string, widget?: string, visible?: boolean, extra?: Record<string, unknown>) => void
  visibilityRef?: MutableRefObject<Record<string, boolean>>
  // Populated by ChatPanel when WS is open so App can send messages to backend
  wsSendRef?: MutableRefObject<((msg: string) => void) | null>
}

export default function ChatPanel({ visible, isAway = false, onStateChange, onAction, visibilityRef, wsSendRef }: ChatPanelProps) {
  const panelRef     = useRef<HTMLDivElement>(null)
  const logRef       = useRef<HTMLDivElement>(null)
  const wsRef        = useRef<WebSocket | null>(null)
  const nextIdRef    = useRef(1)
  const streamingRef = useRef('')
  const rafRef       = useRef<number | null>(null)

  const { dragging, panelStyle, onHeaderMouseDown, initTopLeft } = useDraggable(panelRef, 'maya-pos-chat')

  const fileInputRef  = useRef<HTMLInputElement>(null)
  const onActionRef      = useRef(onAction)
  useEffect(() => { onActionRef.current = onAction }, [onAction])
  const onStateChangeRef = useRef(onStateChange)
  useEffect(() => { onStateChangeRef.current = onStateChange }, [onStateChange])

  const [messages,      setMessages]      = useState<Message[]>([])
  const [input,         setInput]         = useState('')
  const [streamingText, setStreamingText] = useState('')
  const [wakeActive,    setWakeActive]    = useState(false)
  const [wsConnected,   setWsConnected]   = useState(false)
  const [attachedImage, setAttachedImage] = useState<{ b64: string; preview: string } | null>(null)
  const [draggingFile,  setDraggingFile]  = useState(false)

  // Auto-scroll
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [messages, streamingText])

  // Init panel position
  useEffect(() => {
    if (visible) initTopLeft(20, 65)
  }, [visible, initTopLeft])

  // WebSocket with exponential-backoff auto-reconnect
  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>
    const retryRef = { current: TIMING.WS_RECONNECT_MS as number }

    function connect() {
      const ws = new WebSocket(wsUrl(WS_URL))

      ws.onopen = () => {
        retryRef.current = TIMING.WS_RECONNECT_MS   // reset backoff on success
        setWsConnected(true)
        // Give App.tsx a direct send handle so it can sync widget state in real-time
        if (wsSendRef) {
          wsSendRef.current = (msg: string) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(msg)
          }
        }
      }

      ws.onclose = () => {
        setWsConnected(false)
        if (wsSendRef) wsSendRef.current = null
        reconnectTimer = setTimeout(connect, retryRef.current)
        retryRef.current = Math.min(retryRef.current * 2, TIMING.WS_RECONNECT_MAX_MS)
      }

      ws.onerror = () => ws.close()

      ws.onmessage = (e: MessageEvent) => {
        try {
          const msg = JSON.parse(e.data as string)
          switch (msg.type as string) {
            case 'wake_detected':
              setWakeActive(true)
              onStateChangeRef.current('listening')
              break
            case 'transcript':
              setWakeActive(false)
              setMessages(prev => [...prev, { id: nextIdRef.current++, role: 'u', text: msg.text as string }])
              onStateChangeRef.current('thinking')
              break
            case 'token':
              streamingRef.current += msg.text as string
              if (!rafRef.current) {
                rafRef.current = requestAnimationFrame(() => {
                  setStreamingText(streamingRef.current)
                  rafRef.current = null
                })
              }
              break
            case 'done': {
              if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
              const final = streamingRef.current
              if (final) {
                setMessages(prev => [...prev, { id: nextIdRef.current++, role: 'j', text: final }])
              }
              streamingRef.current = ''
              setStreamingText('')
              break
            }
            case 'speaking':
              onStateChangeRef.current('speaking')
              break
            case 'ready':
              setWakeActive(false)
              onStateChangeRef.current('idle')
              break
            case 'action':
              onActionRef.current?.(msg.action as string, msg.widget as string, msg.visible as boolean, msg as Record<string, unknown>)
              break
            case 'capture_screen_request': {
              const maya = window.mayaElectron
              if (maya?.captureScreen) {
                maya.captureScreen().then(dataUrl => {
                  if (dataUrl && wsRef.current?.readyState === WebSocket.OPEN) {
                    const b64 = dataUrl.split(',')[1]
                    wsRef.current.send(JSON.stringify({ type: 'screen_capture', image: b64 }))
                  }
                })
              }
              break
            }
            case 'shutdown': {
              const maya = window.mayaElectron
              if (maya?.shutdown) {
                maya.shutdown()
              } else {
                window.close()
              }
              break
            }
            case 'error':
              console.error('[maya backend]', msg.message)
              onStateChangeRef.current('idle')
              break
          }
        } catch (err) {
          console.error('[ws parse]', err)
        }
      }

      wsRef.current = ws
    }

    connect()
    return () => {
      clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps — callbacks held in refs, WS must not reconnect on re-renders

  function handleFiles(files: FileList | null): void {
    if (!files || files.length === 0) return
    const file = files[0]
    if (!file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      const MAX = 2048
      const scale = Math.min(1, MAX / Math.max(img.width, img.height))
      const w = Math.round(img.width * scale)
      const h = Math.round(img.height * scale)
      const canvas = document.createElement('canvas')
      canvas.width = w
      canvas.height = h
      canvas.getContext('2d')!.drawImage(img, 0, 0, w, h)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85)
      URL.revokeObjectURL(url)
      setAttachedImage({ b64: dataUrl.split(',')[1], preview: dataUrl })
    }
    img.onerror = () => URL.revokeObjectURL(url)
    img.src = url
  }

  function handlePaste(e: React.ClipboardEvent): void {
    const items = e.clipboardData?.items
    if (!items) return
    for (const item of Array.from(items)) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        e.preventDefault()
        handleFiles([item.getAsFile()!] as unknown as FileList)
        break
      }
    }
  }

  function handleDragOver(e: React.DragEvent): void {
    if (e.dataTransfer.types.includes('Files')) { e.preventDefault(); setDraggingFile(true) }
  }
  function handleDragLeave(): void { setDraggingFile(false) }
  function handleDrop(e: React.DragEvent): void {
    e.preventDefault()
    setDraggingFile(false)
    handleFiles(e.dataTransfer.files)
  }

  function clearLog(): void {
    setMessages([])
    streamingRef.current = ''
    setStreamingText('')
  }

  function sendText(): void {
    const text = input.trim()
    if ((!text && !attachedImage) || wsRef.current?.readyState !== WebSocket.OPEN) return
    setMessages(prev => [...prev, {
      id: nextIdRef.current++,
      role: 'u',
      text: text || ' ',
      imagePreview: attachedImage?.preview,
    }])
    setInput('')
    setAttachedImage(null)
    onStateChangeRef.current('thinking')
    const payload: Record<string, string> = { type: 'send_text', text }
    if (attachedImage) payload.image = attachedImage.b64
    wsRef.current.send(JSON.stringify(payload))
  }

  function handleKeyDown(e: React.KeyboardEvent): void {
    if (e.key === 'Enter') {
      if (input.trim() === '/clear') { setInput(''); clearLog(); return }
      sendText()
    }
  }

  return (
    <div
      ref={panelRef}
      className={`${styles.panel} ${visible ? styles.visible : ''} ${dragging ? styles.dragging : ''} ${draggingFile ? styles.draggingFile : ''}`}
      style={{
        ...panelStyle,
        ...(isAway ? { opacity: 0, filter: 'blur(3px)', pointerEvents: 'none', transition: 'opacity 0.8s ease, filter 0.8s ease' } : {}),
      }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onPaste={handlePaste}
    >
      <div className={styles.header} onMouseDown={onHeaderMouseDown}>
        <div className={styles.headerLeft}>
          <span className={`${styles.statusDot} ${wsConnected ? styles.connected : ''}`} />
          <span className={styles.headerTitle}>MAYA</span>
        </div>
        <span className={styles.dragHint}>⠿ ⠿</span>
        <div className={styles.headerRight}>
          <span>{wakeActive ? 'LISTENING...' : 'CONVERSATION LOG'}</span>
          <button
            className={styles.clearBtn}
            onClick={clearLog}
            onMouseDown={e => e.stopPropagation()}
            title="Clear chat log"
          >✕</button>
        </div>
      </div>

      <div className={styles.log} ref={logRef}>
        {messages.map(msg =>
          msg.role === 'u' ? (
            <div key={msg.id} className={styles.msgUser}>
              <span className={styles.msgUserBubble}>
                {msg.imagePreview && (
                  <img src={msg.imagePreview} className={styles.msgImage} alt="uploaded" />
                )}
                {msg.text.trim() && <span>{msg.text}</span>}
              </span>
            </div>
          ) : (
            <div key={msg.id} className={styles.msgMAYA}>
              <span className={styles.msgIcon}>◆</span>
              <span className={styles.msgMAYAText}>{msg.text}</span>
            </div>
          )
        )}
        {streamingText && (
          <div className={styles.msgMAYA}>
            <span className={styles.msgIcon}>◆</span>
            <span className={styles.msgMAYAText}>
              {streamingText}<span className={styles.cursor} />
            </span>
          </div>
        )}
        {!streamingText && !wsConnected && (
          <div className={styles.typingRow}>
            <span className={styles.msgIcon}>◆</span>
            <div className={styles.typingDots}>
              <span className={styles.dot} />
              <span className={styles.dot} />
              <span className={styles.dot} />
            </div>
          </div>
        )}
      </div>

      {attachedImage && (
        <div className={styles.imagePreviewRow}>
          <img src={attachedImage.preview} className={styles.imagePreviewThumb} alt="preview" />
          <button
            className={styles.clearImageBtn}
            onClick={() => setAttachedImage(null)}
            onMouseDown={e => e.stopPropagation()}
            title="Remove image"
          >✕</button>
        </div>
      )}

      <div className={styles.inputRow}>
        {wakeActive ? (
          <span className={styles.listeningPulse}>● LISTENING — say your command...</span>
        ) : (
          <>
            <span className={styles.inputPrefix}>▸</span>
            <input
              className={styles.input}
              type="text"
              placeholder="type a message or say Hey Maya..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onMouseDown={e => e.stopPropagation()}
            />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={e => handleFiles(e.target.files)}
              onClick={e => { (e.target as HTMLInputElement).value = '' }}
            />
            <button
              className={styles.attachBtn}
              onClick={() => fileInputRef.current?.click()}
              onMouseDown={e => e.stopPropagation()}
              title="Attach image"
            >⌁</button>
            <button
              className={styles.sendBtn}
              onClick={sendText}
              onMouseDown={e => e.stopPropagation()}
            >
              SEND ↵
            </button>
          </>
        )}
      </div>
    </div>
  )
}
