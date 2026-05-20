import { useState, useRef, useEffect } from 'react'
import type { RefObject, CSSProperties, MouseEvent } from 'react'
import type { Position } from '../types'

export interface DragResult {
  pos: Position | null
  dragging: boolean
  panelStyle: CSSProperties
  onHeaderMouseDown: (e: MouseEvent) => void
  initBottomCenter: (maxWidth: number, estHeight: number, offset: number) => void
  initTopLeft: (x: number, y: number) => void
}

function loadPos(key: string): Position | null {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return null
    const p = JSON.parse(raw) as Position
    if (typeof p.x !== 'number' || typeof p.y !== 'number') return null
    // Clamp to current viewport so a position saved at a larger resolution
    // doesn't land off-screen.
    return {
      x: Math.max(0, Math.min(p.x, window.innerWidth  - 80)),
      y: Math.max(0, Math.min(p.y, window.innerHeight - 40)),
    }
  } catch {
    return null
  }
}

function savePos(key: string, pos: Position): void {
  try {
    localStorage.setItem(key, JSON.stringify(pos))
  } catch { /* storage full or unavailable — silently ignore */ }
}

export function useDraggable(
  panelRef: RefObject<HTMLDivElement | null>,
  storageKey?: string,
): DragResult {
  const [pos, setPos] = useState<Position | null>(() =>
    storageKey ? loadPos(storageKey) : null
  )
  const [dragging, setDragging] = useState(false)
  const dragOffset = useRef<Position>({ x: 0, y: 0 })

  function initBottomCenter(maxWidth: number, estHeight: number, offset: number): void {
    setPos(prev => {
      if (prev !== null) return prev   // already positioned (dragged or loaded)
      const panelW = Math.min(maxWidth, window.innerWidth - 32)
      return {
        x: Math.round((window.innerWidth - panelW) / 2),
        y: Math.round(window.innerHeight - estHeight - offset),
      }
    })
  }

  function initTopLeft(x: number, y: number): void {
    setPos(prev => prev !== null ? prev : { x, y })
  }

  function onHeaderMouseDown(e: MouseEvent): void {
    e.preventDefault()
    if (!panelRef.current) return
    const rect = panelRef.current.getBoundingClientRect()
    dragOffset.current = { x: e.clientX - rect.left, y: e.clientY - rect.top }
    setDragging(true)
  }

  useEffect(() => {
    if (!dragging) return

    function onMove(e: globalThis.MouseEvent): void {
      const w = panelRef.current?.offsetWidth  ?? 580
      const h = panelRef.current?.offsetHeight ?? 380
      setPos({
        x: Math.max(0, Math.min(window.innerWidth  - w, e.clientX - dragOffset.current.x)),
        y: Math.max(0, Math.min(window.innerHeight - h, e.clientY - dragOffset.current.y)),
      })
    }

    function onUp(): void {
      setDragging(false)
      // Persist final position after drag ends (not on every mousemove)
      if (storageKey) {
        setPos(current => {
          if (current) savePos(storageKey, current)
          return current
        })
      }
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [dragging, panelRef, storageKey])

  const panelStyle: CSSProperties = pos
    ? { left: pos.x, top: pos.y, bottom: 'auto', transform: 'none', ...(dragging ? { zIndex: 100 } : {}) }
    : {}

  return { pos, dragging, panelStyle, onHeaderMouseDown, initBottomCenter, initTopLeft }
}
