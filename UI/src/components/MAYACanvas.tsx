import { useRef } from 'react'
import type { MayaState } from '../types'
import { useMAYACanvas } from '../hooks/useMAYACanvas'

interface MAYACanvasProps {
  mayaState: MayaState
  presence:  'present' | 'away'
}

export default function MAYACanvas({ mayaState, presence }: MAYACanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useMAYACanvas(canvasRef, mayaState, presence)

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 0 }}
    />
  )
}
