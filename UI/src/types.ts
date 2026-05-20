export type MessageRole = 'j' | 'u'
export type MayaState = 'idle' | 'listening' | 'thinking' | 'speaking'

export interface Message {
  id: number
  role: MessageRole
  text: string
  imagePreview?: string
}

export interface Position {
  x: number
  y: number
}

export interface Dot {
  a: number
  distFactor: number
  sz: number
  spd: number
  op: number
}
