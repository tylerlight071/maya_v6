export interface MayaElectron {
  captureScreen?:      () => Promise<string | null>
  shutdown?:           () => void
  saveRecording?:      (buf: Uint8Array, filename: string) => Promise<string | null>
  videoAuthPopup?:     (url: string) => void
  onVideoAuthComplete?:(cb: () => void) => (() => void) | undefined
  getSessionToken?:    () => string
}

declare global {
  interface Window {
    mayaElectron?: MayaElectron
  }
}
