import { useState, useEffect } from 'react'
import styles from './HUDOverlay.module.css'

export default function HUDOverlay() {
  const [time, setTime] = useState('')

  useEffect(() => {
    function tick(): void {
      setTime(new Date().toLocaleTimeString('en-US', { hour12: false }))
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className={styles.overlay}>
      <div className={styles.topBar}>
        <div className={styles.left}>
          <span className={styles.brand}>MAYA</span>
          <span className={styles.sep}>//</span>
          <span className={styles.sub}>AI INTERFACE</span>
        </div>
        <div className={styles.center}>
          <span className={styles.statusDot} />
          <span className={styles.statusLabel}>ONLINE</span>
        </div>
        <div className={styles.right}>
          <span className={styles.time}>{time}</span>
        </div>
      </div>
      <div className={styles.rule} />
    </div>
  )
}
