import { useState } from 'react'
import { motion } from 'framer-motion'

export default function CameraFeed() {
  const [status, setStatus] = useState('connecting') // 'connecting' | 'live' | 'error'

  return (
    <div className="camera-card">
      <div className="card-header">
        <span className="card-title">Cámara en vivo</span>
        <motion.span
          className={`dot dot-${status === 'live' ? 'green' : status === 'error' ? 'red' : 'yellow'}`}
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ repeat: Infinity, duration: 1.4 }}
        />
      </div>

      <div className="camera-frame">
        <img
          className="camera-img"
          src="/api/stream"
          alt="Feed de cámara"
          onLoad={() => setStatus('live')}
          onError={() => setStatus('error')}
        />
        {status !== 'live' && (
          <div className="camera-overlay">
            {status === 'error'
              ? 'Sin señal — camera_pipeline.py corriendo?'
              : 'Conectando…'}
          </div>
        )}
      </div>
    </div>
  )
}
