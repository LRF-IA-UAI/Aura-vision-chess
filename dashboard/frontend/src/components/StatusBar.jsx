import { motion } from 'framer-motion'

function Indicator({ label, value, ok }) {
  return (
    <div className="indicator">
      <motion.span
        className={`dot ${ok ? 'dot-green' : 'dot-red'}`}
        animate={{ opacity: [1, 0.35, 1] }}
        transition={{ repeat: Infinity, duration: ok ? 2.8 : 1.1 }}
      />
      <span className="ind-label">{label}</span>
      <span className="ind-value">{value}</span>
    </div>
  )
}

export default function StatusBar({ trackingStatus, armStatus, wsStatus }) {
  const trackingOk = trackingStatus.startsWith('ON')
  const armOk      = armStatus === 'connected'
  const wsOk       = wsStatus  === 'connected'

  const wsLabel =
    wsStatus === 'connected'    ? 'Conectado'    :
    wsStatus === 'reconnecting' ? 'Reconectando…' : 'Desconectado'

  return (
    <header className="status-bar">
      <div className="brand">
        <span className="brand-accent">AURA</span>
        <span className="brand-sub">Robot Ajedrecista · CAETI UAI</span>
      </div>

      <div className="indicators">
        <Indicator label="Tracking"  value={trackingStatus} ok={trackingOk} />
        <Indicator label="Brazo"     value={armOk ? 'Vinculado' : 'Desvinculado'} ok={armOk} />
        <Indicator label="Dashboard" value={wsLabel} ok={wsOk} />
      </div>
    </header>
  )
}
