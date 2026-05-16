import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const TOOLTIP_TEXT =
  'Notación UCI: pieza origen + pieza destino, p.ej. "e2e4" ' +
  '(peón de e2 a e4). Columnas: a–h. Filas: 1–8.'

function Tooltip({ text }) {
  const [show, setShow] = useState(false)
  return (
    <span className="tooltip-wrap">
      <button
        className="tooltip-btn"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
        aria-label="Ayuda notación"
        type="button"
      >?</button>
      <AnimatePresence>
        {show && (
          <motion.div
            className="tooltip-box"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
          >
            {text}
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  )
}

export default function ControlPanel({
  pendingMove,
  armStatus,
  onSend,
  onCancelPending,
}) {
  const [uciInput, setUciInput] = useState('')
  const [inputErr, setInputErr] = useState('')
  const inputRef = useRef(null)

  // If board selection filled a pending move, mirror it in the input
  const displayUci = pendingMove
    ? (pendingMove.from + pendingMove.to)
    : uciInput

  function validateUci(v) {
    const ok = /^[a-h][1-8][a-h][1-8]$/.test(v)
    setInputErr(ok || v === '' ? '' : 'Formato inválido (ej. e2e4)')
    return ok
  }

  function handleInputChange(e) {
    const v = e.target.value.toLowerCase().trim()
    setUciInput(v)
    validateUci(v)
  }

  function handleSendMove() {
    const uci = pendingMove ? pendingMove.from + pendingMove.to : uciInput
    if (!validateUci(uci)) return
    onSend({ type: 'forcemove', uci })
    setUciInput('')
    setInputErr('')
  }

  const armConnected = armStatus === 'connected'

  return (
    <div className="control-card">
      <div className="card-header">
        <span className="card-title">Control</span>
      </div>

      {/* Camera pipeline commands */}
      <div className="ctrl-section">
        <p className="ctrl-label">Pipeline</p>
        <div className="btn-row">
          <motion.button
            className="btn btn-primary"
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSend({ type: 'cmd', cmd: 'calibrate' })}
          >
            ◎ Calibrar
          </motion.button>
          <motion.button
            className="btn btn-secondary"
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSend({ type: 'cmd', cmd: 'analyze' })}
          >
            ⚡ Analizar
          </motion.button>
          <motion.button
            className="btn btn-ghost"
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSend({ type: 'cmd', cmd: 'photo' })}
          >
            📸 Foto
          </motion.button>
        </div>
      </div>

      <div className="ctrl-divider" />

      {/* Forced move */}
      <div className="ctrl-section">
        <p className="ctrl-label">
          Forzar jugada <Tooltip text={TOOLTIP_TEXT} />
        </p>

        {pendingMove ? (
          <AnimatePresence mode="wait">
            <motion.div
              className="pending-move-row"
              key="pending"
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 8 }}
            >
              <span className="pending-badge">
                {pendingMove.from} → {pendingMove.to}
              </span>
              <motion.button
                className="btn btn-confirm"
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                onClick={handleSendMove}
              >
                Confirmar
              </motion.button>
              <motion.button
                className="btn btn-ghost btn-small"
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                onClick={onCancelPending}
              >
                ✕
              </motion.button>
            </motion.div>
          </AnimatePresence>
        ) : (
          <div className="move-input-row">
            <div className="input-wrap">
              <input
                ref={inputRef}
                className={`move-input ${inputErr ? 'input-error' : ''}`}
                type="text"
                maxLength={4}
                placeholder="e2e4"
                value={uciInput}
                onChange={handleInputChange}
                onKeyDown={e => e.key === 'Enter' && handleSendMove()}
                spellCheck={false}
              />
              {inputErr && <span className="input-hint">{inputErr}</span>}
            </div>
            <motion.button
              className="btn btn-confirm"
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              disabled={uciInput.length !== 4 || !!inputErr}
              onClick={handleSendMove}
            >
              Enviar
            </motion.button>
          </div>
        )}
      </div>

      <div className="ctrl-divider" />

      {/* Arm status */}
      <div className="ctrl-section">
        <p className="ctrl-label">Brazo robótico</p>
        <div className={`arm-status ${armConnected ? 'arm-ok' : 'arm-off'}`}>
          <motion.span
            className={`dot ${armConnected ? 'dot-green' : 'dot-red'}`}
            animate={{ opacity: [1, 0.35, 1] }}
            transition={{ repeat: Infinity, duration: armConnected ? 2.5 : 1.2 }}
          />
          {armConnected ? 'Brazo vinculado' : 'Brazo no vinculado aún'}
        </div>
      </div>
    </div>
  )
}
