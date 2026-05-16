import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'

import { useWebSocket }  from './hooks/useWebSocket.js'
import Board             from './components/Board.jsx'
import CameraFeed        from './components/CameraFeed.jsx'
import ControlPanel      from './components/ControlPanel.jsx'
import StatusBar         from './components/StatusBar.jsx'

export default function App() {
  const {
    boardState, bestMove,
    trackingStatus, armStatus, wsStatus,
    send,
  } = useWebSocket()

  const [pendingMove, setPendingMove] = useState(null)   // {from, to}
  const [flipped,     setFlipped]     = useState(false)

  // Board requested a move (drag/click)
  const handleBoardMove = useCallback(({ from, to }) => {
    setPendingMove({ from, to })
  }, [])

  // ControlPanel confirmed or typed a move
  const handleSend = useCallback((msg) => {
    send(msg)
    if (msg.type === 'forcemove') setPendingMove(null)
  }, [send])

  const handleCancelPending = useCallback(() => setPendingMove(null), [])

  return (
    <div className="app">
      <StatusBar
        trackingStatus={trackingStatus}
        armStatus={armStatus}
        wsStatus={wsStatus}
      />

      <main className="main-layout">
        {/* ── Left column: board ── */}
        <section className="board-section">
          <div className="board-header">
            <span className="section-title">Tablero</span>
            <motion.button
              className="btn-flip"
              onClick={() => setFlipped(f => !f)}
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.94 }}
              title="Girar tablero"
            >
              ⇅
            </motion.button>
          </div>

          <Board
            boardState={boardState}
            bestMove={bestMove}
            pendingMove={pendingMove}
            onMove={handleBoardMove}
            flipped={flipped}
          />

          {bestMove && (
            <motion.div
              className="bestmove-banner"
              key={bestMove.move}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <span className="bm-label">Jugada sugerida</span>
              <span className="bm-san">{bestMove.san || bestMove.move}</span>
              <span className="bm-uci">({bestMove.move})</span>
            </motion.div>
          )}
        </section>

        {/* ── Right column: camera + controls ── */}
        <aside className="sidebar">
          <CameraFeed />
          <ControlPanel
            pendingMove={pendingMove}
            armStatus={armStatus}
            onSend={handleSend}
            onCancelPending={handleCancelPending}
          />
        </aside>
      </main>
    </div>
  )
}
