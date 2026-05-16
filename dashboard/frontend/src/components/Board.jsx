import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

// Square name from display coords
function toSquareName(displayRow, displayCol, flipped) {
  const r = flipped ? 7 - displayRow : displayRow
  const c = flipped ? 7 - displayCol : displayCol
  const file = FILES[c]
  const rank = 8 - r
  return file + rank
}

// "e2" → { row: 6, col: 4 } in standard display (rank-1 at bottom)
function parseSquare(sq) {
  if (!sq || sq.length < 2) return null
  const col  = sq.charCodeAt(0) - 97
  const rank = parseInt(sq[1], 10)
  const row  = 8 - rank
  return { row, col }
}

function flipPos(pos) {
  return { row: 7 - pos.row, col: 7 - pos.col }
}

// Parse UCI string "e2e4" into arrow coords (display space)
function parseArrow(uci, flipped) {
  if (!uci || uci.length < 4) return null
  let from = parseSquare(uci.slice(0, 2))
  let to   = parseSquare(uci.slice(2, 4))
  if (!from || !to) return null
  if (flipped) { from = flipPos(from); to = flipPos(to) }
  return { from, to }
}

// Piece visual component
function Piece({ team, draggable, onDragStart }) {
  return (
    <motion.div
      className={`piece piece-${team.toLowerCase()}`}
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0, opacity: 0 }}
      transition={{ type: 'spring', stiffness: 600, damping: 28 }}
      draggable={draggable}
      onDragStart={onDragStart}
      whileHover={{ scale: 1.08 }}
      whileDrag={{ scale: 1.15, zIndex: 20 }}
    >
      <span className="piece-label">{team === 'RED' ? 'R' : 'G'}</span>
    </motion.div>
  )
}

// Best move SVG arrow overlay
function BestMoveArrow({ arrow }) {
  if (!arrow) return null
  const { from, to } = arrow
  const x1 = from.col + 0.5
  const y1 = from.row + 0.5
  const x2 = to.col   + 0.5
  const y2 = to.row   + 0.5

  // Shorten line slightly so arrowhead doesn't overlap piece
  const dx = x2 - x1, dy = y2 - y1
  const len = Math.sqrt(dx * dx + dy * dy)
  const ux = dx / len, uy = dy / len
  const ax2 = x2 - ux * 0.3, ay2 = y2 - uy * 0.3

  const d = `M${x1},${y1} L${ax2},${ay2}`

  return (
    <svg className="board-arrow-svg" viewBox="0 0 8 8" preserveAspectRatio="none">
      <defs>
        <marker id="ah" markerWidth="3" markerHeight="3" refX="3" refY="1.5" orient="auto">
          <polygon points="0 0, 3 1.5, 0 3" fill="rgba(0,229,192,0.92)" />
        </marker>
      </defs>
      {/* glow layer */}
      <motion.path
        d={d}
        stroke="rgba(0,229,192,0.35)"
        strokeWidth="0.55"
        fill="none"
        strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.35 }}
      />
      {/* solid layer with arrowhead */}
      <motion.path
        d={d}
        stroke="rgba(0,229,192,0.92)"
        strokeWidth="0.22"
        fill="none"
        strokeLinecap="round"
        markerEnd="url(#ah)"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.35 }}
      />
    </svg>
  )
}

export default function Board({
  boardState,
  bestMove,
  pendingMove,
  onMove,
  flipped = false,
}) {
  const [selected, setSelected] = useState(null)
  const dragFromRef = useRef(null)

  const arrow = bestMove
    ? parseArrow(bestMove.move || bestMove.uci || '', flipped)
    : null

  // Highlight squares from pending move
  const pendingFrom = pendingMove?.from ?? null
  const pendingTo   = pendingMove?.to   ?? null

  const handleClick = useCallback((sq) => {
    if (!selected) {
      setSelected(sq)
    } else {
      if (selected !== sq) onMove({ from: selected, to: sq })
      setSelected(null)
    }
  }, [selected, onMove])

  const handleDragStart = useCallback((sq) => {
    dragFromRef.current = sq
    setSelected(sq)
  }, [])

  const handleDrop = useCallback((e, sq) => {
    e.preventDefault()
    const from = dragFromRef.current
    if (from && from !== sq) onMove({ from, to: sq })
    dragFromRef.current = null
    setSelected(null)
  }, [onMove])

  const handleDragOver = (e) => e.preventDefault()

  const rows    = Array.from({ length: 8 }, (_, i) => i)
  const cols    = Array.from({ length: 8 }, (_, i) => i)
  const ranks   = flipped
    ? ['1','2','3','4','5','6','7','8']
    : ['8','7','6','5','4','3','2','1']
  const files   = flipped ? [...FILES].reverse() : FILES

  return (
    <div className="board-container">
      <div className="board-with-ranks">
        {/* Rank labels */}
        <div className="rank-labels">
          {ranks.map(r => (
            <span key={r} className="coord-label">{r}</span>
          ))}
        </div>

        {/* Board grid + overlay */}
        <div className="board-inner">
          <div className="board-grid">
            {rows.map(displayRow =>
              cols.map(displayCol => {
                const sq        = toSquareName(displayRow, displayCol, flipped)
                const dataRow   = flipped ? 7 - displayRow : displayRow
                const dataCol   = flipped ? 7 - displayCol : displayCol
                const state     = boardState?.[dataRow]?.[dataCol] ?? 'EMPTY'
                const isLight   = (displayRow + displayCol) % 2 === 0
                const isSel     = selected === sq
                const isPFrom   = pendingFrom === sq
                const isPTo     = pendingTo   === sq

                // Arrow highlight (subtle bg on from/to squares)
                const isAFrom   = arrow && arrow.from.row === displayRow && arrow.from.col === displayCol
                const isATo     = arrow && arrow.to.row   === displayRow && arrow.to.col   === displayCol

                const cls = [
                  'square',
                  isLight ? 'sq-light' : 'sq-dark',
                  isSel   ? 'sq-selected'  : '',
                  isPFrom ? 'sq-pending-from' : '',
                  isPTo   ? 'sq-pending-to'   : '',
                  isAFrom ? 'sq-arrow-from'   : '',
                  isATo   ? 'sq-arrow-to'     : '',
                ].filter(Boolean).join(' ')

                return (
                  <div
                    key={sq}
                    className={cls}
                    onClick={() => handleClick(sq)}
                    onDragOver={handleDragOver}
                    onDrop={e => handleDrop(e, sq)}
                  >
                    <AnimatePresence mode="wait">
                      {state !== 'EMPTY' && (
                        <Piece
                          key={state + sq}
                          team={state}
                          draggable
                          onDragStart={() => handleDragStart(sq)}
                        />
                      )}
                    </AnimatePresence>
                  </div>
                )
              })
            )}
          </div>

          {/* Arrow overlay */}
          <BestMoveArrow arrow={arrow} />
        </div>
      </div>

      {/* File labels */}
      <div className="file-labels-row">
        <div className="rank-labels-spacer" />
        <div className="file-labels">
          {files.map(f => (
            <span key={f} className="coord-label">{f}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
