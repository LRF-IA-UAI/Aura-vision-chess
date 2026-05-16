import { useEffect, useRef, useState, useCallback } from 'react'

const RECONNECT_MS = 3000
const EMPTY_BOARD  = () => Array.from({ length: 8 }, () => Array(8).fill('EMPTY'))

export function useWebSocket() {
  const [boardState,      setBoardState]      = useState(EMPTY_BOARD)
  const [bestMove,        setBestMove]        = useState(null)
  const [trackingStatus,  setTrackingStatus]  = useState('OFF')
  const [armStatus,       setArmStatus]       = useState('disconnected')
  const [wsStatus,        setWsStatus]        = useState('connecting')

  const wsRef      = useRef(null)
  const timerRef   = useRef(null)
  const mountedRef = useRef(true)

  const send = useCallback((data) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true

    function connect() {
      if (!mountedRef.current) return
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const url   = `${proto}://${window.location.host}/ws`
      const ws    = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) return
        setWsStatus('connected')
        clearTimeout(timerRef.current)
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setWsStatus('reconnecting')
        timerRef.current = setTimeout(connect, RECONNECT_MS)
      }

      ws.onerror = () => {
        ws.close()
      }

      ws.onmessage = (evt) => {
        if (!mountedRef.current) return
        let msg
        try { msg = JSON.parse(evt.data) } catch { return }

        switch (msg.type) {
          case 'board':
            if (Array.isArray(msg.state)) setBoardState(msg.state)
            break
          case 'bestmove':
            setBestMove(msg)
            break
          case 'status':
            if (msg.tracking) setTrackingStatus(msg.tracking)
            break
          case 'arm':
            if (msg.status) setArmStatus(msg.status)
            break
          case 'forcemove':
            // optimistic update handled by App
            break
        }
      }
    }

    connect()

    return () => {
      mountedRef.current = false
      clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [])

  return { boardState, bestMove, trackingStatus, armStatus, wsStatus, send }
}
