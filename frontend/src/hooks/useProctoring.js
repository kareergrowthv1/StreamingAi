/**
 * STEP 4 & 5 — AI Proctoring in Browser + WebSocket
 * MediaPipe Face Detection (presence, multiple faces) + FaceMesh (468 landmarks)
 * Head pose from landmarks → send events: no_face, multiple_faces, head_turned
 * WebSocket sends only event JSON to backend.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

function getProctoringWsUrl() {
  const base = import.meta.env.VITE_WS_BASE || (typeof location !== 'undefined' ? location.origin : '')
  const url = base ? base.replace(/^http/, 'ws').replace(/\/$/, '') + '/ws/proctoring' : ''
  return url || (typeof location !== 'undefined' ? (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws/proctoring' : 'ws://127.0.0.1:9000/ws/proctoring')
}

// Head pose thresholds (radians)
const YAW_THRESHOLD = 0.35
const PITCH_THRESHOLD = 0.35

const NOSE_TIP = 1
const LEFT_EYE = 33
const RIGHT_EYE = 263
const LEFT_FACE = 234
const RIGHT_FACE = 454

function estimateHeadPose(landmarks) {
  if (!landmarks || landmarks.length < 468) return { yaw: 0, pitch: 0 }
  const nose = landmarks[NOSE_TIP]
  const leftEye = landmarks[LEFT_EYE]
  const rightEye = landmarks[RIGHT_EYE]
  const leftFace = landmarks[LEFT_FACE]
  const rightFace = landmarks[RIGHT_FACE]
  if (!nose || !leftEye || !rightEye) return { yaw: 0, pitch: 0 }
  const eyeCenterX = (leftEye.x + rightEye.x) / 2
  const eyeCenterY = (leftEye.y + rightEye.y) / 2
  const faceWidth = rightFace && leftFace ? Math.abs(rightFace.x - leftFace.x) : 0.3
  const yaw = faceWidth > 0 ? (nose.x - eyeCenterX) / faceWidth : 0
  const pitch = (nose.y - eyeCenterY) / 0.2
  return { yaw, pitch }
}

export function useProctoring(clientId, positionId, candidateId, enabled = false) {
  const [status, setStatus] = useState('idle')
  const [lastEvent, setLastEvent] = useState(null)
  const wsRef = useRef(null)
  const faceLandmarkerRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const rafRef = useRef(null)
  const sentNoFaceRef = useRef(false)

  const sendEvent = useCallback((event, confidence) => {
    const payload = JSON.stringify({
      event,
      confidence: typeof confidence === 'number' ? confidence : 1,
      timestamp: Date.now() / 1000,
      clientId,
      positionId,
      candidateId,
    })
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(payload)
      setLastEvent({ event, confidence, at: Date.now() })
    }
  }, [clientId, positionId, candidateId])

  useEffect(() => {
    if (!enabled || !clientId || !positionId || !candidateId) return
    const wsUrl = getProctoringWsUrl()
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws
    setStatus('connecting')
    ws.onopen = () => setStatus('active')
    ws.onerror = () => setStatus('error')
    ws.onclose = () => setStatus('idle')
    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [enabled, clientId, positionId, candidateId])

  useEffect(() => {
    if (!enabled || status !== 'active') return
    let cancelled = false
    const runDetection = async () => {
      try {
        const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision')
        const wasmPath = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        const vision = await FilesetResolver.forVisionTasks(wasmPath)
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          },
          numFaces: 2,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrix: false,
          runningMode: 'VIDEO',
        })
        faceLandmarkerRef.current = faceLandmarker
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
        streamRef.current = stream
        const video = document.createElement('video')
        video.srcObject = stream
        video.playsInline = true
        video.muted = true
        await video.play()
        videoRef.current = video

        function detect() {
          if (cancelled || !faceLandmarkerRef.current || !videoRef.current) return
          const results = faceLandmarkerRef.current.detectForVideo(videoRef.current, performance.now())
          const faces = results?.faceLandmarks ?? []
          if (faces.length === 0) {
            if (!sentNoFaceRef.current) {
              sendEvent('no_face', 1)
              sentNoFaceRef.current = true
            }
          } else {
            sentNoFaceRef.current = false
            if (faces.length > 1) sendEvent('multiple_faces', 0.95)
            const landmarks = faces[0]
            const { yaw, pitch } = estimateHeadPose(landmarks)
            if (Math.abs(yaw) > YAW_THRESHOLD || Math.abs(pitch) > PITCH_THRESHOLD) {
              sendEvent('head_turned', 0.9)
            }
          }
          rafRef.current = requestAnimationFrame(detect)
        }
        rafRef.current = requestAnimationFrame(detect)
      } catch (err) {
        console.warn('MediaPipe proctoring init failed:', err)
        setStatus('error')
      }
    }
    runDetection()
    return () => {
      cancelled = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      videoRef.current = null
    }
  }, [enabled, status, sendEvent])

  return { status, lastEvent }
}
