/**
 * STEP 2 — Capture Screenshot
 * getUserMedia → camera, Canvas API → frame as image, Fetch → send to backend
 */
import { useCallback, useRef, useState } from 'react'
import axios from 'axios'
import { API_BASE } from '../config'

export function useScreenshot(clientId, positionId, candidateId) {
  const [status, setStatus] = useState('idle') // idle | capturing | uploading | done | error
  const [error, setError] = useState(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  const captureAndUpload = useCallback(async () => {
    if (!clientId || !positionId || !candidateId) {
      setError('Fill clientId, positionId, candidateId')
      return
    }
    setStatus('capturing')
    setError(null)
    let stream = null
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
      streamRef.current = stream
      const video = document.createElement('video')
      video.srcObject = stream
      video.muted = true
      await video.play()
      const w = video.videoWidth
      const h = video.videoHeight
      const canvas = document.createElement('canvas')
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0)
      stream.getTracks().forEach((t) => t.stop())
      streamRef.current = null

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'))
      setStatus('uploading')
      const form = new FormData()
      form.append('client_id', clientId)
      form.append('position_id', positionId)
      form.append('candidate_id', candidateId)
      form.append('file', blob, 'screenshot.png')
      const res = await axios.post(`${API_BASE}/api/screenshot`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      if (res.data?.success) {
        setStatus('done')
        return res.data
      }
      throw new Error(res.data?.error || 'Upload failed')
    } catch (err) {
      setError(err.message || 'Screenshot failed')
      setStatus('error')
      if (stream) stream.getTracks().forEach((t) => t.stop())
      throw err
    }
  }, [clientId, positionId, candidateId])

  return { captureAndUpload, status, error }
}
