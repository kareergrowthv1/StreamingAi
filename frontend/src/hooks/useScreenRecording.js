/**
 * STEP 3 — Screen Recording (10-sec chunks)
 * getDisplayMedia → screen, MediaRecorder → stop/restart every 10s so each chunk
 * starts with a keyframe and is independently playable.
 */
import { useCallback, useRef, useState } from 'react'
import axios from 'axios'
import { API_BASE } from '../config'

const CHUNK_DURATION_MS = 10_000

export function useScreenRecording(clientId, positionId, candidateId) {
  const [isRecording, setIsRecording] = useState(false)
  const [chunkIndex, setChunkIndex] = useState(0)
  const [error, setError] = useState(null)
  const [mergeSuccess, setMergeSuccess] = useState(false)
  const mediaRecorderRef = useRef(null)
  const streamRef = useRef(null)
  const audioContextRef = useRef(null)
  const micStreamRef = useRef(null)
  const chunkIndexRef = useRef(0)
  const chunkTimerRef = useRef(null)
  const stillRecordingRef = useRef(false)

  const uploadChunk = useCallback(
    async (blob) => {
      const form = new FormData()
      form.append('client_id', clientId)
      form.append('position_id', positionId)
      form.append('candidate_id', candidateId)
      form.append('index', chunkIndexRef.current)
      form.append('file', blob, `chunk_${chunkIndexRef.current}.webm`)
      await axios.post(`${API_BASE}/api/chunk`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      chunkIndexRef.current += 1
      setChunkIndex(chunkIndexRef.current)
    },
    [clientId, positionId, candidateId]
  )

  const startStreaming = useCallback(async () => {
    if (!clientId || !positionId || !candidateId) {
      setError('Fill clientId, positionId, candidateId')
      return
    }
    setError(null)
    setMergeSuccess(false)
    chunkIndexRef.current = 0
    setChunkIndex(0)
    stillRecordingRef.current = true
    try {
      // 1. Get Screen Stream (System Audio)
      const screenStream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'browser' },
        audio: true, // Request system audio
      })

      // 2. Get Microphone Stream
      let micStream
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: true,
          video: false,
        })
      } catch (err) {
        console.warn('Mic access denied or unavailable', err)
      }

      // 3. Setup Audio Mixing
      const audioContext = new (window.AudioContext || window.webkitAudioContext)()
      const dest = audioContext.createMediaStreamDestination()

      // Mix Screen Audio
      if (screenStream.getAudioTracks().length > 0) {
        const screenSource = audioContext.createMediaStreamSource(screenStream)
        screenSource.connect(dest)
      }

      // Mix Mic Audio
      if (micStream && micStream.getAudioTracks().length > 0) {
        micStreamRef.current = micStream
        const micSource = audioContext.createMediaStreamSource(micStream)
        micSource.connect(dest)
      }

      audioContextRef.current = audioContext

      // 4. Combine Video + Mixed Audio
      const mixedAudioTracks = dest.stream.getAudioTracks()
      const combinedStream = new MediaStream([
        ...screenStream.getVideoTracks(),
        ...mixedAudioTracks
      ])

      streamRef.current = screenStream // Keep ref to screen stream for stopping

      const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm'

      const startNextChunk = () => {
        if (!stillRecordingRef.current || !streamRef.current) return
        // Use combinedStream for recording
        const recorder = new MediaRecorder(combinedStream, {
          mimeType: mime,
          videoBitsPerSecond: 2_500_000,
        })
        mediaRecorderRef.current = recorder
        let chunkBlob = null
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunkBlob = e.data
        }
        recorder.onstop = async () => {
          if (chunkBlob && chunkBlob.size > 0) {
            try {
              await uploadChunk(chunkBlob)
            } catch (err) {
              setError(err.message || 'Chunk upload failed')
            }
          }
          if (!stillRecordingRef.current) return
          chunkTimerRef.current = setTimeout(startNextChunk, 0)
        }
        recorder.start()
        chunkTimerRef.current = setTimeout(() => {
          if (recorder.state === 'recording') recorder.stop()
        }, CHUNK_DURATION_MS)
      }
      startNextChunk()
      setIsRecording(true)
    } catch (err) {
      setError(err.message || 'Start streaming failed')
      stillRecordingRef.current = false
    }
  }, [clientId, positionId, candidateId, uploadChunk])

  const endTest = useCallback(async () => {
    stillRecordingRef.current = false
    if (chunkTimerRef.current) {
      clearTimeout(chunkTimerRef.current)
      chunkTimerRef.current = null
    }
    const recorder = mediaRecorderRef.current
    const screenStream = streamRef.current
    const micStream = micStreamRef.current
    const audioContext = audioContextRef.current

    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
      mediaRecorderRef.current = null
    }

    // Stop Screen Stream
    if (screenStream) {
      screenStream.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }

    // Stop Mic Stream
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop())
      micStreamRef.current = null
    }

    // Close Audio Context
    if (audioContext) {
      audioContext.close()
      audioContextRef.current = null
    }

    setIsRecording(false)
    setError(null)
    try {
      const form = new FormData()
      form.append('client_id', clientId)
      form.append('position_id', positionId)
      form.append('candidate_id', candidateId)
      const res = await axios.post(`${API_BASE}/api/end-test`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      if (res.data?.success) setMergeSuccess(true)
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'End test request failed')
    }
  }, [clientId, positionId, candidateId])

  return { startStreaming, endTest, isRecording, chunkIndex, error, mergeSuccess }
}
