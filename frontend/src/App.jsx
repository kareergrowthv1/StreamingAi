/**
 * STEP 1 — UI
 * Inputs: clientId, positionId, candidateId
 * Buttons: Capture Screenshot, Start Streaming, End Test
 * Wires: screenshot, screen recording (10-sec chunks), AI proctoring (MediaPipe + WebSocket)
 */
import { useState } from 'react'
import { useScreenshot } from './hooks/useScreenshot'
import { useScreenRecording } from './hooks/useScreenRecording'
import { useProctoring } from './hooks/useProctoring'

export default function App() {
  const [clientId, setClientId] = useState('')
  const [positionId, setPositionId] = useState('')
  const [candidateId, setCandidateId] = useState('')
  // New state for video player
  const [showVideo, setShowVideo] = useState(false)

  const { captureAndUpload, status: screenshotStatus, error: screenshotError } = useScreenshot(
    clientId,
    positionId,
    candidateId
  )
  const {
    startStreaming,
    endTest,
    isRecording,
    chunkIndex,
    error: recordingError,
    mergeSuccess,
  } = useScreenRecording(clientId, positionId, candidateId)
  const { status: proctoringStatus, lastEvent } = useProctoring(
    clientId,
    positionId,
    candidateId,
    isRecording
  )

  const handleCapture = async () => {
    try {
      await captureAndUpload()
    } catch (_) { }
  }

  return (
    <div style={{ maxWidth: 560, margin: '0 auto' }}>
      <h1 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Streaming Proctoring</h1>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.5rem' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={{ color: '#94a3b8' }}>Client ID</span>
          <input
            type="text"
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
            placeholder="clientId"
            style={{
              padding: '0.5rem 0.75rem',
              borderRadius: 8,
              border: '1px solid #334155',
              background: '#1e293b',
              color: '#e2e8f0',
            }}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={{ color: '#94a3b8' }}>Position ID</span>
          <input
            type="text"
            value={positionId}
            onChange={(e) => setPositionId(e.target.value)}
            placeholder="positionId"
            style={{
              padding: '0.5rem 0.75rem',
              borderRadius: 8,
              border: '1px solid #334155',
              background: '#1e293b',
              color: '#e2e8f0',
            }}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={{ color: '#94a3b8' }}>Candidate ID</span>
          <input
            type="text"
            value={candidateId}
            onChange={(e) => setCandidateId(e.target.value)}
            placeholder="candidateId"
            style={{
              padding: '0.5rem 0.75rem',
              borderRadius: 8,
              border: '1px solid #334155',
              background: '#1e293b',
              color: '#e2e8f0',
            }}
          />
        </label>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
        <button
          onClick={() => setShowVideo(!showVideo)}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: 8,
            border: '1px solid #475569',
            background: showVideo ? '#eab308' : '#334155', // Yellow if active
            color: showVideo ? '#000' : '#e2e8f0',
            cursor: 'pointer',
          }}
        >
          {showVideo ? 'Hide Test Video' : 'Test Audio (Play Video)'}
        </button>
        <button
          onClick={handleCapture}
          disabled={screenshotStatus === 'capturing' || screenshotStatus === 'uploading'}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: 8,
            border: '1px solid #475569',
            background: screenshotStatus === 'done' ? '#166534' : '#334155',
            color: '#e2e8f0',
            cursor: screenshotStatus === 'capturing' || screenshotStatus === 'uploading' ? 'not-allowed' : 'pointer',
          }}
        >
          {screenshotStatus === 'capturing' || screenshotStatus === 'uploading'
            ? 'Capturing…'
            : screenshotStatus === 'done'
              ? 'Screenshot saved'
              : 'Capture Screenshot'}
        </button>
        <button
          onClick={startStreaming}
          disabled={isRecording}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: 8,
            border: '1px solid #475569',
            background: isRecording ? '#166534' : '#334155',
            color: '#e2e8f0',
            cursor: isRecording ? 'not-allowed' : 'pointer',
          }}
        >
          {isRecording ? `Streaming (chunk ${chunkIndex})` : 'Start Streaming'}
        </button>
        <button
          onClick={endTest}
          disabled={!isRecording}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: 8,
            border: '1px solid #475569',
            background: !isRecording ? '#1e293b' : '#991b1b',
            color: '#e2e8f0',
            cursor: !isRecording ? 'not-allowed' : 'pointer',
          }}
        >
          End Test
        </button>
      </div>

      {/* Video Player Section */}
      {showVideo && (
        <div style={{ marginBottom: '1.5rem', border: '1px solid #475569', borderRadius: 8, overflow: 'hidden' }}>
          <div style={{ padding: '0.5rem', background: '#334155', fontSize: '0.875rem' }}>
            System Audio Test: Play this video and ensure "Share System Audio" is checked when starting stream.
          </div>
          <iframe
            width="100%"
            height="315"
            src="https://www.youtube.com/embed/jfKfPfyJRdk?autoplay=1"
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          ></iframe>
        </div>
      )}

      {(screenshotError || recordingError) && (
        <p style={{ color: '#f87171', fontSize: '0.875rem', marginBottom: '0.5rem' }}>
          {screenshotError || recordingError}
        </p>
      )}

      {mergeSuccess && clientId && positionId && candidateId && (
        <p style={{ marginBottom: '0.5rem' }}>
          <a
            href={`/api/merged/${encodeURIComponent(clientId)}/${encodeURIComponent(positionId)}/${encodeURIComponent(candidateId)}`}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: '#38bdf8' }}
          >
            Play recording
          </a>
        </p>
      )}

      <div style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
        Proctoring: {proctoringStatus}
        {lastEvent && (
          <span style={{ marginLeft: 8 }}>
            — Last: {lastEvent.event} ({lastEvent.confidence})
          </span>
        )}
      </div>
    </div>
  )
}
