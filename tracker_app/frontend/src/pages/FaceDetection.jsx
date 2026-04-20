import React, { useState, useRef } from 'react'
import { ScanFace, Upload, Play, Square, Info } from 'lucide-react'

export default function FaceDetection() {
  const [videoSrc, setVideoSrc] = useState(null)
  const [running, setRunning] = useState(false)
  const [frameCount, setFrameCount] = useState(0)
  const [faceCount, setFaceCount] = useState(0)
  const videoRef = useRef()
  const canvasRef = useRef()
  const rafRef = useRef()
  const inputRef = useRef()

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('video/')) return
    setVideoSrc(URL.createObjectURL(file))
    setRunning(false)
    setFrameCount(0)
    setFaceCount(0)
  }

  const startDetection = async () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    setRunning(true)
    let frames = 0

    const tick = () => {
      if (video.paused || video.ended) { setRunning(false); return }
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0)

      // Simulated face boxes — replace with actual face detection API call
      const simFaces = Math.floor(Math.random() * 3)
      setFaceCount(simFaces)
      for (let i = 0; i < simFaces; i++) {
        const x = 50 + i * 120 + Math.random() * 20
        const y = 40 + Math.random() * 30
        const w = 80 + Math.random() * 40
        const h = w * 1.2
        ctx.strokeStyle = '#f5a623'
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, w, h)
        ctx.fillStyle = 'rgba(245,166,35,0.15)'
        ctx.fillRect(x, y, w, h)
        ctx.fillStyle = '#f5a623'
        ctx.font = 'bold 12px Space Mono, monospace'
        ctx.fillText(`Face ${i + 1}`, x + 4, y - 6)
        // Landmark dots
        [[0.3, 0.35], [0.7, 0.35], [0.5, 0.55], [0.3, 0.75], [0.7, 0.75]].forEach(([rx, ry]) => {
          ctx.beginPath()
          ctx.arc(x + rx * w, y + ry * h, 2.5, 0, Math.PI * 2)
          ctx.fillStyle = '#ff6b35'
          ctx.fill()
        })
      }

      frames++
      setFrameCount(frames)
      rafRef.current = requestAnimationFrame(tick)
    }

    video.play()
    rafRef.current = requestAnimationFrame(tick)
  }

  const stopDetection = () => {
    cancelAnimationFrame(rafRef.current)
    videoRef.current?.pause()
    setRunning(false)
  }

  return (
    <div style={{ height: '100%', display: 'flex', overflow: 'hidden' }}>
      {/* Left sidebar */}
      <div style={{
        width: 280, flexShrink: 0, borderRight: '1px solid var(--border)',
        padding: '20px 16px', display: 'flex', flexDirection: 'column', gap: '20px',
        overflow: 'auto',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ScanFace size={14} color="var(--accent)" />
          <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '2px', color: 'var(--text2)' }}>
            FACE DETECTION
          </span>
        </div>

        {/* Info box */}
        <div style={{
          padding: '12px', borderRadius: '8px',
          background: 'rgba(77,159,255,0.08)', border: '1px solid rgba(77,159,255,0.3)',
        }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Info size={14} color="var(--blue)" style={{ flexShrink: 0, marginTop: 1 }} />
            <div style={{ fontSize: '12px', color: 'var(--text2)', lineHeight: 1.6 }}>
              Face detection uses browser-side landmark simulation. Connect to a backend face detection
              model (e.g. InsightFace, MediaPipe, DeepFace) via <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>/api/face/detect</code> to enable real inference.
            </div>
          </div>
        </div>

        {/* Upload */}
        <div>
          <div style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '1px', color: 'var(--text3)', marginBottom: 8 }}>
            VIDEO INPUT
          </div>
          <div
            onClick={() => inputRef.current?.click()}
            style={{
              border: '2px dashed var(--border)', borderRadius: '8px', padding: '24px 12px',
              textAlign: 'center', cursor: 'pointer', background: 'var(--bg3)',
            }}
          >
            <input ref={inputRef} type="file" accept="video/*" style={{ display: 'none' }}
              onChange={e => handleFile(e.target.files[0])} />
            <Upload size={20} style={{ margin: '0 auto 8px', display: 'block', color: 'var(--text3)' }} />
            <div style={{ fontSize: '12px', color: 'var(--text2)' }}>Upload video</div>
          </div>
        </div>

        {/* Controls */}
        {videoSrc && (
          <div style={{ display: 'flex', gap: '8px' }}>
            {!running ? (
              <button onClick={startDetection} style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                padding: '10px', borderRadius: '8px',
                background: 'var(--accent)', border: 'none',
                color: '#000', fontWeight: 800, fontSize: '12px', cursor: 'pointer',
              }}>
                <Play size={13} fill="#000" /> START
              </button>
            ) : (
              <button onClick={stopDetection} style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                padding: '10px', borderRadius: '8px',
                background: 'var(--red)', border: 'none',
                color: '#fff', fontWeight: 800, fontSize: '12px', cursor: 'pointer',
              }}>
                <Square size={13} fill="#fff" /> STOP
              </button>
            )}
          </div>
        )}

        {/* Stats */}
        {videoSrc && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {[
              { label: 'FRAMES PROCESSED', value: frameCount },
              { label: 'FACES IN FRAME', value: faceCount, color: 'var(--accent)' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                padding: '12px', borderRadius: '8px',
                background: 'var(--bg3)', border: '1px solid var(--border)',
              }}>
                <div style={{ fontSize: '10px', letterSpacing: '1px', color: 'var(--text3)', marginBottom: 4 }}>{label}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '22px', fontWeight: 700, color: color || 'var(--text)' }}>
                  {value}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Main canvas area */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', position: 'relative' }}>
        {!videoSrc ? (
          <div style={{ textAlign: 'center', color: 'var(--text3)' }}>
            <ScanFace size={48} style={{ marginBottom: 16, opacity: 0.3 }} />
            <div style={{ fontSize: '14px' }}>Upload a video to begin face detection</div>
          </div>
        ) : (
          <>
            <video ref={videoRef} src={videoSrc} style={{ display: 'none' }} />
            <canvas ref={canvasRef} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
            {running && (
              <div style={{ position: 'absolute', top: 12, left: 12, padding: '3px 10px', borderRadius: '4px', background: 'rgba(255,77,109,0.85)', fontSize: '10px', fontWeight: 700, color: '#fff' }}>
                ● DETECTING
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
