import React, { useState } from 'react'
import Navbar from './components/Navbar.jsx'
import ObjectTracking from './pages/ObjectTracking.jsx'
import FaceDetection from './pages/FaceDetection.jsx'

export default function App() {
  const [page, setPage] = useState('tracking')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', minHeight: 0 }}>
      <Navbar page={page} setPage={setPage} />
      <main style={{ flex: 1, minHeight: 0, minWidth: 0, overflow: 'auto' }}>
        {page === 'tracking' ? <ObjectTracking /> : <FaceDetection />}
      </main>
    </div>
  )
}
