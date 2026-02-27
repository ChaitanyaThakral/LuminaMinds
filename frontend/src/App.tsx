import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import ProtectedRoute from './components/ProtectedRoute'
import Home from './pages/Home.tsx'
import TextChat from './pages/TextChat.tsx'
import VoiceChat from './pages/VoiceChat.tsx'
import Privacy from './pages/Privacy.tsx'
import GetHelp from './pages/GetHelp.tsx'
import Feedback from './pages/Feedback.tsx'
import Reports from './pages/Reports.tsx'
import Login from './pages/Login.tsx'
import Signup from './pages/Signup.tsx'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/" element={<ProtectedRoute><Home /></ProtectedRoute>} />
        <Route path="/text" element={<ProtectedRoute><TextChat /></ProtectedRoute>} />
        <Route path="/voice" element={<ProtectedRoute><VoiceChat /></ProtectedRoute>} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/help" element={<GetHelp />} />
        <Route path="/feedback" element={<ProtectedRoute><Feedback /></ProtectedRoute>} />
        <Route path="/feedback/:id" element={<ProtectedRoute><Feedback /></ProtectedRoute>} />
        <Route path="/reports" element={<ProtectedRoute><Reports /></ProtectedRoute>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}

export default App
