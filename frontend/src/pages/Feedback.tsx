import { useState, useEffect } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { getAuthHeaders } from '../contexts/AuthContext'

const API_BASE = '/api'

interface Report {
  id: string
  content: string
  createdAt: string
  source: string
}

export default function Feedback() {
  const location = useLocation()
  const navigate = useNavigate()
  const { id } = useParams()
  const [report, setReport] = useState<Report | undefined>(location.state?.report as Report | undefined)
  const [loading, setLoading] = useState(Boolean(id) && !report)

  useEffect(() => {
    if (id && !report) {
      fetch(`${API_BASE}/feedback/reports/${id}`, { headers: getAuthHeaders() })
        .then((res) => res.json())
        .then((data) => data.report && setReport(data.report))
        .catch(console.error)
        .finally(() => setLoading(false))
    }
  }, [id, report])

  if (loading) {
    return (
      <div className="flex-1 flex justify-center items-center">
        <div className="flex gap-1">
          <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="flex-1 flex flex-col max-w-3xl mx-auto w-full px-4 py-6">
        <Link to="/" className="text-sm text-lumina-terracotta hover:underline">
          ← Back
        </Link>
        <div className="mt-8 p-6 lumina-card text-center">
          <p className="text-gray-600">No feedback report available.</p>
          <Link
            to="/reports"
            className="mt-4 inline-block text-lumina-terracotta hover:underline font-medium"
          >
            View your reports
          </Link>
        </div>
      </div>
    )
  }

  const date = new Date(report.createdAt).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  })

  return (
    <div className="flex-1 flex flex-col max-w-3xl mx-auto w-full px-4 py-6">
      <Link to="/" className="text-sm text-lumina-terracotta hover:underline">
        ← Back
      </Link>
      <div className="mt-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-serif font-semibold text-lumina-terracotta">
            Session Feedback Report
          </h1>
          <span className="text-xs text-gray-500 uppercase">
            {report.source === 'voice' ? 'Voice' : 'Text'} · {date}
          </span>
        </div>
        <div className="lumina-card p-6">
          <div className="prose prose-gray max-w-none text-gray-700 whitespace-pre-wrap">
            {report.content}
          </div>
        </div>
        <div className="mt-6 flex gap-4">
          <Link
            to="/reports"
            className="rounded-xl bg-lumina-terracotta text-white px-5 py-3 font-medium hover:bg-lumina-coral transition-colors"
          >
            View all reports
          </Link>
          <button
            onClick={() => navigate('/')}
            className="rounded-xl border border-lumina-warm/50 bg-white text-gray-700 px-5 py-3 font-medium hover:bg-lumina-sand/40 transition-colors"
          >
            Home
          </button>
        </div>
      </div>
    </div>
  )
}
