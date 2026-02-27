import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getAuthHeaders } from '../contexts/AuthContext'

const API_BASE = '/api'

interface Report {
  id: string
  content: string
  createdAt: string
  source: string
}

export default function Reports() {
  const [reports, setReports] = useState<Report[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API_BASE}/feedback/reports`, { headers: getAuthHeaders() })
      .then((res) => res.json().then((data) => ({ res, data })))
      .then(({ res, data }) => {
        if (res.ok && data.reports) setReports(data.reports)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="flex-1 flex flex-col max-w-3xl mx-auto w-full px-4 py-6">
      <Link to="/" className="text-sm text-lumina-terracotta hover:underline">
        ← Back
      </Link>
      <h1 className="text-2xl font-serif font-semibold text-lumina-terracotta mt-2">
        Feedback Reports
      </h1>
      <p className="text-gray-600 text-sm mt-1">
        Your session feedback reports, ordered by date.
      </p>

      {loading ? (
        <div className="mt-8 flex justify-center">
          <div className="flex gap-1">
            <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
      ) : reports.length === 0 ? (
        <div className="mt-8 p-6 lumina-card text-center">
          <p className="text-gray-600">No feedback reports yet.</p>
          <p className="text-sm text-gray-500 mt-2">
            Complete a session with at least 5 messages and click “Finish Session” to generate a report.
          </p>
          <Link
            to="/text"
            className="mt-4 inline-block text-lumina-terracotta hover:underline font-medium"
          >
            Start a text chat
          </Link>
        </div>
      ) : (
        <div className="mt-6 space-y-4">
          {reports.map((r) => {
            const date = new Date(r.createdAt).toLocaleString(undefined, {
              dateStyle: 'medium',
              timeStyle: 'short',
            })
            const preview = r.content.slice(0, 150) + (r.content.length > 150 ? '...' : '')
            return (
              <Link
                key={r.id}
                to={`/feedback/${r.id}`}
                state={{ report: r } as { report: Report }}
                className="lumina-card p-5 block text-left hover:shadow-warm transition-all"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-lumina-forest uppercase">
                    {r.source === 'voice' ? 'Voice' : 'Text'} · {date}
                  </span>
                </div>
                <p className="text-sm text-gray-700 line-clamp-3">{preview}</p>
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
