import { useState, useEffect, useCallback, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Vapi from '@vapi-ai/web'
import MicIcon from '../components/MicIcon'
import { getAuthHeaders } from '../contexts/AuthContext'

const VAPI_PUBLIC_KEY = import.meta.env.VITE_VAPI_PUBLIC_KEY
const VAPI_ASSISTANT_ID = import.meta.env.VITE_VAPI_ASSISTANT_ID
const API_BASE = '/api'
const MIN_INPUTS = 5

export default function VoiceChat() {
  const navigate = useNavigate()
  const [vapi, setVapi] = useState<Vapi | null>(null)
  const [status, setStatus] = useState<'idle' | 'connecting' | 'active' | 'ended'>('idle')
  const [transcripts, setTranscripts] = useState<{ role: string; content: string; clientId?: string; analysis?: { label: string; scores: Record<string, number> } }[]>([])
  const [liveTranscript, setLiveTranscript] = useState<{ role: string; content: string } | null>(null)
  const [overall, setOverall] = useState<{ label: string; scores: Record<string, number> } | null>(null)
  const [finishing, setFinishing] = useState(false)
  const sessionIdRef = useRef<string | null>(null)
  const transcriptEndRef = useRef<HTMLDivElement>(null)

  const userInputCount = transcripts.filter((t) => t.role === 'user').length
  const canFinish = userInputCount >= MIN_INPUTS

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [transcripts, liveTranscript])

  useEffect(() => {
    if (!VAPI_PUBLIC_KEY || !VAPI_ASSISTANT_ID) return

    const vapiInstance = new Vapi(VAPI_PUBLIC_KEY)

    vapiInstance.on('error', (e: unknown) => {
      console.error('VAPI error:', e)
      setStatus('idle')
    })

    vapiInstance.on('call-start', () => {
      sessionIdRef.current = crypto.randomUUID()
      setStatus('active')
      setOverall(null)
      setLiveTranscript(null)
    })
    vapiInstance.on('call-end', async () => {
      setStatus('ended')
      setLiveTranscript(null)
      const sid = sessionIdRef.current
      if (sid) {
        try {
          const res = await fetch(`${API_BASE}/conversations/${sid}/overall`)
          if (res.ok) {
            const data = await res.json()
            if (data.overall) setOverall(data.overall)
          }
        } catch (e) {
          console.error('Failed to fetch overall:', e)
        }
      }
    })

    vapiInstance.on('message', (message: { type?: string; role?: string; transcript?: string; transcriptType?: string; message?: { role?: string; content?: string } }) => {
      const isPartial = message.transcriptType === 'partial'

      const addOrUpdateTranscript = (role: string, content: string) => {
        const clientId = role === 'user' ? crypto.randomUUID() : undefined
        setTranscripts((prev) => {
          const last = prev[prev.length - 1]
          if (last?.role !== role) return [...prev, { role, content, ...(clientId ? { clientId } : {}) }]
          const prevContent = last.content
          if (content.startsWith(prevContent) || prevContent.startsWith(content)) {
            const merged = content.length >= prevContent.length ? content : prevContent
            return [...prev.slice(0, -1), { role, content: merged, ...(clientId ? { clientId } : {}) }]
          }
          let i = 0
          while (i < Math.min(prevContent.length, content.length) && prevContent[i] === content[i]) i++
          if (i >= 3 && i > Math.min(prevContent.length, content.length) * 0.5) {
            const merged = content.length >= prevContent.length ? content : prevContent
            return [...prev.slice(0, -1), { role, content: merged, ...(clientId ? { clientId } : {}) }]
          }
          return [...prev, { role, content, ...(clientId ? { clientId } : {}) }]
        })
        if (role === 'user' && sessionIdRef.current) {
          fetch(`${API_BASE}/transcripts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              transcript: content,
              source: 'voice',
              conversationId: sessionIdRef.current,
              clientId,
            }),
          })
            .then((res) => res.json())
            .then((data) => {
              if (data.clientId != null) {
                setTranscripts((prev) =>
                  prev.map((t) =>
                    t.clientId === data.clientId ? { ...t, analysis: data.analysis ?? undefined } : t
                  )
                )
              }
              if (data.overall) setOverall(data.overall)
            })
            .catch(console.error)
        }
      }

      if (message.type === 'transcript' && message.role && message.transcript) {
        if (isPartial) {
          setLiveTranscript({ role: message.role, content: message.transcript })
          return
        }
        setLiveTranscript(null)
        addOrUpdateTranscript(message.role, message.transcript)
        return
      }
      if (message.message) {
        const { role, content } = message.message
        if (role && content) addOrUpdateTranscript(role, content)
      }
    })

    setVapi(vapiInstance)
    return () => {
      vapiInstance.stop()
    }
  }, [])

  const startCall = useCallback(async () => {
    if (!vapi || !VAPI_ASSISTANT_ID) return
    setStatus('connecting')
    try {
      await vapi.start(VAPI_ASSISTANT_ID)
    } catch (err) {
      console.error('Failed to start call:', err)
      setStatus('idle')
    }
  }, [vapi])

  const endCall = useCallback(() => {
    vapi?.stop()
    setStatus('idle')
  }, [vapi])

  const handleFinishSession = async () => {
    if (!canFinish || finishing) return

    const sid = sessionIdRef.current
    if (!sid) return

    setFinishing(true)

    try {
      if (status === 'active' || status === 'connecting') {
        vapi?.stop()
        setStatus('ended')
        await new Promise((r) => setTimeout(r, 1500))
      }

      let mlOutput = overall
      if (!mlOutput) {
        const res = await fetch(`${API_BASE}/conversations/${sid}/overall`)
        if (res.ok) {
          const data = await res.json()
          mlOutput = data.overall
        }
      }

      if (!mlOutput) {
        alert('Unable to get conversation analysis. Please try again.')
        setFinishing(false)
        return
      }

      const transcript = transcripts
        .map((t) => `${t.role === 'user' ? 'User' : 'Assistant'}: ${t.content}`)
        .join('\n\n')

      const finishRes = await fetch(`${API_BASE}/feedback/finish-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          conversationId: sid,
          source: 'voice',
          transcript,
          mlOutput,
        }),
      })

      const data = await finishRes.json()
      if (!finishRes.ok) throw new Error(data.error || 'Failed to generate report')

      setTranscripts([])
      setOverall(null)
      sessionIdRef.current = null
      navigate('/feedback', { state: { report: data.report } })
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to generate feedback report')
    } finally {
      setFinishing(false)
    }
  }

  const isConfigured = Boolean(VAPI_PUBLIC_KEY && VAPI_ASSISTANT_ID)

  return (
    <div className="flex-1 flex flex-col max-w-3xl mx-auto w-full px-4 py-6">
      <div className="mb-6">
        <Link to="/" className="text-sm text-lumina-terracotta hover:underline">
          ← Back
        </Link>
        <h1 className="text-2xl font-serif font-semibold text-lumina-terracotta mt-2">
          Voice Chat
        </h1>
        <p className="text-gray-600 text-sm mt-1">
          Speak naturally. Have a voice conversation with our supportive assistant.
        </p>
      </div>

      {!isConfigured && (
        <div className="lumina-card p-6 mb-6 border-amber-200 bg-amber-50/50">
          <p className="text-amber-800">
            Voice chat requires VAPI keys. Add <code className="bg-amber-100 px-1 rounded">VITE_VAPI_PUBLIC_KEY</code> and <code className="bg-amber-100 px-1 rounded">VITE_VAPI_ASSISTANT_ID</code> to <code className="bg-amber-100 px-1 rounded">frontend/.env</code>.
          </p>
        </div>
      )}

      <div className="lumina-card flex flex-col h-[min(60vh,500px)] overflow-hidden shrink-0">
        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden p-6 space-y-2 scroll-smooth">
          {transcripts.length === 0 && status === 'idle' && (
            <div className="text-center py-12 text-gray-500">
              <p className="font-serif italic">Tap the button below to start a voice conversation.</p>
              <p className="text-sm mt-2">Find a quiet place and speak when you're ready.</p>
            </div>
          )}

          {transcripts.map((t, i) => (
            <div
              key={i}
              className={`flex ${t.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                  t.role === 'user'
                    ? 'bg-lumina-terracotta/15 text-gray-800'
                    : 'bg-lumina-sand/60 text-gray-700'
                }`}
              >
                <span className="text-xs text-gray-500 font-medium uppercase">{t.role === 'user' ? 'You' : 'Assistant'}</span>
                <p className="text-sm whitespace-pre-wrap mt-1">{t.content}</p>
              </div>
            </div>
          ))}

          {liveTranscript && (
            <div
              className={`flex ${liveTranscript.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                  liveTranscript.role === 'user'
                    ? 'bg-lumina-terracotta/15 text-gray-800'
                    : 'bg-lumina-sand/60 text-gray-700'
                }`}
              >
                <span className="text-xs text-gray-500 font-medium uppercase">{liveTranscript.role === 'user' ? 'You' : 'Assistant'}</span>
                <p className="text-sm whitespace-pre-wrap mt-1">{liveTranscript.content}</p>
              </div>
            </div>
          )}

          {status === 'connecting' && (
            <div className="flex justify-center py-4">
              <div className="flex gap-1">
                <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          )}
          <div ref={transcriptEndRef} />
        </div>

        <div className="p-6 border-t border-lumina-warm/30 flex flex-col gap-4">
          {canFinish && (status === 'active' || status === 'ended') && (
            <button
              type="button"
              onClick={handleFinishSession}
              disabled={finishing}
              className="w-full rounded-xl border-2 border-lumina-forest/30 bg-lumina-sage/20 text-lumina-forest px-4 py-3 font-medium hover:bg-lumina-sage/40 disabled:opacity-50 transition-colors"
            >
              {finishing ? 'Generating report...' : 'Finish Session'}
            </button>
          )}
          <div className="flex justify-center">
          {(status === 'idle' || status === 'ended') && (
            <button
              onClick={startCall}
              disabled={!isConfigured}
              className="w-20 h-20 rounded-full bg-lumina-terracotta text-white flex items-center justify-center hover:bg-lumina-coral disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 shadow-soft"
            >
              <MicIcon className="w-8 h-8" />
            </button>
          )}
          {(status === 'active' || status === 'connecting') && (
            <button
              onClick={endCall}
              className="w-20 h-20 rounded-full bg-red-500 text-white flex items-center justify-center hover:bg-red-600 transition-all hover:scale-105 shadow-soft animate-pulse"
            >
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
            </button>
          )}
          </div>
        </div>
      </div>
    </div>
  )
}