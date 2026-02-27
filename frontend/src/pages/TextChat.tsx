import { useState, useRef, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { getAuthHeaders } from '../contexts/AuthContext'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  analysis?: { label: string; scores: Record<string, number> }
}

const API_BASE = '/api'
const MIN_INPUTS = 5

export default function TextChat() {
  const navigate = useNavigate()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [finishing, setFinishing] = useState(false)
  const [threadId, setThreadId] = useState<string | null>(null)
  const [overall, setOverall] = useState<{ label: string; scores: Record<string, number> } | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const userInputCount = messages.filter((m) => m.role === 'user').length
  const canFinish = userInputCount >= MIN_INPUTS

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input.trim(),
    }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/chat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content, threadId }),
      })

      const data = await res.json()

      if (!res.ok) throw new Error(data.error || 'Failed to get response')

      setThreadId(data.threadId)

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.message,
      }
      if (data.overall) setOverall(data.overall)
      setMessages((prev) => {
        const updated = prev.map((m) =>
          m.id === userMessage.id && data.analysis
            ? { ...m, analysis: data.analysis }
            : m
        )
        return [...updated, assistantMessage]
      })
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `Sorry, something went wrong: ${err instanceof Error ? err.message : 'Please try again.'}`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleFinishSession = async () => {
    if (!canFinish || finishing) return

    const transcript = messages
      .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
      .join('\n\n')

    if (!overall) {
      alert('Please wait for the conversation analysis to complete before finishing.')
      return
    }

    setFinishing(true)
    try {
      const res = await fetch(`${API_BASE}/feedback/finish-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          conversationId: threadId,
          source: 'text',
          transcript,
          mlOutput: overall,
        }),
      })

      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Failed to generate report')

      setMessages([])
      setThreadId(null)
      setOverall(null)
      navigate('/feedback', { state: { report: data.report } })
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to generate feedback report')
    } finally {
      setFinishing(false)
    }
  }

  return (
    <div className="flex-1 flex flex-col max-w-3xl mx-auto w-full px-4 py-6">
      <div className="mb-6">
        <Link to="/" className="text-sm text-lumina-terracotta hover:underline">
          ← Back
        </Link>
        <h1 className="text-2xl font-serif font-semibold text-lumina-terracotta mt-2">
          Text Chat
        </h1>
        <p className="text-gray-600 text-sm mt-1">
          Reflect at your own pace. Share what's on your mind.
        </p>
      </div>

      <div className="lumina-card flex flex-col h-[min(60vh,500px)] overflow-hidden shrink-0">
        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden p-6 space-y-4 scroll-smooth">
          {messages.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              <p className="font-serif italic">Start a conversation whenever you're ready.</p>
              <p className="text-sm mt-2">This is a safe space to reflect and explore.</p>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-lumina-terracotta/15 text-gray-800'
                    : 'bg-lumina-sand/60 text-gray-700'
                }`}
              >
                <span className="text-xs text-gray-500 font-medium uppercase">{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                <p className="text-sm whitespace-pre-wrap mt-1">{msg.content}</p>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start animate-fade-in">
              <div className="bg-lumina-sand/60 rounded-2xl px-4 py-3">
                <div className="flex gap-1">
                  <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 rounded-full bg-lumina-terracotta/60 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="flex flex-col gap-3 p-4 border-t border-lumina-warm/30">
          {canFinish && (
            <button
              type="button"
              onClick={handleFinishSession}
              disabled={finishing}
              className="w-full rounded-xl border-2 border-lumina-forest/30 bg-lumina-sage/20 text-lumina-forest px-4 py-3 font-medium hover:bg-lumina-sage/40 disabled:opacity-50 transition-colors"
            >
              {finishing ? 'Generating report...' : 'Finish Session'}
            </button>
          )}
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 rounded-xl border border-lumina-warm/50 bg-white px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-lumina-terracotta/30 focus:border-lumina-terracotta"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="rounded-xl bg-lumina-terracotta text-white px-5 py-3 font-medium hover:bg-lumina-coral disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
