import { Link } from 'react-router-dom'
import MicIcon from '../components/MicIcon'

export default function Home() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 py-16">
      <div className="max-w-2xl mx-auto text-center space-y-8">
        <h1 className="text-4xl md:text-5xl font-serif font-semibold text-lumina-terracotta">
          Welcome to Lumina Mind
        </h1>
        <p className="text-lg text-gray-600 font-sans">
          A warm, reflective space for wellness and self-care. Choose how you'd like to connect today.
        </p>

        <div className="grid md:grid-cols-2 gap-6 pt-8">
          <Link
            to="/text"
            className="lumina-card p-8 block text-left hover:shadow-warm transition-all duration-300 group"
          >
            <div className="w-12 h-12 rounded-xl bg-lumina-sky/40 flex items-center justify-center mb-4 group-hover:bg-lumina-sky/60 transition-colors">
              <svg className="w-6 h-6 text-lumina-forest" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <h2 className="text-xl font-serif font-semibold text-gray-800 mb-2">Text Chat</h2>
            <p className="text-gray-600 text-sm">
              Write and reflect at your own pace. Our AI assistant is here to listen and support.
            </p>
          </Link>

          <Link
            to="/voice"
            className="lumina-card p-8 block text-left hover:shadow-warm transition-all duration-300 group"
          >
            <div className="w-12 h-12 rounded-xl bg-lumina-lavender/40 flex items-center justify-center mb-4 group-hover:bg-lumina-lavender/60 transition-colors">
              <MicIcon className="w-6 h-6 text-lumina-forest" />
            </div>
            <h2 className="text-xl font-serif font-semibold text-gray-800 mb-2">Voice Chat</h2>
            <p className="text-gray-600 text-sm">
              Speak naturally. Have a voice conversation with our supportive assistant.
            </p>
          </Link>
        </div>

        <div className="pt-12 flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/reports"
            className="inline-flex items-center gap-2 text-lumina-terracotta hover:text-lumina-coral font-medium transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            My Feedback Reports
          </Link>
          <Link
            to="/privacy"
            className="inline-flex items-center gap-2 text-lumina-terracotta hover:text-lumina-coral font-medium transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            Privacy Policy
          </Link>
        </div>
      </div>
    </div>
  )
}
