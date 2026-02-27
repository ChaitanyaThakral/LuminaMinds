import { Link } from 'react-router-dom'
import Disclaimer from './Disclaimer'
import { useAuth } from '../contexts/AuthContext'

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user, token, logout } = useAuth()
  const isLoggedIn = Boolean(token && user)

  return (
    <div
      className="min-h-screen flex flex-col bg-cover bg-center bg-fixed bg-no-repeat"
      style={{ backgroundImage: 'url(/lumina-bg.png)', backgroundColor: 'rgb(250, 248, 245)' }}
    >
      <Disclaimer />
      <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-lumina-warm/30">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 group">
            <img src="/lumina-logo.png" alt="Lumina Mind" className="h-8 object-contain" />
            <span className="text-2xl font-serif font-semibold text-lumina-terracotta group-hover:text-lumina-coral transition-colors">
              Lumina Mind
            </span>
          </Link>
          <nav className="flex items-center gap-4">
            {isLoggedIn ? (
              <>
                <Link
                  to="/text"
                  className="text-sm font-medium text-gray-600 hover:text-lumina-terracotta transition-colors"
                >
                  Text Chat
                </Link>
                <Link
                  to="/voice"
                  className="text-sm font-medium text-gray-600 hover:text-lumina-terracotta transition-colors"
                >
                  Voice Chat
                </Link>
                <span className="text-sm text-gray-500">{user?.email}</span>
                <button
                  onClick={logout}
                  className="text-sm font-medium text-gray-600 hover:text-lumina-terracotta transition-colors"
                >
                  Sign out
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className="text-sm font-medium text-gray-600 hover:text-lumina-terracotta transition-colors"
                >
                  Sign in
                </Link>
                <Link
                  to="/signup"
                  className="text-sm font-medium text-lumina-terracotta hover:text-lumina-coral transition-colors"
                >
                  Sign up
                </Link>
              </>
            )}
          </nav>
        </div>
      </header>

      <main className="flex-1 flex flex-col">
        {children}
      </main>

      <footer className="border-t border-lumina-warm/30 bg-white/60 py-6">
        <div className="max-w-4xl mx-auto px-4 text-center text-sm text-gray-500">
          <p className="mb-2">
            Lumina Mind is not a substitute for professional medical advice, diagnosis, or treatment.
            Always seek the advice of your physician or qualified health provider.
          </p>
          <span className="text-lumina-terracotta font-medium cursor-default">
            Get Help
          </span>
        </div>
      </footer>
    </div>
  )
}
