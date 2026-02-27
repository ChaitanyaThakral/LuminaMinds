import { createContext, useContext, useState, useEffect, useCallback } from 'react'

const TOKEN_KEY = 'lumina_token'
const USER_KEY = 'lumina_user'
const STORAGE = sessionStorage

interface User {
  id: string
  email: string
}

interface AuthContextValue {
  user: User | null
  token: string | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (email: string, password: string) => Promise<void>
  logout: () => void
  setToken: (token: string | null) => void
  setUser: (user: User | null) => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

const API_BASE = '/api'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const stored = STORAGE.getItem(USER_KEY)
      return stored ? JSON.parse(stored) : null
    } catch {
      return null
    }
  })
  const [token, setTokenState] = useState<string | null>(() => STORAGE.getItem(TOKEN_KEY))
  const [loading, setLoading] = useState(false)

  const setToken = useCallback((t: string | null) => {
    if (t) {
      STORAGE.setItem(TOKEN_KEY, t)
    } else {
      STORAGE.removeItem(TOKEN_KEY)
    }
    setTokenState(t)
  }, [])

  useEffect(() => {
    if (user) {
      STORAGE.setItem(USER_KEY, JSON.stringify(user))
    } else {
      STORAGE.removeItem(USER_KEY)
    }
  }, [user])

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Login failed')
      setToken(data.token)
      setUser(data.user)
    } finally {
      setLoading(false)
    }
  }, [setToken])

  const signup = useCallback(async (email: string, password: string) => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Signup failed')
      setToken(data.token)
      setUser(data.user)
    } finally {
      setLoading(false)
    }
  }, [setToken])

  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
  }, [setToken])

  const value: AuthContextValue = {
    user,
    token,
    loading,
    login,
    signup,
    logout,
    setToken,
    setUser,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}

export function getAuthHeaders(): Record<string, string> {
  const token = STORAGE.getItem(TOKEN_KEY)
  if (!token) return {}
  return { Authorization: `Bearer ${token}` }
}
