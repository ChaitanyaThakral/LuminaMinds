/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        lumina: {
          cream: '#FDF8F3',
          sand: '#F5E6D3',
          warm: '#E8D4C4',
          peach: '#E8B4A0',
          coral: '#D4846A',
          terracotta: '#C47050',
          sage: '#9CAF88',
          forest: '#6B7B5C',
          sky: '#B8D4E3',
          lavender: '#D4C4E8',
        },
      },
      fontFamily: {
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        serif: ['Lora', 'Georgia', 'serif'],
      },
      boxShadow: {
        soft: '0 4px 20px rgba(212, 132, 106, 0.12)',
        warm: '0 8px 32px rgba(212, 132, 106, 0.15)',
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
