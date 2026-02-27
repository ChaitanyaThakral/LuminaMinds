import { Link } from 'react-router-dom'

const RESOURCES = [
  { name: 'Emergency', number: '911', desc: 'Police, fire, medical emergencies' },
  { name: 'Suicide & Crisis Lifeline', number: '988', desc: 'Call or text 24/7 for mental health crisis support' },
]

export default function GetHelp() {
  return (
    <div className="flex-1 max-w-2xl mx-auto px-4 py-8">
      <Link to="/" className="text-sm text-lumina-terracotta hover:underline">
        ← Back
      </Link>

      <div className="mt-6 lumina-card p-8">
        <h1 className="text-2xl font-serif font-semibold text-lumina-terracotta mb-2">
          Get Help
        </h1>
        <p className="text-gray-600 mb-6">
          If you or someone you know is in crisis or needs immediate support, please reach out to one of these resources.
        </p>

        <div className="space-y-4">
          {RESOURCES.map((r) => (
            <div
              key={r.name}
              className="rounded-xl bg-lumina-cream/80 p-4 border border-lumina-warm/30"
            >
              <h3 className="font-semibold text-gray-800">{r.name}</h3>
              <a
                href={`tel:${r.number.replace(/\D/g, '')}`}
                className="text-lg font-medium text-lumina-terracotta hover:text-lumina-coral mt-1 block"
              >
                {r.number}
              </a>
              <p className="text-sm text-gray-600 mt-1">{r.desc}</p>
            </div>
          ))}
        </div>

        <div className="mt-8 p-4 rounded-xl bg-amber-50/80 border border-amber-200/50">
          <h3 className="font-semibold text-amber-900 mb-2">Important</h3>
          <p className="text-sm text-amber-800">
            Lumina Mind is not a substitute for professional medical advice, diagnosis, or treatment.
            If you think you may have a medical emergency, call 911 or go to the nearest emergency room immediately.
          </p>
        </div>

        <Link
          to="/privacy"
          className="inline-block mt-6 text-sm text-gray-500 hover:text-lumina-terracotta transition-colors"
        >
          Privacy Policy
        </Link>
      </div>
    </div>
  )
}
