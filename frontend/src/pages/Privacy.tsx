import { Link } from 'react-router-dom'

export default function Privacy() {
  return (
    <div className="flex-1 max-w-2xl mx-auto px-4 py-8">
      <Link to="/" className="text-sm text-lumina-terracotta hover:underline">
        ← Back
      </Link>

      <div className="mt-6 lumina-card p-8">
        <h1 className="text-2xl font-serif font-semibold text-lumina-terracotta mb-2">
          Privacy Policy
        </h1>
        <p className="text-sm text-gray-500 mb-6">
          Last updated: February 2026
        </p>

        <div className="prose prose-gray max-w-none space-y-6 text-gray-700">
          <section>
            <h2 className="text-lg font-semibold text-gray-800 mb-2">Our Commitment to Your Privacy</h2>
            <p className="text-sm leading-relaxed">
              Lumina Mind is committed to protecting your personal information and health-related data.
              This policy explains how we collect, use, and safeguard your data when you use our service.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-800 mb-2">Data We Collect</h2>
            <p className="text-sm leading-relaxed">
              When you use Lumina Mind, we collect only what is necessary to provide and improve our service:
            </p>
            <ul className="list-disc list-inside text-sm mt-2 space-y-1 text-gray-600">
              <li>Conversation content you share in text or voice sessions</li>
              <li>Session feedback reports generated with your consent</li>
              <li>Technical data such as session identifiers (stored locally in your browser)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-800 mb-2">What We Never Store</h2>
            <p className="text-sm leading-relaxed">
              To protect your privacy, we explicitly do not store:
            </p>
            <ul className="list-disc list-inside text-sm mt-2 space-y-1 text-gray-600">
              <li>Raw transcripts of your conversations — these are used only to generate feedback reports and are never saved</li>
              <li>Recordings of voice calls — see Voice Call Privacy below</li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-800 mb-2">Voice Call Privacy & HIPAA Compliance</h2>
            <p className="text-sm leading-relaxed">
              We use VAPI for voice conversations. <strong>HIPAA compliance is enabled on our VAPI integration</strong>,
              which means:
            </p>
            <ul className="list-disc list-inside text-sm mt-2 space-y-1 text-gray-600">
              <li><strong>Web calls are never recorded</strong> — your voice conversations are not stored or retained by us or our voice provider</li>
              <li>Voice data is processed in real time only for the duration of your session</li>
              <li>VAPI operates under HIPAA-compliant safeguards when handling health-related conversations</li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-800 mb-2">How We Protect Your Data</h2>
            <p className="text-sm leading-relaxed">
              We protect your data through:
            </p>
            <ul className="list-disc list-inside text-sm mt-2 space-y-1 text-gray-600">
              <li>Secure transmission (HTTPS) for all data in transit</li>
              <li>Storing feedback reports in a controlled, encrypted environment</li>
              <li>Minimal data retention — we keep only what is needed to provide your feedback history</li>
              <li>No sale or sharing of your personal or health data with third parties for marketing</li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-800 mb-2">Your Rights</h2>
            <p className="text-sm leading-relaxed">
              You may request access to, correction of, or deletion of your stored feedback reports.
              Contact us using the information provided in the app if you have questions or requests.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-800 mb-2">Updates</h2>
            <p className="text-sm leading-relaxed">
              We may update this privacy policy from time to time. We will notify you of material changes
              by posting the updated policy and revising the date above.
            </p>
          </section>
        </div>
      </div>
    </div>
  )
}
