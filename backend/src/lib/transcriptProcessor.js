/**
 * Transcript processing for two-stage mental health classifier.
 * - Mood model: Anxiety, Depression, Normal, Stress
 * - Risk model: Suicidal
 * - Cleans text, extracts user lines, chunks for model (MAX_LEN=256 ~= ~180 words)
 */

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const WORDS_PER_CHUNK = 180; // Conservative for MAX_LEN=256

const USER_PREFIXES = [
  /^\s*user\s*:\s*/i,
  /^\s*you\s*:\s*/i,
  /^\s*Customer\s*:\s*/i,
];
const ASSISTANT_PREFIXES = [
  /^\s*assistant\s*:\s*/i,
  /^\s*agent\s*:\s*/i,
  /^\s*AI\s*:\s*/i,
  /^\s*Assistant\s*:\s*/i,
  /^\s*bot\s*:\s*/i,
];

function cleanText(text) {
  if (!text || typeof text !== 'string') return '';
  let t = text;
  // Remove URLs
  t = t.replace(/https?:\/\/[^\s]+/g, '');
  t = t.replace(/www\.[^\s]+/g, '');
  // Lowercase
  //t = t.toLowerCase();
  // Normalize whitespace
  t = t.replace(/\s+/g, ' ').trim();
  return t;
}

function extractUserLines(transcript) {
  const lines = transcript.split(/\n|(?<=[.!?])\s+/).map((l) => l.trim()).filter(Boolean);
  const userLines = [];

  for (const line of lines) {
    const stripped = line.replace(/^["']|["']$/g, '');
    const isAssistant = ASSISTANT_PREFIXES.some((re) => re.test(stripped));
    const isUser = USER_PREFIXES.some((re) => re.test(stripped));

    if (isAssistant) continue;
    if (isUser) {
      const cleaned = stripped.replace(/^\s*(user|you|Customer)\s*:\s*/i, '').trim();
      if (cleaned) userLines.push(cleaned);
      continue;
    }
    // If no prefix, assume user (single-speaker transcripts)
    if (line.length > 0) userLines.push(line);
  }

  const hasExplicitAssistant = lines.some((l) =>
    ASSISTANT_PREFIXES.some((re) => re.test(l))
  );
  if (hasExplicitAssistant || userLines.length > 0) return userLines.join(' ');
  return transcript;
}

function chunkByWords(text, maxWords = WORDS_PER_CHUNK) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  for (let i = 0; i < words.length; i += maxWords) {
    chunks.push(words.slice(i, i + maxWords).join(' '));
  }
  return chunks.filter(Boolean);
}

async function predictChunk(text) {
  const res = await fetch(`${ML_SERVICE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`ML service error: ${res.status}`);
  return res.json();
}

async function processTranscript(transcript) {
  const cleaned = cleanText(transcript);
  const userOnly = extractUserLines(cleaned);
  const cleanedUserText = cleanText(userOnly);

  if (!cleanedUserText) {
    return {
      processed: '',
      analysis: {
        label: 'Normal',
        scores: { Anxiety: 0.25, Depression: 0.25, Normal: 0.25, Stress: 0.25, Suicidal: 0 },
        perChunk: [],
      },
    };
  }

  const chunks = chunkByWords(cleanedUserText);
  const perChunk = [];
  const labelOrder = ['Anxiety', 'Depression', 'Normal', 'Stress', 'Suicidal'];
  const avgScores = Object.fromEntries(labelOrder.map((l) => [l, 0]));

  for (const chunk of chunks) {
    const result = await predictChunk(chunk);
    perChunk.push({ text: chunk.slice(0, 80) + (chunk.length > 80 ? '...' : ''), ...result });
    for (const label of labelOrder) {
      avgScores[label] += result.scores[label] ?? 0;
    }
  }

  const n = chunks.length;
  for (const label of labelOrder) {
    avgScores[label] /= n;
  }

  const finalLabel = labelOrder.reduce((a, b) =>
    avgScores[a] >= avgScores[b] ? a : b
  );

  return {
    processed: cleanedUserText,
    analysis: {
      label: finalLabel,
      scores: avgScores,
      perChunk: perChunk.length > 1 ? perChunk : undefined,
    },
  };
}

export { processTranscript, cleanText, extractUserLines, chunkByWords };
