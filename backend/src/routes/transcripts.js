import express from 'express';
import crypto from 'crypto';
import { processTranscript } from '../lib/transcriptProcessor.js';
import { addTranscript, getAllTranscripts, getTranscriptsByConversation } from '../lib/transcriptStore.js';

const router = express.Router();

router.post('/', async (req, res) => {
  try {
    const { transcript, source, conversationId, clientId } = req.body;

    if (!transcript) {
      return res.status(400).json({ error: 'Transcript is required' });
    }

    const id = crypto.randomUUID();
    let processedText = transcript;
    let analysis = null;

    try {
      const result = await processTranscript(transcript);
      processedText = result.processed;
      analysis = result.analysis;
    } catch (err) {
      console.error('ML processing error:', err);
    }

    const record = {
      id,
      conversationId: conversationId || null,
      transcript,
      source: source || 'voice',
      processed: processedText,
      analysis,
      createdAt: new Date().toISOString(),
    };

    addTranscript(record);

    let overall = null;
    if (conversationId) {
      const convTranscripts = getTranscriptsByConversation(conversationId);
      const userTexts = convTranscripts
        .filter((t) => t.source === 'text' || t.source === 'voice')
        .map((t) => t.processed || t.transcript)
        .filter(Boolean);
      if (userTexts.length > 0) {
        try {
          const result = await processTranscript(userTexts.join(' '));
          overall = result.analysis ? { label: result.analysis.label, scores: result.analysis.scores } : null;
        } catch (err) {
          console.error('Overall ML error:', err);
        }
      }
    }

    const payload = {
      id,
      message: 'Transcript saved.',
      analysis: analysis ? { label: analysis.label, scores: analysis.scores } : null,
    };
    if (clientId != null) payload.clientId = clientId;
    if (overall) payload.overall = overall;

    res.status(201).json(payload);
  } catch (error) {
    console.error('Transcript save error:', error);
    res.status(500).json({ error: error.message || 'Failed to save transcript' });
  }
});

router.get('/', (req, res) => {
  res.json({ transcripts: getAllTranscripts() });
});

export default router;
