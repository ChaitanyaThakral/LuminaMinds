import express from 'express';
import { getTranscriptsByConversation } from '../lib/transcriptStore.js';
import { processTranscript } from '../lib/transcriptProcessor.js';

const router = express.Router();

router.get('/:id/overall', async (req, res) => {
  try {
    const { id } = req.params;
    const transcripts = getTranscriptsByConversation(id);

    if (transcripts.length === 0) {
      return res.status(404).json({ error: 'No transcripts found for this conversation' });
    }

    const userTexts = transcripts
      .filter((t) => t.source === 'text' || t.source === 'voice')
      .map((t) => t.processed || t.transcript)
      .filter(Boolean);

    if (userTexts.length === 0) {
      return res.json({
        conversationId: id,
        messageCount: 0,
        overall: null,
      });
    }

    const combined = userTexts.join(' ');
    const result = await processTranscript(combined);

    res.json({
      conversationId: id,
      messageCount: transcripts.length,
      overall: result.analysis,
    });
  } catch (err) {
    console.error('Overall analysis error:', err);
    res.status(500).json({ error: err.message || 'Failed to compute overall analysis' });
  }
});

export default router;
