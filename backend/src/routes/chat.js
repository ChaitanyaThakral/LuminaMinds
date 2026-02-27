import express from 'express';
import crypto from 'crypto';
import OpenAI from 'openai';
import { processTranscript } from '../lib/transcriptProcessor.js';
import { addTranscript, getTranscriptsByConversation } from '../lib/transcriptStore.js';

const router = express.Router();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// In-memory thread storage (in production, use a database)
const threads = new Map();

router.post('/message', async (req, res) => {
  try {
    const { message, threadId } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    let currentThreadId = threadId;

    if (!currentThreadId) {
      const thread = await openai.beta.threads.create();
      currentThreadId = thread.id;
      threads.set(currentThreadId, []);
    }

    await openai.beta.threads.messages.create(currentThreadId, {
      role: 'user',
      content: message,
    });

    const run = await openai.beta.threads.runs.create(currentThreadId, {
      assistant_id: process.env.OPENAI_ASSISTANT_ID,
    });

    let runStatus = await openai.beta.threads.runs.retrieve(currentThreadId, run.id);

    while (runStatus.status === 'queued' || runStatus.status === 'in_progress') {
      await new Promise((resolve) => setTimeout(resolve, 500));
      runStatus = await openai.beta.threads.runs.retrieve(currentThreadId, run.id);
    }

    if (runStatus.status === 'completed') {
      const messages = await openai.beta.threads.messages.list(currentThreadId);
      const assistantMessage = messages.data.find((m) => m.role === 'assistant');
      const text = assistantMessage?.content?.[0]?.text?.value || '';

      let analysis = null;
      let processedText = message;
      try {
        const result = await processTranscript(message);
        analysis = result.analysis;
        processedText = result.processed;
      } catch (err) {
        console.error('ML analysis error:', err);
      }

      const record = {
        id: crypto.randomUUID(),
        conversationId: currentThreadId,
        transcript: message,
        source: 'text',
        processed: processedText,
        analysis,
        createdAt: new Date().toISOString(),
      };
      addTranscript(record);

      let overall = null;
      const threadTranscripts = getTranscriptsByConversation(currentThreadId);
      if (threadTranscripts.length > 1) {
        try {
          const combined = threadTranscripts.map((t) => t.processed || t.transcript).join(' ');
          const overallResult = await processTranscript(combined);
          overall = overallResult.analysis;
        } catch (err) {
          console.error('Overall analysis error:', err);
        }
      } else if (threadTranscripts.length === 1) {
        overall = analysis;
      }

      return res.json({
        message: text,
        threadId: currentThreadId,
        analysis: analysis ? { label: analysis.label, scores: analysis.scores } : null,
        overall: overall ? { label: overall.label, scores: overall.scores } : null,
      });
    }

    res.status(500).json({ error: 'Assistant run failed', status: runStatus.status });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: error.message || 'Chat failed' });
  }
});

export default router;
