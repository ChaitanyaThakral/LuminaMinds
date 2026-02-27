import express from 'express';
import crypto from 'crypto';
import OpenAI from 'openai';
import { addReport, getReportsByUserId, getReportById } from '../lib/feedbackReportStore.js';
import { authMiddleware } from '../lib/auth.js';

const FEEDBACK_ASSISTANT_ID = process.env.FEEDBACK_ASSISTANT_ID;
const MIN_USER_INPUTS = 5;

const router = express.Router();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

router.post('/finish-session', authMiddleware, async (req, res) => {
  try {
    if (!FEEDBACK_ASSISTANT_ID) {
      return res.status(500).json({
        error: 'FEEDBACK_ASSISTANT_ID is not configured. Add it to your .env file.',
      });
    }

    const { conversationId, source, transcript, mlOutput } = req.body;
    const userId = req.user.id;

    if (!transcript || !mlOutput) {
      return res.status(400).json({
        error: 'Transcript and ML output are required',
      });
    }

    const userInputCount = (transcript.match(/^(user|you)\s*:/gim) || []).length;
    if (userInputCount < MIN_USER_INPUTS) {
      return res.status(400).json({
        error: `Minimum ${MIN_USER_INPUTS} user inputs required to finish session. You have ${userInputCount}.`,
      });
    }

    const thread = await openai.beta.threads.create();
    const inputMessage = `Please analyze this session and provide a feedback report.

## Conversation Transcript
(Do not store or reproduce the transcript in your response - use it only for analysis.)

${transcript}

## ML Classification Output (JSON)
${JSON.stringify(mlOutput, null, 2)}

Please summarize the conversation and analyze the ML output to classify the user into the appropriate categories (Anxiety, Depression, Stress, Normal, Suicidal) if applicable. Provide a supportive, actionable feedback report.`;

    await openai.beta.threads.messages.create(thread.id, {
      role: 'user',
      content: inputMessage,
    });

    const run = await openai.beta.threads.runs.create(thread.id, {
      assistant_id: FEEDBACK_ASSISTANT_ID,
    });

    let runStatus = await openai.beta.threads.runs.retrieve(thread.id, run.id);
    let attempts = 0;
    const maxAttempts = 60;

    while (
      (runStatus.status === 'queued' || runStatus.status === 'in_progress') &&
      attempts < maxAttempts
    ) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      runStatus = await openai.beta.threads.runs.retrieve(thread.id, run.id);
      attempts++;
    }

    if (runStatus.status !== 'completed') {
      return res.status(500).json({
        error: 'Feedback assistant did not complete',
        status: runStatus.status,
      });
    }

    const messages = await openai.beta.threads.messages.list(thread.id);
    const assistantMessage = messages.data.find((m) => m.role === 'assistant');
    const reportContent =
      assistantMessage?.content?.[0]?.text?.value || 'No feedback generated.';

    const report = {
      id: crypto.randomUUID(),
      userId,
      conversationId: conversationId || null,
      source: source || 'unknown',
      content: reportContent,
      mlOutput,
      createdAt: new Date().toISOString(),
    };

    await addReport(report);

    res.json({
      report: {
        id: report.id,
        content: report.content,
        createdAt: report.createdAt,
        source: report.source,
      },
    });
  } catch (error) {
    console.error('Feedback finish-session error:', error);
    res.status(500).json({
      error: error.message || 'Failed to generate feedback report',
    });
  }
});

router.get('/reports', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;

    const reports = await getReportsByUserId(userId);
    res.json({
      reports: reports.map((r) => ({
        id: r.id,
        content: r.content,
        createdAt: r.createdAt,
        source: r.source,
      })),
    });
  } catch (error) {
    console.error('Feedback reports error:', error);
    res.status(500).json({ error: error.message || 'Failed to fetch reports' });
  }
});

router.get('/reports/:id', authMiddleware, async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;
    const report = await getReportById(id);

    if (!report) {
      return res.status(404).json({ error: 'Report not found' });
    }

    if (report.userId !== userId) {
      return res.status(403).json({ error: 'Access denied' });
    }

    res.json({
      report: {
        id: report.id,
        content: report.content,
        createdAt: report.createdAt,
        source: report.source,
      },
    });
  } catch (error) {
    console.error('Feedback report error:', error);
    res.status(500).json({ error: error.message || 'Failed to fetch report' });
  }
});

export default router;
