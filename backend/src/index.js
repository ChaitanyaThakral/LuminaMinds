import './load-env.js';
import express from 'express';
import cors from 'cors';
import chatRoutes from './routes/chat.js';
import transcriptRoutes from './routes/transcripts.js';
import conversationRoutes from './routes/conversations.js';
import feedbackRoutes from './routes/feedback.js';
import authRoutes from './routes/auth.js';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors({ origin: true }));
app.use(express.json());

app.use('/api/auth', authRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/transcripts', transcriptRoutes);
app.use('/api/conversations', conversationRoutes);
app.use('/api/feedback', feedbackRoutes);

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Lumina Mind API' });
});

app.listen(PORT, () => {
  console.log(`Lumina Mind backend running on http://localhost:${PORT}`);
});
