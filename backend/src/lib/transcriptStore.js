const transcripts = new Map();

export function addTranscript(record) {
  transcripts.set(record.id, record);
  return record;
}

export function getAllTranscripts() {
  return Array.from(transcripts.values()).sort(
    (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
  );
}

export function getTranscriptsByConversation(conversationId) {
  return getAllTranscripts().filter((t) => t.conversationId === conversationId);
}
