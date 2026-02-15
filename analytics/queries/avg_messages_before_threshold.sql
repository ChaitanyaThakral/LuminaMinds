-- Average messages per session before the 5-message threshold triggers
-- (sessions that never reached 5 messages)
SELECT
    AVG(message_count) AS avg_messages_before_threshold,
    COUNT(*)           AS sessions_below_threshold,
    MIN(message_count) AS min_messages,
    MAX(message_count) AS max_messages
FROM session_events
WHERE threshold_triggered = false;
