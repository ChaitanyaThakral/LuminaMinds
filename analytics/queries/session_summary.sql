-- Overall session summary statistics
SELECT
    COUNT(*)                                AS total_sessions,
    COUNT(DISTINCT user_id)                 AS unique_users,
    ROUND(AVG(message_count), 2)            AS avg_messages,
    ROUND(AVG(session_duration_sec), 1)     AS avg_duration_sec,
    ROUND(AVG(risk_score_final), 4)         AS avg_risk_score,
    SUM(CASE WHEN threshold_triggered THEN 1 ELSE 0 END) AS sessions_with_threshold,
    ROUND(SUM(CASE WHEN threshold_triggered THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS threshold_pct,
    SUM(CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END) AS critical_sessions
FROM session_events;
