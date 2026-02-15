-- Cohort table: sessions bucketed by risk level × session length
SELECT
    risk_level,
    CASE
        WHEN message_count < 5  THEN 'short (1-4)'
        WHEN message_count < 10 THEN 'medium (5-9)'
        WHEN message_count < 20 THEN 'long (10-19)'
        ELSE 'very_long (20+)'
    END AS session_length_bucket,
    COUNT(*)                       AS session_count,
    ROUND(AVG(risk_score_final), 4) AS avg_risk_score,
    ROUND(AVG(session_duration_sec), 1) AS avg_duration_sec
FROM session_events
GROUP BY risk_level, session_length_bucket
ORDER BY
    CASE risk_level
        WHEN 'low' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'high' THEN 3
        WHEN 'critical' THEN 4
    END,
    session_length_bucket;
