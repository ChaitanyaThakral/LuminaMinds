-- False negative rate by user session length
-- A "false negative" here = session with high true risk (risk_score >= 0.5)
-- but classified as low/medium risk level
SELECT
    CASE
        WHEN message_count < 5  THEN 'short (1-4)'
        WHEN message_count < 10 THEN 'medium (5-9)'
        WHEN message_count < 20 THEN 'long (10-19)'
        ELSE 'very_long (20+)'
    END AS session_length_bucket,
    COUNT(*) AS total_sessions,
    SUM(CASE WHEN risk_score_final >= 0.5 THEN 1 ELSE 0 END) AS true_high_risk,
    SUM(CASE WHEN risk_score_final >= 0.5 AND risk_level IN ('low', 'medium') THEN 1 ELSE 0 END) AS false_negatives,
    ROUND(
        CASE
            WHEN SUM(CASE WHEN risk_score_final >= 0.5 THEN 1 ELSE 0 END) = 0 THEN 0
            ELSE SUM(CASE WHEN risk_score_final >= 0.5 AND risk_level IN ('low', 'medium') THEN 1 ELSE 0 END)
                 * 100.0
                 / SUM(CASE WHEN risk_score_final >= 0.5 THEN 1 ELSE 0 END)
        END,
    2) AS fn_rate_pct
FROM session_events
GROUP BY session_length_bucket
ORDER BY session_length_bucket;
