-- Distribution of risk score at session end
-- Bucketed into 10 equal-width bins
SELECT
    FLOOR(risk_score_final * 10) / 10 AS risk_bucket,
    COUNT(*)                          AS session_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM session_events
GROUP BY risk_bucket
ORDER BY risk_bucket;
