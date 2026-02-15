-- Risk score escalation patterns across sessions per user
-- Users with more than one session: how does risk evolve?
SELECT
    user_id,
    COUNT(*)                                    AS session_count,
    ROUND(MIN(risk_score_final), 4)             AS min_risk,
    ROUND(MAX(risk_score_final), 4)             AS max_risk,
    ROUND(MAX(risk_score_final) - MIN(risk_score_final), 4) AS risk_range,
    ROUND(AVG(risk_score_final), 4)             AS avg_risk
FROM session_events
GROUP BY user_id
HAVING COUNT(*) > 1
ORDER BY risk_range DESC
LIMIT 20;
