WITH aggregated_data AS (
    SELECT 
        DATE(CAST(SUBSTR(SQLDATE, 1, 4) AS INT64), 
             CAST(SUBSTR(SQLDATE, 5, 2) AS INT64), 
             CAST(SUBSTR(SQLDATE, 7, 2) AS INT64)) AS date,
        SUM(CASE WHEN EventCode IN ('141', '145', '172') THEN 1 ELSE 0 END) AS disturbances,
        SUM(CASE WHEN EventCode IN ('061', '0331', '0256', '091') THEN 1 ELSE 0 END) AS economic_events,
        SUM(CASE WHEN EventCode IN ('0233', '073', '182') THEN 1 ELSE 0 END) AS medical,
        AVG(AvgTone) AS avg_tone
    FROM 
        `GDELT.event-data`
    WHERE 
        Actor1Geo_CountryCode = 'US'
        AND SQLDATE BETWEEN '20200101' AND '20211231'
    GROUP BY 
        date
    ORDER BY 
        date
)

-- Calculate all correlations in one go
SELECT
    type1, type2, correlation
FROM (
    SELECT 
        'Disturbances' AS type1, 
        'Disturbances' AS type2, 
        1.0 AS correlation -- Self-correlation
    UNION ALL
    SELECT 
        'Economic Events' AS type1, 
        'Economic Events' AS type2, 
        1.0 AS correlation -- Self-correlation
    UNION ALL
    SELECT 
        'Medical' AS type1, 
        'Medical' AS type2, 
        1.0 AS correlation -- Self-correlation
    UNION ALL
    SELECT 
        'AvgTone' AS type1, 
        'AvgTone' AS type2, 
        1.0 AS correlation -- Self-correlation
    UNION ALL
    SELECT 
        'Disturbances' AS type1, 
        'Economic Events' AS type2, 
        CORR(disturbances, economic_events) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'Disturbances' AS type1, 
        'Medical' AS type2, 
        CORR(disturbances, medical) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'Disturbances' AS type1, 
        'AvgTone' AS type2, 
        CORR(disturbances, avg_tone) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'Economic Events' AS type1, 
        'Disturbances' AS type2, 
        CORR(economic_events, disturbances) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'Economic Events' AS type1, 
        'Medical' AS type2, 
        CORR(economic_events, medical) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'Economic Events' AS type1, 
        'AvgTone' AS type2, 
        CORR(economic_events, avg_tone) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'Medical' AS type1, 
        'Disturbances' AS type2, 
        CORR(medical, disturbances) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'Medical' AS type1, 
        'Economic Events' AS type2, 
        CORR(medical, economic_events) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'Medical' AS type1, 
        'AvgTone' AS type2, 
        CORR(medical, avg_tone) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'AvgTone' AS type1, 
        'Disturbances' AS type2, 
        CORR(avg_tone, disturbances) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'AvgTone' AS type1, 
        'Economic Events' AS type2, 
        CORR(avg_tone, economic_events) AS correlation
    FROM aggregated_data
    UNION ALL
    SELECT 
        'AvgTone' AS type1, 
        'Medical' AS type2, 
        CORR(avg_tone, medical) AS correlation
    FROM aggregated_data
) AS correlation_matrix
ORDER BY type1, type2;



