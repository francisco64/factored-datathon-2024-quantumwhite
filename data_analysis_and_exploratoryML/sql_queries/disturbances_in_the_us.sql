SELECT 
    PARSE_DATE('%Y%m%d', SQLDATE) AS date,
    AvgTone AS avgtone,
    CASE
        WHEN EventCode IN ('0233', '073', '182') THEN 'Medical'
        WHEN EventCode IN ('0334', '0254', '0342') THEN 'Governmental Regulations'
        WHEN EventCode IN ('141', '145', '172') THEN 'Disturbances'
        WHEN EventCode IN ('061', '0331', '0256', '091') THEN 'Economic Events'
        ELSE NULL
    END AS event_type,
    ActionGeo_Lat AS latitude,
    ActionGeo_Long AS longitude
FROM 
     `GDELT.event-data`
WHERE 
    ActionGeo_CountryCode = 'US'--Actor1Geo_CountryCode = 'US'
    AND SQLDATE BETWEEN '20200101' AND '20211231'
    AND EventCode IN ('0233', '073', '182', '0334', '0254', '0342', '141', '145', '172', '061', '0331', '0256', '091')
    --AND ActionGeo_Lat IS NOT NULL
    --AND ActionGeo_Long IS NOT NULL;
    
