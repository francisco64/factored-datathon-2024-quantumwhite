CREATE TABLE `factoreddatathon2014.GDELT.actor_event_covid` AS
SELECT
GLOBALEVENTID,  
SQLDATE,
Actor1Name,
Actor2Name,
EventRootCode,
EventBaseCode,
EventCode,
NumMentions,
SOURCEURL

FROM `factoreddatathon2014.GDELT.event-data`  
WHERE ActionGeo_CountryCode = 'US' 
AND Actor2Name in (
    SELECT Actor2Name FROM (
      SELECT Actor2Name, count(*) as qty
      FROM `factoreddatathon2014.GDELT.event-data` 
      WHERE ActionGeo_CountryCode = 'US' 
      and Actor2Name is not null
      and SQLDATE BETWEEN '20200101' AND '20211231'
      GROUP BY Actor2Name
      ORDER BY count(*) desc
  )
  WHERE qty>45000
)
AND Actor1Name in (
    SELECT Actor1Name FROM (
      SELECT Actor1Name, count(*) as qty
      FROM `factoreddatathon2014.GDELT.event-data` 
      WHERE ActionGeo_CountryCode = 'US' 
      and Actor1Name is not null
      and SQLDATE BETWEEN '20200101' AND '20211231'
      GROUP BY Actor1Name
      ORDER BY count(*) desc
  )
  WHERE qty>45000
)
