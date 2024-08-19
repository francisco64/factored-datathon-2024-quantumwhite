from google.cloud import bigquery
schema = [
    bigquery.SchemaField("GLOBALEVENTID", "STRING"),
    bigquery.SchemaField("SQLDATE", "STRING"),
    bigquery.SchemaField("MonthYear", "STRING"),
    bigquery.SchemaField("Year", "STRING"),
    bigquery.SchemaField("FractionDate", "FLOAT"),
    bigquery.SchemaField("Actor1Code", "STRING"),
    bigquery.SchemaField("Actor1Name", "STRING"),
    bigquery.SchemaField("Actor1CountryCode", "STRING"),
    bigquery.SchemaField("Actor1KnownGroupCode", "STRING"),
    bigquery.SchemaField("Actor1EthnicCode", "STRING"),
    bigquery.SchemaField("Actor1Religion1Code", "STRING"),
    bigquery.SchemaField("Actor1Religion2Code", "STRING"),
    bigquery.SchemaField("Actor1Type1Code", "STRING"),
    bigquery.SchemaField("Actor1Type2Code", "STRING"),
    bigquery.SchemaField("Actor1Type3Code", "STRING"),
    bigquery.SchemaField("Actor2Code", "STRING"),
    bigquery.SchemaField("Actor2Name", "STRING"),
    bigquery.SchemaField("Actor2CountryCode", "STRING"),
    bigquery.SchemaField("Actor2KnownGroupCode", "STRING"),
    bigquery.SchemaField("Actor2EthnicCode", "STRING"),
    bigquery.SchemaField("Actor2Religion1Code", "STRING"),
    bigquery.SchemaField("Actor2Religion2Code", "STRING"),
    bigquery.SchemaField("Actor2Type1Code", "STRING"),
    bigquery.SchemaField("Actor2Type2Code", "STRING"),
    bigquery.SchemaField("Actor2Type3Code", "STRING"),
    bigquery.SchemaField("IsRootEvent", "STRING"),
    bigquery.SchemaField("EventCode", "STRING"),
    bigquery.SchemaField("EventBaseCode", "STRING"),
    bigquery.SchemaField("EventRootCode", "STRING"),
    bigquery.SchemaField("QuadClass", "INTEGER"),
    bigquery.SchemaField("GoldsteinScale", "FLOAT"),
    bigquery.SchemaField("NumMentions", "INTEGER"),
    bigquery.SchemaField("NumSources", "INTEGER"),
    bigquery.SchemaField("NumArticles", "INTEGER"),
    bigquery.SchemaField("AvgTone", "FLOAT"),
    bigquery.SchemaField("Actor1Geo_Type", "INTEGER"),
    bigquery.SchemaField("Actor1Geo_FullName", "STRING"),
    bigquery.SchemaField("Actor1Geo_CountryCode", "STRING"),
    bigquery.SchemaField("Actor1Geo_ADM1Code", "STRING"),
    bigquery.SchemaField("Actor1Geo_Lat", "FLOAT"),
    bigquery.SchemaField("Actor1Geo_Long", "FLOAT"),
    bigquery.SchemaField("Actor1Geo_FeatureID", "STRING"),
    bigquery.SchemaField("Actor2Geo_Type", "INTEGER"),
    bigquery.SchemaField("Actor2Geo_FullName", "STRING"),
    bigquery.SchemaField("Actor2Geo_CountryCode", "STRING"),
    bigquery.SchemaField("Actor2Geo_ADM1Code", "STRING"),
    bigquery.SchemaField("Actor2Geo_Lat", "FLOAT"),
    bigquery.SchemaField("Actor2Geo_Long", "FLOAT"),
    bigquery.SchemaField("Actor2Geo_FeatureID", "STRING"),
    bigquery.SchemaField("ActionGeo_Type", "INTEGER"),
    bigquery.SchemaField("ActionGeo_FullName", "STRING"),
    bigquery.SchemaField("ActionGeo_CountryCode", "STRING"),
    bigquery.SchemaField("ActionGeo_ADM1Code", "STRING"),
    bigquery.SchemaField("ActionGeo_Lat", "FLOAT"),
    bigquery.SchemaField("ActionGeo_Long", "FLOAT"),
    bigquery.SchemaField("ActionGeo_FeatureID", "STRING"),
    bigquery.SchemaField("DATEADDED", "STRING"),
    bigquery.SchemaField("SOURCEURL", "STRING")
]

