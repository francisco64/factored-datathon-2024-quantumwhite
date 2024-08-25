CREATE OR REPLACE MODEL `factoreddatathon2014.GDELT.kmeans_model_15`
OPTIONS(
  MODEL_TYPE='KMEANS',
  NUM_CLUSTERS=15,  -- You can adjust the number of clusters
  DISTANCE_TYPE='COSINE',
  STANDARDIZE_FEATURES=TRUE  -- Standardizes features before training
) AS
SELECT *
FROM `factoreddatathon2014.GDELT.training_kmeans`;
