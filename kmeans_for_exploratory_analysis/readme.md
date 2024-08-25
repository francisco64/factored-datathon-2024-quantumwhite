# Clustering Analysis of GDELT Events Data Using BigQuery and KMeans

## Overview

This document outlines the process of clustering actors and events from the GDELT (Global Database of Events, Language, and Tone) dataset using BigQuery and KMeans clustering. The objective of this analysis is to uncover relationships between different actors and the actions that connect them during the COVID-19 period, revealing patterns and insights that are significant in understanding the social, political, and economic landscape during that time.

## Methodology

### Tools and Technologies Used

- **BigQuery:** Google BigQuery was utilized to manage and process the vast amount of data from the GDELT dataset. The data was loaded into BigQuery, where SQL queries were used for data manipulation and preparation for the clustering analysis.
- **Multi-Hot Encoding:** This encoding technique was used to convert categorical data (actors and events) into a format suitable for machine learning models.
- **KMeans Clustering:** KMeans is a popular clustering algorithm that was applied to group similar data points (events and actors) into clusters. Cosine distance was used as the distance metric in KMeans, which is effective for handling high-dimensional data like multi-hot encoded vectors.

### Step-by-Step Process

1. **Data Loading:**
   - The GDELT dataset was loaded into BigQuery. This dataset contains millions of events that capture interactions between various actors and actions (events).
   - The dataset includes both structured data (like dates, actors, and locations) and unstructured data (like textual descriptions of events).

2. **Data Preparation:**
   - Using SQL queries in BigQuery, the raw data was cleaned and preprocessed. This involved filtering relevant columns, removing any irrelevant or noisy data, and creating a new table that specifically focuses on the relationships between actors and events.
   - **Multi-Hot Encoding:** 
     - The actor and event columns were transformed into multi-hot encoded vectors. This encoding creates a binary vector for each data point, where each dimension corresponds to a unique actor or event, and the presence of that actor or event in the data point is marked with a 1.
     - This transformation was crucial as it allowed the categorical data to be represented in a numerical format suitable for clustering.

3. **Clustering with KMeans:**
   - **KMeans Algorithm:** The KMeans algorithm was used to cluster the multi-hot encoded data. KMeans works by initializing a predefined number of centroids, then iteratively refining the position of these centroids to minimize the distance between each data point and its assigned centroid.
   - **Cosine Distance:** 
     - Cosine distance was chosen as the distance metric for this clustering task. Unlike Euclidean distance, which measures the straight-line distance between points, cosine distance measures the cosine of the angle between two vectors. This makes it well-suited for high-dimensional data like multi-hot encoded vectors, where the direction of the vector is more important than its magnitude.
     - The use of cosine distance helps in identifying clusters based on the similarity of actors and events, rather than their absolute counts.

4. **Cluster Analysis:**
   - The KMeans algorithm identified 10 distinct clusters within the dataset. Each cluster represents a group of events and actors that are closely related in terms of the actions taken and the entities involved.
   - The results of the clustering were analyzed to understand the key characteristics of each cluster, such as the most prominent actors and the most frequent events.

### Key Considerations

- **Choice of Clustering Method:** KMeans was chosen for its simplicity and effectiveness in partitioning the data into distinct groups. The use of cosine distance as the metric was particularly important for this dataset, where the focus is on the relationship between actors and actions rather than the raw counts.
- **Data Dimensionality:** The high dimensionality of the dataset, due to the multi-hot encoding, was handled effectively by BigQuery, which provided the computational power needed to process and analyze such large datasets.

## Conclusions

### Summary of Findings

The clustering analysis revealed 10 distinct clusters, each representing different aspects of the global response to the COVID-19 pandemic. Here are the key findings:

1. **Cluster 1: Civil Unrest and Law Enforcement:**
   - **Key Actors:** United States, Media, Police, Criminals
   - **Key Events:** Coerce, Investigate, Make Public Statement
   - **Insight:** This cluster captured the intense period of civil unrest in the United States, particularly related to events like the George Floyd protests. The frequent occurrence of the "Coerce" event indicates the use of force by law enforcement, while the involvement of media reflects the significant coverage of these events.

2. **Cluster 2: Educational and Public Safety Responses:**
   - **Key Actors:** United States, Schools, Police
   - **Key Events:** Provide Aid, Make Public Statement, Engage in Diplomatic Cooperation
   - **Insight:** This cluster represents the response of educational institutions and public safety agencies to the pandemic, with a focus on maintaining order and providing support during the crisis.

3. **Cluster 3: Governmental and Military Responses:**
   - **Key Actors:** President, United States, Military
   - **Key Events:** Engage in Material Cooperation, Engage in Diplomatic Cooperation
   - **Insight:** This cluster likely reflects national security concerns and international military cooperation, with a strong emphasis on governmental authority and strategic decision-making.

4. **Cluster 4: Legislative Response to Unrest:**
   - **Key Actors:** Senators, Police, Government
   - **Key Events:** Protest, Make Public Statement, Investigate
   - **Insight:** The legislative and governmental response to civil unrest is the focus of this cluster, highlighting the role of policymakers in addressing public demands for justice.

5. **Cluster 5: International Economic Diplomacy:**
   - **Key Actors:** United States, American, Africa
   - **Key Events:** Engage in Economic Cooperation, Make Public Statement, Engage in Diplomatic Cooperation
   - **Insight:** This cluster emphasizes economic diplomacy during the pandemic, particularly the efforts to maintain and strengthen international relations amidst global challenges.

6. **Cluster 6: Domestic Unrest and Educational Impact:**
   - **Key Actors:** Police, Media, Schools
   - **Key Events:** Investigate, Protest, Make Public Statement
   - **Insight:** Reflecting domestic unrest and its impact on educational institutions, this cluster highlights the significant role of media and law enforcement in these events.

7. **Cluster 7: State-Level Management of the Pandemic:**
   - **Key Actors:** Governors, Schools, Media
   - **Key Events:** Protest, Make Public Statement, Engage in Diplomatic Cooperation
   - **Insight:** State-level actions, particularly the role of Governors in managing the pandemic, are central to this cluster. The presence of protests indicates public dissatisfaction with certain policies.

8. **Cluster 8: Community and Educational Response:**
   - **Key Actors:** Schools, Students, Community
   - **Key Events:** Engage in Economic Cooperation, Make Public Statement, Investigate
   - **Insight:** This cluster focuses on the community and educational response to the pandemic, emphasizing the economic and social adaptations made by these groups.

9. **Cluster 9: Social Movements and Activism:**
   - **Key Actors:** United States, Community, Students
   - **Key Events:** Protest, Make Public Statement, Engage in Diplomatic Cooperation
   - **Insight:** This cluster reflects social movements and activism, particularly those driven by younger populations and community organizations during the pandemic.

10. **Cluster 10: Political Activity During the U.S. Presidential Election:**
    - **Key Actors:** Voters, President, United States
    - **Key Events:** Make Public Statement, Engage in Diplomatic Cooperation, Coerce
    - **Insight:** Focused on the 2020 U.S. Presidential election, this cluster underscores the political activity and pressures associated with the election process, as well as the broader implications for the pandemic response.

### Conclusion

The analysis conducted using BigQuery and KMeans clustering provided valuable insights into the relationships between actors and events during the COVID-19 pandemic. By grouping similar data points, we were able to identify key patterns and trends that highlight the various facets of global responses to the pandemic. The choice of multi-hot encoding and cosine distance in KMeans proved to be effective in handling the high-dimensional data, leading to meaningful and interpretable clusters.

This analysis not only enhances our understanding of the complex dynamics during the pandemic but also demonstrates the power of combining advanced data processing tools with machine learning techniques to derive actionable insights from large datasets.
