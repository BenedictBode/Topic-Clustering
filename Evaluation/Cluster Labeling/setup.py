import pandas as pd
import random

# Assuming df is your DataFrame with 'level 1', 'level 2', and 'level 3' representing clusters

CLUSTERS_PER_LEVEL = 100
CLUSTER_LEVELS = 2

df = pd.read_csv("/Database/Output/clusteredTopicsRaw.csv")

# Create a dictionary to hold questions
questions = []

# Loop through each cluster level (level 1 to level 3)
for cluster_level_num in range(1, CLUSTER_LEVELS+1):
    # Group topics by the current cluster level
    cluster_level = "level " + str(cluster_level_num)
    prev_cluster_level = "level 0" #+ str(cluster_level_num-1)
    print(cluster_level, prev_cluster_level)

    all_clusters = df[cluster_level].unique().tolist()
    clusters = random.sample(all_clusters, min(CLUSTERS_PER_LEVEL, len(all_clusters)))

    for cluster in clusters:
        sorted_by_cohesion = 'cohesion_'+cluster_level
        sorted_by_appearances = "appearances"
        topics = df[df[cluster_level] == cluster].sort_values(by=[sorted_by_cohesion])[prev_cluster_level].tolist()

        if len(topics) < 2:
            continue


        # Prepare a data row for each cluster level and cluster
        row = {
            'cluster_level': cluster_level,
            'cluster': cluster,
            'choice': ''
        }

        for i, topic in enumerate(topics):
            row[f'topic{i+1}'] = topic

        questions.append(row)

# Save all questions to CSV
random.shuffle(questions)
questions_df = pd.DataFrame(questions)
questions_df.to_csv('questions.csv', index=False)

