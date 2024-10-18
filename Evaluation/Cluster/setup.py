import pandas as pd
import random

CLUSTER_LEVELS = 5
MAX_CLUSTERS_PER_LEVEL = 50

# Assuming df is your DataFrame with 'level 1', 'level 2', and 'level 3' representing clusters
df = pd.read_csv("../../Database/Output/clusteredTopicsRaw.csv")

# Create a dictionary to hold questions
questions = []

# Loop through each cluster level (level 1 to level 3)
for cluster_level_num in range(1, CLUSTER_LEVELS+1):
    # Group topics by the current cluster level
    cluster_level = "level " + str(cluster_level_num)
    prev_cluster_level = "level 0" #+ str(cluster_level_num-1)
    print(cluster_level, prev_cluster_level)

    clustered_topics = df.groupby(cluster_level)[prev_cluster_level].apply(list).to_dict()

    all_clusters = list(clustered_topics.keys())
    clusters = random.sample(all_clusters, min(MAX_CLUSTERS_PER_LEVEL, len(all_clusters)))

    for cluster in clusters:
        topics = list(set(clustered_topics[cluster]))[:8]  # Limit to 8 topics
        if len(topics) < 2:
            continue
        other_clusters = [t for c, t_list in clustered_topics.items() if c != cluster for t in t_list]
        random_topic = random.choice(other_clusters)

        # Prepare a data row for each cluster level and cluster
        row = {
            'cluster_level': cluster_level,
            'cluster': cluster,
            'random_topic': random_topic
        }
        for i, topic in enumerate(topics):
            row[f'topic{i+1}'] = topic

        # Add placeholders for choice and correctness
        row['choice'] = ''
        row['is_right'] = ''

        questions.append(row)

# Save all questions to CSV
random.shuffle(questions)
questions_df = pd.DataFrame(questions)
questions_df.to_csv('questions.csv', index=False)

