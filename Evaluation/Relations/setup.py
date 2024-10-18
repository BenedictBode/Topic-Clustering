import pandas as pd
import random

CLUSTER_LEVELS = 5
N_RELATIONS = 150  # Number of relations to test

# Assuming df is your DataFrame with 'level 0', 'level 1', ... 'level 5' representing clusters
df = pd.read_csv("../../Database/Output/clusteredTopicsLabeled.csv")

# Create a set to hold unique relations
relations = set()

# Loop through each cluster level (level 1 to level CLUSTER_LEVELS)
for cluster_level_num in range(1, CLUSTER_LEVELS + 1):
    cluster_level = "level " + str(cluster_level_num)
    level_0 = "level "  + str(cluster_level_num-1)

    # Group topics by the current cluster level and level 0
    clustered_topics = df.groupby([level_0, cluster_level]).size().reset_index().rename(columns={0: 'count'})

    # Randomly select n relations between level 0 and the current cluster level
    while len(relations) < cluster_level_num*(N_RELATIONS/CLUSTER_LEVELS):
        sampled_row = clustered_topics.sample(n=1)
        level_0_label = sampled_row[level_0].values[0]
        cluster_level_label = sampled_row[cluster_level].values[0]

        # Ensure level 0 and the current cluster level are not the same
        if level_0_label != cluster_level_label:
            relation = (level_0_label, cluster_level_label, cluster_level)
            if relation not in relations:
                relations.add(relation)

# Convert the relations set into a list of dictionaries for CSV export
relation_rows = [{'cluster_level': relation[2], 'level_0': relation[0], 'level_x': relation[1]} for relation in relations]

# Save all unique relations to CSV
relations_df = pd.DataFrame(relation_rows)
relations_df.to_csv('questions.csv', index=False)

print(f"{len(relations)} relations saved to 'relations.csv'")