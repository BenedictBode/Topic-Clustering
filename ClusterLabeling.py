import pandas as pd
import numpy as np

df = pd.read_csv("Database/Output/clusteredTopicsRaw5.csv")
df.fillna(1, inplace=True)

PRIMARY_LEVEL = "level 0"
CLUSTER_LEVELS = 5

def clusterLabels(
        cluster_level="level 1",
        sil_w=0.0,
        cohesion_w=0.0,
        seperation_w=0.0,
        appearances_w=0.0):
    #df['log_appearances_' + cluster_level] = df['appearances'].apply(np.log1p)  # Log transform (log(1 + x))
    df['log_norm_appearances_' + cluster_level] = df.groupby(cluster_level)[
        'appearances'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    df['norm_cohesion_' + cluster_level] = df.groupby(cluster_level)[
        'cohesion_' + cluster_level].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    df['norm_separation_' + cluster_level] = df.groupby(cluster_level)[
        'separation_' + cluster_level].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    df['norm_silhouette_' + cluster_level] = df.groupby(cluster_level)[
        'silhouette_' + cluster_level].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))

    df.fillna(1, inplace=True)
    df['labelScore_' + cluster_level] = ((df['norm_cohesion_' + cluster_level] * -cohesion_w) +
                                         (df['norm_separation_' + cluster_level] * seperation_w) +
                                         (df['norm_silhouette_' + cluster_level] * sil_w) +
                                         (df['log_norm_appearances_' + cluster_level] * appearances_w))

    # Get the cluster label with the maximum score for each cluster_level
    cluster_labels = df.loc[df.groupby(cluster_level)['labelScore_' + cluster_level].idxmax()][
        [cluster_level, PRIMARY_LEVEL]].copy()

    return cluster_labels


def labelAllLevels(appearances_w=0.2,
                                  cohesion_w=1.0,
                                  seperation_w=0.0,
                                  sil_w=0.1):
    labeledClusters = pd.DataFrame()
    labeledClusters[PRIMARY_LEVEL] = df[PRIMARY_LEVEL]
    labeledClusters["appearances"] = df["appearances"]

    for i in range(1, CLUSTER_LEVELS + 1):
        cluster_level = "level " + str(i)
        predicted = clusterLabels(cluster_level=cluster_level,
                                  appearances_w=appearances_w,
                                  cohesion_w=cohesion_w,
                                  seperation_w=seperation_w,
                                  sil_w=sil_w)

        # Merge the predicted labels for this cluster level back into the original dataframe
        df_merged = df.merge(predicted, on=cluster_level, how="left", suffixes=('', '_label'))

        # Assign the predicted label for this level into labeledClusters
        labeledClusters[cluster_level] = df_merged[PRIMARY_LEVEL + '_label']

    return labeledClusters


labelAllLevels(appearances_w=0.5, cohesion_w=1.0, sil_w=0).to_csv("Output/clusteredTopicsLabeled5.csv", index=False)
