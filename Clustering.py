import pandas as pd
import DataPrep
from scipy.cluster.hierarchy import fcluster, linkage
from ClusterMetrics import cluster_metrics
from scipy.cluster.hierarchy import maxdists
import math

import numpy as np

LEVELS = 5
AVERAGE_ELEMENTS_PER_CLUSTER = 5

topics, topics_appearances = DataPrep.prep_keywords(min_appearances=5)
embeddings = DataPrep.embed_keywords(topics)
pd.DataFrame(embeddings).to_csv('Database/Precomputed/embeddings5.csv', header=False, index=False)
linkage = linkage(embeddings, method='ward')
pd.DataFrame(linkage).to_csv('Database/Precomputed/linkage5.csv', header=False, index=False)

#linkage = pd.read_csv("Output/linkage.csv", header=None).values
#embeddings = pd.read_csv("Output/embeddings.csv", header=None).values

n = len(topics)

def generateRawClusters():
    # Maybe also add
    df = pd.DataFrame({"level 0": topics, "appearances": topics_appearances})

    for i in range(1, LEVELS + 1):
        level_descr = f"level {i}"
        cluster_count = round(len(linkage) / (AVERAGE_ELEMENTS_PER_CLUSTER ** i))

        print(level_descr, cluster_count)

        labels = fcluster(linkage, cluster_count, criterion='maxclust')
        sil, a, b, closest = cluster_metrics(embeddings, labels)

        print("silhouette_score: ", np.mean(sil))

        level_df = pd.DataFrame(
            {level_descr: labels,
             'silhouette_' + level_descr: sil,
             'cohesion_' + level_descr: a,
             'separation_' + level_descr: b,
             'closest_' + level_descr: closest,
             })

        df = pd.concat([df, level_df], axis=1)

    df.to_csv("Output/clusteredTopicsRaw5.csv", index=False)

generateRawClusters()

def getCuttingDistances(elem_per_cluster):
    n = len(topics)
    for i in range(1, int(math.log(n, elem_per_cluster))):
        level_descr = f"level{i}"
        cluster_size = (elem_per_cluster ** i)
        cluster_count = round(len(linkage) / cluster_size)

        #labels = fcluster(linkage, cluster_count, criterion='maxclust')
        #df = pd.DataFrame({"cluster": labels})
        #df["cluster"].value_counts().std()

        cutting_distance = linkage[-(cluster_count-1), 2]
        merges = n - cluster_count
        print(merges/cutting_distance)

#getCuttingDistances(elem_per_cluster=7)


def plot_merges_vs_distance(epc1, epc2):
    import matplotlib.pyplot as plt

    epc_range = np.arange(epc1, epc2, ((epc2 - epc1) / 300))
    y = []
    for epc in epc_range:
        m = n - round(n / epc)
        y.append(m / (linkage[m, 2] + 0.0000001))


    # Plotting merges vs. cluster distance
    plt.figure(figsize=(5.2, 4))
    plt.plot(epc_range, y, marker='o')
    #plt.yscale("log")
    plt.axvline(x=3.6, color='red', linestyle='--', label='Optimum')
    plt.axvline(x=5, color='blue', linestyle='--', label='Selected')
    #plt.axvline(x=25, color='red', linestyle='--')
    #plt.axvline(x=125, color='red', linestyle='--')

    plt.xlabel('Topics per cluster')
    plt.ylabel('Merges per distance')
    plt.grid(True)
    plt.show()

#plot_merges_vs_distance(1, 20)