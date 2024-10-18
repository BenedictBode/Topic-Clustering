import pandas as pd
import DataPrep
import scipy.cluster.hierarchy as sch

linkage = pd.read_csv("../Database/Output/linkage.csv", header=None).values
embeddings = pd.read_csv("../Database/Output/embeddings.csv", header=None).values

t_vals = [23, 24, 25, 26, 27]
for i in range(len(t_vals)):
    t_vals[i] = int(len(linkage)/t_vals[i])

print(t_vals)
print(DataPrep.plot_silhouette_scores(linkage, embeddings, t_vals, "maxclust"))