import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

EMBEDDING_MODEL = 'paraphrase-MiniLM-L6-v2'
RESULT_PATH = "../../Database/NewsGroup20Comp"
#EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


data_set = "newsgroup"
if data_set == "newsgroup":
    # Load 20newsgroups dataset (use all categories)
    categories = ['alt.atheism', 'comp.graphics', 'sci.space', 'talk.politics.guns']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories)
    data = newsgroups.data
    y_true = newsgroups.target
    print(len(data))
    RESULT_PATH = "../../Database/NewsGroup20MethodsComp"
else:
    # self labeled
    wikiCat10 = pd.read_csv("../../Database/wikiCat10.csv", sep=";")
    data = wikiCat10["level 0"]
    y_true = wikiCat10["level 1"]
    RESULT_PATH = '../../Database/wikiCat10MethodsComp.csv'


# SentenceTransformer model for embeddings
model = SentenceTransformer(EMBEDDING_MODEL)

# Generate embeddings from text using SentenceTransformer
embeddings = model.encode(data, convert_to_tensor=False)

#dimensionality reduction
svd = TruncatedSVD(n_components=50, random_state=42)
tsne = TSNE(n_components=2, random_state=42)

reduction_method = "tsne"

if reduction_method == "tsne":
    X_embeddings_reduced = tsne.fit_transform(embeddings)
else:
    X_embeddings_reduced = svd.fit_transform(embeddings)

# Function to evaluate clustering performance (ARI and Silhouette)
def evaluate_clustering(model, X, y_true):
    y_pred = model.fit_predict(X)
    ari = adjusted_rand_score(y_true, y_pred)
    sil_score = silhouette_score(X, y_pred, metric='euclidean') if len(set(y_pred)) > 1 else -1  # Avoid error when all labels are same
    return ari, sil_score

# Clustering models with different agglomerative linkage criteria
linkage_criteria = ['ward', 'complete', 'average', 'single']
results = []

# Agglomerative Clustering with embeddings (no reduction)
for linkage in linkage_criteria:
    agg_model = AgglomerativeClustering(n_clusters=len(set(y_true)), linkage=linkage)
    ari, sil_score = evaluate_clustering(agg_model, embeddings, y_true)
    results.append([linkage, 'embedding', ari, sil_score])

# Agglomerative Clustering with embeddings (with reduction)
for linkage in linkage_criteria:
    agg_model = AgglomerativeClustering(n_clusters=len(set(y_true)), linkage=linkage)
    ari, sil_score = evaluate_clustering(agg_model, X_embeddings_reduced, y_true)
    results.append([linkage, 'embedding_reduced', ari, sil_score])

# Agglomerative Clustering with TF-IDF (with reduction)
TFIDF = False
if TFIDF:
    # Baseline TF-IDF Vectorization + Dimensionality Reduction
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_tfidf = vectorizer.fit_transform(data)
    X_tfidf_reduced = svd.fit_transform(X_tfidf)

    for linkage in linkage_criteria:
        agg_model = AgglomerativeClustering(n_clusters=len(set(y_true)), linkage=linkage)
        ari, sil_score = evaluate_clustering(agg_model, X_tfidf_reduced, y_true)
        results.append([linkage, 'tfidf_reduced', ari, sil_score])

# KMeans as baseline comparison (Embeddings)
kmeans = KMeans(n_clusters=len(set(y_true)), random_state=42)
ari, sil_score = evaluate_clustering(kmeans, embeddings, y_true)
results.append(['kmeans', 'embedding', ari, sil_score])

# KMeans as baseline comparison (Embeddings Reduced)
ari, sil_score = evaluate_clustering(kmeans, X_embeddings_reduced, y_true)
results.append(['kmeans', 'embedding_reduced', ari, sil_score])

# DBSCAN as baseline comparison (Embeddings)
dbscan = DBSCAN(eps=0.5, min_samples=5)
ari, sil_score = evaluate_clustering(dbscan, embeddings, y_true)
results.append(['dbscan', 'embedding', ari, sil_score])

# DBSCAN as baseline comparison (Embeddings Reduced)
ari, sil_score = evaluate_clustering(dbscan, X_embeddings_reduced, y_true)
results.append(['dbscan', 'embedding_reduced', ari, sil_score])

# Save results to CSV
df_results = pd.DataFrame(results, columns=['method', 'variant', 'adj_rand', 'sil_score'])
df_results.to_csv(RESULT_PATH, index=False)

# Print summary of results
print("\nClustering Performance Comparison: ", RESULT_PATH)
print(df_results)
