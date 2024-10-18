import pandas as pd
import sqlite3
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

DATABASE_PATH = "Database/modules.db"
TOPICS_TABLE_NAME = "NERD_MODULES_NEW"
MODULES_TABLE_NAME = "SCORED_MODULES_NEW"
MODULE_COLUMN = "NAME"
KEYWORD_COLUMN = "TOPIC"

def getModuleTopics():
    db = sqlite3.connect(DATABASE_PATH)
    module_topics = pd.read_sql_query(f"SELECT * FROM {TOPICS_TABLE_NAME}", db)
    db.close()

    return module_topics

def getModules():
    db = sqlite3.connect(DATABASE_PATH)
    modules = pd.read_sql_query(f"SELECT * FROM {MODULES_TABLE_NAME}", db)
    db.close()

    return modules

def cleanTopics(module_topics, min_appearances, max_appearances_per_module=7):
    topic_appearances_per_module = module_topics.groupby([MODULE_COLUMN, KEYWORD_COLUMN]).size().reset_index(
        name="count")
    topic_appearances_per_module["count"] = topic_appearances_per_module["count"].clip(upper=max_appearances_per_module)

    unique_topics_count = topic_appearances_per_module.groupby(KEYWORD_COLUMN)["count"].sum().reset_index(name="count")
    unique_filtered_topics = unique_topics_count[unique_topics_count["count"] >= min_appearances].reset_index()

    return unique_filtered_topics


# Method to prepare keywords
def prep_keywords(min_appearances, max_appearances_per_module=7):
    module_topics = getModuleTopics()

    unique_filtered_topics = cleanTopics(module_topics, min_appearances, max_appearances_per_module)

    return unique_filtered_topics[KEYWORD_COLUMN], unique_filtered_topics["count"]


# Method to embed keywords
def embed_keywords(keywords, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(keywords, convert_to_tensor=True).cpu().numpy()
    return embeddings


def plot_silhouette_scores(linkage, keyword_embeddings, t_values, criterion='distance'):
    silhouette_scores = []

    for t in t_values:
        clusters = fcluster(linkage, t=t, criterion=criterion)
        score = silhouette_score(keyword_embeddings, clusters)
        print(score)
        silhouette_scores.append(score)

    # Plot the silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different t Values')
    plt.xlabel('t value')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()
