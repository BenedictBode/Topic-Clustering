import pandas as pd
import plotly.express as px
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt


# Method to find the parent keyword for a single cluster
def cluster_parent(cluster_keywords, cluster_appearances, cluster_embeddings, dist_weight=0.9):
    mean_embedding = np.mean(cluster_embeddings, axis=0)
    distances = cdist([mean_embedding], cluster_embeddings, 'euclidean').flatten()

    epsilon = 1e-10  # to prevent devision by zero

    max_count = max(cluster_appearances) + epsilon
    max_dist = distances.max() + epsilon

    norm_distances = distances / max_dist
    norm_counts = 1 - (np.array(cluster_appearances) / max_count)

    scores = dist_weight * norm_distances + (1 - dist_weight) * norm_counts
    best_idx = np.argmin(scores)

    return cluster_keywords[best_idx]


# Method to find the parent keywords for all clusters
def cluster_parents(keywords, keyword_appearances, keyword_embeddings, keyword_cluster_mapping, dist_weight=0.7, add_silhouette_naming=False):
    cluster_parent_mapping = {}
    clusters = np.unique(keyword_cluster_mapping)

    silhouette_scores = silhouette_samples(keyword_embeddings, keyword_cluster_mapping)

    for cluster in clusters:
        clusters_indices = np.where(keyword_cluster_mapping == cluster)[0]
        clusters_embeddings = keyword_embeddings[clusters_indices]
        clusters_keywords = [keywords[i] for i in clusters_indices]
        clusters_appearances = [keyword_appearances[i] for i in clusters_indices]

        silhouette_scores_cluster = silhouette_scores[clusters_indices]

        cluster_name = cluster_parent(clusters_keywords, clusters_appearances,
                                                         clusters_embeddings,
                                                         dist_weight)

        if add_silhouette_naming:
            best_sil_idx = np.argmax(silhouette_scores_cluster)
            cluster_name += " / " + clusters_keywords[best_sil_idx]

        cluster_parent_mapping[cluster] = cluster_name




    return cluster_parent_mapping, silhouette_scores

def cluster_keywords(keywords, keyword_appearances, keyword_embeddings, distance_threshold=8.5, Z=None,
                     dist_weight=0.7):
    if Z is None:
        Z = linkage(keyword_embeddings, method='ward')

        # calculate optimal silhoutte score
    # plot_silhouette_scores(Z, keyword_embeddings)

    keyword_cluster_mapping = fcluster(Z, t=distance_threshold, criterion='distance')
    cluster_parent_mapping, silhouette_scores = cluster_parents(keywords, keyword_appearances, keyword_embeddings,
                                                                keyword_cluster_mapping, dist_weight)

    parents = map(lambda x: cluster_parent_mapping[x], keyword_cluster_mapping)
    return list(parents), silhouette_scores


def keyword_hierarchy_unbiased(keywords, keyword_embeddings, keyword_appearances, height, distance_threshold=8.5,
                               distance_increment=0.5):
    current_keywords = keywords
    current_keyword_embeddings = keyword_embeddings
    current_keyword_appearances = keyword_appearances
    hierarchy_df = pd.DataFrame({f'level 0': keywords})

    for level in range(1, height, 1):
        parents = cluster_keywords(current_keywords, current_keyword_appearances, current_keyword_embeddings,
                                   distance_threshold=level * distance_increment + distance_threshold)
        keyword_parent_map = dict(zip(current_keywords, parents))

        # Add parent clusters to the DataFrame
        hierarchy_df[f'level {level}'] = hierarchy_df[f'level {level - 1}'].map(keyword_parent_map)

        # Prepare for the next level
        parent_set = set(parents)
        current_keywords = list(parent_set)

        # Avoid using list comprehension directly in np.array
        current_keyword_embeddings = np.array([keyword_embeddings[keywords.index(kw)] for kw in current_keywords])
        current_keyword_appearances = [keyword_appearances[keywords.index(kw)] for kw in current_keywords]

        print(len(current_keyword_embeddings))

    return hierarchy_df


def find_best_silhouette_cutoff(Z, embeddings):
    import scipy.cluster.hierarchy as sch
    # Possible cutoff distances (unique distances from the linkage matrix)
    distances = np.unique(Z[:, 2])

    # Initialize the best silhouette score and corresponding cutoff distance
    best_score = -1
    best_cutoff = None

    for d in distances:
        # Form clusters at this cutoff distance
        clusters = sch.fcluster(Z, t=d, criterion='distance')

        # Calculate the silhouette score
        if len(set(clusters)) > 1:  # Silhouette score is not defined for a single cluster
            score = silhouette_score(embeddings, clusters)
            if score > best_score:
                best_score = score
                best_cutoff = d

    return best_cutoff, best_score

def plot_silhouette(df, cluster_label_col='cluster_label', silhouette_score_col='silhouette score', outlier_percentile=5, plot_height=6):
    """
    Plots a silhouette diagram based on the cluster labels and silhouette scores in the dataframe,
    while trimming outliers.

    Parameters:
    df (DataFrame): The dataframe containing the data.
    cluster_label_col (str): The name of the column containing cluster labels.
    silhouette_score_col (str): The name of the column containing silhouette scores.
    outlier_percentile (int): The percentile threshold for outlier trimming (default is 5).
    plot_height (int): The height of the plot (default is 6).
    """

    # Calculating the bounds for trimming
    lower_bound = df[silhouette_score_col].quantile(outlier_percentile / 100)
    upper_bound = df[silhouette_score_col].quantile(1 - outlier_percentile / 100)

    # Trimming the silhouette scores
    df[silhouette_score_col] = np.clip(df[silhouette_score_col], lower_bound, upper_bound)

    # Calculate average silhouette score for each cluster
    cluster_avg_scores = df.groupby(cluster_label_col)[silhouette_score_col].mean().sort_values()

    # Sort the dataframe by cluster labels based on average silhouette score
    sorted_cluster_labels = cluster_avg_scores.index
    df[cluster_label_col] = pd.Categorical(df[cluster_label_col], categories=sorted_cluster_labels, ordered=True)
    df = df.sort_values(by=[cluster_label_col, silhouette_score_col])

    # Plotting
    plt.figure(figsize=(10, plot_height))
    y_lower = 10

    for i in sorted_cluster_labels:
        ith_cluster_silhouette_values = df[df[cluster_label_col] == i][silhouette_score_col]
        ith_cluster_silhouette_values = ith_cluster_silhouette_values.sort_values()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            alpha=0.7
        )
        avg_score = ith_cluster_silhouette_values.mean()
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f"{str(i)} ({avg_score:.3f})")
        y_lower = y_upper + 10  # 10 for the 0 samples gap between clusters

    plt.xlabel("Silhouette Score")
    plt.ylabel("Cluster Label")
    plt.title("Silhouette Plot")
    plt.axvline(x=df[silhouette_score_col].mean(), color="red", linestyle="--")
    plt.show()

def keyword_hierachy(keywords, keyword_embeddings, keyword_appearances, cutting_distances, Z=None, dist_weight=0.7):
    hierarchy_df = pd.DataFrame({f'level 0': keywords})
    hierarchy_df["appearances"] = keyword_appearances

    if Z is None:
        Z = linkage(keyword_embeddings, method='ward')

    level = 0

    print(f"keyword count: {len(keywords)}")

    for cutting_distance in cutting_distances:
        level = level + 1

        parents, silhouette_scores = cluster_keywords(keywords, keyword_appearances, keyword_embeddings,
                                                      distance_threshold=cutting_distance, Z=Z, dist_weight=dist_weight)

        # average cluster size
        print(f"cluster count: {len(set(parents))}")

        hierarchy_df[f'level {level}'] = parents
        hierarchy_df[f'level {level} score'] = silhouette_scores

    return hierarchy_df

def map_cluster_labels(true_labels, predicted_labels):
    """
    Maps predicted cluster labels to true cluster labels using the Hungarian algorithm.

    Parameters:
    - true_labels: list of true cluster labels (strings or integers)
    - predicted_labels: list of predicted cluster labels (strings or integers)

    Returns:
    - mapping: dict mapping predicted labels to true labels
    - mapped_predictions: list of predicted labels mapped to the true labels
    """

    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    # Step 1: Encode string labels to numeric labels
    le_true = LabelEncoder()
    le_pred = LabelEncoder()

    true_labels_encoded = le_true.fit_transform(true_labels)
    predicted_labels_encoded = le_pred.fit_transform(predicted_labels)

    # Step 2: Create the confusion matrix
    conf_matrix = confusion_matrix(true_labels_encoded, predicted_labels_encoded)

    # Step 3: Apply Hungarian algorithm (linear_sum_assignment) to find the best mapping
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    # Step 4: Create a mapping from predicted labels to true labels
    mapping = {le_pred.inverse_transform([col])[0]: le_true.inverse_transform([row])[0]
               for row, col in zip(row_ind, col_ind)}

    # Apply mapping to predicted labels
    mapped_predictions = [mapping[label] for label in predicted_labels]

    return mapping, mapped_predictions


def plot_tsne_clusters(cluster_labels, tsne_result, labels, score):
    """
    Plots t-SNE clusters, showing a fixed number of labels per cluster to reduce clutter.
    All labels are still visible in the hover tooltip.

    Parameters:
    - cluster_labels: Array or list of cluster labels
    - tsne_result: 2D t-SNE result array (shape: [n_samples, 2])
    - labels: Array or list of labels to display on the plot
    - labels_per_cluster: Number of labels to display per cluster (default is 5)

    Returns:
    - fig: Plotly figure object
    """
    # Combine tsne_result, cluster_labels, and labels into a DataFrame
    plot_data = pd.DataFrame({
        'tsne_x': tsne_result[:, 0],
        'tsne_y': tsne_result[:, 1],
        'cluster': cluster_labels,
        'score': score,
        'label': labels
    })

    # Sort the data by cluster labels
    plot_data = plot_data.sort_values(by='cluster').reset_index(drop=True)

    # Create a scatter plot with Plotly
    fig = px.scatter(
        plot_data,
        x='tsne_x', y='tsne_y',
        color='cluster',
        text="label",
        hover_name='label',  # This ensures all labels are visible on hover
        labels={'color': 'Cluster Label', 'score': 'score'},
        template='plotly_white'
    )

    # Adjust the layout to ensure labels are visible
    fig.update_traces(textposition='top center')

    # Show the plot
    fig.show()

def remove_noisy_clusters(df, cluster_label_col='level 1', score_col='score', top_percent=0.1):
    # Calculate the mean score for each cluster with observed=True
    cluster_means = df.groupby(cluster_label_col, observed=True)[score_col].mean()

    # Determine the threshold for the worst clusters
    threshold = cluster_means.quantile(top_percent)

    # Identify the clusters to remove
    clusters_to_remove = cluster_means[cluster_means <= threshold].index

    # Filter the DataFrame using .loc to avoid SettingWithCopyWarning
    filtered_df = df.loc[~df[cluster_label_col].isin(clusters_to_remove)]

    return filtered_df