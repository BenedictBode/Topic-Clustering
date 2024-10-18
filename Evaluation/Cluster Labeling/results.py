

def clusterLabels(
        cluster_level = "level 1",
        sil_w = 0.0,
        cohesion_w = 0.0,
        seperation_w = 0.0,
        appearances_w = 0.0):


    df.fillna(1, inplace=True)
    df['labelScore_' + cluster_level] = ((df['norm_cohesion_' + cluster_level] * -cohesion_w) +
                                         (df['norm_separation_' + cluster_level] * seperation_w) +
                                         (df['norm_silhouette_' + cluster_level] * sil_w) +
                                         (df['log_norm_appearances_' + cluster_level] * appearances_w))

    #df['labelScore_' + cluster_level] = (df['separation_' + cluster_level] * w) - (df['cohesion_' + cluster_level] * (1 - w))


    cluster_labels = df.loc[df.groupby(cluster_level)['labelScore_' + cluster_level].idxmax()][[cluster_level, PRIMARY_LEVEL]].copy()
    return cluster_labels

def calcAccuracy(testSet, predicted, cluster_level = "level 1"):

    # Ensure the renaming is applied correctly (not chained)
    testSet = testSet[testSet["cluster_level"] == cluster_level].copy()
    testSet.rename(columns={"choice 1": "level 0", "cluster": cluster_level}, inplace=True)

    # Merge the predicted labels with the testSet on 'level 1'
    merged_df = testSet.merge(predicted, on=cluster_level, how="left", suffixes=('_test', '_predicted'))
    #print(merged_df.columns)
    # Check how many 'level 0' elements match
    #print(merged_df[["level 0_test", "level 0_predicted"]])
    matched = merged_df[merged_df["level 0_test"] == merged_df["level 0_predicted"]]

    #not_matched = merged_df[merged_df["level 0_test"] != merged_df["level 0_predicted"]]
    #print(not_matched[["level 0_test", "level 0_predicted"]])

    # Calculate the accuracy (number of matches / total test set size)
    num_matches = matched.shape[0]
    total = merged_df.shape[0]

    accuracy = num_matches / total if total > 0 else 0

    #print(f"Number of matches: {num_matches}")
    #print(f"Total entries: {total}")
    return accuracy

def matchingTestSet():

    testSet = pd.read_csv("Evaluation/Cluster Labeling/questions.csv")
    #testSet = testSet[testSet["cluster_level"] == "level 2"]

    testSet["overlap 1"] = testSet["choice 1"] == testSet["choice 2"]
    testSet["overlap 2"] = testSet["choice 1"] == testSet["choice 3"]
    testSet["overlap 3"] = testSet["choice 2"] == testSet["choice 3"]
    testSet["overlap 4"] = testSet["overlap 1"] & testSet["overlap 2"] & testSet["overlap 3"]

    print(len(testSet[testSet["overlap 2"]]), len(testSet[testSet["overlap 4"]]))

    return pd.DataFrame(testSet[testSet["overlap 4"]])

matchingTestSet = matchingTestSet()
def evaluate_parameters_and_plot():
    import matplotlib.pyplot as plt
    import seaborn as sns

    parameters = ['appearances', 'cohesion', 'separation', 'silhouette']
    levels = ['level 1', 'level 2']

    # Store the results in a DataFrame
    results = pd.DataFrame(columns=['Parameter', 'Level', 'Accuracy'])

    for param in parameters:
        # Set the weights
        appearances_w = 1.0 if param == 'appearances' else 0.0
        cohesion_w = 1.0 if param == 'cohesion' else 0.0
        seperation_w = 1.0 if param == 'separation' else 0.0
        sil_w = 1.0 if param == 'silhouette' else 0.0

        for level in levels:
            # Get the predicted labels
            predicted = clusterLabels(cluster_level=level,
                                      appearances_w=appearances_w,
                                      cohesion_w=cohesion_w,
                                      seperation_w=seperation_w,
                                      sil_w=sil_w)
            # Calculate accuracy
            acc = calcAccuracy(testSet=matchingTestSet, predicted=predicted, cluster_level=level)
            # Append to results
            results = pd.concat([results, pd.DataFrame([{'Parameter': param,
                                      'Level': level,
                                      'Accuracy': acc}])], ignore_index=True)

def evaluate_parameters_and_plot2():
    import matplotlib.pyplot as plt
    import seaborn as sns

    levels = ['level 1', 'level 2']

    # Store the results in a DataFrame
    results = pd.DataFrame(columns=['Parameter', 'Level', 'Accuracy'])

    for param in [0.1, 0.2, 0.5, 0.6, 0.9, 1.0]:
        # Set the weights
        appearances_w = 1-param
        cohesion_w = param
        seperation_w = 1.0 if param == 'separation' else 0.0
        sil_w = 1.0 if param == 'silhouette' else 0.0

        for level in levels:
            # Get the predicted labels
            predicted = clusterLabels(cluster_level=level,
                                      appearances_w=appearances_w,
                                      cohesion_w=cohesion_w,
                                      seperation_w=seperation_w,
                                      sil_w=sil_w)
            # Calculate accuracy
            acc = calcAccuracy(testSet=matchingTestSet, predicted=predicted, cluster_level=level)
            # Append to results
            results = pd.concat([results, pd.DataFrame([{'Parameter': param,
                                                         'Level': level,
                                                         'Accuracy': acc}])], ignore_index=True)


    print(results)
    # Plot the bar chart
    plt.figure(figsize=(5, 5))
    sns.barplot(data=results, x='Parameter', y='Accuracy', hue='Level', palette="Blues")
    plt.ylabel('Accuracy')
    plt.xlabel('Parameter')
    plt.legend(title='Level')
    plt.tight_layout()
    plt.show()


def playingAroundWithConfigs():
    cluster_level = "level 2"

    configs = [[0.00, 0.0, 0.0, 1.0]]

    results = pd.DataFrame(columns=["config", "level", "accuracy"])

    for config in configs:
        for level in ["level 1", "level 2"]:
            predicted = clusterLabels(cluster_level=level, appearances_w=config[0], cohesion_w=config[1],
                                      seperation_w=config[2], sil_w=config[3])
            acc = calcAccuracy(testSet=matchingTestSet, predicted=predicted, cluster_level=level)

            # Append the result to the DataFrame
            results = results.append({"config": tuple(config), "level": level, "accuracy": acc}, ignore_index=True)

    print(results)

evaluate_parameters_and_plot2()