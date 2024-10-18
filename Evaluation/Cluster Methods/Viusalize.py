import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visCompResults(data):
    plt.figure(figsize=(4, 6))
    data = data.sort_values(by="adj_rand", ascending=False)
    sns.barplot(data=data, x='method', y='adj_rand', hue='variant', palette="Blues")

    # Add title and labels
    plt.title('')
    plt.ylabel('Adjusted Rand Index')
    plt.ylim(0, 0.62)
    plt.xlabel('Clustering Method')
    plt.xticks(rotation=90)

    # Display the plot
    plt.tight_layout()
    plt.show()


NewsGroupMethodsComp = pd.read_csv("database/NewsGroup20MethodsComp")
wikiCat10Comp = pd.read_csv("database/wikiCat10MethodsComp.csv")

visCompResults(NewsGroupMethodsComp)
print("NewsGroupMethodsComp")
visCompResults(wikiCat10Comp)
print("wikiCat10Comp")