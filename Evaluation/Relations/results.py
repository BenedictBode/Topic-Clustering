import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("questions1.csv")
df = df.groupby("cluster_level")["reaction"].value_counts().unstack(fill_value=0)

# Sum of 'approved' and 'disapproved' (ignoring 'unsure')
total_without_unsure = df[['approved', 'disapproved']].sum(axis=1)

# Calculate the percentage of 'approved' reactions
df['approved_percentage'] = (df['approved'] / total_without_unsure) * 100

# Display the result
df[['approved', 'disapproved', 'approved_percentage']]

print(df)

weights = [5**4, 5**3, 5**2, 5**1, 5**0]
weighted_average = (df['approved_percentage'] * weights).sum() / sum(weights)

approved_average = df['approved_percentage'].mean()
# Create the bar plot
plt.figure(figsize=(3, 3))
sns.barplot(x=df.index, y=df['approved_percentage'], color="#5682AB")

# Add labels and title
plt.xlabel('')
plt.ylabel('Positive relation (%)')
plt.ylim(0, 100)
plt.axhline(weighted_average, color='blue', linestyle='--', label=f'Weighted average: {weighted_average:.2f}%')
plt.axhline(approved_average, color='black', linestyle='--', label=f'Average: {approved_average:.2f}%')

print(approved_average, weighted_average)

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()