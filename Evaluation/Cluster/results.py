import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("questions1.csv")

# Ensure 'is_right' column is boolean (if not already)
df['is_right'] = df['is_right'].astype(bool)

# Group by 'cluster_level' and count 'True' and 'False'
df_grouped = df.groupby("cluster_level")["is_right"].value_counts().unstack(fill_value=0)

# Sum of 'True' and 'False' (ignoring any potential 'unsure')
total_without_unsure = df_grouped[True] + df_grouped[False]

# Calculate the percentage of 'approved' (True) reactions
df_grouped['percentage'] = (df_grouped[True] / total_without_unsure) * 100

# Define the weights
weights = [5**4, 5**3, 5**2, 5**1, 5**0]

# Calculate weighted average and simple average
weighted_average = (df_grouped['percentage'] * weights).sum() / sum(weights)
average = df_grouped['percentage'].mean()

# Create the bar plot
plt.figure(figsize=(3, 3))
sns.barplot(x=df_grouped.index, y=df_grouped['percentage'], color="#5682AB")

# Add labels and title
plt.xlabel('')
plt.ylabel('Positive cohesion (%)')
plt.axhline(weighted_average, color='blue', linestyle='--', label=f'Weighted average: {weighted_average:.2f}%')
plt.axhline(average, color='black', linestyle='--', label=f'Average: {average:.2f}%')

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print averages
print(f"Average: {average:.2f}%")
print(f"Weighted average: {weighted_average:.2f}%")
