import matplotlib.pyplot as plt
import pandas as pd

# Data from the table provided by the user
data = {
    'Scenario': ['TrueSkill', 'XGBoost\n(All features)', 'MLP NN\n(18 features)', 'XGBoost\n(7 features)'],
    'Accuracy': [57.9, 60, 59.8, 59.6],
    'F1 Score': [59.7, 63.5, 62.3, 63.5],
    'ROC AUC': [61.6, 64.1, 63.7, 64.1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the style
plt.style.use('ggplot')

# Set up the bar width
barWidth = 0.25

# Set position of bar on X axis
r1 = range(len(df['Accuracy']))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Colors for the bars
colors = ['#abc9ea', '#efb792', '#d7d7d7']  # Third color is a light grey, similar to the first two

# Make the plot with updated aesthetics
plt.figure(figsize=(12,7))
plt.bar(r1, df['Accuracy'], color=colors[0], width=barWidth, edgecolor='grey', label='Accuracy')
plt.bar(r2, df['F1 Score'], color=colors[1], width=barWidth, edgecolor='grey', label='F1 Score')
plt.bar(r3, df['ROC AUC'], color=colors[2], width=barWidth, edgecolor='grey', label='ROC AUC')

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(df['Accuracy']))], df['Scenario'])
plt.grid(axis='y', linewidth=0.7, color='gray')

# Set the y-axis limit starting at 50
plt.ylim([54, None])

# Set the background to be completely white
plt.gca().set_facecolor('white')
plt.gcf().set_facecolor('white')

# Create legend & Show graphic
plt.legend()
plt.title('Model Comparison')

# Show the plot
plt.savefig('figures/results-bars.png', bbox_inches='tight')
