import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('csv/df_full_diff.csv')
print(df.shape)
y = df['win']

# X = df['ts_win_prob']
# predictions = (X > 0.5).astype(int)

# X = df['elo_win_prob']
# predictions = (X > 0.5).astype(int)

# X = df['team2_rank'] - df['team1_rank']
# predictions = (X > 0).astype(int)

X = df['map_wr']
df['predictions'] = 0  # Initialize all predictions to 0

# Case where map_wr is greater than 0
df.loc[df['map_wr'] > 0, 'predictions'] = 1

# Case where map_wr is equal to 0 and wr_diff is positive or zero
df.loc[(df['map_wr'] == 0) & (df['map_rwr'] >= 0), 'predictions'] = 1

predictions = df['predictions']

accuracy = accuracy_score(y, predictions)
f1 = f1_score(y, predictions)
cm = confusion_matrix(y, predictions)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
cm_df = pd.DataFrame(cm, 
                     index=['True 0', 'True 1'], 
                     columns=['Pred 0', 'Pred 1'])
print(cm_df)


# auc_roc = roc_auc_score(y, X)
# print(f'AUC ROC: {auc_roc}')

# fpr, tpr, thresholds = roc_curve(y, X)
# auc = roc_auc_score(y, X)

# # Plotting
# plt.figure(figsize=(9, 8))
# plt.plot(fpr, tpr, color='#73879d', lw=2, label=f'ROC curve (area = {auc:.2f})')
# plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic Curve')
# # plt.legend(loc="lower right")
# plt.savefig("figures/ts-auc.png", bbox_inches='tight')
