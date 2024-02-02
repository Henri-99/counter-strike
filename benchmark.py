import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('csv/df_full.csv')
y = df['win']

# X = df['ts_win_prob']
# predictions = (X > 0.5).astype(int)

# X = df['elo_win_prob']
# predictions = (X > 0.5).astype(int)

X = df['team2_rank'] - df['team1_rank']
predictions = (X > 0).astype(int)

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
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.savefig("figures/ts-auc.png")
