import pandas as pd

# df = pd.read_csv('csv/df_full_diff.csv', index_col=0)
# print(df.shape)

# rank_threshold = 15
# df_filtered = df[(df['team1_rank'] <= rank_threshold) | (df['team2_rank'] <= rank_threshold)]
# df_filtered = df[(df['format'] >= 3)]
# df_filtered = df[(df['lan'] == 1)]
# print(df_filtered.shape)
# df_filtered.to_csv("csv/df_lan.csv", index=False)

# df_full = pd.read_csv('csv/df_full.csv', index_col=0)
df_full = pd.read_csv('csv/df_full_diff.csv', index_col=0)
# df_full = pd.read_csv('csv/df_30.csv', index_col=0)
# df_full = pd.read_csv('csv/df_bo3.csv', index_col=0)
# df_full = pd.read_csv('csv/df_lan.csv', index_col=0)
data = df_full.drop(['match_id', 'datetime', 'team1_id', 'team2_id','team1', 'team2',  't1_score', 't2_score'], axis=1)

# data = df_full.drop([''])
data['lan'] = data['lan'].astype('category')
data['elim'] = data['elim'].astype('category')
data['format'] = data['format'].astype('category')

# Summary of descriptive statistics
# summary = data.describe()

X = data.drop(['win'], axis=1)
y = data['win']

print(X.shape)

split_index = int(0.8 * len(data))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(X_train.shape)
print(X_test.shape)