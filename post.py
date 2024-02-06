import pandas as pd

df = pd.read_csv('csv/df_full_diff.csv', index_col=0)
print(df.shape)

# rank_threshold = 15
# df_filtered = df[(df['team1_rank'] <= rank_threshold) | (df['team2_rank'] <= rank_threshold)]
# df_filtered = df[(df['format'] >= 3)]
df_filtered = df[(df['lan'] == 1)]
print(df_filtered.shape)
df_filtered.to_csv("csv/df_lan.csv", index=False)