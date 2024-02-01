import pandas as pd

df = pd.read_csv('csv/df_full.csv', index_col=0)
print(df.shape)

rank_threshold = 30
df_filtered = df[(df['team1_rank'] <= rank_threshold) & (df['team2_rank'] <= rank_threshold)]
print(df_filtered.shape)
df_filtered.to_csv("csv/filtered_df.csv", index=False)


def generate_diff_features():
    # Calculate the difference and create a new column
    df['rank_diff'] = df['team2_rank'] - df['team1_rank']
    df['win_rate_diff'] = df['t1_wr'] - df['t2_wr']
    df['round_wr_diff'] = df['t1_wr'] - df['t2_wr']
    df['pistol_wr_diff'] = df['t1_avg_pistol_wr'] - df['t2_avg_pistol_wr']
    df['pl_rating_diff'] = df['t1_avg_pl_rating'] - df['t2_avg_pl_rating']

    # Save the modified dataframe to a new CSV file
    df.to_csv('csv/modified_df_full.csv', index=False)  # Replace with your desired file path