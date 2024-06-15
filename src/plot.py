from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import os
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("csv/df_full_diff.csv", index_col=0)

# font_manager.fontManager.addfont("C:/Users/9903165908084/AppData/Local/Microsoft/Windows/Fonts/Palatino.ttf")
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'Palatino'
# mpl.rcParams['font.size'] = 12

sns.set_style("whitegrid")
sns.set_palette("pastel")

def plot_box_ratings():
	# Boxlot for TrueSkill, Elo, HLTV ranking difference
	plt.figure(figsize=(18, 9), facecolor='white')

	plt.subplot(1, 3, 1)
	plt.ylim(0, 1)
	sns.boxplot(x='win', y='ts_win_prob', hue="win", data=df, showfliers=False, legend=None)
	plt.title(f'TrueSkill win probability')
	plt.xlabel('Class')
	plt.ylabel('')

	plt.subplot(1, 3, 2)
	plt.ylim(0, 1)
	sns.boxplot(x='win', y='elo_win_prob', hue="win", data=df, showfliers=False, legend=None)
	plt.title(f'Elo win probability')
	plt.xlabel('Class')
	plt.ylabel('')

	plt.subplot(1, 3, 3)
	sns.boxplot(x='win', y='rank_diff', hue="win", data=df, showfliers=False, legend=None)
	plt.title(f'HLTV ranking difference')
	plt.xlabel('Class')
	plt.ylabel('')

	plt.savefig("figures/class-box-1.png", bbox_inches='tight', facecolor=plt.gcf().get_facecolor())

def plot_box_team_stats():
	# Boxplot for win-rate
	plt.figure(figsize=(18, 9), facecolor='white')

	plt.subplot(1, 3, 1)
	# plt.ylim(0, 1)
	sns.boxplot(x='win', y='wr_diff', hue="win", data=df, showfliers=False, legend=None)
	plt.title(f'Difference in overall match win-rate')
	plt.xlabel('Class')
	plt.ylabel('')

	plt.subplot(1, 3, 2)
	# plt.ylim(0, 1)
	sns.boxplot(x='win', y='map_wr', hue="win", data=df, showfliers=False, legend=None)
	plt.title(f'Number of maps where team had a higher win-rate')
	plt.xlabel('Class')
	plt.ylabel('')
	
	df_filtered = df[df['h2h_maps'] != 0]
	print(df.shape)
	print(df_filtered.shape)
	plt.subplot(1, 3, 3)
	sns.boxplot(x='win', y='h2h_wr', hue="win", data=df_filtered, showfliers=False, legend=None)
	plt.title(f'Head-to-head historical map win-rate')
	plt.xlabel('Class')
	plt.ylabel('')
	
	plt.savefig("figures/class-box-2.png", bbox_inches='tight', facecolor=plt.gcf().get_facecolor())

	# columns_to_plot_updated = ['rank_diff', 'win_rate_diff']
	# for i, col in enumerate(columns_to_plot_updated, 1):
	#     plt.subplot(1, 2, i)
	#     sns.boxplot(x='win', y=col, hue="win", data=df, showfliers=False)
	#     plt.title(f'Boxplot of {col} by Match Outcome')
	#     plt.xlabel('Winning Team (1: Team 1, 0: Team 2)')
	#     plt.ylabel(col)
	# plt.savefig("figures/class-box-2.png", bbox_inches='tight', facecolor=plt.gcf().get_facecolor())

	# # Boxplot for average player rating, pistol win-rate diff
	# plt.figure(figsize=(18, 9), facecolor='white')
	# columns_to_plot_updated = ['pl_rating_diff', 'pistol_wr_diff']
	# for i, col in enumerate(columns_to_plot_updated, 1):
	#     plt.subplot(1, 2, i)
	#     sns.boxplot(x='win', y=col, hue="win", data=df, showfliers=False)
	#     plt.title(f'Boxplot of {col} by Match Outcome')
	#     plt.xlabel('Winning Team (1: Team 1, 0: Team 2)')
	#     plt.ylabel(col)

	# plt.savefig("figures/class-box-3.png", bbox_inches='tight', facecolor=plt.gcf().get_facecolor())

def plot_scatter():
	class1 = df[df['win'] == 0]
	class2 = df[df['win'] == 1]

	# Creating the scatter plot
	plt.figure(figsize=(10, 6))

	plt.scatter(class1['ts_win_prob'], class1['age_diff'], color='blue', label='Class 1')
	plt.scatter(class2['ts_win_prob'], class2['age_diff'], color='red', label='Class 2')

	plt.title('Scatter Plot of ts_win_prob vs xp_diff')
	plt.xlabel('ts_win_prob')
	plt.ylabel('xp_diff')
	plt.legend()
	
	plt.savefig("figures/scatter.png", bbox_inches='tight', facecolor=plt.gcf().get_facecolor())

def corr_plot():
	ml_df = df.drop(['match_id', 'datetime', 'team1_id', 'team2_id', 'team1', 'team2', 't1_score', 't2_score'], axis = 1)
	ml_df = ml_df[[ 'win', 'team1_rank', 'team2_rank', 'rank_diff', 't1_mu', 't1_sigma', 't2_mu', 't2_sigma', 'ts_win_prob',
       't1_elo', 't2_elo', 'elo_win_prob', 't1_wr', 't2_wr', 'map_xp', 'map_wr',	'map_rwr', 
	   'age_diff', 'xp_diff', 'mp_diff', 'wr_diff', 'ws_diff', 'rust_diff',
	   'avg_hltv_rating_diff', 'avg_fk_pr_diff', 'avg_cl_pr_diff', 'avg_pl_rating_diff', 'avg_pl_adr_diff', 'avg_plr_kast_diff', 'avg_pistol_wr_diff']]
	
	corr_matrix = ml_df.corr()

	# Filter out features with a correlation below the threshold
	# threshold = 0.05
	# win_corr = corr_matrix['win']
	# relevant_features = win_corr[abs(win_corr) >= threshold].index.tolist()
	# filtered_corr_matrix = ml_df[relevant_features].corr()

	plt.figure(figsize=(12, 10))
	sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm')
	plt.title("Correlation Matrix with 'win'")
	plt.savefig("figures/corr_plot.png", bbox_inches='tight')

corr_plot()
# plot_scatter()