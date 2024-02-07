from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import os
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("csv/modified_df_full.csv", index_col=0)

font_manager.fontManager.addfont("C:/Users/9903165908084/AppData/Local/Microsoft/Windows/Fonts/Palatino.ttf")
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['font.size'] = 12

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
    plt.figure(figsize=(18, 18), facecolor='white')

    plt.subplot(1, 3, 1)
    plt.ylim(0, 1)
    sns.boxplot(x='win', y='win_rate_diff', hue="win", data=df, showfliers=False, legend=None)
    plt.title(f'Difference in match win-rate')
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

    columns_to_plot_updated = ['rank_diff', 'win_rate_diff']
    for i, col in enumerate(columns_to_plot_updated, 1):
        plt.subplot(1, 2, i)
        sns.boxplot(x='win', y=col, hue="win", data=df, showfliers=False)
        plt.title(f'Boxplot of {col} by Match Outcome')
        plt.xlabel('Winning Team (1: Team 1, 0: Team 2)')
        plt.ylabel(col)
    plt.savefig("figures/class-box-2.png", bbox_inches='tight', facecolor=plt.gcf().get_facecolor())

    # Boxplot for average player rating, pistol win-rate diff
    plt.figure(figsize=(18, 9), facecolor='white')
    columns_to_plot_updated = ['pl_rating_diff', 'pistol_wr_diff']
    for i, col in enumerate(columns_to_plot_updated, 1):
        plt.subplot(1, 2, i)
        sns.boxplot(x='win', y=col, hue="win", data=df, showfliers=False)
        plt.title(f'Boxplot of {col} by Match Outcome')
        plt.xlabel('Winning Team (1: Team 1, 0: Team 2)')
        plt.ylabel(col)

    plt.savefig("figures/class-box-3.png", bbox_inches='tight', facecolor=plt.gcf().get_facecolor())

def corr_plot():
    ml_df = df.drop(['match_id', 'datetime', 'team1_id', 'team2_id', 'team1', 'team2', 't1_score', 't2_score'], axis = 1)
    corr_matrix = ml_df.corr()

    # Filter out features with a correlation below the threshold
    threshold = 0.05
    win_corr = corr_matrix['win']
    relevant_features = win_corr[abs(win_corr) >= threshold].index.tolist()
    filtered_corr_matrix = ml_df[relevant_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Filtered Correlation Matrix with 'win'")
    plt.show()
    plt.savefig("figures/corr_plot.png", bbox_inches='tight')

corr_plot()