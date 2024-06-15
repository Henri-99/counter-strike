from database.operations import get_date_range
from database.setup import session
from database.models import Match, Lineup, Map, PlayerStats
from sqlalchemy import func, and_
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import os
import numpy as np
import pandas as pd
from collections import Counter

def get_maps(start_date=None, end_date=None, limit=None):
	query = session.query(Map)
	if start_date is not None:
		query = query.filter(Map.datetime >= start_date)
	if end_date is not None:
		query = query.filter(Map.datetime <= end_date)
	if limit is not None:
		query = query.limit(limit)
	
	map_data = query.order_by(Map.id).all()

	return map_data

def get_matches(start_date=None, end_date=None, limit=None):
	query = session.query(Map)
	if start_date and end_date is not None:
		query = query.filter(
			and_(
				Match.datetime >= start_date,
				Match.datetime <= end_date
			)
		).order_by(Match.id)
	
	if limit is not None:
		query = query.limit(limit)
	
	match_data = query.all()

	return match_data

def get_maps_count(start_date=None, end_date=None, cs2=None, lan=None):
	query = session.query(Map)

	if cs2 is not None or lan is not None:
		query = query.join(Match, Map.match_id == Match.id)

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)

	if start_date and end_date is not None:
		query = query.filter(
			and_(
				Map.datetime >= start_date,
				Map.datetime <= end_date
			)
		)
	
	map_count = query.count()
	
	return map_count

def get_matches_count(start_date=None, end_date=None, cs2=None, lan=None):
	query = session.query(Match)

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)

	if start_date and end_date is not None:
		query = query.filter(
			and_(
				Match.datetime >= start_date,
				Match.datetime <= end_date
			)
		)
	
	match_count = query.count()
	
	return match_count

def records_per_month(cs2 = None, lan = None):
	data = []
	start_date, end_date = get_date_range()
	end_date = end_date + timedelta(days = 30)
	first_days = []
	current_date = start_date
	
	while current_date <= end_date:
		first_days.append(current_date)

		# Move to the first day of the next month
		if current_date.month == 12:
			current_date = datetime(current_date.year + 1, 1, 1)
		else:
			current_date = datetime(current_date.year, current_date.month + 1, 1)

	date_pairs = [(first_days[i], first_days[i + 1]) for i in range(len(first_days) - 1)]

	for start, end in date_pairs:
		maps = get_maps_count(start, end, cs2, lan)
		matches = get_matches_count(start, end, cs2, lan)
		data.append({
			"date": start,
			"match_count" : matches,
			"map_count": maps
		})

	# Create a Pandas DataFrame from the data list
	df = pd.DataFrame(data)

	return df

def records_per_week():
	week_maps_list = []

	start_date, end_date = get_date_range()
	end_date = datetime(start_date.year + 1, 1, 1)
	first_days = []
	current_date = start_date

	while current_date <= end_date:
		first_days.append(current_date)

		# Move to the first day of the next week, but not beyond end_date
		current_date += timedelta(days=7)
		if current_date > end_date:
			current_date = end_date
	
	date_pairs = [(first_days[i], first_days[i + 1]) for i in range(len(first_days) - 1)]

	for start, end in date_pairs:
		maps = get_maps_count(start, end)
		matches = get_matches_count(start, end)
		week_maps_list.append({
			"date": start,
			"match_count" : matches,
			"map_count": maps
		})

	df = pd.DataFrame(week_maps_list)

	return df

def map_distribution(cs2 = None, lan = None, min_rank = None, start_date = None, end_date = None):
	# Filter Maps based on Match.cs2
	query = session.query(Map.map_name, func.count(Map.id)).group_by(Map.map_name)

	if cs2 is not None or lan is not None:
		query = query.join(Match, Map.match_id == Match.id)

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)
		
	map_counts = query.all()

	df = pd.DataFrame(map_counts, columns=["map_name", "count"])
	df = df.sort_values(by="count", ascending=False)

	return df

def format_distribution(cs2 = None, lan = None):
	query = session.query(Match.best_of, func.count(Match.id)).group_by(Match.best_of)

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)

	best_of_counts = query.all()

	df = pd.DataFrame(best_of_counts, columns = ["format", "count"])

	return df

def rank_distribution(cs2=True, lan=True):
	query = (
		session.query(Match.team1_rank, Match.team2_rank)
	)

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)

	# Fetch the team ranks
	team_ranks = query.all()
	
	df = pd.DataFrame(team_ranks)

	df['average_rank'] = df[['team1_rank', 'team2_rank']].mean(axis=1)

	count_df = df['average_rank'].value_counts().reset_index()
	count_df.columns = ['average_rank', 'count']

	return count_df.sort_values(by="average_rank", ascending=True)

def rank_differential_distribution(cs2 = None, lan = None):
	query = session.query(Match.team1_rank, Match.team2_rank).filter(Match.team1_rank.isnot(None), Match.team2_rank.isnot(None))

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)

	matches = query.all()

	differences = [abs(team1_rank - team2_rank) for team1_rank, team2_rank in matches]
	difference_counts = {}
	for diff in differences:
		if diff not in difference_counts:
			difference_counts[diff] = 1
		else:
			difference_counts[diff] += 1

	result_df = pd.DataFrame(list(difference_counts.items()), columns=["Rank_Difference", "Frequency"])
	result_df = result_df.sort_values(by="Rank_Difference")

	return result_df

def score_distribution(cs2 = None, lan = None):
	query = session.query(Map.t1_score, Map.t2_score, func.count(Map.id)).group_by(Map.t1_score, Map.t2_score)

	if cs2 is not None or lan is not None:
		query = query.join(Match, Map.match_id == Match.id)

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)
	
	score_combinations = query.all()
	df = pd.DataFrame(score_combinations, columns=["t1_score", "t2_score", "count"])
	df["score_pair"] = df.apply(lambda row: tuple(sorted([row["t1_score"], row["t2_score"]])), axis=1)

	result_df = df.groupby("score_pair")["count"].sum().reset_index()
	result_df.rename(columns={"score_pair": "Score", "count": "Count"}, inplace=True)

	return result_df

def winner_rank_diff_dist(cs2 = None, lan = None):
	# Get team ids, ranks, winner_id. Calculate winning team rank - losing team rank
	query = session.query(Match.team1_rank, Match.team2_rank, Match.winner, Match.team1_id).filter(
		Match.team1_rank.isnot(None), Match.team2_rank.isnot(None), Match.winner.isnot(None)
	)

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)

	matches = query.all()

	differences = [team2_rank - team1_rank if winner == team1_id else team1_rank - team2_rank for team1_rank, team2_rank, winner, team1_id in matches]
	difference_counts = {}
	
	for diff in differences:
		if diff not in difference_counts:
			difference_counts[diff] = 1
		else:
			difference_counts[diff] += 1

	result_df = pd.DataFrame(list(difference_counts.items()), columns=["Rank_Difference", "Frequency"])
	result_df = result_df.sort_values(by="Rank_Difference")

	return result_df

def score_win_probability(cs2 = None, lan = None, map_name = None):
	query = session.query(Map.t1_score, Map.t2_score, Map.t1_round_history, Map.t2_round_history, Map.map_name)

	query = query.filter(Map.overtime == 0) #Modify later to support OT maps

	if cs2 is not None or lan is not None:
		query = query.join(Match, Map.match_id == Match.id)

	if cs2 is not None:
		query = query.filter(Match.cs2 == 1 if cs2 else Match.cs2 == 0)

	if lan is not None:
		query = query.filter(Match.lan == 1 if lan else Match.lan == 0)

	if map_name is not None:
		query = query.filter(Map.map_name == map_name)
		
	map_counts = query.limit(10).all()

	winner = 1 if Map.t1_score > Map.t2_score else 0

	

	return map_counts

def score_progression(t1_round_history, t2_round_history):
	team1_score = 0
	team2_score = 0
	score_progression = []

	for i in range(t1_round_history):
		if t1_round_history[i] == 'C' or t1_round_history[i] == 'T':
			team1_score += 1
		elif t2_round_history[i] == 'C' or t2_round_history[i] == 'T':
			team2_score += 1

		score_progression.append(f"{team1_score}-{team2_score}")

	return score_progression

def configure_plot_settings():
	font_manager.fontManager.addfont("C:/Users/9903165908084/AppData/Local/Microsoft/Windows/Fonts/Palatino.ttf")
	mpl.rcParams['font.family'] = 'serif'
	mpl.rcParams['font.serif'] = 'Palatino'
	mpl.rcParams['font.size'] = 12

def plot_matches_per_month():
	configure_plot_settings()

	df = records_per_month()[['date', 'match_count']]

	df['date'] = pd.to_datetime(df['date'])
	df['year'] = df['date'].dt.year
	df['month'] = df['date'].dt.month

	pivot_table = pd.pivot_table(df, values='match_count', index='month', columns='year', aggfunc='sum', fill_value=0)
	pivot_table.plot(kind='bar', stacked=False, figsize=(10, 6), width=0.8, colormap='cividis')

	plt.xlabel('Month')
	plt.ylabel('Number of Matches Played')
	plt.title('Number of Matches Played Each Month (by Year)')
	plt.legend(title='Year')
	plt.savefig(os.path.join('figures','matches_per_month_total.png'))

def plot_rank_distribution():
	configure_plot_settings()

	df = rank_distribution()[['team_rank', 'count']]

	plt.plot(df['team_rank'], df['count'], label='Team Rank')
	plt.xlabel('Rank')
	plt.ylabel('Number of Matches Played')
	plt.title('Number of Matches Played')
	plt.savefig(os.path.join('figures','match_rank_distribution.png'))

def elo_plot():
	configure_plot_settings()
	# Assuming a logistic curve function as a base for the Elo rating system
	def elo_curve(rating_diff, max_elo=400):
		return 1 / (1 + 10 ** ((-rating_diff) / max_elo))

	# Generate a range of rating differences
	rating_diffs = np.linspace(-700, 700, 1000)

	# Calculate the expected score for each rating difference
	scores = elo_curve(rating_diffs)

	# Plotting the curve
	plt.figure(figsize=(10, 5))
	plt.plot(rating_diffs, scores, color='gray')

	# Adding title and labels
	plt.title('Rating Difference vs Expected Score')
	plt.xlabel('Difference in ratings')
	plt.ylabel('Expected Score')

	# Setting the y-axis ticks to be in percentages
	plt.yticks(np.arange(0, 1.1, 0.1), np.round(np.arange(0, 1.1, 0.1),1))

	# Adding the grid
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)

	plt.rcParams['axes.unicode_minus'] = False

	plt.tight_layout()

	# Show the plot
	plt.savefig('figures/elo.png', bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

	# print(records_per_week())

	# print(map_distribution(lan=True))

	# print(format_distribution())

	# print(rank_distribution(cs2 = False, lan = True))

	# print(rank_differential_distribution(lan = True))

	# print(score_distribution(cs2=False, lan = True))

	# print(winner_rank_diff_dist(cs2 = False))

	# print(score_win_probability())

	# print(score_progression("TBB_______________SDC"))

	# plot_rank_distribution()

	elo_plot()