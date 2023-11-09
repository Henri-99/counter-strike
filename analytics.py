from database.operations import get_date_range
from database.setup import session
from database.models import Match, Lineup, Map, PlayerStats
from sqlalchemy import func, and_
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_maps(start_date=None, end_date=None, limit=None):
	query = session.query(Map)
	if start_date and end_date is not None:
		query = query.filter(
			and_(
				Map.datetime >= start_date,
				Map.datetime <= end_date
			)
		).order_by(Map.id)
	
	if limit is not None:
		query = query.limit(limit)
	
	map_data = query.all()

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

def get_maps_count(start_date=None, end_date=None):
	query = session.query(Map)
	if start_date and end_date is not None:
		query = query.filter(
			and_(
				Map.datetime >= start_date,
				Map.datetime <= end_date
			)
		)
	
	map_count = query.count()
	
	return map_count

def get_matches_count(start_date=None, end_date=None):
	query = session.query(Match)
	if start_date and end_date is not None:
		query = query.filter(
			and_(
				Match.datetime >= start_date,
				Match.datetime <= end_date
			)
		)
	
	match_count = query.count()
	
	return match_count

def records_per_month():
	data = []
	start_date, end_date = get_date_range()
	
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
		maps = get_maps_count(start, end)
		matches = get_matches_count(start, end)
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

def map_distribution():
	map_counts = session.query(Map.map_name, func.count(Map.id)).group_by(Map.map_name).all()

	df = pd.DataFrame(map_counts, columns=["map_name", "count"])
	df = df.sort_values(by="count", ascending=False)

	return df

def format_distribution():
	best_of_counts = session.query(Match.best_of, func.count(Match.id)).group_by(Match.best_of).all()

	df = pd.DataFrame(best_of_counts, columns = ["format", "count"])

	return df

def rank_distribution():
	team1_rank_counts = session.query(Match.team1_rank, func.count(Match.id)).group_by(Match.team1_rank).all()
	team2_rank_counts = session.query(Match.team2_rank, func.count(Match.id)).group_by(Match.team2_rank).all()

	team1_rank_df = pd.DataFrame(team1_rank_counts, columns=["team_rank", "count"])
	team2_rank_df = pd.DataFrame(team2_rank_counts, columns=["team_rank", "count"])
	combined_df = pd.concat([team1_rank_df, team2_rank_df])

	result_df = combined_df.groupby("team_rank")["count"].sum().reset_index()

	return result_df

def rank_differential_distribution():
	matches = session.query(Match.team1_rank, Match.team2_rank).filter(Match.team1_rank.isnot(None), Match.team2_rank.isnot(None)).all()

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

def score_distribution(cs2=False):
	query = session.query(Map.t1_score, Map.t2_score, func.count(Map.id)).group_by(Map.t1_score, Map.t2_score)

	# Filter Maps based on Match.cs2
	if cs2:
		query = query.join(Match, Map.match_id == Match.id).filter(Match.cs2 == 1)
	else:
		query = query.join(Match, Map.match_id == Match.id).filter(Match.cs2 == 0)
	
	score_combinations = query.all()
	df = pd.DataFrame(score_combinations, columns=["t1_score", "t2_score", "count"])
	df["score_pair"] = df.apply(lambda row: tuple(sorted([row["t1_score"], row["t2_score"]])), axis=1)

	result_df = df.groupby("score_pair")["count"].sum().reset_index()
	result_df.rename(columns={"score_pair": "Score", "count": "Count"}, inplace=True)

	return result_df

def plot_maps_per_month():
	data = records_per_month()
	months = []
	map_count = []
	for month in data:
		months.append(month["date"])
		map_count.append(month["map_count"])
	
	
	plt.bar(months, map_count, width=10)
	plt.xlabel('Month')
	plt.ylabel('Maps played')
	plt.title('Map-frequency')
	plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
	plt.tight_layout()  # Ensure labels fit in the plot
	plt.savefig('my_bar_plot.png')  # Save the bar plot to a file

def plot_maps_per_week():
	data = records_per_week()
	weeks = []
	map_count = []
	for week in data:
		weeks.append(week["date"])
		map_count.append(week["map_count"])
	
	plt.plot(weeks, map_count)
	plt.xlabel('Month')
	plt.ylabel('Maps played')
	plt.title('Map-frequency')
	plt.savefig('my_plot.png')

if __name__ == "__main__":
	pd.set_option("display.max_rows", None)

	# print(map_distribution())

	# print(records_per_month())

	# print(records_per_week())

	# print(format_distribution())

	# print(rank_distribution())

	# print(rank_differential_distribution())

	print(score_distribution(cs2=False))