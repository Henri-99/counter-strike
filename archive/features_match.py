from database.setup import session
from database.models import Match, Map, Lineup, PlayerMatchTrueSkill, LineupAge
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import or_, and_, func
import scipy.stats as stats

def generate_match_dataframe(start_date = "2023-11-01", n_matches=None):

	# Joining Match and Map tables with Match on the left side
	query = session.query(Match)\
		.filter(Match.datetime >= start_date)\
		.filter(and_(Match.team1_rank.isnot(None), Match.team2_rank.isnot(None)))\
		.filter(or_(Match.team1_rank <= 30, Match.team2_rank <= 30))\
		# .filter(Match.best_of >= 3)\
		# .filter(Match.lan == 1)\

	if n_matches:
		query = query.limit(n_matches)

	result = query.all()
	print(f"{len(result)} records queried")

	# Creating a DataFrame from the result
	columns = [
		"match_id", "datetime", "team1_id", "team2_id", "team1", "team2", "team1_rank", "team2_rank", "rank_diff", "lan", "elim", "t1_score", "t2_score", "win"
	]
	
	words_to_check = ["elimination", "eliminated", "lower", "consolidation", "quarter", "semi", "grand"]

	data = [
		(
			match.id,
			match.datetime,
			match.team1_id,
			match.team2_id,
			match.team1,
			match.team2,
			match.team1_rank,
			match.team2_rank,
			match.team2_rank - match.team1_rank,
			1 if match.lan else 0,
			1 if any(word in match.box_str.lower() for word in words_to_check) else 0,
			match.team1_score,
			match.team2_score,
			1 if match.team1_id == match.winner else 0
		)
		for match in result
	]

	df = pd.DataFrame(data, columns=columns)

	return df

# Function to calculate win-rate for a team in the previous 3 months
def calculate_win_rates(row, team_id_column):   
	# Extract team ID and match datetime
	team_id = row[team_id_column]
	match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")

	# Define the time range for the previous 3 months
	start_date = match_datetime - timedelta(days=90)

	# Query maps for the specified team within the time range
	team_match_history_query = session.query(Match)\
		.filter(
			Match.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date),
			Match.id < row['match_id'],
			(Match.team1_id == team_id) | (Match.team2_id == team_id)
		)
	
	# Calculate the win-rate
	matches_played = team_match_history_query.count()
	if matches_played == 0:
		win_rate = 0.0
	else:
		win_count = team_match_history_query.filter(Match.winner == team_id).count()
		win_rate = win_count / matches_played

	return win_rate, matches_played

# Function to calculate head-to-head stats in the previous 3 months
def calculate_head_to_head_stats(row):
	match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
	start_date = match_datetime - timedelta(days=90)

	team1_id = row['team1_id']
	team2_id = row['team2_id']

	team_maps_query = session.query(Map)\
		.filter(
			Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date),
			Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime),
			((Map.t1_id == team1_id) & (Map.t2_id == team2_id)) | ((Map.t1_id == team2_id) & (Map.t2_id == team1_id))
		)
	
	maps_considered = team_maps_query.all()

	h2h_maps_played_count = team_maps_query.count()
	if h2h_maps_played_count == 0:
		t1_h2h_win_rate = 0.0
		t2_h2h_win_rate = 0.0
	else:
		t1_h2h_win_rate = team_maps_query.filter(Map.winner_id == team1_id).count() / h2h_maps_played_count
		t2_h2h_win_rate = team_maps_query.filter(Map.winner_id == team2_id).count() / h2h_maps_played_count
	
	return h2h_maps_played_count, t1_h2h_win_rate, t2_h2h_win_rate

def fetch_trueskill_ratings(row):
	# Extract team IDs and match date from the row
	team1_id, team2_id = row['team1_id'], row['team2_id']
	match_id = row['match_id']

	# Helper function to calculate team TrueSkill
	def calculate_team_trueskill(team_id):
		# Get the lineup for the given match
		lineup = session.query(Lineup).filter(
			and_(Lineup.team_id == team_id, Lineup.match_id == match_id)
		).first()

		if not lineup:
			return None, None

		players = [lineup.player1_id, lineup.player2_id, lineup.player3_id, lineup.player4_id, lineup.player5_id]

		total_mu, total_sigma, count = 0, 0, 0

  		# Get latest trueskill for lineup
		for player_id in players:
			player_skill = session.query(PlayerMatchTrueSkill).filter(
				and_(PlayerMatchTrueSkill.player_id == player_id, PlayerMatchTrueSkill.match_id < match_id)
			).order_by(PlayerMatchTrueSkill.match_id.desc()).first()

			if player_skill:
				total_mu += player_skill.mu
				total_sigma += player_skill.sigma
				count += 1

		# Calculate team Trueskill (weighted average)
		if count > 0:
			return total_mu / count, total_sigma / count
		else:
			return None, None 
	
	# Fetch TrueSkill ratings for both teams
	t1_ts_mu, t1_ts_sigma = calculate_team_trueskill(team1_id)
	t2_ts_mu, t2_ts_sigma = calculate_team_trueskill(team2_id)

	# Calculate win probabilities
	
	def calculate_win_probability(mu1, sigma1, mu2, sigma2):
		"""
		Calculate the probability of Team 1 winning against Team 2.

		:param mu1: Mean skill level of Team 1
		:param sigma1: Standard deviation of Team 1's skill
		:param mu2: Mean skill level of Team 2
		:param sigma2: Standard deviation of Team 2's skill
		:return: Probability of Team 1 winning
		"""
		# Performance difference is normally distributed
		performance_diff_mean = mu1 - mu2
		performance_diff_variance = sigma1**2 + sigma2**2

		# Cumulative distribution function for the difference
		win_probability = 1 - stats.norm.cdf(0, loc=performance_diff_mean, scale=performance_diff_variance**0.5)

		return win_probability
	
	t1_ts_win_prob = calculate_win_probability(t1_ts_mu, t1_ts_sigma, t2_ts_mu, t2_ts_sigma)
	t2_ts_win_prob = 1 - t1_ts_win_prob

	return t1_ts_mu, t1_ts_sigma, t2_ts_mu, t2_ts_sigma, t1_ts_win_prob, t2_ts_win_prob
	team1_id, team2_id = row['team1_id'], row['team2_id']
	match_id = row['match_id']

def fetch_lineup_age_stats(row):
	team1_id, team2_id = row['team1_id'], row['team2_id']
	match_id = row['match_id']
	lineup_age_object_1 = session.query(LineupAge).filter(LineupAge.match_id == match_id).filter(LineupAge.team_id == team1_id).first()
	lineup_age_object_2 = session.query(LineupAge).filter(LineupAge.match_id == match_id).filter(LineupAge.team_id == team2_id).first()

	return lineup_age_object_1.age_days, lineup_age_object_1.matches_together, lineup_age_object_2.age_days, lineup_age_object_2.matches_together

def calculate_winstreak(row):
	team1_id, team2_id = row['team1_id'], row['team2_id'] 
	match_id = row['match_id']
	date = row['datetime']
	t1_winstreak, t2_winstreak, t1_days_since_last_match, t2_days_since_last_match = 0, 0, 0, 0

	def get_last_match_winners(date, team_id, match_id):
		match_winners = session.query(Match.winner, Match.datetime)\
			.filter(Match.datetime < func.strftime('%Y-%m-%d %H:%M', date),
			Match.id < match_id,
			(Match.team1_id == team_id) | (Match.team2_id == team_id))\
			.order_by(Match.datetime.desc())
		return match_winners.limit(50).all()
	t1_history = get_last_match_winners(date, team1_id, match_id)
	t2_history = get_last_match_winners(date, team2_id, match_id)

	def calculate_days(start_date, end_date):
			start = datetime.strptime(start_date.split(' ')[0], '%Y-%m-%d')
			end = datetime.strptime(end_date.split(' ')[0], '%Y-%m-%d')
			difference = (end - start).days

			return difference

	for match in t1_history:
		t1_days_since_last_match = calculate_days(match.datetime, date)
		if match.winner == team1_id:
			t1_winstreak += 1
		else:
			break
	for match in t2_history:
		t2_days_since_last_match = calculate_days(match.datetime, date)
		if match.winner == team2_id:
			t2_winstreak += 1
		else:
			break

	return t1_winstreak, t2_winstreak, t1_days_since_last_match, t2_days_since_last_match

def generate_new_dataset_csv():
	start_time = datetime.now()

	df = generate_match_dataframe()
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Time to generate match map dataframe:  {elapsed_time:.2f} seconds")
	start_time = datetime.now()

	df[['winrate_t1', 'playcount_t1']] = df.apply(lambda row: pd.Series(calculate_win_rates(row, 'team1_id')), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Time to calculate win rates for team1: {elapsed_time:.2f} seconds")
	start_time = datetime.now()

	
	df[['winrate_t2', 'playcount_t2']] = df.apply(lambda row: pd.Series(calculate_win_rates(row, 'team2_id')), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Time to calculate win rates for team2: {elapsed_time:.2f} seconds")
	start_time = datetime.now()

	df[['h2h_maps', 'h2h_wr_t1', 'h2h_wr_t2']] = df.apply(lambda row: pd.Series(calculate_head_to_head_stats(row)), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Time to calculate head-to-head stats: {elapsed_time:.2f} seconds")
	start_time = datetime.now()

	df[['t1_ts_mu', 't1_ts_sigma', 't2_ts_mu', 't2_ts_sigma', 't1_ts_win_prob', 't2_ts_win_prob']] =  df.apply(lambda row: pd.Series(fetch_trueskill_ratings(row)), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Time to calculate TrueSkill data: {elapsed_time:.2f} seconds")

	df[['age', 'lineup_xp', 'opp_age', 'opp_xp']] = df.apply(lambda row: pd.Series(fetch_lineup_age_stats(row)), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Time to get lineup age stats: {elapsed_time:.2f} seconds")
	start_time = datetime.now()

	df[['t1_ws', 't2_ws', 't1_cooldown', 't2_cooldown']] = df.apply(lambda row: pd.Series(calculate_winstreak(row)), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Time to calculate winstreaks: {elapsed_time:.2f} seconds")
	start_time = datetime.now()

	df[['t1_ws', 't2_ws', 't1_cooldown', 't2_cooldown']] = df.apply(lambda row: pd.Series(calculate_winstreak(row)), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Time to calculate pistol round success: {elapsed_time:.2f} seconds")

	ml_df = df[
				[	'team1_rank', 'team2_rank', 'rank_diff', 
					'lan', 'elim', 
					'playcount_t1', 'winrate_t1', 'playcount_t2', 'winrate_t2', 
					't1_ws', 't2_ws', 't1_cooldown', 't2_cooldown',
					'h2h_maps', 'h2h_wr_t1', 'h2h_wr_t2',
					't1_ts_mu', 't1_ts_sigma', 't2_ts_mu', 't2_ts_sigma', 't1_ts_win_prob', 
					'age', 'lineup_xp', 'opp_age', 'opp_xp', 
					'win'
				]
			]
	ml_df.columns = [
					'rank', 'opp_rank', 'rank_diff', 
					'lan', 'elim', 
					'matches_played', 'match_winrate', 'opp_matches_played', 'opp_winrate', 
					't1_ws', 't2_ws', 'cooldown', 'opp_cooldown',
					'h2h_maps', 'h2h_wr', 'h2h_opp_wr',
					'ts_mu', 'ts_sigma', 'opp_ts_mu', 'opp_ts_sigma', 'ts_win_prob', 
					'age', 'lineup_xp', 'opp_age', 'opp_xp', 
					'win'
					]
	print(ml_df.head(30))

	ml_df.to_csv("temp_df.csv")

def get_recent_matches(row):
	match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
	start_date = match_datetime - timedelta(days=90)
	team1_id = row['team1_id']
	team2_id = row['team2_id']

	# Get all match-map items
	t1_mm_objects = session.query(Match, Map)\
		.join(Map, Match.id == Map.match_id)\
		.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
		.filter((Map.t1_id == team1_id) | (Map.t2_id == team1_id))\
		.all()
	t2_mm_objects = session.query(Match, Map)\
		.join(Map, Match.id == Map.match_id)\
		.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
		.filter((Map.t1_id == team2_id) | (Map.t2_id == team2_id))\
		.all()
	
	return t1_mm_objects, t2_mm_objects

def get_recent_map_matches(row):
	match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
	start_date = match_datetime - timedelta(days=90)
	team1_id = row['team1_id']
	team2_id = row['team2_id']

	# Get all match-map items
	t1_mm_objects = session.query(Match, Map)\
		.join(Map, Match.id == Map.match_id)\
		.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
		.filter((Map.t1_id == team1_id) | (Map.t2_id == team1_id))\
		.all()
	t2_mm_objects = session.query(Match, Map)\
		.join(Map, Match.id == Map.match_id)\
		.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
		.filter((Map.t1_id == team2_id) | (Map.t2_id == team2_id))\
		.all()
	
	return t1_mm_objects, t2_mm_objects

def calculate_pistol_round_win_rate(map_match_objects, team_id):
	# Compute pistol round success
	def calculate_pistol_round_win_rate_for_map(round_history_string, cs2):
		round_wins = 0
		if round_history_string[0] != '_':
			round_wins += 1
		if round_history_string[12 if cs2 else 15] != '_':
			round_wins += 1
		return round_wins/2
	
	pistol_win_rate_list = []
	for mm in map_match_objects:
		if mm.Map.t1_id == team_id:
			pistol_win_rate_list.append(calculate_pistol_round_win_rate_for_map(mm.Map.t1_round_history, mm.Match.cs2))
		else:
			pistol_win_rate_list.append(calculate_pistol_round_win_rate_for_map(mm.Map.t2_round_history, mm.Match.cs2))
	pistol_win_rate = sum(pistol_win_rate_list) / len(pistol_win_rate_list)
	return pistol_win_rate

def calculate_map_specific_win_rates(map_match_objects, team_id):
	# count up 
	map_stats = []
	# { map: 'mirage', playcount = 0, wins = 0 }

	for mm in map_match_objects:
		map_name = mm.Map.map_name



def generate_features_for_row(row):
	team1_id = row['team1_id']
	team2_id = row['team2_id']

	mm_1, mm_2 = get_recent_map_matches(row)

	t1_pistol_wr = calculate_pistol_round_win_rate(mm_1, team1_id)
	t2_pistol_wr = calculate_pistol_round_win_rate(mm_2, team2_id)

	return t1_pistol_wr, t2_pistol_wr

def main():

	df = generate_match_dataframe(n_matches=100)
	df[['t1_pistol_wr', 't2_pistol_wr']] = df.apply(lambda row: pd.Series(generate_features_for_row(row)), axis=1)

	# ml_df = df[
	# 			[	'team1_rank', 'team2_rank', 'rank_diff', 
	# 				'playcount_t1', 'winrate_t1', 'playcount_t2', 'winrate_t2', 
	# 				'h2h_maps', 'h2h_wr_t1', 'h2h_wr_t2',
	# 				'lan', 'elim', 
	# 				't1_ts_mu', 't1_ts_sigma', 't2_ts_mu', 't2_ts_sigma', 't1_ts_win_prob', 
	# 				'age', 'lineup_xp', 'opp_age', 'opp_xp', 
	# 				't1_ws', 't2_ws', 't1_cooldown', 't2_cooldown',
	# 				'win'
	# 			]
	# 		]
	# ml_df.columns = [
	# 				'rank', 'opp_rank', 'rank_diff', 
	# 				'matches_played', 'match_winrate', 'opp_matches_played', 'opp_winrate', 
	# 				'h2h_maps', 'h2h_wr', 'h2h_opp_wr',
	# 				'lan', 'elim', 
	# 				'ts_mu', 'ts_sigma', 'opp_ts_mu', 'opp_ts_sigma', 'ts_win_prob', 
	# 				'age', 'lineup_xp', 'opp_age', 'opp_xp', 
	# 				't1_ws', 't2_ws', 'cooldown', 'opp_cooldown',
	# 				'win'
	# 				]
	print(df.head(30))


if __name__ == "__main__":
	# generate_new_dataset_csv()

	# df = generate_match_dataframe()
	# # df[['winrate_t1', 'playcount_t1']] = df.apply(lambda row: pd.Series(calculate_win_rates(row, 'team1_id')), axis=1)
	# df.apply(lambda row: pd.Series(get_recent_map_matches(row)), axis=1)
	# print(df)

	main()