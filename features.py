# Transforms a dataframe of match records into a dataframe of ML features

from database.setup import session
from database.models import Match, Map, Lineup, PlayerMatchTrueSkill, LineupAge, PlayerElo
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import or_, and_, func
import scipy.stats as stats
from elo import calculate_expected_outcome, calculate_team_elo

# First let's get the matches

def generate_match_dataframe( start_date = "2019-07-01",
							  rank_threshold = 30, 
							  lan = None, 
							  min_format = None, 
							  n_matches = None, 
							  require_hltv_rank = False ):
	
	query = session.query(Match)\
		.filter(Match.datetime >= start_date)\

	if lan:
		query = query.filter(Match.lan == lan)
	if min_format:
		query = query.filter(Match.best_of >= min_format)
	if require_hltv_rank:
		query = query.filter(and_(Match.team1_rank.isnot(None), Match.team2_rank.isnot(None)))
	if rank_threshold:
		query = query.filter(or_(Match.team1_rank <= 30, Match.team2_rank <= 30))
	if n_matches:
		query = query.limit(n_matches)

	result = query.all()

	columns = [ "match_id", "datetime", "team1_id", "team2_id", "team1", "team2", "t1_score", "t2_score", "win", "format", "team1_rank", "team2_rank", "rank_diff", "lan", "elim" ]
	data = [
		(
			match.id,
			match.datetime,
			match.team1_id,
			match.team2_id,
			match.team1,
			match.team2,
			match.team1_score,
			match.team2_score,
			1 if match.team1_id == match.winner else 0,	# y
			match.best_of,
			match.team1_rank,
			match.team2_rank,
			match.team2_rank - match.team1_rank,
			1 if match.lan else 0,
			1 if any(word in match.box_str.lower() for word in ["elimination", "eliminated", "lower", "consolidation", "quarter", "semi", "grand"]) else 0
		)
		for match in result
	]

	df = pd.DataFrame(data, columns=columns)

	return df

# Next, let's get the TrueSkill, Elo, and Lineup features for both teams from db

import trueskill
env = trueskill.TrueSkill()

def get_trueskill(row):
	team1_id, team2_id, match_id, date = row['team1_id'], row['team2_id'], row['match_id'], row['datetime']
	date = date.split(" ")[0]

	# Get TrueSkill ratings for a given team_id
	def get_team_trueskill(team_id, date):
		lineup = session.query(Lineup)\
			.filter((Lineup.team_id == team_id) & (Lineup.match_id == match_id)).first()

		if not lineup:
			print(f"Lineup not found for team {team_id} in match {match_id}")
			return None, None

		players = [lineup.player1_id, lineup.player2_id, lineup.player3_id, lineup.player4_id, lineup.player5_id]

		latest_player_ts = []

  		# Get latest trueskill for lineup
		for player_id in players:
			player_ts = session.query(PlayerMatchTrueSkill)\
				.filter(PlayerMatchTrueSkill.player_id == player_id)\
				.filter(PlayerMatchTrueSkill.date < date)\
				.order_by(PlayerMatchTrueSkill.date.desc()).first()
			
			if player_ts is None:
				player_ts = session.query(PlayerMatchTrueSkill)\
					.filter(PlayerMatchTrueSkill.player_id == player_id)\
					.filter(PlayerMatchTrueSkill.match_id < match_id)\
					.order_by(PlayerMatchTrueSkill.date.desc()).first()


			if player_ts:
				latest_player_ts.append(env.create_rating(mu = player_ts.mu, sigma = player_ts.sigma ))

		# Convert the TrueSkill ratings into the required format
		team = [env.create_rating(mu=player_ts.mu, sigma=player_ts.sigma) for player_ts in latest_player_ts]

		if len(team) > 0:
			mean_mu = sum(player.mu for player in team) / len(team)
			mean_sigma = sum(player.sigma for player in team) / len(team)

			return team, mean_mu, mean_sigma
		else:
			return None, None, None

	def calculate_win_probability(team1, team2):
		delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
		sum_sigma = sum(r.sigma ** 2 for r in team1) + sum(r.sigma ** 2 for r in team2)
		player_count = len(team1) + len(team2)
		denominator = player_count * (env.beta ** 2) + sum_sigma
		ts = trueskill.global_env()
		return ts.cdf(delta_mu / (denominator ** 0.5))

	team1, t1_mu, t1_sigma = get_team_trueskill(team1_id, date)
	team2, t2_mu, t2_sigma = get_team_trueskill(team2_id, date)
	ts_win_prob = calculate_win_probability(team1, team2)

	return t1_mu, t1_sigma, t2_mu, t2_sigma, ts_win_prob

def get_elo(row):
	team1_id, team2_id, match_id, date = row['team1_id'], row['team2_id'], row['match_id'], row['datetime']
	date = date.split(" ")[0]
	
	# Get lineups for given match
	lineup_A, lineup_B = session.query(Lineup).filter(match_id == Lineup.match_id).all()

	# Ensure lineups are correct order
	if lineup_A.team_id != team1_id:
		print("Swapping lineups")
		temp_lineup = lineup_A
		lineup_A = lineup_B
		lineup_B = temp_lineup

	def get_player_elos(lineup):
		# Extract player IDs from the lineup
		players = [lineup.player1_id, lineup.player2_id, lineup.player3_id, lineup.player4_id, lineup.player5_id]
		player_elos = []

		for player in players:
			player_elo = session.query(PlayerElo)\
				.filter(PlayerElo.player_id == player)\
				.filter(PlayerElo.date < date)\
				.order_by(PlayerElo.date.desc()).first()

			if player_elo:
				player_elos.append({'player_id': player_elo.player_id, 'elo': player_elo.elo, 'matches_played': player_elo.matches_played, 'date' : player_elo.date})

		return player_elos

	# Get player ELO dicts for both teams: [ {'player_id': 1950, 'elo': 1000.0, 'maps_played': 0}, ... ]
	elo_list_A = get_player_elos(lineup_A)
	elo_list_B = get_player_elos(lineup_B)

	team1_elo = calculate_team_elo(elo_list_A)
	team2_elo = calculate_team_elo(elo_list_B)

	# Calculate expected result
	probA, probB = calculate_expected_outcome(team1_elo, team2_elo)

	return team1_elo, team2_elo, probA

def get_lineup_xp(row):
	team1_id, team2_id = row['team1_id'], row['team2_id']
	match_id = row['match_id']
	lineup_age_object_1 = session.query(LineupAge).filter(LineupAge.match_id == match_id).filter(LineupAge.team_id == team1_id).first()
	lineup_age_object_2 = session.query(LineupAge).filter(LineupAge.match_id == match_id).filter(LineupAge.team_id == team2_id).first()

	return lineup_age_object_1.age_days, lineup_age_object_1.matches_together, lineup_age_object_2.age_days, lineup_age_object_2.matches_together

# Now let's get features from just their match history
# Matches played, match-winrate, winstreak, days since last match

def get_recent_match_stats(row, time_period_days, team_id):
	match_id = row['match_id']
	date = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
	start_date = date - timedelta(days=time_period_days)

	# Get all matches
	matches = session.query(Match)\
		.filter(Match.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Match.datetime < func.strftime('%Y-%m-%d %H:%M', date))\
		.filter((Match.team1_id == team_id) | (Match.team2_id == team_id))\
		.order_by(Match.datetime.desc())\
		.all()
	
	matches_played = len(matches)
	
	for match in matches:
		if match.winner == team_id:
			match_wr += 1
	match_wr = match_wr/matches_played

	def get_last_match_winners(date, team_id, match_id):
		match_winners = session.query(Match.winner, Match.datetime)\
			.filter(Match.datetime < func.strftime('%Y-%m-%d %H:%M', date),
			Match.id < match_id,
			(Match.team1_id == team_id) | (Match.team2_id == team_id))\
			.order_by(Match.datetime.desc())
		return match_winners.limit(50).all()
	history = get_last_match_winners(date, team_id, match_id)

	def calculate_days(start_date, end_date):
			start = datetime.strptime(start_date.split(' ')[0], '%Y-%m-%d')
			end = datetime.strptime(end_date.split(' ')[0], '%Y-%m-%d')
			difference = (end - start).days

			return difference

	for match in history:
		days_since_last_match = calculate_days(match.datetime, date)
		if match.winner == team_id:
			winstreak += 1
		else:
			break
	
	return matches_played, match_wr, winstreak, days_since_last_match

# Finally let's calculate features from their map history and player performances

def get_map_features(row):
	match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
	start_date = match_datetime - timedelta(days=90)
	team1_id = row['team1_id']
	team2_id = row['team2_id']

	# Get all match-map items
	t1_mm_objects = session.query(Match, Map)\
		.join(Map, Match.id == Map.match_id)\
		.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
		.filter((Map.t1_id == team1_id) | (Map.t2_id == team1_id))
	t2_mm_objects = session.query(Match, Map)\
		.join(Map, Match.id == Map.match_id)\
		.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
		.filter((Map.t1_id == team2_id) | (Map.t2_id == team2_id))
	
	return t1_mm_objects, t2_mm_objects

def main():
	LOOKBACK_DAYS = 90

	# team1_rank, team2_rank, rank_diff, lan, elim
	df = generate_match_dataframe(start_date="2023-10-01", require_hltv_rank=True, min_format=3)#, n_matches=50)
	print(df.shape)

	# trueskill features
	start_time = datetime.now()
	df[['t1_mu', 't1_sigma', 't2_mu', 't2_sigma', 'ts_win_prob']] =  df.apply(lambda row: pd.Series(get_trueskill(row)), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"TrueSkill features: {elapsed_time:.2f} seconds")
	
	# elo features
	start_time = datetime.now()
	df[['t1_elo', 't2_elo', 'elo_win_prob']] =  df.apply(lambda row: pd.Series(get_elo(row)), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Elo features: {elapsed_time:.2f} seconds")

	# lineup history
	start_time = datetime.now()
	df[['t1_age', 't1_xp', 't2_age', 't2_xp']] = df.apply(lambda row: pd.Series(get_lineup_xp(row)), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Lineup XP features: {elapsed_time:.2f} seconds")
	start_time = datetime.now()

	# matches played, win-rate, win-streak, days since last match
	start_time = datetime.now()
	df[['t1_mp, t1_wr, t1_ws, t1_rust']] = df.apply(lambda row: pd.Series(get_recent_match_stats(row, LOOKBACK_DAYS, row['team1_id'])), axis=1)
	df[['t2_mp, t2_wr, t2_ws, t2_rust']] = df.apply(lambda row: pd.Series(get_recent_match_stats(row, LOOKBACK_DAYS, row['team2_id'])), axis=1)
	end_time = datetime.now()
	elapsed_time = (end_time - start_time).total_seconds()
	print(f"Match history features: {elapsed_time:.2f} seconds")
	start_time = datetime.now()


	# Output
	print(df.shape)
	print(df.head(30))

	ml_df = df[
				[	'format', 'team1_rank', 'team2_rank', 'rank_diff', 'lan', 'elim', 
					't1_mp, t1_wr, t1_ws, t1_rust',
					't2_mp, t2_wr, t2_ws, t2_rust',
					# 'h2h_maps', 'h2h_wr_t1', 'h2h_wr_t2',
					't1_mu', 't1_sigma', 't2_mu', 't2_sigma', 'ts_win_prob', 
					't1_elo', 't2_elo', 'elo_win_prob',
					't1_age', 't1_xp', 't2_age', 't2_xp', 
					'win'
				]
			]
	# ml_df.columns = [
	# 				'rank', 'opp_rank', 'rank_diff', 
	# 				'lan', 'elim', 
	# 				'matches_played', 'match_winrate', 'opp_matches_played', 'opp_winrate', 
	# 				't1_ws', 't2_ws', 'cooldown', 'opp_cooldown',
	# 				'h2h_maps', 'h2h_wr', 'h2h_opp_wr',
	# 				'ts_mu', 'ts_sigma', 'opp_ts_mu', 'opp_ts_sigma', 'ts_win_prob', 
	# 				'age', 'lineup_xp', 'opp_age', 'opp_xp', 
	# 				'win'
	# 				]
	print(ml_df.head(30))

	df.to_csv("df_full.csv")
	ml_df.to_csv("df_ml.csv")

if __name__ == "__main__":
	main()