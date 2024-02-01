# Transforms a dataframe of match records into a dataframe of ML features
import numpy as np
from database.setup import session
from database.models import Match, Map, Lineup, PlayerMatchTrueSkill, LineupAge, PlayerElo, PlayerStats
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import or_, and_, func
import scipy.stats as stats
from elo import calculate_expected_outcome, calculate_team_elo

# First let's get the matches

def generate_match_dataframe( start_date = "2019-07-01",
							  rank_threshold = None, 
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
		query = query.filter(or_(Match.team1_rank <= rank_threshold, Match.team2_rank <= rank_threshold))
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

def impute_hltv_ranks(df):
	intercept = 262.34
	coefficients = [-5.64, -0.081]

	imputed_ranks_count = 0

	for index, row in df.iterrows():
		if pd.isna(row['team1_rank']):
			df.at[index, 'team1_rank'] = int(intercept + coefficients[0] * row['t1_mu'] + coefficients[1] * row['t1_elo'])
			imputed_ranks_count += 1
		if pd.isna(row['team2_rank']):
			df.at[index, 'team2_rank'] = int(intercept + coefficients[0] * row['t2_mu'] + coefficients[1] * row['t2_elo'])
			imputed_ranks_count += 1
		
		df.at[index, 'rank_diff'] = df.at[index, 'team1_rank'] - df.at[index, 'team2_rank']

	print(f"{imputed_ranks_count} HLTV ranks imputed")
	return df

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
	matches_won = 0

	for match in matches:
		if match.winner == team_id:
			matches_won += 1
	match_wr = matches_won/matches_played if matches_played > 0 else 0

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
			return (end_date - start).days

	winstreak = 0
	days_since_last_match = None
	for match in history:
		if days_since_last_match is None:
			days_since_last_match = calculate_days(match.datetime, date)
		if match.winner == team_id:
			winstreak += 1
		else:
			break
	
	if days_since_last_match is None:
		days_since_last_match = 100
	
	return matches_played, match_wr, winstreak, days_since_last_match

# Finally let's calculate features from their map history and player performances

def get_head_to_head_stats(row, time_period_days):
	match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
	start_date = match_datetime - timedelta(days=time_period_days)
	match_id = row['match_id']

	team1_id = row['team1_id']
	team2_id = row['team2_id']

	team_maps_query = session.query(Map)\
		.filter(
			Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date),
			Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime),
			Map.match_id != match_id,
			((Map.t1_id == team1_id) & (Map.t2_id == team2_id)) | ((Map.t1_id == team2_id) & (Map.t2_id == team1_id))
		)

	h2h_maps_played = team_maps_query.count()

	# Initialize variables
	h2h_wr, avg_round_wp = 0, 0
	avg_round_wp_list = []

	if h2h_maps_played > 0:
		# Win rate calculation
		h2h_wr = team_maps_query.filter(Map.winner_id == team1_id).count() / h2h_maps_played

		# Round win-percentage calculation for each map
		for map_ in team_maps_query.all():
			total_rounds = map_.t1_score + map_.t2_score
			if total_rounds > 0:
				if map_.t1_id == team1_id:
					avg_round_wp_list.append(map_.t1_score / total_rounds)
				else:
					avg_round_wp_list.append(map_.t2_score / total_rounds)

		# Average round win-percentage calculation
		avg_round_wp = sum(avg_round_wp_list) / len(avg_round_wp_list) if avg_round_wp_list else 0

	else:
		h2h_maps_played, avg_round_wp, = 0, 0
	
	return h2h_maps_played, h2h_wr, avg_round_wp

def get_map_features(row, team_id, time_period_days):
	def calculate_pistol_round_win_rate_for_map(round_history_string, cs2):
		round_wins = 0
		if round_history_string[0] != '_':
			round_wins += 1
		if round_history_string[12 if cs2 else 15] != '_':
			round_wins += 1
		return round_wins/2
	
	def get_avg_player_stats(map_id, team_id):
		player_stats = session.query(PlayerStats).filter(PlayerStats.map_id == map_id, PlayerStats.team_id == team_id).all()
		ratings, adr, kast = 0, 0, 0
		n_players = len(player_stats)
		for player in player_stats:
			ratings += player.rating
			adr += player.adr
			kast += player.kast
		return ratings/n_players, adr/n_players, kast/n_players
	
	match_id = row['match_id']
	match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
	start_date = match_datetime - timedelta(days=time_period_days)

	# Get all match-map items
	mm_objects = session.query(Match, Map)\
		.join(Map, Match.id == Map.match_id)\
		.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
		.filter(Match.id != match_id)\
		.filter((Map.t1_id == team_id) | (Map.t2_id == team_id))\
		.all()
	
	hltv_rating, fk_pr, cl_pr, pl_rating, pl_adr, plr_kast, pistol_wr_list = [], [], [], [], [], [], []

	for mm in mm_objects:
		total_rounds = mm.Map.t1_score + mm.Map.t2_score
		avg_player_rating, avg_player_adr, avg_player_kast = get_avg_player_stats(mm.Map.id, team_id)
		if mm.Map.t1_id == team_id:
			team_rating = mm.Map.t1_rating
			first_kills = mm.Map.t1_first_kills
			clutches = mm.Map.t1_clutches
			pistol_wr = calculate_pistol_round_win_rate_for_map(mm.Map.t1_round_history, mm.Match.cs2)
		else:
			team_rating = mm.Map.t2_rating
			first_kills = mm.Map.t2_first_kills
			clutches = mm.Map.t2_clutches
			pistol_wr = calculate_pistol_round_win_rate_for_map(mm.Map.t2_round_history, mm.Match.cs2)
		
		hltv_rating.append(team_rating)
		fk_pr.append(first_kills)
		cl_pr.append(clutches)
		pl_rating.append(avg_player_rating)
		pl_adr.append(avg_player_adr)
		plr_kast.append(avg_player_kast)
		pistol_wr_list.append(pistol_wr)

	if len(hltv_rating) > 0:
		avg_hltv_rating, sd_hltv_rating = np.mean(hltv_rating), np.std(hltv_rating)
		avg_fk_pr, sd_fk_pr = np.mean(fk_pr), np.std(fk_pr)
		avg_cl_pr, sd_cl_pr = np.mean(cl_pr), np.std(cl_pr)
		avg_pl_rating, sd_pl_rating = np.mean(pl_rating), np.std(pl_rating)
		avg_pl_adr, sd_pl_adr = np.mean(pl_adr), np.std(pl_adr)
		avg_plr_kast, sd_plr_kast = np.mean(plr_kast), np.std(plr_kast)
		avg_pistol_wr, sd_pistol_wr = np.mean(pistol_wr_list), np.std(pistol_wr_list)
		return avg_hltv_rating, sd_hltv_rating, avg_fk_pr, sd_fk_pr, avg_cl_pr, sd_cl_pr, avg_pl_rating, sd_pl_rating, avg_pl_adr, sd_pl_adr, avg_plr_kast, sd_plr_kast, avg_pistol_wr, sd_pistol_wr
	else:
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def get_mapwise_features(row, time_period_days):
	match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
	start_date = match_datetime - timedelta(days=time_period_days)
	match_id, team1_id, team2_id  = row['match_id'], row['team1_id'], row['team2_id']
	
	t1_mm_map_names = session.query(Map.map_name)\
	.join(Match, Match.id == Map.match_id)\
	.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
	.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
	.filter(Match.id != match_id)\
	.filter((Map.t1_id == team1_id) | (Map.t2_id == team1_id))\
	.all()

	t2_mm_map_names = session.query(Map.map_name)\
	.join(Match, Match.id == Map.match_id)\
	.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
	.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
	.filter(Match.id != match_id)\
	.filter((Map.t1_id == team2_id) | (Map.t2_id == team2_id))\
	.all()

	t1_map_names = {result[0] for result in t1_mm_map_names}
	t2_map_names = {result[0] for result in t2_mm_map_names}
	map_names = list(t1_map_names.union(t2_map_names))

	if 'Cache' in map_names and 'Vertigo' in map_names:
		map_names.remove('Cache')
	if 'Train' in map_names and 'Ancient' in map_names:
		map_names.remove('Train')
	if 'Dust2' in map_names and 'Anubis' in map_names:
		map_names.remove('Dust2')

	
	# Get all match-map items
	t1_mm_objects = session.query(Match, Map)\
		.join(Map, Match.id == Map.match_id)\
		.filter(Map.datetime >= func.strftime('%Y-%m-%d %H:%M', start_date))\
		.filter(Map.datetime < func.strftime('%Y-%m-%d %H:%M', match_datetime))\
		.filter(Match.id != match_id)\
		.filter((Map.t1_id == team1_id) | (Map.t2_id == team1_id))\
		.all()

	


# Generate difference features

def generate_diff_features(df):
	df['rank_diff'] = df['team2_rank'] - df['team1_rank']
	df['age_diff'] = df['t1_age'] - df['t2_age']
	df['xp_diff'] = df['t1_xp'] - df['t2_xp']
	df['mp_diff'] = df['t1_mp'] - df['t2_mp']
	df['wr_diff'] = df['t1_wr'] - df['t2_wr']
	df['ws_diff'] = df['t1_ws'] - df['t2_ws']
	df['rust_diff'] = df['t1_rust'] - df['t2_rust']
	df['avg_hltv_rating_diff'] = df['t1_avg_hltv_rating'] - df['t2_avg_hltv_rating']
	df['sd_hltv_rating_diff'] = df['t1_sd_hltv_rating'] - df['t2_sd_hltv_rating']
	df['avg_fk_pr_diff'] = df['t1_avg_fk_pr'] - df['t2_avg_fk_pr']
	df['sd_fk_pr_diff'] = df['t1_sd_fk_pr'] - df['t2_sd_fk_pr']
	df['avg_cl_pr_diff'] = df['t1_avg_cl_pr'] - df['t2_avg_cl_pr']
	df['sd_cl_pr_diff'] = df['t1_sd_cl_pr'] - df['t2_sd_cl_pr']
	df['avg_pl_rating_diff'] = df['t1_avg_pl_rating'] - df['t2_avg_pl_rating']
	df['sd_pl_rating_diff'] = df['t1_sd_pl_rating'] - df['t2_sd_pl_rating']
	df['avg_pl_adr_diff'] = df['t1_avg_pl_adr'] - df['t2_avg_pl_adr']
	df['sd_pl_adr_diff'] = df['t1_sd_pl_adr'] - df['t2_sd_pl_adr']
	df['avg_plr_kast_diff'] = df['t1_avg_plr_kast'] - df['t2_avg_plr_kast']
	df['sd_plr_kast_diff'] = df['t1_sd_plr_kast'] - df['t2_sd_plr_kast']
	df['avg_pistol_wr_diff'] = df['t1_avg_pistol_wr'] - df['t2_avg_pistol_wr']
	df['sd_pistol_wr_diff'] = df['t1_sd_pistol_wr'] - df['t2_sd_pistol_wr']
	return df


def main():
	LOOKBACK_DAYS = 90

	# team1_rank, team2_rank, rank_diff, lan, elim
	df = generate_match_dataframe(start_date="2019-06-07")#, n_matches=10)
	print(df.shape)
 
 
	# trueskill features
	start_time = datetime.now()
	df[['t1_mu', 't1_sigma', 't2_mu', 't2_sigma', 'ts_win_prob']] =  df.apply(lambda row: pd.Series(get_trueskill(row)), axis=1)
	elapsed_time = (datetime.now() - start_time).total_seconds()
	print(f"{"TrueSkill features".ljust(25)}: {elapsed_time:.2f} seconds")
	
	# elo features
	start_time = datetime.now()
	df[['t1_elo', 't2_elo', 'elo_win_prob']] =  df.apply(lambda row: pd.Series(get_elo(row)), axis=1)
	elapsed_time = (datetime.now() - start_time).total_seconds()
	print(f"{"Elo features".ljust(25)}: {elapsed_time:.2f} seconds")

	# impute missing ranks
	df = impute_hltv_ranks(df)

	# lineup history
	start_time = datetime.now()
	df[['t1_age', 't1_xp', 't2_age', 't2_xp']] = df.apply(lambda row: pd.Series(get_lineup_xp(row)), axis=1)
	elapsed_time = (datetime.now() - start_time).total_seconds()
	print(f"{"Lineup features".ljust(25)}: {elapsed_time:.2f} seconds")

	# matches played, win-rate, win-streak, days since last match
	start_time = datetime.now()
	df[['t1_mp', 't1_wr', 't1_ws', 't1_rust']] = df.apply(lambda row: pd.Series(get_recent_match_stats(row, LOOKBACK_DAYS, row['team1_id'])), axis=1)
	df[['t2_mp', 't2_wr', 't2_ws', 't2_rust']] = df.apply(lambda row: pd.Series(get_recent_match_stats(row, LOOKBACK_DAYS, row['team2_id'])), axis=1)
	elapsed_time = (datetime.now() - start_time).total_seconds()
	print(f"{"Match history features".ljust(25)}: {elapsed_time:.2f} seconds")

	# head-to-head, maps, winrate, round win %
	start_time = datetime.now()
	df[['h2h_maps', 'h2h_wr', 'h2h_rwp']] = df.apply(lambda row: pd.Series(get_head_to_head_stats(row, LOOKBACK_DAYS)), axis=1)
	elapsed_time = (datetime.now() - start_time).total_seconds()
	print(f"{"Head-to-head features".ljust(25)}: {elapsed_time:.2f} seconds")

	start_time = datetime.now()
	# Map-specific features
	df[
		[	't1_avg_hltv_rating', 
	 		't1_sd_hltv_rating', 
	 		't1_avg_fk_pr', 
	 		't1_sd_fk_pr', 
	 		't1_avg_cl_pr', 
	 		't1_sd_cl_pr', 
	 		't1_avg_pl_rating', 
	 		't1_sd_pl_rating', 
	 		't1_avg_pl_adr', 
	 		't1_sd_pl_adr', 
	 		't1_avg_plr_kast', 
	 		't1_sd_plr_kast', 
	 		't1_avg_pistol_wr', 
	 		't1_sd_pistol_wr']] = df.apply(lambda row: pd.Series(get_map_features(row, row['team1_id'], LOOKBACK_DAYS)), axis=1)
	df[
		[	't2_avg_hltv_rating', 
	 		't2_sd_hltv_rating', 
	 		't2_avg_fk_pr', 
	 		't2_sd_fk_pr', 
	 		't2_avg_cl_pr', 
	 		't2_sd_cl_pr', 
	 		't2_avg_pl_rating', 
	 		't2_sd_pl_rating', 
	 		't2_avg_pl_adr', 
	 		't2_sd_pl_adr', 
	 		't2_avg_plr_kast', 
	 		't2_sd_plr_kast', 
	 		't2_avg_pistol_wr', 
	 		't2_sd_pistol_wr']] = df.apply(lambda row: pd.Series(get_map_features(row, row['team2_id'], LOOKBACK_DAYS)), axis=1)
	elapsed_time = (datetime.now() - start_time).total_seconds()
	print(f"{"Map perf features".ljust(25)}: {elapsed_time:.2f} seconds")

	# Output
	print(df.shape)
	print(df.head(30))

	ml_df = df[
				[	'format', 'team1_rank', 'team2_rank', 'rank_diff', 'lan', 'elim', 
					't1_mp', 't1_wr', 't1_ws', 't1_rust',
					't2_mp', 't2_wr', 't2_ws', 't2_rust',
					'h2h_maps', 'h2h_wr', 'h2h_rwp',
					't1_mu', 't1_sigma', 't2_mu', 't2_sigma', 'ts_win_prob', 
					't1_elo', 't2_elo', 'elo_win_prob',
					't1_age', 't1_xp', 't2_age', 't2_xp', 
					't1_avg_hltv_rating', 
			 		't1_sd_hltv_rating', 't1_avg_fk_pr', 't1_sd_fk_pr', 't1_avg_cl_pr', 't1_sd_cl_pr', 't1_avg_pl_rating', 't1_sd_pl_rating', 't1_avg_pl_adr', 't1_sd_pl_adr', 't1_avg_plr_kast', 't1_sd_plr_kast', 't1_avg_pistol_wr', 't1_sd_pistol_wr',
			 		't2_sd_hltv_rating', 't2_avg_fk_pr', 't2_sd_fk_pr', 't2_avg_cl_pr', 't2_sd_cl_pr', 't2_avg_pl_rating', 't2_sd_pl_rating', 't2_avg_pl_adr', 't2_sd_pl_adr', 't2_avg_plr_kast', 't2_sd_plr_kast', 't2_avg_pistol_wr', 't2_sd_pistol_wr',
					'win'
				]
			]

	df.to_csv("csv/df_full.csv")
	ml_df.to_csv("csv/df_ml.csv", na_rep = 'NULL')

if __name__ == "__main__":
	# main()

	df = pd.read_csv('csv/filtered_df.csv')
	df = generate_diff_features(df)
	df.to_csv('csv/diff.csv')