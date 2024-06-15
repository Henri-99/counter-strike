import trueskill
from database.setup import session
from database.models import Match, Lineup, PlayerMatchTrueSkill
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import func, and_, asc
import logging

main_logger = logging.getLogger('main_logger')
main_logger.setLevel(logging.INFO)
main_handler = logging.FileHandler('main.log')
main_handler.setLevel(logging.INFO)
main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
main_logger.addHandler(main_handler)

mu = 1200  # Initial mu (mean skill level)
sigma = 400  # Initial sigma (standard deviation)
beta = 200  # Beta (skill variability)
tau = 20  # Tau (dynamics factor)

# env = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau)
env = trueskill.TrueSkill()

def add_new_players(lineup):
	player_id_list = [lineup.player1_id, lineup.player2_id, lineup.player3_id, lineup.player4_id, lineup.player5_id]
	existing_players = session.query(PlayerMatchTrueSkill.player_id).filter(PlayerMatchTrueSkill.player_id.in_(player_id_list)).all()
	existing_players = {player[0] for player in existing_players}

	# Identify new players
	new_players = [player_id for player_id in player_id_list if player_id not in existing_players]

	default_trueskill = env.create_rating()
	
	new_player_trueskill = []
	# Insert records for new players
	for player_id in new_players:
		new_player_trueskill.append(PlayerMatchTrueSkill(player_id=player_id, match_id=0, date=datetime.strptime(lineup.date, "%Y-%m-%d %H:%M"), mu=default_trueskill.mu, sigma = default_trueskill.sigma, matches_played=0))
	insert_player_trueskill(new_player_trueskill)

def insert_player_trueskill(player_elo_list):
	try:
		session.add_all(player_elo_list)
		session.commit()
	except Exception as e:
		session.rollback()
		print(f"Error inserting records: {e}")

def get_player_trueskill(lineup):
	# Extract player IDs from the lineup
	players = [lineup.player1_id, lineup.player2_id, lineup.player3_id, lineup.player4_id, lineup.player5_id]

	latest_match_ids = (
		session.query(PlayerMatchTrueSkill.player_id, func.max(PlayerMatchTrueSkill.match_id).label('max_match_id'))
		.filter(PlayerMatchTrueSkill.player_id.in_(players))
		.group_by(PlayerMatchTrueSkill.player_id)
		.subquery()
	)

	# Query to get player ELOs for the latest map_id
	ts_query = (
		session.query(
			PlayerMatchTrueSkill.player_id, 
			PlayerMatchTrueSkill.mu, 
			PlayerMatchTrueSkill.sigma, 
			PlayerMatchTrueSkill.matches_played,
			PlayerMatchTrueSkill.date)
		.join(latest_match_ids, and_(PlayerMatchTrueSkill.player_id == latest_match_ids.c.player_id, PlayerMatchTrueSkill.match_id == latest_match_ids.c.max_match_id))
	)

	# Fetch all results in one query
	results = ts_query.all()

	# Create a list of dictionaries for each player
	player_trueskill = [
		{
			'player_id': result.player_id, 
			'mu': result.mu, 
			'sigma': result.sigma, 
			'matches_played': result.matches_played, 
			'date' : result.date}
		for result in results
	]

	return player_trueskill

def update_trueskill(match):
	if match is None:
		main_logger.error(f"{match.id} has no associated Match")
		return

	# Get lineups for given match
	lineup_A, lineup_B = session.query(Lineup).filter(match.id == Lineup.match_id).all()

	# Ensure lineups are correct order
	if lineup_A.team_id != match.team1_id:
		print("Swapping lineups")
		temp_lineup = lineup_A
		lineup_A = lineup_B
		lineup_B = temp_lineup

	# Initiate any new players to the database
	add_new_players(lineup_A)
	add_new_players(lineup_B)

	# Get player TS dicts for both teams: [ {'player_id': 1950, 'mu': 25.0, 'maps_played': 0}, ... ]
	ts_list_A = get_player_trueskill(lineup_A)
	ts_list_B = get_player_trueskill(lineup_B)

	ts_team_A = [env.create_rating(player['mu'], player['sigma']) for player in ts_list_A]
	ts_team_B = [env.create_rating(player['mu'], player['sigma']) for player in ts_list_B]

	outcome = [0,1] if match.team1_id == match.winner else [1,0]

	updated_team1, updated_team2 = env.rate([ts_team_A, ts_team_B], ranks=outcome)

	# Insert new TrueSkills into database
	true_skill_objects = []

	for i, player in enumerate(ts_list_A):
		true_skill_objects.append(PlayerMatchTrueSkill(
							   player_id = player['player_id'], 
							   match_id = match.id,
							   date = datetime.strptime(match.datetime, "%Y-%m-%d %H:%M"), 
							   mu = updated_team1[i].mu, 
							   sigma = updated_team1[i].sigma, 
							   matches_played = player['matches_played'] + 1))
	for i, player in enumerate(ts_list_B):
		true_skill_objects.append(PlayerMatchTrueSkill(
							   player_id = player['player_id'], 
							   match_id = match.id,
							   date = datetime.strptime(match.datetime, "%Y-%m-%d %H:%M"),
							   mu = updated_team2[i].mu, 
							   sigma = updated_team2[i].sigma, 
							   matches_played = player['matches_played'] + 1))
	
	insert_player_trueskill(true_skill_objects)

if __name__ == "__main__":
	# Get some maps
	subquery = session.query(PlayerMatchTrueSkill.match_id).distinct()

	# Determine the total number of records
	total_count = session.query(Match).filter(Match.id.notin_(subquery)).count()

	batch_size = 500
	processed_count = 0
	incomplete = True

	while incomplete:
		query = (
			session.query(Match)
			.filter(Match.id.notin_(subquery))
			.order_by(asc(Match.id))
			.limit(batch_size)
		)
		match_objects = query.all()

		if len(match_objects) < batch_size:
			incomplete = False

		for i, match in enumerate(match_objects):
			update_trueskill(match)
			processed_count += 1
			if processed_count % 10 == 0 or processed_count == total_count:
				print(f"{str(processed_count).rjust(5)}/{total_count}: {(float(processed_count) * 100 / total_count):.2f}%")