# Get list of map data
# Get all players current ELO ratings for that match up using player id + largest mapID
#   if new player, set ELO score to 1000
#   for first 10 matches, set K factor to (10*(10-n))
# Adjust each players ELO after map result with K=10, use average team ELO for expected outcome
# Store player id, new ELO, mapID, and datetime in DB

from database.setup import session
from database.models import Match, Map, Lineup, PlayerElo
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

def insert_player_elos(player_elo_list):
	try:
		session.add_all(player_elo_list)
		session.commit()
	except Exception as e:
		session.rollback()
		print(f"Error inserting records: {e}")

def get_player_elos(lineup):
	# Extract player IDs from the lineup
	players = [lineup.player1_id, lineup.player2_id, lineup.player3_id, lineup.player4_id, lineup.player5_id]

	latest_match_ids = (
		session.query(PlayerElo.player_id, func.max(PlayerElo.match_id).label('max_map_id'))
		.filter(PlayerElo.player_id.in_(players))
		.group_by(PlayerElo.player_id)
		.subquery()
	)

	# Query to get player ELOs for the latest map_id
	elo_query = (
		session.query(PlayerElo.player_id, PlayerElo.elo, PlayerElo.matches_played, PlayerElo.date)
		.join(latest_match_ids, and_(PlayerElo.player_id == latest_match_ids.c.player_id, PlayerElo.match_id == latest_match_ids.c.max_map_id))
	)

	# Fetch all results in one query
	results = elo_query.all()

	# Create a list of dictionaries for each player
	player_elos = [
		{'player_id': result.player_id, 'elo': result.elo, 'matches_played': result.matches_played, 'date' : result.date}
		for result in results
	]

	return player_elos

def calculate_expected_outcome(rating1, rating2):
	expected_outcome1 = 1 / (1 + 10**((rating2 - rating1) / 400))
	expected_outcome2 = 1 / (1 + 10**((rating1 - rating2) / 400))
	return expected_outcome1, expected_outcome2

def update_ratings(team_ratings1, team_ratings2, expected_outcome_A, expected_outcome_B, k_factor, winner):
	updated_ratings1, updated_ratings2 = [], []
	
	for player in team_ratings1:
		if player['matches_played'] <= 10:
			dynamic_k_factor = 50 - 4 * player['matches_played']
			new_rating = player['elo'] + dynamic_k_factor * (winner - expected_outcome_A)
		else:
			new_rating = player['elo'] + k_factor * (winner - expected_outcome_A)
		updated_ratings1.append(new_rating)

	for player in team_ratings2:
		if player['matches_played'] <= 10:
			dynamic_k_factor = 50 - 4 * player['matches_played']
			new_rating = player['elo'] + dynamic_k_factor * (winner - expected_outcome_B)
		else:
			new_rating = player['elo'] + k_factor * (winner - expected_outcome_B)
		updated_ratings2.append(new_rating)

	return updated_ratings1, updated_ratings2

def add_new_players(lineup):
	player_id_list = [lineup.player1_id, lineup.player2_id, lineup.player3_id, lineup.player4_id, lineup.player5_id]
	existing_players = session.query(PlayerElo.player_id).filter(PlayerElo.player_id.in_(player_id_list)).all()
	existing_players = {player[0] for player in existing_players}

	# Identify new players
	new_players = [player_id for player_id in player_id_list if player_id not in existing_players]

	new_player_elos = []
	# Insert records for new players
	for player_id in new_players:
		new_player_elos.append(PlayerElo(player_id=player_id, match_id=0, date=datetime.strptime(lineup.date, "%Y-%m-%d %H:%M"), elo=1000.0, matches_played=0))
	insert_player_elos(new_player_elos)

def calculate_team_elo(elo_list):
	total_elo = 0
	for player in elo_list:
		total_elo += player['elo']
	team_elo = total_elo / 5
	return team_elo

def update_elos(match):
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

	# Get player ELO dicts for both teams: [ {'player_id': 1950, 'elo': 1000.0, 'maps_played': 0}, ... ]
	elo_list_A = get_player_elos(lineup_A)
	elo_list_B = get_player_elos(lineup_B)

	# Calculate expected result
	probA, probB = calculate_expected_outcome(calculate_team_elo(elo_list_A), calculate_team_elo(elo_list_B))

	# Get actual result
	map_outcome = 1 if match.team1_id == match.winner else 0

	# Compute new ELOs
	new_elo_list_A, new_elo_list_B = update_ratings(elo_list_A, elo_list_B, probA, probB, 10, map_outcome)

	# Insert new ELOs into database
	elo_objects = []

	for i, player in enumerate(elo_list_A):
		if match.id == 0:
			print()
		elo_objects.append(PlayerElo(match_id = match.id, 
							   player_id = player['player_id'], 
							   date = datetime.strptime(match.datetime, "%Y-%m-%d %H:%M"), 
							   elo = new_elo_list_A[i], 
							   matches_played = player['matches_played'] + 1))
	for i, player in enumerate(elo_list_B):
		elo_objects.append(PlayerElo(match_id = match.id, 
							   player_id = player['player_id'], 
							   date = datetime.strptime(match.datetime, "%Y-%m-%d %H:%M"), 
							   elo = new_elo_list_B[i], 
							   matches_played = player['matches_played'] + 1))
	
	insert_player_elos(elo_objects)


	elo_list_A = get_player_elos(lineup_A)
	elo_list_B = get_player_elos(lineup_B)


if __name__ == "__main__":
	# Get some maps
	subquery = session.query(PlayerElo.match_id).distinct().subquery()
	query = (
		session.query(Match)
		.filter(Match.id.notin_(subquery))
		.order_by(asc(Match.datetime))
		# .limit(1000)
	)
	match_objects = query.all()

	for i, match in enumerate(match_objects):
		if i%10 == 0:
			print(f"{str(i).rjust(5)}/{len(match_objects)}: {(float(i)*100/len(match_objects)):2f}%")
		update_elos(match)