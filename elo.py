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
from sqlalchemy import func, and_

def insert_player_elos(player_elo_list):
	try:
		session.add_all(player_elo_list)
		session.commit()
		print("Records inserted successfully.")
	except Exception as e:
		session.rollback()
		print(f"Error inserting records: {e}")

def get_player_elos(lineup):
	# Extract player IDs from the lineup
	players = [lineup.player1_id, lineup.player2_id, lineup.player3_id, lineup.player4_id, lineup.player5_id]

	latest_map_ids = (
		session.query(PlayerElo.player_id, func.max(PlayerElo.map_id).label('max_map_id'))
		.filter(PlayerElo.player_id.in_(players))
		.group_by(PlayerElo.player_id)
		.subquery()
	)

	# Query to get player ELOs for the latest map_id
	elo_query = (
		session.query(PlayerElo.player_id, PlayerElo.elo, PlayerElo.maps_played, PlayerElo.date)
		.join(latest_map_ids, and_(PlayerElo.player_id == latest_map_ids.c.player_id, PlayerElo.map_id == latest_map_ids.c.max_map_id))
	)

	# Fetch all results in one query
	results = elo_query.all()

	# Create a list of dictionaries for each player
	player_elos = [
		{'player_id': result.player_id, 'elo': result.elo, 'maps_played': result.maps_played, 'date' : result.date}
		for result in results
	]

	return player_elos

def calculate_expected_outcome(rating1, rating2):
	expected_outcome1 = 1 / (1 + 10**((rating2 - rating1) / 400))
	expected_outcome2 = 1 / (1 + 10**((rating1 - rating2) / 400))
	return expected_outcome1, expected_outcome2

def update_ratings(team_ratings1, team_ratings2, expected_outcome_A, expected_outcome_B, k_factor, winner):
	updated_ratings1 = [
		player['elo'] + k_factor * (winner - expected_outcome_A)
		for player in team_ratings1
	]
	updated_ratings2 = [
		player['elo'] + k_factor * (1-winner - expected_outcome_B)
		for player in team_ratings2
	]
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
		new_player_elos.append(PlayerElo(player_id=player_id, map_id=0, date=datetime.strptime(lineup.date, "%Y-%m-%d %H:%M"), elo=1000.0, maps_played=0))
	insert_player_elos(new_player_elos)

def calculate_team_elo(elo_list):
	total_elo = 0
	for player in elo_list:
		total_elo += player['elo']
	team_elo = total_elo / 5
	return team_elo

def update_elos(map_):
	# Get lineups for given match
	lineup_A, lineup_B = session.query(Lineup).filter(map_.Match.id == Lineup.match_id).all()

	# Initiate any new players to the database
	add_new_players(lineup_A)
	add_new_players(lineup_B)

	# Get player ELO dicts for both teams: [ {'player_id': 1950, 'elo': 1000.0, 'maps_played': 0}, ... ]
	elo_list_A = get_player_elos(lineup_A)
	elo_list_B = get_player_elos(lineup_B)

	print("Player ELOs before update: ")
	print([f"{player['player_id']}:{player['elo']}" for player in elo_list_A])
	print([f"{player['player_id']}:{player['elo']}" for player in elo_list_B])

	# Calculate expected result
	probA, probB = calculate_expected_outcome(calculate_team_elo(elo_list_A), calculate_team_elo(elo_list_B))

	# Get actual result
	map_outcome = 1 if map_.Map.t1_id == map_.Map.winner_id else 0

	# Compute new ELOs
	new_elo_list_A, new_elo_list_B = update_ratings(elo_list_A, elo_list_B, probA, probB, 10, map_outcome)

	# Insert new ELOs into database
	elo_objects = []

	for i, player in enumerate(elo_list_A):
		elo_objects.append(PlayerElo(map_id = map_.Map.id, 
							   player_id = player['player_id'], 
							   date = datetime.strptime(map_.Map.datetime, "%Y-%m-%d %H:%M"), 
							   elo = new_elo_list_A[i], 
							   maps_played = player['maps_played'] + 1))
	for i, player in enumerate(elo_list_B):
		elo_objects.append(PlayerElo(map_id = map_.Map.id, 
							   player_id = player['player_id'], 
							   date = datetime.strptime(map_.Map.datetime, "%Y-%m-%d %H:%M"), 
							   elo = new_elo_list_B[i], 
							   maps_played = player['maps_played'] + 1))
	
	insert_player_elos(elo_objects)


	elo_list_A = get_player_elos(lineup_A)
	elo_list_B = get_player_elos(lineup_B)
	print("Player ELOs after update: ")
	print([f"{player['player_id']}:{player['elo']}" for player in elo_list_A], end = " and ")
	print([f"{player['player_id']}:{player['elo']}" for player in elo_list_B])
	print("\n")


if __name__ == "__main__":
	# Get some maps
	query = session.query(Map, Match).join(Match, Map.match_id == Match.id, isouter=True)
	map_objects = query.limit(1000).all()

	for map_ in map_objects:
		update_elos(map_)