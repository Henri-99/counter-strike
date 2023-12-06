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
        session.query(PlayerElo.player_id, PlayerElo.elo, PlayerElo.maps_played)
        .join(latest_map_ids, and_(PlayerElo.player_id == latest_map_ids.c.player_id, PlayerElo.map_id == latest_map_ids.c.max_map_id))
    )

    # Fetch all results in one query
    results = elo_query.all()

    # Fetch all results in one query
    results = elo_query.all()

    # Create a list of dictionaries for each player
    player_elos = [
        {'player_id': result.player_id, 'elo': result.elo, 'maps_played': result.maps_played}
        for result in results
    ]

    return player_elos

def calculate_expected_outcome(rating1, rating2):
    expected_outcome1 = 1 / (1 + 10**((rating2 - rating1) / 400))
    expected_outcome2 = 1 / (1 + 10**((rating1 - rating2) / 400))
    return expected_outcome1, expected_outcome2

def update_ratings(rating1, rating2, k_factor, outcome1, outcome2):
    updated_rating1 = rating1 + k_factor * (outcome1 - calculate_expected_outcome(rating1, rating2)[0])
    updated_rating2 = rating2 + k_factor * (outcome2 - calculate_expected_outcome(rating1, rating2)[1])
    return updated_rating1, updated_rating2


query = session.query(Map, Match).join(Match, Map.match_id == Match.id, isouter=True)

result = query.limit(10).all()

for record in result:
    print(record)
    lineup_query = session.query(Lineup).filter(record.Match.id == Lineup.match_id).all()
    for lineup in lineup_query:
        print(get_player_elos(lineup))

    break