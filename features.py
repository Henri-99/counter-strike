from database.setup import session
from database.models import Match, Map
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import or_, and_, func

# Added

# HLTV rank
# opponent's HLTV rank
# number of times times the map has been played*
# number of times times the opponent has played the map*
# win-rate on this map*
# opponent's win-rate on this map*
# number of maps played*
# number of maps the opponent has played*

# historical map win-rate against this opponent*
# historical match win-rate against this opponent*

# days since last played a match
# days since opponent last played a match
# days since the map was last played
# days since opponent last played the map
# days since last roster change
# days since opponent's last roster change
# whether the map is played on LAN or online
# the tournament prize pool in USD
# whether the match is an elimination match or not
# last match-up with same opponent won or lost
# days since last match-up with same opponent
# match win-rate*
# opponent match win-rate*
# current map win-streak*
# opponent map win-streak*
# match format (bo1/bo3/bo5)


# Advanced

# average player ELO
# opponent's average player ELO
# average player TrueSkill
# average opponent TrueSkill

# was the map picked or not
# was the map picked by the opponent

# first map won/loss/unplayed
# second map won/loss/unplayed
# third map won/loss/unplayed
# fourth map won/loss/unplayed

def generate_match_map_dataframe():

    # Joining Match and Map tables with Match on the left side
    query = session.query(Match, Map)\
        .join(Map, Match.id == Map.match_id)\
        .filter(Match.datetime > "2023-01-01")\
        .filter(and_(Match.team1_rank.isnot(None), Match.team2_rank.isnot(None)))\
        .filter(and_(Match.team1_rank < 30, Match.team2_rank < 30))\
        # .filter(or_(Match.team1 == "astralis", Match.team2 == "astralis"))\

    result = query.all()
    print(f"{len(result)} records queried")

    # Creating a DataFrame from the result
    columns = [
        "match_id", "map_id", "datetime", "team1_id", "team2_id", "team1", "team2", "team1_rank", "team2_rank",
        "map_name", "t1_score", "t2_score", "win"
    ]

    data = [
        (
            match.id,
            map_.id,
            map_.datetime,
            map_.t1_id,
            map_.t2_id,
            match.team1,
            match.team2,
            match.team1_rank,
            match.team2_rank,
            map_.map_name,
            map_.t1_score,
            map_.t2_score,
            0 if map_.t2_id == map_.winner_id else 1
        )
        for match, map_ in result
    ]

    df = pd.DataFrame(data, columns=columns)

    return df

# Function to calculate win-rate for a team in the previous 3 months
def calculate_stats(row, team_id_column):   
    # Extract team ID and match datetime
    team_id = row[team_id_column]
    match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")

    # Define the time range for the previous 3 months
    start_date = match_datetime - timedelta(days=90)

    # Query maps for the specified team within the time range
    team_maps_query = session.query(Map)\
        .filter(
            Map.datetime >= start_date,
            Map.datetime < match_datetime,
            (Map.t1_id == team_id) | (Map.t2_id == team_id)
        )
    
    # Calculate the win-rate
    all_maps_played_count = team_maps_query.count()
    if all_maps_played_count == 0:
        all_maps_win_rate = 0.0
    else:
        win_count = team_maps_query.filter(
            ((Map.t1_id == team_id) & (Map.winner_id == Map.t1_id)) |
            ((Map.t2_id == team_id) & (Map.winner_id == Map.t2_id))
        ).count()
        all_maps_win_rate = win_count / all_maps_played_count


    this_map_only_query = team_maps_query.filter(Map.map_name == row['map_name'])

    # Calculate the win-rate
    map_played_count = this_map_only_query.count()
    if map_played_count == 0:
        win_rate = 0.0
    else:
        win_count = this_map_only_query.filter(
            ((Map.t1_id == team_id) & (Map.winner_id == Map.t1_id)) |
            ((Map.t2_id == team_id) & (Map.winner_id == Map.t2_id))
        ).count()
        win_rate = win_count / map_played_count

    return win_rate, map_played_count, all_maps_win_rate, all_maps_played_count

df = generate_match_map_dataframe()
df[['map_winrate_t1', 'map_played_count_t1', 'all_maps_win_rate_t1', 'all_maps_played_count_t1']] = df.apply(lambda row: pd.Series(calculate_stats(row, 'team1_id')), axis=1)
df[['map_winrate_t2', 'map_played_count_t2', 'all_maps_win_rate_t2', 'all_maps_played_count_t2']] = df.apply(lambda row: pd.Series(calculate_stats(row, 'team2_id')), axis=1)

ml_df = df[['team1_rank', 'team2_rank', 'map_played_count_t1', 'map_played_count_t2', 'map_winrate_t1', 'map_winrate_t2', 'all_maps_win_rate_t1', 'all_maps_played_count_t1', 'all_maps_win_rate_t2', 'all_maps_played_count_t2', 'win']]
ml_df.columns = ['rank', 'opp_rank', 'map_playcount', 'opp_map_playcount', 'map_winrate', 'opp_map_winrate', 'all_maps_winrate', 'all_maps_playcount', 'opp_all_maps_winrate', 'opp_all_maps_playcount', 'win']
print(ml_df.head(20))

ml_df.to_csv("temp_df.csv")