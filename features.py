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
# whether the map is played on LAN or online
# historical map win-rate against this opponent*
# match format (bo1/bo3/bo5)

# days since last played a match
# days since opponent last played a match
# days since the map was last played
# days since opponent last played the map
# days since last roster change
# days since opponent's last roster change

# the tournament prize pool in USD
# whether the match is an elimination match or not
# last match-up with same opponent won or lost
# days since last match-up with same opponent
# match win-rate*
# opponent match win-rate*
# current map win-streak*
# opponent map win-streak*


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
        .filter(and_(Match.team1_rank <= 30, Match.team2_rank <= 30))\
        # .filter(or_(Match.team1 == "astralis", Match.team2 == "astralis"))\

    result = query.all()
    print(f"{len(result)} records queried")

    # Creating a DataFrame from the result
    columns = [
        "match_id", "map_id", "datetime", "team1_id", "team2_id", "team1", "team2", "team1_rank", "team2_rank", "rank_diff", "map_name", "lan", "format", "t1_score", "t2_score", "win"
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
            match.team2_rank - match.team1_rank,
            map_.map_name,
            match.lan,
            match.best_of,
            map_.t1_score,
            map_.t2_score,
            0 if map_.t2_id == map_.winner_id else 1
        )
        for match, map_ in result
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


# Function to calculate head-to-head stats in the previous 6 months
def calculate_head_to_head_stats(row):
    match_datetime = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M")
    start_date = match_datetime - timedelta(days=180)

    team1_id = row['team1_id']
    team2_id = row['team2_id']

    team_maps_query = session.query(Map)\
        .filter(
            Map.datetime >= start_date,
            Map.datetime < match_datetime,
            ((Map.t1_id == team1_id) & (Map.t2_id == team2_id)) | ((Map.t1_id == team2_id) & (Map.t2_id == team1_id))
        )
    h2h_maps_played_count = team_maps_query.count()
    if h2h_maps_played_count == 0:
        t1_h2h_win_rate = 0.0
        t2_h2h_win_rate = 0.0
    else:
        t1_h2h_win_rate = team_maps_query.filter(Map.winner_id == team1_id).count() / h2h_maps_played_count
        t2_h2h_win_rate = team_maps_query.filter(Map.winner_id == team2_id).count() / h2h_maps_played_count
    
    return h2h_maps_played_count, t1_h2h_win_rate, t2_h2h_win_rate
    
def generate_new_dataset_csv():
    df = generate_match_map_dataframe()
    df[['map_winrate_t1', 'map_played_count_t1', 'all_maps_win_rate_t1', 'all_maps_played_count_t1']] = df.apply(lambda row: pd.Series(calculate_win_rates(row, 'team1_id')), axis=1)
    df[['map_winrate_t2', 'map_played_count_t2', 'all_maps_win_rate_t2', 'all_maps_played_count_t2']] = df.apply(lambda row: pd.Series(calculate_win_rates(row, 'team2_id')), axis=1)
    df[['h2h_maps', 'h2h_wr_t1', 'h2h_wr_t2']] = df.apply(lambda row: pd.Series(calculate_head_to_head_stats(row)), axis=1)

    ml_df = df[['team1_rank', 'team2_rank', 'rank_diff', 'map_played_count_t1', 'map_played_count_t2', 'map_winrate_t1', 'map_winrate_t2', 'all_maps_win_rate_t1', 'all_maps_played_count_t1', 'all_maps_win_rate_t2', 'all_maps_played_count_t2','h2h_maps', 'h2h_wr_t1', 'h2h_wr_t2', 'lan', 'format', 'win']]
    ml_df.columns = ['rank', 'opp_rank', 'rank_diff', 'map_playcount', 'opp_map_playcount', 'map_winrate', 'opp_map_winrate', 'all_maps_winrate', 'all_maps_playcount', 'opp_all_maps_winrate', 'opp_all_maps_playcount', 'h2h_maps', 'h2h_wr', 'opp_h2h_wr', 'lan', 'format','win']
    print(ml_df.head(30))

    ml_df.to_csv("temp_df.csv")


if __name__ == "__main__":
    generate_new_dataset_csv()