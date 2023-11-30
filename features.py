from database.setup import session
from database.models import Match, Map
import pandas as pd
from datetime import datetime, timedelta

# Joining Match and Map tables with Match on the left side
query = session.query(Match, Map)\
    .join(Map, Match.id == Map.match_id)\
    .filter(Match.datetime > "2023-11-25") 

result = query.all()
print(len(result))

# Creating a DataFrame from the result
columns = [
    "datetime", "match_id", "map_id", "team1_id", "team2_id", "team1_rank", "team2_rank",
    "map_name", "t1_score", "t2_score", "win"
]

data = [
    (
        map_.datetime, match.id, map_.id, map_.t1_id, map_.t2_id, match.team1_rank, match.team2_rank,
        map_.map_name, map_.t1_score, map_.t2_score,
        0 if map_.t2_id == map_.winner_id else 1
    )
    for match, map_ in result
]

df = pd.DataFrame(data, columns=columns)

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
    total_maps = team_maps_query.count()
    if total_maps == 0:
        win_rate = 0.0
    else:
        win_count = team_maps_query.filter(
            ((Map.t1_id == team_id) & (Map.winner_id == Map.t1_id)) |
            ((Map.t2_id == team_id) & (Map.winner_id == Map.t2_id))
        ).count()
        win_rate = win_count / total_maps

    # Calculate the number of times the specific map was played
    map_played_count = team_maps_query.filter(Map.map_name == row['map_name']).count()

    return win_rate, map_played_count

# Apply the function to create new columns
df[['map_winrate_t1', 'map_played_count_t1']] = df.apply(lambda row: pd.Series(calculate_stats(row, 'team1_id')), axis=1)
df[['map_winrate_t2', 'map_played_count_t2']] = df.apply(lambda row: pd.Series(calculate_stats(row, 'team2_id')), axis=1)

print(df)

# # Calculate the percentage of win==expected
# percentage = (df['win'] == df['expected']).mean() * 100

# # Print the percentage
# print(f"Percentage of win==expected: {percentage:.2f}%")

