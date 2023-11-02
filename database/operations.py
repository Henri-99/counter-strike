from database.setup import session
from database.models import MatchURL, Match, Lineup, Map
from sqlalchemy import func, and_
from datetime import datetime

def get_match_urls(downloaded=False, processed=None, limit=None):
    # Query the database to get a list of MatchURLs where downloaded matches the specified value
    query = session.query(MatchURL).filter(MatchURL.downloaded == downloaded)
    
    if processed is not None:
        query = query.filter(MatchURL.processed == processed)

    # Optionally, limit the number of results
    if limit is not None:
        query = query.limit(limit)

    # Execute the query and get the results
    match_urls = query.all()
    
    return match_urls

def get_maps(start_date=None, end_date=None, limit=None):
    query = session.query(Map)
    if start_date and end_date is not None:
        query = query.filter(
			and_(
				Map.datetime >= start_date,
				Map.datetime <= end_date
			)
		).order_by(Map.id)
    
    if limit is not None:
        query = query.limit(limit)
    
    map_data = query.all()

    return map_data

def get_date_range():
    # Query for the minimum and maximum datetime values
    min_date = session.query(func.min(Map.datetime)).scalar()
    max_date = session.query(func.max(Map.datetime)).scalar()
    date_format = "%Y-%m-%d %H:%M"

    return datetime.strptime(min_date, date_format), datetime.strptime(max_date, date_format)

def update_match_url(id_list: list, flag: str):
    if flag not in ['downloaded', 'processed']:
        raise ValueError("Flag must be 'downloaded' or 'processed'")

    # Create a dictionary to specify which flag to update
    flag_to_update = {flag: 1}

    # Use SQLAlchemy's update() to set the flag to 1 for the specified IDs
    session.query(MatchURL).filter(MatchURL.id.in_(id_list)).update(flag_to_update, synchronize_session=False)

    # Commit the changes to the database
    session.commit()
    
def insert_matches(match_data_list):
    """
    Insert a list of match data dictionaries into the Match table.

    Args:
        match_data_list (list of dict): List of dictionaries containing match data.

    Returns:
        None
    """
    for data in match_data_list:
        match_entry = Match(
            id=data['id'],
            url=data['url'],
            datetime=data['datetime'],
            team1_id=data['team1_id'],
            team2_id=data['team2_id'],
            team1=data['team1'],
            team2=data['team2'],
            team1_score=data['team1_score'],
            team2_score=data['team2_score'],
            team1_rank=data['team1_rank'],
            team2_rank=data['team2_rank'],
            winner=data['winner'],
            event_id=data['event_id'],
            event=data['event'],
            lan=data['lan'],
            best_of=data['best_of'],
            box_str=data['box_str'],
            veto=data['veto']
        )
        session.add(match_entry)

    session.commit()
    
def insert_lineups(lineup_data_list):
    """
    Insert a list of lineup data dictionaries into the Lineup table.

    Args:
        lineup_data_list (list of dict): List of dictionaries containing lineup data.

    Returns:
        None
    """
    for data in lineup_data_list:
        lineup_entry = Lineup(
            team_id=data['team_id'],
            team_name=data['team_name'],
            rank=data['rank'],
            date=data['date'],
            match_id=data['match_id'],
            player1_id=data['player1_id'],
            player1=data['player1'],
            player2_id=data['player2_id'],
            player2=data['player2'],
            player3_id=data['player3_id'],
            player3=data['player3'],
            player4_id=data['player4_id'],
            player4=data['player4'],
            player5_id=data['player5_id'],
            player5=data['player5']
        )
        session.add(lineup_entry)

    session.commit()