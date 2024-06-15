from database.setup import session
from database.models import MatchURL, Match, Lineup, Map, MapURL, PlayerStats
from sqlalchemy import func, and_
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import logging

db_logger = logging.getLogger('db_logger')
db_logger.setLevel(logging.INFO)
db_handler = logging.FileHandler('db.log')
db_handler.setLevel(logging.INFO)
db_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
db_logger.addHandler(db_handler)

def set_map_url_processed_false():
	session.query(MapURL).update({MapURL.processed: False})
	session.commit()

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

def get_map_urls(downloaded=False, processed=None, limit=None):
	# Query the database to get a list of MatchURLs where downloaded matches the specified value
	query = session.query(MapURL).filter(MapURL.downloaded == downloaded)
	
	if processed is not None:
		query = query.filter(MapURL.processed == processed)

	# Exclude records with IDs in the exclusion_list
	exclusion_list = [111065, 107386, 102400, 97830]
	query = query.filter(~MapURL.id.in_(exclusion_list))

	# Optionally, limit the number of results
	if limit is not None:
		query = query.limit(limit)

	# Execute the query and get the results
	map_urls = query.all()

	return map_urls

def get_date_range():
	# Query for the minimum and maximum datetime values
	min_date = session.query(func.min(Map.datetime)).scalar()
	max_date = session.query(func.max(Map.datetime)).scalar()
	date_format = "%Y-%m-%d %H:%M"

	return datetime.strptime(min_date, date_format), datetime.strptime(max_date, date_format)

def get_unscraped_date_range():
	max_date_id = session.query(func.max(MapURL.id)).scalar() 
	max_date = session.query(Map.datetime).filter(Map.id == max_date_id).scalar()
	last_scraped_date = max_date.split(" ")[0]
	now = datetime.now().strftime("%Y-%m-%d")
	return last_scraped_date, now
	
def update_match_url(id_list: list, flag: str):
	if flag not in ['downloaded', 'processed']:
		raise ValueError("Flag must be 'downloaded' or 'processed'")

	# Create a dictionary to specify which flag to update
	flag_to_update = {flag: 1}

	# Use SQLAlchemy's update() to set the flag to 1 for the specified IDs
	session.query(MatchURL).filter(MatchURL.id.in_(id_list)).update(flag_to_update, synchronize_session=False)

	# Commit the changes to the database
	session.commit()

def update_map_url_status(id_list: list, flag: str):
	if flag not in ['downloaded', 'processed']:
		raise ValueError("Flag must be 'downloaded' or 'processed'")

	# Create a dictionary to specify which flag to update
	flag_to_update = {flag: 1}

	# Use SQLAlchemy's update() to set the flag to 1 for the specified IDs
	session.query(MapURL).filter(MapURL.id.in_(id_list)).update(flag_to_update, synchronize_session=False)

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
	count_before = session.query(Match).count()

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
			veto=data['veto'],
			cs2 = data['cs2']
		)
		session.add(match_entry)

	session.commit()

	new_records_added = session.query(Match).count() - count_before
	db_logger.info(f"{new_records_added} matches added to database")
	
def insert_lineups(lineup_data_list):
	"""
	Insert a list of lineup data dictionaries into the Lineup table.

	Args:
		lineup_data_list (list of dict): List of dictionaries containing lineup data.

	Returns:
		None
	"""
	count_before = session.query(Lineup).count()
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

	new_records_added = session.query(Lineup).count() - count_before
	db_logger.info(f"{new_records_added} lineups added to database")
	
def insert_map_urls(map_url_list):
	count_before = session.query(MapURL).count()
	for map_url_dict in map_url_list:
		map_url = MapURL(**map_url_dict)
		session.merge(map_url)
	session.commit()
	count_after = session.query(MapURL).count()
	new_records_added = count_after - count_before
	db_logger.info(f"{new_records_added} MapURLs added to database")

def insert_maps(map_list):
	count_before = session.query(Map).count()
	for map_dict in map_list:
		map_ = Map(**map_dict)
		session.merge(map_)
	session.commit()
	count_after = session.query(Map).count()
	new_records_added = count_after - count_before
	db_logger.info(f"{new_records_added} maps added to database")

def insert_player_performances(player_stats_list):
	count_before = session.query(PlayerStats).count()
	for map_ in player_stats_list:
		for player_data in map_:
			player_stat = PlayerStats(
				player_id = player_data['player_id'],
				player_name = player_data['player'],
				map_id = player_data['map_id'],
				team = player_data['team'],
				team_id = player_data['teamID'],
				kills = player_data['kills'],
				hs = player_data['hs'],
				assists = player_data['assists'],
				flashes = player_data['flashes'],
				deaths = player_data['deaths'],
				kast = player_data['kast'],
				adr = player_data['adr'],
				first_kd = player_data['first_kd'],
				rating = player_data['rating']
			)
			session.merge(player_stat)
	session.commit()
	count_after = session.query(PlayerStats).count()
	new_records_added = count_after - count_before
	db_logger.info(f"{new_records_added} player performances added to database")

def update_map_status(id_list: list, flag: str):
	if flag not in ['downloaded', 'processed']:
		raise ValueError("Flag must be 'downloaded' or 'processed'")

	# Create a dictionary to specify which flag to update
	flag_to_update = {flag: 1}

	# Use SQLAlchemy's update() to set the flag to 1 for the specified IDs
	session.query(Map).filter(MapURL.id.in_(id_list)).update(flag_to_update, synchronize_session=False)

	# Commit the changes to the database
	session.commit()

def create_match_url_records():
	# Get match_ids that are already in the match_url table
	existing_match_ids = session.query(MatchURL.id).all()
	existing_match_ids = [match[0] for match in existing_match_ids]

	# Query distinct match_ids from the map table, excluding those already in match_url
	unique_matches = (session.query(Map.match_id, Map.match_page_url)
						.filter(Map.match_id.notin_(existing_match_ids))
						.group_by(Map.match_id)
						.all())

	# For each unique match, create a match_url record
	for match in unique_matches:
		new_record = MatchURL(
			id=match.match_id,
			url=match.match_page_url,
			downloaded=False,
			processed=False 
		)
		session.add(new_record)

	session.commit()

def update_cs2_field():
	session.query(Match).update({"cs2" : 0})

	# Check if the date is after 8 October 2023 and update cs2 to 1
	session.query(Match).filter(Match.datetime > datetime(2023, 10, 8)).update({"cs2": 1})

	# Check if box_str contains "Counter-Strike 2" and update cs2 to 1
	session.query(Match).filter(Match.box_str.like('%Counter-Strike 2%')).update({"cs2": 1})

	# Commit the changes to the database
	session.commit()