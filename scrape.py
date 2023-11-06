from bs4 import BeautifulSoup
from datetime import datetime
import re
import logging
import os

scraper_logger = logging.getLogger('scraper_logger')
scraper_logger.setLevel(logging.INFO)
scraper_handler = logging.FileHandler('scraper.log')
scraper_handler.setLevel(logging.INFO)
scraper_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
scraper_logger.addHandler(scraper_handler)

def extract_match_data(match):
	with open(f"download/match/{match.id}.html", "r", encoding='utf-8') as html:
		soup = BeautifulSoup(html, "html.parser")
		
		teambox = soup.find("div", class_ = "teamsBox")
		try:
			team1_div = teambox.find("div", class_ = "team1-gradient")
		except Exception as e:
			scraper_logger.info(f"{match.id}: {e}")
			return None, None


		team1_id = team1_div.find("a")['href'].split("/")[2]
		team1 = team1_div.find("a")['href'].split("/")[3]
		team1_score = team1_div.find_all("div")[1].text

		team2_div = teambox.find("div", class_ = "team2-gradient")
		team2_id = team2_div.find("a")['href'].split("/")[2]
		team2 = team2_div.find("a")['href'].split("/")[3]
		team2_score = team2_div.find_all("div")[1].text

		time_unix = int(teambox.find("div", class_ = "time")['data-unix'])/1000
		datetime_ = datetime.fromtimestamp(time_unix).strftime("%Y-%m-%d %H:%M")
		event_id = teambox.find("div", class_ = "event").find("a")['href'].split("/")[2]
		event = teambox.find("div", class_ = "event").find("a")['href'].split("/")[3]


		boxes = soup.find_all("div", class_ = "veto-box")

		lineups = []

		lineup_boxes = soup.find_all("div", class_ = "lineup")
		try:
			team1_rank = int(lineup_boxes[0].find("div", class_="teamRanking").find("a").text.strip("World rank: #"))
		except:
			scraper_logger.info(f"Anomolous team rank {match.id}/{match.url}")
			team1_rank = None
		try:
			team2_rank = int(lineup_boxes[1].find("div", class_="teamRanking").find("a").text.strip("World rank: #"))
		except:
			scraper_logger.info(f"Anomolous team rank {match.id}/{match.url}")
			team2_rank = None

		for j, lineup_box in enumerate(lineup_boxes):
			lineup = {
				"team_id" : team1_id if j == 0 else team2_id,
				"team_name" : team1 if j == 0 else team2,
				"rank" : team1_rank if j == 0 else team2_rank,
				"date" : datetime_,
				"match_id" : match.id,
			}
			player_links = lineup_box.find_all("td", class_='player')[:5]
			for i, link in enumerate(player_links):
				link = link.find("a")
				player_id = link['href'].split('/')[2]  # Extract the player id from the href attribute
				player_name = link['href'].split('/')[3]  # Extract the player id from the href attribute
				lineup[f'player{i+1}_id'] = player_id
				lineup[f'player{i+1}'] = player_name

			lineups.append(lineup)
		
		# Start building stats dict
		match_data = {
			"id" : match.id,
			"url" : match.url,
			"datetime" : datetime_,
			"team1_id" : team1_id,
			"team2_id" : team2_id,
			"team1" : team1,
			"team2" : team2,
			"team1_score" : team1_score,
			"team2_score" : team2_score,
			"team1_rank" : team1_rank,
			"team2_rank" : team2_rank,
			"winner" : team1_id if team1_score > team2_score else team2_id,
			"event_id" : event_id,
			"event" : event,
		}
		match_data['lan'] = True if "LAN" in boxes[0].text else False
		pattern = r'(Best of|BO)\s*(\d+)'
		format_ = [match[1] for match in re.findall(pattern, boxes[0].text)]
		if len(format_) > 0:
			match_data['best_of'] = int(format_[0])
		else:
			match_data['best_of'] = 0
		match_data['box_str'] = "\n".join([s.strip("* ") for s in boxes[0].text.strip().split("\n") if s])
		try:
			match_data['veto'] = boxes[1].text.strip()
		except Exception as e:
			scraper_logger.info(f"{match.id} has no veto data")
			match_data['veto'] = ""

		scraper_logger.info(f"Match {match.id} processed successfully")
	return match_data, lineups

def extract_map_page_list(start_date, end_date):
	with open(f"download/maplist/{end_date}_0.html", "r", encoding='utf-8') as html:
		soup = BeautifulSoup(html, "html.parser")
		pagination_data = soup.find("span", {"class": "pagination-data"}).text.split(" of ")
		total_map_count = int(pagination_data[1])
		no_pages_to_download = int(total_map_count/50) + 1

		page_list = []
		for page_no in range(1,no_pages_to_download):
			page_list.append(
				{
					'url': f"https://www.hltv.org/stats/matches?startDate={start_date}&endDate={end_date}&offset={page_no*50}&rankingFilter=Top50",
					'path': f"maplist/{end_date}_{page_no}"
				}
			)
		return page_list

def extract_map_url_data(scrape_date):
	filenames = [f for f in os.listdir("download/maplist/") if scrape_date in f]
	map_url_data = []
	for filename in filenames:
		with open(f"download/maplist/{filename}", "r", encoding='utf-8') as html:
			soup = BeautifulSoup(html, "html.parser")
			# Loop through each row	and record map URLs
			table_rows  = soup.table.contents[3].find_all("tr")
			for i in range(len(table_rows)):
				map_url = table_rows[i].find_all("a", href=True)
				map_object = {
					'id'   : map_url[0]['href'].split('/')[4],
					'url'  : map_url[0]['href'].split('?')[0].split("/")[-1],
					'downloaded' : 0,
					'processed' : 0
					}
				map_url_data.append(map_object)
	return map_url_data

#------ extract map data

def extract_map_data(map_id):
	map_stats = {}
	player_stats = []
	with open(f"download/map/{map_id}.html", "r", encoding='utf-8') as html:	
		soup = BeautifulSoup(html, "html.parser")

		map_stats.update({'id': map_id})

		match_box_elem = soup.find("div", class_="match-info-box-con")
		map_stats.update({
			'match_id': int(match_box_elem.find("a", class_="match-page-link")['href'].split('/')[2]),
			'match_page_url': match_box_elem.find("a", class_="match-page-link")['href'].split('/')[3],
			'datetime': match_box_elem.find("span").string
		})

		box_links = match_box_elem.find_all("a")
		map_stats.update(
				{
					'event_id': int(box_links[0]['href'].split("=")[1]),
					'event_name': box_links[0].text.strip()
				}
			)
		map_stats.update(
			{
				't1_id': int(box_links[1]['href'].split('/')[3]),
				't2_id': int(box_links[2]['href'].split('/')[3]),
				't1': box_links[1]['href'].split('/')[4],
				't2': box_links[2]['href'].split('/')[4]
			}
		)
		map_stats['map_name'] = soup.find("div", class_="small-text").next_sibling.strip()

		match_box_rows = match_box_elem.find_all("div", class_="match-info-row")
		
		score_breakdown = match_box_rows[0].find_all("span")
		
		t1_start = score_breakdown[2]['class'][0].split('-')[0]
		mapping = {
			't1_t_score': 4,
			't1_ct_score': 2,
			't2_t_score': 3,
			't2_ct_score': 5
		}
		if t1_start == 't':
			mapping = {
				't1_t_score': 2,
				't1_ct_score': 4,
				't2_t_score': 5,
				't2_ct_score': 3
			}
			
		map_stats.update(
			{ key: int(score_breakdown[value].text) for key, value in mapping.items() }
		)

		ot_text = score_breakdown[5].next_sibling.strip()
		ot_stats = {'overtime': False, 't1_ot_score': 0, 't2_ot_score': 0}
		if '(' in ot_text:
			ot_stats['overtime'] = True
			ot_score = [int(score) for score in ot_text.split() if score.isdigit()]
			ot_stats['t1_ot_score'], ot_stats['t2_ot_score'] = ot_score
		map_stats.update(ot_stats)

		map_stats.update(
			{
			't1_rating': float(match_box_rows[1].find("div").string.split(":")[0]),
			't2_rating': float(match_box_rows[1].find("div").string.split(":")[1]),
			't1_first_kills': int(match_box_rows[2].find("div").string.split(":")[0]),
			't2_first_kills': int(match_box_rows[2].find("div").string.split(":")[1]),
			't1_clutches': int(match_box_rows[3].find("div").string.split(":")[0]),
			't2_clutches': int(match_box_rows[3].find("div").string.split(":")[1])
		})
		
		# Round history
		t1_round_icons, t2_round_icons = get_round_icons(soup, "round-history-con")
		map_stats['t1_round_history'], map_stats['t2_round_history'] = process_round_history(t1_round_icons, t2_round_icons)

		# Overtime round history
		if map_stats['overtime']:
			t1_round_icons, t2_round_icons = get_round_icons(soup, "round-history-overtime")
			map_stats['t2_ot_round_history'], map_stats['t1_ot_round_history'] = process_round_history(t1_round_icons, t2_round_icons)
		else:
			map_stats['t2_ot_round_history'] = ""
			map_stats['t1_ot_round_history'] = ""

		t1_score = map_stats['t1_t_score'] + map_stats['t1_ct_score'] + map_stats['t1_ot_score']
		t2_score = map_stats['t2_t_score'] + map_stats['t2_ct_score'] + map_stats['t2_ot_score']
		winner_id = map_stats['t1_id'] if t1_score > t2_score else map_stats['t2_id']

		map_stats.update(
			{
				't1_score' : t1_score,
				't2_score' : t2_score,
				'winner_id' : winner_id,
			}
		)

		# Player statistics

		player_stats_tables = soup.find_all("table", {"class" : "totalstats"})
		for i in range(2):
			player_stats_rows = player_stats_tables[i].find_all("tr")[1:] # skip heading row
			player_stats = player_stats + process_player_stats(player_stats_rows, i+1, map_stats)

		scraper_logger.info(f"Map {map_stats['id']} processed successfully")
	return map_stats, player_stats

def extract_performance(player, map_id, team, team_id):
	"""
	Extracts performance metrics for a given player.
	
	Args:
		player (Tag): A BeautifulSoup tag representing a player's stats row.
		map_id (int) : Map ID
		team (str): Team name
		team_id (int): Team ID

	Returns:
		dict: A dictionary containing performance metrics for the player.
	"""
	cells = player.find_all("td")
	
	player_link = cells[0].find("a")['href'].split('/')
	player_id = int(player_link[3])
	player_name = player_link[4]

	kills_data = cells[1].text.split(' ')
	assists_data = cells[2].text.split(' ')
	
	kills = int(kills_data[0])
	assists = int(assists_data[0])

	hs_span = cells[1].find("span")
	flashes_span = cells[2].find("span")

	hs = int(hs_span.string.split('(')[1][:-1]) if hs_span else None
	flashes = int(flashes_span.string.split('(')[1][:-1]) if flashes_span else None

	deaths = int(cells[3].text)
	kast = float(cells[4].text[:-1])
	adr = float(cells[6].text) if cells[6].text != '-' else 0
	first_kd = int(cells[7].text)
	rating = float(cells[8].text)

	return {
		"player_id": player_id,
		"player": player_name,
		"map_id": int(map_id),
		"team": team,
		"teamID": int(team_id),
		"kills": kills,
		"hs": hs,
		"assists": assists,
		"flashes": flashes,
		"deaths": deaths,
		"kast": kast,
		"adr": adr,
		"first_kd": first_kd,
		"rating": rating
	}

def process_player_stats(player_stats_rows, team, map_stats):
	"""
	Processes player stats and extracts performance metrics.
	
	Args:
		player_stats_rows (list): List of BeautifulSoup tags representing players' stats rows.
		team (int): Team number 1 or 2
		map_stats (dict): Dictionary containing map statistics.

	Returns:
		list: A list of dictionaries containing performance metrics for each player.
	"""
	player_stats = []
	for player in player_stats_rows:
		team_name, team_id = (map_stats["t1"], map_stats["t1_id"]) if team == 1 else (map_stats["t2"], map_stats["t2_id"])
		performance = extract_performance(player, map_stats['id'], team_name, team_id)
		player_stats.append(performance)

	return player_stats

def get_round_history(round_icons):
	outcome_mapping = {
		"t_win": "T",
		"emptyHistory": "_",
		"bomb_exploded": "B",
		"ct_win": "C",
		"bomb_defused": "D",
		"stopwatch": "S"
	}

	round_history = ""

	for round_img in round_icons:
		round_outcome = round_img['src'].split('/')[4][:-4]
		round_history += outcome_mapping.get(round_outcome, "")

	return round_history

def get_round_icons(soup, class_name):
	round_hist_rows = soup.find("div", class_=class_name).find_all("div", class_="round-history-team-row")
	t1_round_icons = round_hist_rows[0].find_all("img")[1:]
	t2_round_icons = round_hist_rows[1].find_all("img")[1:]
	return t1_round_icons, t2_round_icons

def process_round_history(t1_round_icons, t2_round_icons):
	t1_rounds_won = get_round_history(t1_round_icons)
	t2_rounds_won = get_round_history(t2_round_icons)

	for i in range(16, len(t1_rounds_won)):
		if t1_rounds_won[i] == t2_rounds_won[i]:
			t1_rounds_won = t1_rounds_won[:i-1]
			t2_rounds_won = t2_rounds_won[:i-1]
			break

	return t1_rounds_won, t2_rounds_won