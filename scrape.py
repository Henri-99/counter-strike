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

		scraper_logger.info(f"{match.id} processed successfully")
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







