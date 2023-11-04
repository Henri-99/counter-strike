from database.setup import create_tables
from database.operations import get_match_urls, update_match_url, insert_matches, insert_lineups, get_unscraped_date_range, insert_map_urls
from download import download_pages
from scrape import extract_match_data, extract_map_page_list, extract_map_url_data
import os
import logging
import asyncio

main_logger = logging.getLogger('main_logger')
main_logger.setLevel(logging.INFO)
main_handler = logging.FileHandler('main.log')
main_handler.setLevel(logging.INFO)
main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
main_logger.addHandler(main_handler)

def update_downloaded_matches_status():
	file_list = []
	size_threshold = 100 * 1024
	directory_path = "./download/match"
	for filename in os.listdir(directory_path):
		file_path = os.path.join(directory_path, filename)
		if os.path.isfile(file_path) and os.path.getsize(file_path) > size_threshold:
			file_list.append(int(filename.strip('.html')))
	
	update_match_url(file_list, "downloaded")

def get_latest_map_urls():
	last_date, today = get_unscraped_date_range()
	asyncio.run(download_pages([
		{
			'url': f"https://www.hltv.org/stats/matches?startDate={last_date}&endDate={today}&rankingFilter=Top50",
			'path': f"maplist/{today}_0"
		}
	]
	))

	asyncio.run(download_pages(extract_map_page_list(last_date, today)))
	return extract_map_url_data(today)
	




def main(update_tables = False,
		 download_matches = False,
		 process_matches = False,
		 update_all = False):

	if update_tables:
		create_tables()

	if download_matches:
		# Get list of matches to download
		update_downloaded_matches_status()
		matches_to_download = get_match_urls(downloaded=False)
		download_list = [
			{
				'url': f"https://www.hltv.org/matches/{match.id}/{match.url}",
				'path': f"match/{match.id}"
			}
			for match in matches_to_download
		]

		# Download the match pages and update the 'downloaded' flag in database
		asyncio.run(download_pages(download_list))
		update_downloaded_matches_status()

	if process_matches:
		match_data = []
		lineup_data = []
		success_ids = []
		# Get list of matches to process
		matches_to_process = get_match_urls(downloaded=True, processed=False)
		for match in matches_to_process:
			try:
				this_match, lineups = extract_match_data(match)
			except Exception as e:
				main_logger.error(f"Failed to process {match.id}: {e}")
				continue
			match_data.append(this_match)
			lineup_data = lineup_data + lineups
			success_ids.append(this_match["id"])
		insert_matches(match_data)
		insert_lineups(lineup_data)
		update_match_url(success_ids, 'processed')

	if update_all:
		insert_map_urls(get_latest_map_urls())

		# Download all maps
		# Process maps
		# Update match_urls
		# Download matches
		# Process matches



if __name__ == '__main__':
	main(update_all = True)
