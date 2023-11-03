from database.setup import create_tables
from database.operations import get_match_urls, update_match_url, insert_matches, insert_lineups
from download import download_pages
from scrape import extract_match_data
import random
import os
import logging

main_logger = logging.getLogger('main_logger')
main_logger.setLevel(logging.INFO)
main_handler = logging.FileHandler('main.log')
main_handler.setLevel(logging.INFO)
main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
main_logger.addHandler(main_handler)

def check_downloaded_matches():
	file_list = []
	size_threshold = 100 * 1024
	directory_path = "./download/match"
	for filename in os.listdir(directory_path):
		file_path = os.path.join(directory_path, filename)
		if os.path.isfile(file_path) and os.path.getsize(file_path) > size_threshold:
			file_list.append(int(filename.strip('.html')))
	
	update_match_url(file_list, "downloaded")

def main(update_tables = False,
		 download_matches = False,
		 process_matches = False):

	if update_tables:
		create_tables()

	if download_matches:
		# Get list of matches to download
		matches_to_download = get_match_urls(downloaded=False)
		matches_to_download = random.sample(matches_to_download, len(matches_to_download))
		download_list = [
			{
				'url': f"https://www.hltv.org/matches/{match.id}/{match.url}",
				'path': f"match/{match.id}"
			}
			for match in matches_to_download
		]
		
		

		# Download the match pages and update the 'downloaded' flag in database
		success = download_pages(download_list)
		success_ids = [int(match.split("/")[4]) for match in success]
		update_match_url(success_ids, 'downloaded')

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


if __name__ == '__main__':
	check_downloaded_matches()
	main(process_matches = True)
