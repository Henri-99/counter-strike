from database.operations import get_match_urls, update_match_url, insert_matches, insert_lineups, get_unscraped_date_range, insert_map_urls, get_map_urls, update_map_url_status, insert_maps, insert_player_performances, update_map_status, create_match_url_records, update_cs2_field, set_map_url_processed_false
from database.models import setup_tables
from download import download_pages
from scrape import extract_match_data, extract_map_page_list, extract_map_url_data, extract_map_data
import os
import logging
import asyncio

main_logger = logging.getLogger('main_logger')
main_logger.setLevel(logging.INFO)
main_handler = logging.FileHandler('main.log')
main_handler.setLevel(logging.INFO)
main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
main_logger.addHandler(main_handler)

def download_map_urls(start_date, end_date):
	asyncio.run(download_pages([
		{
			'url': f"https://www.hltv.org/stats/matches?startDate={start_date}&endDate={end_date}&rankingFilter=Top50",
			'path': f"maplist/{end_date}_0"
		}
	]
	))
	pages_to_download = extract_map_page_list(start_date, end_date)
	asyncio.run(download_pages(pages_to_download))

def download_maps():
	# Get list of maps to download
	update_downloaded_maps_status()
	maps_to_download = get_map_urls(downloaded=False, processed=False)
	download_list = [
		{
			'url': f"https://www.hltv.org/stats/matches/mapstatsid/{map_.id}/{map_.url}",
			'path': f"map/{map_.id}"
		}
		for map_ in maps_to_download
	]

	# Download the match pages and update the 'downloaded' flag in database
	asyncio.run(download_pages(download_list))
	update_downloaded_maps_status()

def download_matches():
	# Get list of matches to download
	update_downloaded_matches_status()
	matches_to_download = get_match_urls(downloaded=False, processed=False)
	download_list = [
		{
			'url': f"https://www.hltv.org/matches/{match.id}/{match.url}",
			'path': f"match/{match.id}"
		}
		for match in matches_to_download
	]

	# Download the match pages and update the 'downloaded' flag in database+
	asyncio.run(download_pages(download_list))
	update_downloaded_matches_status()

def update_downloaded_matches_status():
	file_list = []
	size_threshold = 100 * 1024
	directory_path = "./download/match"
	for filename in os.listdir(directory_path):
		file_path = os.path.join(directory_path, filename)
		if os.path.isfile(file_path) and os.path.getsize(file_path) > size_threshold:
			file_list.append(int(filename.strip('.html')))
	
	update_match_url(file_list, "downloaded")

def update_downloaded_maps_status():
	file_list = []
	size_threshold = 100 * 1024
	directory_path = "./download/map"
	for filename in os.listdir(directory_path):
		file_path = os.path.join(directory_path, filename)
		if os.path.isfile(file_path) and os.path.getsize(file_path) > size_threshold:
			file_list.append(int(filename.strip('.html')))
	
	update_map_status(file_list, "downloaded")

def update_downloaded_maps_status():
	file_list = []
	size_threshold = 100 * 1024
	directory_path = "./download/map"
	for filename in os.listdir(directory_path):
		file_path = os.path.join(directory_path, filename)
		if os.path.isfile(file_path) and os.path.getsize(file_path) > size_threshold:
			file_list.append(int(filename.strip('.html')))
	
	update_map_url_status(file_list, "downloaded")

def process_map_url_pages(date):
	map_urls = extract_map_url_data(date)
	insert_map_urls(map_urls)

import concurrent.futures

def process_map(map_info, total_maps):
	map_index, map_ = map_info
	try:
		map_stats, player_stats_list = extract_map_data(map_.id)
		remaining_maps = total_maps - map_index - 1
		progress_percent = ((total_maps - remaining_maps) / total_maps) * 100
		print(f"Processed {map_.id} ({progress_percent:.2f}% complete, {remaining_maps} remaining)")
		return map_stats, player_stats_list
	except Exception as e:
		main_logger.error(f"Failed to process {map_.id}: {e}")
		return None, None

def process_map_pages():
	map_data = []
	player_performance_data = []
	success_ids = []

	while True:
		# Get list of maps to process
		maps_to_process = get_map_urls(downloaded=True, processed=False, limit=1000)
		total_maps = len(maps_to_process)
		if total_maps == 0:
			break

		with concurrent.futures.ThreadPoolExecutor() as executor:
			# Use executor.map to process maps concurrently
			results = executor.map(process_map, enumerate(maps_to_process), [total_maps] * total_maps)

		for result in results:
			if result is not None:
				map_stats, player_stats_list = result
				map_data.append(map_stats)
				player_performance_data.append(player_stats_list)
				success_ids.append(map_stats['id'])

		insert_maps(map_data)
		insert_player_performances(player_performance_data)
		update_map_url_status(success_ids, 'processed')

def process_match_pages():
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

def main():
	# Get date range to scrape
	last_date, today = get_unscraped_date_range()
	# last_date, today = ("2019-11-01", "2019-12-31")
	print(f"Syncing with HLTV... (last update: {last_date})\n")

	# Download pages with map URLs
	print("Fetching links to new maps")
	download_map_urls(last_date, today)
	
	# Extract list of map URLs from pages and insert into database
	print("Finding new map URLs")
	process_map_url_pages(today)
	print(f"Map URLs added to database")

	# Download map pages
	print("Fetching new maps")
	download_maps()

	# Extract map, player data from map pages and insert into database
	print("Reading map data")
	process_map_pages()
	print(f"Map data added to database")

	# Create match URL records from map records
	create_match_url_records()

	# Download match pages
	print("Fetching new matches")
	download_matches()

	# Extract match, lineup data from match pages and insert into database
	print("Reading match data")
	process_match_pages()

	print("Matches added to database")

	print("\nSync complete.")


if __name__ == '__main__':
	main()
	# setup_tables()
	# process_map_pages()
	# download_maps() 
	# process_map_pages()