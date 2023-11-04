"""
Download

This module provides functions for downloading web pages using ZenRowsClient and tracking the download progress. It allows you to specify a list of pages to download, including their URLs and paths for saving the downloaded content.

Author: Henri du Plessis
Date: 02/11/2023
"""

from zenrows import ZenRowsClient
import time
import logging
import random
import asyncio
from urllib.parse import urlparse, parse_qs

ZENROWS_API_KEY = "5771f6aa356840a80c50a447b69d8d0ad31ab888"
DEFAULT_PARAMS = {"premium_proxy":"true"}

download_logger = logging.getLogger('download_logger')
download_logger.setLevel(logging.INFO)
download_handler = logging.FileHandler('download.log')
download_handler.setLevel(logging.INFO)
download_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
download_logger.addHandler(download_handler)

async def download_page(client, page, counter):
	response = await client.get_async(page['url'], params = DEFAULT_PARAMS)
	elapsed_time = time.time() - counter['start_time']
	with open(f"./download/{page['path']}.html", 'w', encoding='utf-8') as file:
		file.write(response.text)
	# print(f"Downloaded {page['url']} and saved to ./download/{page['path']}.html")
	download_logger.info(f"Successfully downloaded {page['url']}")
	# Increase the counter by 1 for each successful download
	counter['downloaded'] += 1
	print(f"{counter['downloaded']} / {counter['total']} •", end=" ", flush=True)

	# Calculate the average time per page download and estimate time to completion
	average_time_per_page = elapsed_time / counter['downloaded']
	remaining_pages = counter['total'] - counter['downloaded']
	estimated_time_to_completion = format_time(average_time_per_page * remaining_pages)
	print(f"{estimated_time_to_completion}")

async def download_pages(pages):
	client = ZenRowsClient(ZENROWS_API_KEY, concurrency=10, retries=3)
	counter = {
		'downloaded': 0,
		'total': len(pages),
		'start_time': time.time()  # Record the start time here
	}
	tasks = [download_page(client, page, counter) for page in pages]
	await asyncio.gather(*tasks)



def format_time(total_seconds):
	"""
	Formats a time value into 'x:y:z' format.

	Parameters:
	total_seconds (int or float): The total time in seconds.

	Returns:
	str: The formatted time string 'x:y:z' where x is hours, y is minutes, and z is seconds.
	"""
	# Ensure total_seconds is a non-negative number
	if total_seconds < 0:
		raise ValueError("Total seconds must be a non-negative number")

	# Calculate hours, minutes, and seconds
	hours = int(total_seconds // 3600)
	remaining_seconds = total_seconds % 3600
	minutes = int(remaining_seconds // 60)
	seconds = int(remaining_seconds % 60)

	# Format and return the time string
	return f'{hours}:{minutes:02d}:{seconds:02d}'

# def download_page(client, page, params):
# 	"""
# 	Downloads a web page using the ZenRowsClient.

# 	Parameters:
# 	client (ZenRowsClient): An instance of ZenRowsClient for making HTTP requests.
# 	page (dict): A dictionary containing the URL to download and the path to save the content.
# 	params (dict): Parameters to pass in the GET request.

# 	Returns:
# 	float: The time taken to download the page in seconds, or None if an error occurs.
# 	"""
# 	start_time = time.time()

# 	tries = 3
# 	while tries > 0:
# 		try:
# 			download_logger.info(f"Downloading {page['url']}")
# 			print(page['url'])
# 			response = client.get(page['url'], params)
# 			with open(f"./download/{page['path']}.html", 'w', encoding='utf-8') as file:
# 				data = response.text
# 				if "Operation timeout exceeded (CTX0002)" in data:
# 					raise TimeoutError
# 				file.write(response.text)
# 			download_logger.info(f"Successfully downloaded {page['url']}")
# 			break
# 		except Exception as e:
# 			download_logger.error(f"Failed to download {page['url']}: {e}")
# 			tries -= 1

# 	return time.time() - start_time

# def download_pages(page_list, params = DEFAULT_PARAMS):
# 	"""
# 	Download a list of web pages and track the download progress.

# 	Parameters:
# 	page_list (list of dict): A list of dictionaries, where each dictionary contains the URL to download
# 		and the path to save the content.
# 	params (dict): Optional parameters to pass in the GET request.

# 	Returns:
# 	list: A list of URLs for pages that were successfully downloaded.
# 	"""
# 	client = ZenRowsClient(ZENROWS_API_KEY, concurrency=10, retries=2)
# 	total_pages = len(page_list)
# 	times = []
# 	success = []

# 	# Randomize order of pages to download
# 	page_list = random.sample(page_list, len(page_list))

# 	for count, page in enumerate(page_list):
# 		progress_percent = f"{round(100 * count / total_pages, 2):.2f}".rjust(5)
# 		progress_absolute = f"{str(count).rjust(4)}/{total_pages}"
# 		progress_string = f"{progress_absolute} • {progress_percent}%"
# 		print(progress_string, end=" ", flush=True)
		
# 		if times:
# 			avg_time = sum(times) / len(times)
# 			time_remaining = format_time(avg_time * (total_pages - count)).rjust(8)
# 			avg_time_string = f"{round(avg_time,1)}".rjust(4)
# 			print(f"• {avg_time_string}s • {time_remaining} •", end=" ", flush=True)
# 		else:
# 			print("•       •          •", end=" ", flush=True)

# 		time_taken = download_page(client, page, params)
# 		if time_taken is not None:
# 			times.append(time_taken)
# 			success.append(page['url'])
	
# 	return success
		
if __name__ == '__main__':
	page_list = [
		{"url": "https://www.hltv.org/matches/2338764/fate-vs-skade-ggbet-winter-cup", "path": "match/2338764.html"},
	]
	success = download_pages(page_list)
	print(success)