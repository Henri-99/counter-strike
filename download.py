"""
Download

This module provides functions for downloading web pages using ZenRowsClient and tracking the download progress. It allows you to specify a list of pages to download, including their URLs and paths for saving the downloaded content.

Author: Henri du Plessis
Date: 02/11/2023
"""

from zenrows import ZenRowsClient
import time
import logging

ZENROWS_API_KEY = "5771f6aa356840a80c50a447b69d8d0ad31ab888"
DEFAULT_PARAMS = {"premium_proxy":"true"}

logging.basicConfig(filename='download.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def download_page(client, page, params):
    """
    Downloads a web page using the ZenRowsClient.

    Parameters:
    client (ZenRowsClient): An instance of ZenRowsClient for making HTTP requests.
    page (dict): A dictionary containing the URL to download and the path to save the content.
    params (dict): Parameters to pass in the GET request.

    Returns:
    float: The time taken to download the page in seconds, or None if an error occurs.
    """
    start_time = time.time()

    try:
        logging.info(f"Downloading {page['url']}")
        print(page['url'])
        response = client.get(page['url'], params)
        with open("./download/"+page['path'], 'w', encoding='utf-8') as file:
            file.write(response.text)
        logging.info(f"Successfully downloaded {page['url']}")
    except Exception as e:
        logging.error(f"Failed to download {page['url']}: {e}")
        return None

    return time.time() - start_time

def download_pages(page_list, params = DEFAULT_PARAMS):
    """
    Download a list of web pages and track the download progress.

    Parameters:
    page_list (list of dict): A list of dictionaries, where each dictionary contains the URL to download
        and the path to save the content.
    params (dict): Optional parameters to pass in the GET request.

    Returns:
    list: A list of URLs for pages that were successfully downloaded.
    """
    client = ZenRowsClient(ZENROWS_API_KEY)
    total_pages = len(page_list)
    times = []
    success = []

    for count, page in enumerate(page_list):
        progress_string = f"{str(count).rjust(4)}/{total_pages}: {round(100*(count)/total_pages, 2):.2f}%"
        print(progress_string, end=" ", flush=True)
        
        if times:
            avg_time = sum(times) / len(times)
            time_remaining = format_time(avg_time * (total_pages - count + 1))
            print(f"{time_remaining} •", end=" ", flush=True)
        else:
            print("h:mm:ss •", end=" ", flush=True)

        time_taken = download_page(client, page, params)
        if time_taken is not None:
            times.append(time_taken)
            success.append(page['url'])
    
    return success
        
if __name__ == '__main__':
    page_list = [
        {"url": "https://www.hltv.org/matches/2338764/fate-vs-skade-ggbet-winter-cup", "path": "match/2338764.html"},
    ]
    success = download_pages(page_list)
    print(success)