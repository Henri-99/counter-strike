from database.models import MatchURL
from database.setup import session, create_tables
from database.operations import get_match_urls, update_match_url
from download import download_pages

def main(create_tables = False,
         download_matches = False):

    if create_tables:
        create_tables()

    # Get list of matches to download
    matches_to_download = get_match_urls(downloaded=False, limit=98)
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

    # Get list of matches to process
    matches_to_process = get_match_urls(processed=False)
    for match in matches_to_process:
        print("...")
        # Extract match info, save to 
    # Store in db
    # Flag as processed

    

if __name__ == '__main__':
    main()
