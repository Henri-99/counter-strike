from database.models import MatchURL
from database.setup import session, create_tables
from database.operations import get_match_urls, update_match_url
from download import download_pages

def main():
    # create_tables()

    matches_to_download = get_match_urls(downloaded=False, limit=1)
    download_list = []
    for match_ in matches_to_download:
        download_list.append({
            'url' : f"https://www.hltv.org/matches/{match_.id}/{match_.url}",
            'path' : f"match/{match_.id}"
        })
    success = download_pages(download_list)
    success_ids = [int(match.split("/")[4]) for match in success]
    update_match_url(success_ids, 'downloaded')




    # Use the MatchURL model and session to perform database operations
    # new_match = MatchURL(id = 12345, url='https://example.com', downloaded=False, processed=False)
    # session.add(new_match)
    # session.commit()

    # 1. Download new pages
    # 2. Scrape downloaded pages
    # 3. Store in database

if __name__ == '__main__':
    main()
