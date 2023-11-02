from database.models import MatchURL
from database.setup import session, create_tables

def main():
    create_tables()

    # Use the MatchURL model and session to perform database operations
    # new_match = MatchURL(id = 12345, url='https://example.com', downloaded=False, processed=False)
    # session.add(new_match)
    # session.commit()

    # 1. Download new pages
    # 2. Scrape downloaded pages
    # 3. Store in database

if __name__ == '__main__':
    main()
