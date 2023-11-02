from database.setup import session
from database.models import MatchURL

def get_match_urls(downloaded=False, processed=None, limit=None):
    # Query the database to get a list of MatchURLs where downloaded matches the specified value
    query = session.query(MatchURL).filter(MatchURL.downloaded == downloaded)
    
    if processed is not None:
        query = query.filter(MatchURL.processed == processed)

    # Optionally, limit the number of results
    if limit is not None:
        query = query.limit(limit)

    # Execute the query and get the results
    match_urls = query.all()
    
    return match_urls

def update_match_url(id_list: list, flag: str):
    if flag not in ['downloaded', 'processed']:
        raise ValueError("Flag must be 'downloaded' or 'processed'")

    # Create a dictionary to specify which flag to update
    flag_to_update = {flag: 1}

    # Use SQLAlchemy's update() to set the flag to 1 for the specified IDs
    session.query(MatchURL).filter(MatchURL.id.in_(id_list)).update(flag_to_update, synchronize_session=False)

    # Commit the changes to the database
    session.commit()