from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, and_
from datetime import datetime, timedelta
from database.setup import session
from database.models import Lineup, LineupAge  # Import your model definitions

# Query Lineups: The function will start by querying Lineup objects that don't already have corresponding LineupAge objects, ordered by date.

# Process Each Lineup: For each Lineup object, the function will:

# 2(a) Check for an existing LineupAge object for the same team_id that is the most recent.
# 2(b) If no existing LineupAge is found, create a new LineupAge with age_days set to 0 and matches_together set to 1.
# 2(c) If an existing LineupAge is found, compare the current Lineup with the Lineup associated with the found LineupAge's match_id. If all 5 players match, increment age_days and create a new LineupAge.
# 2(d) If the players differ, look for a Lineup at least one month older and repeat the check. If players match, create a new LineupAge accordingly.
# 2(e) If no matching lineup is found, create a new LineupAge with age_days set to 0 and matches_together set to 1.

def process_lineups():
	# Query Lineups
	subquery = session.query(LineupAge.match_id).distinct()

	# Determine the total number of records to process
	total_count = session.query(Lineup).filter(Lineup.match_id.notin_(subquery)).count()

	# Set the batch size
	batch_size = 500
	processed_count = 0
	incomplete = True

	while incomplete:
		# Query Lineups in batches
		query = (
			session.query(Lineup)
			.filter(Lineup.match_id.notin_(subquery))
			.order_by(Lineup.date.asc())
			.limit(batch_size)
		)
		lineup_objects = query.all()

		# Check if this is the last batch
		if len(lineup_objects) < batch_size:
			incomplete = False

		# Process each lineup in the current batch
		for lineup in lineup_objects:
			process_lineup(lineup)
			processed_count += 1
			if processed_count % 10 == 0 or processed_count == total_count:
				print(f"{str(processed_count).rjust(5)}/{total_count}: {(float(processed_count) * 100 / total_count):.2f}%")

def process_lineup(lineup):
	# 2(a) Check for existing LineupAge
	latest_lineup_age = session.query(LineupAge).filter(LineupAge.team_id == lineup.team_id).order_by(LineupAge.date.desc()).first()

	# 2(b) No existing LineupAge
	if not latest_lineup_age:
		insert_lineup_age(lineup, 0, 1)
		return

	# 2(c) Compare with the latest lineup
	previous_match_lineup = session.query(Lineup).filter(latest_lineup_age.team_id == Lineup.team_id).filter(latest_lineup_age.date == Lineup.date).first()
	if is_matching_lineup(previous_match_lineup, lineup):
		age_days = latest_lineup_age.age_days + calculate_age_days(latest_lineup_age.date, lineup.date)
		insert_lineup_age(lineup, age_days, latest_lineup_age.matches_together + 1)
	else:
		# 2(d) Find older lineup
		older_lineup = find_older_lineup(lineup)
		if older_lineup and is_matching_lineup(older_lineup, lineup):
			age_days = latest_lineup_age.age_days + calculate_age_days(latest_lineup_age.date, lineup.date)
			insert_lineup_age(lineup, age_days, latest_lineup_age.matches_together +1)
		else:
			# 2(e) No matching lineup found
			insert_lineup_age(lineup, 0, 1)

def is_matching_lineup(lineup1, lineup2):
	players1 = [lineup1.player1_id, lineup1.player2_id, lineup1.player3_id, lineup1.player4_id, lineup1.player5_id]
	players2 = [lineup2.player1_id, lineup2.player2_id, lineup2.player3_id, lineup2.player4_id, lineup2.player5_id]

	for player in players1:
		if player not in players2:
			return False

	return True

def calculate_age_days(start_date, end_date):
	start = datetime.strptime(start_date.split(' ')[0], '%Y-%m-%d')
	end = datetime.strptime(end_date.split(' ')[0], '%Y-%m-%d')
	difference = (end - start).days

	return difference

def insert_lineup_age(lineup, age_days, matches_together):
	try:
		session.add(LineupAge(
			   team_id = lineup.team_id,
			   date = lineup.date,
			   match_id = lineup.match_id,
			   age_days = age_days,
			   matches_together = matches_together
		))
		session.commit()
	except Exception as e:
		session.rollback()
		print(f"Error inserting records: {e}")

def find_older_lineup(current_lineup):
	# Convert the current lineup date to a datetime object
	current_date = datetime.strptime(current_lineup.date, '%Y-%m-%d %H:%M')

	# Calculate the date one month earlier
	one_month_earlier = current_date - timedelta(days=30)

	# Query to find an older lineup
	older_lineup = session.query(Lineup)\
						  .filter(Lineup.team_id == current_lineup.team_id)\
						  .filter(Lineup.date < one_month_earlier.strftime('%Y-%m-%d'))\
						  .order_by(Lineup.date.desc())\
						  .first()

	return older_lineup

if __name__ == "__main__":
	process_lineups()