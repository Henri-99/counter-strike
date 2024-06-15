import sqlite3

# Connect to the new database (playerstats)
conn = sqlite3.connect('../counter-strike.db')  # Update with your new database file path
cursor = conn.cursor()

try:
    # Fetch all records from the old table
    cursor.execute("SELECT player_id, map_id, player_name, team_id, team, kills, hs, assists, flashes, deaths, kast, adr, first_kd, rating FROM playerstats_old")
    old_records = cursor.fetchall()

    for old_record in old_records:
        # Insert each record into the new table
        cursor.execute("INSERT INTO playerstats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", old_record)

    # Commit the changes to the new database
    conn.commit()

except sqlite3.Error as e:
    # Handle any exceptions that may occur during the migration
    print(f"An error occurred during migration: {str(e)}")
    conn.rollback()

finally:
    # Close database connections
    conn.close()
    conn.close()
