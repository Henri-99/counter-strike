import sqlite3

old_conn = sqlite3.connect('old.db')
old_cursor = old_conn.cursor()
new_conn = sqlite3.connect('counter-strike.db')
new_cursor = new_conn.cursor()
old_cursor.execute("SELECT match_id, match_page_url FROM match_url")
old_data = old_cursor.fetchall()
for row in old_data:
    new_cursor.execute("INSERT INTO match_url (id, url, downloaded, processed) VALUES (?, ?, 0, 0)", (row[0], row[1]))
new_conn.commit()
new_conn.close()

old_conn.close()
