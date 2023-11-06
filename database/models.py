from sqlalchemy import Column, Integer, String, Boolean, Float, Date, ForeignKey
from sqlalchemy.orm import relationship
from database.setup import Base

class MatchURL(Base):
	__tablename__ = 'match_url'

	id = Column(Integer, primary_key=True)
	url = Column(String, nullable=False)
	downloaded = Column(Boolean, default=False)
	processed = Column(Boolean, default=False)

	def __init__(self, id, url, downloaded=False, processed=False):
		self.id = id
		self.url = url
		self.downloaded = downloaded
		self.processed = processed

	def __repr__(self):
		return f"<MatchURL(ID={self.id}, URI='{self.url}', downloaded={self.downloaded}, processed={self.processed})>"

class Match(Base):
	__tablename__ = 'match'

	id = Column(Integer, primary_key=True)
	url = Column(String)
	datetime = Column(String)
	team1_id = Column(Integer)
	team2_id = Column(Integer)
	team1 = Column(String)
	team2 = Column(String)
	team1_score = Column(Integer)
	team2_score = Column(Integer)
	team1_rank = Column(Integer)
	team2_rank = Column(Integer)
	winner = Column(Integer)
	event_id = Column(Integer)
	event = Column(String)
	lan = Column(Boolean)
	best_of = Column(Integer)
	box_str = Column(String)
	veto = Column(String)

class MapURL(Base):
	__tablename__ = 'map_url'

	id = Column(Integer, primary_key=True)
	url = Column(String, nullable=False)
	downloaded = Column(Boolean, default=False)
	processed = Column(Boolean, default=False)

	def __repr__(self):
		return f"<MapURL(ID={self.id}, URI='{self.url}', downloaded={self.downloaded}, processed={self.processed})>"

class Map(Base):
	__tablename__ = 'map'

	id = Column(Integer, primary_key=True)
	match_id = Column(Integer)
	match_page_url = Column(String)
	datetime = Column(String)
	event_id = Column(Integer)
	event_name = Column(String)
	t1_id = Column(Integer)
	t1 = Column(String)
	t2_id = Column(Integer)
	t2 = Column(String)
	map_name = Column(String)
	t1_t_score = Column(Integer)
	t1_ct_score = Column(Integer)
	t2_t_score = Column(Integer)
	t2_ct_score = Column(Integer)
	overtime = Column(Integer)
	t1_ot_score = Column(Integer)
	t2_ot_score = Column(Integer)
	t1_rating = Column(Float)
	t2_rating = Column(Float)
	t1_first_kills = Column(Integer)
	t2_first_kills = Column(Integer)
	t1_clutches = Column(Integer)
	t2_clutches = Column(Integer)
	t1_round_history = Column(String)
	t2_round_history = Column(String)
	t2_ot_round_history = Column(String)
	t1_ot_round_history = Column(String)
	t1_score = Column(Integer)
	t2_score = Column(Integer)
	winner_id = Column(Integer)
	
	def __repr__(self):
		return f"<Map(ID={self.id}, date={self.datetime}, t1='{self.t1}', t2={self.t2}, map_name={self.map_name})>"

class PlayerStats(Base):
	__tablename__ = 'playerstats'

	player_id = Column(Integer, primary_key=True)
	map_id = Column(Integer, primary_key=True)
	player_name = Column(String)
	team_id = Column(Integer)
	team = Column(String)
	kills = Column(Integer)
	hs = Column(Integer)
	assists = Column(Integer)
	flashes = Column(Integer)
	deaths = Column(Integer)
	kast = Column(Float)
	adr = Column(Float)
	first_kd = Column(Integer)
	rating = Column(Float)

class Ranking(Base):
	__tablename__ = 'ranking'

	id = Column(Integer, primary_key=True)
	date = Column(Date, nullable=False)
	rank = Column(Integer, nullable=False)
	team_id = Column(Integer, nullable=False)
	team_name = Column(String, nullable=False)

class Team(Base):
	__tablename__ = 'team'

	id = Column(Integer, primary_key=True)
	name = Column(String, unique=True)

class Lineup(Base):
	__tablename__ = 'lineup'

	id = Column(Integer, primary_key=True)
	team_id = Column(Integer)
	team_name = Column(String)
	rank = Column(Integer)
	date = Column(String)
	match_id = Column(Integer)
	player1_id = Column(Integer)
	player1 = Column(String)
	player2_id = Column(Integer)
	player2 = Column(String)
	player3_id = Column(Integer)
	player3 = Column(String)
	player4_id = Column(Integer)
	player4 = Column(String)
	player5_id = Column(Integer)
	player5 = Column(String)