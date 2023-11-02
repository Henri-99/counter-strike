from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URI = 'sqlite:///counter-strike.db'
engine = create_engine(DATABASE_URI, echo=True)

Base = declarative_base()

Session = sessionmaker(bind=engine)
session = Session()

def create_tables():
	Base.metadata.create_all(engine)