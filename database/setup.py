from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URI = 'sqlite:///counter-strike.db'
engine = create_engine(DATABASE_URI, echo=False)

Base = declarative_base()

Session = sessionmaker(bind=engine)
session = Session()

if __name__ == "__main__":
	Base.metadata.create_all(engine)