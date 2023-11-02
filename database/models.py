from sqlalchemy import Column, Integer, String, Boolean
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
