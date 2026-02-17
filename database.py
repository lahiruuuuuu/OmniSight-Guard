from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class SafetyViolation(Base):
    __tablename__ = 'violations'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    violation_type = Column(String) 
    confidence = Column(Float)

# Setup database
engine = create_engine('sqlite:///omnisight_audit.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()