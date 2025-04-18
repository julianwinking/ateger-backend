"""
Script to reset the database tables
This will delete all existing data and recreate the tables with the new schema.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
from database import engine

# Drop all tables
Base.metadata.drop_all(bind=engine)

# Create all tables
Base.metadata.create_all(bind=engine)

print("Database has been reset successfully!")