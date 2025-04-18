from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from database import Base

class TeaserStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class Teaser(Base):
    __tablename__ = "teasers"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    extracted_text = Column(Text, nullable=True)
    entities = Column(JSON, nullable=True)
    gpt_analysis = Column(JSON, nullable=True)  # Store the structured MECE analysis from GPT
    status = Column(Enum(TeaserStatus), default=TeaserStatus.PENDING, nullable=False)
    report_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())