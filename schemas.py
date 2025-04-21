from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class TeaserStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class TeaserBase(BaseModel):
    filename: str
    
class TeaserCreate(TeaserBase):
    pass

class Entity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int

class TeaserResponse(TeaserBase):
    id: int
    extracted_text: Optional[str] = None
    entities: Optional[Dict[str, List[Entity]]] = None
    gpt_analysis: Optional[Dict[str, Any]] = None
    status: TeaserStatus = TeaserStatus.PENDING
    report_path: Optional[str] = None
    created_at: datetime
    
    class Config:
        orm_mode = True
        from_attributes = True

class TeaserList(BaseModel):
    teasers: List[TeaserResponse]

class TeaserProcessRequest(BaseModel):
    building_blocks: List[str]