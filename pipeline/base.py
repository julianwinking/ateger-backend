import os
from typing import Dict, List, Optional, Any
from models import Teaser, TeaserStatus
from sqlalchemy.orm import Session
from abc import ABC, abstractmethod

class Pipeline(ABC):
    """Base abstract class for processing pipelines"""
    
    def __init__(self, db: Session):
        self.db = db
    
    @abstractmethod
    async def _run_pipeline_steps(self, teaser_id: int, selected_blocks: Optional[List[str]] = None) -> bool:
        """Execute the pipeline steps on a teaser"""
        pass
    
    async def _generate_report(self, teaser_id: int) -> Optional[str]:
        """Generate a report based on the processed data"""
        pass
    
    async def process(self, teaser_id: int, selected_blocks: Optional[List[str]] = None) -> bool:
        """
        Process a teaser through the pipeline
        
        Args:
            teaser_id: The ID of the teaser to process
            selected_blocks: Optional list of building block IDs to include in the processing
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Run the pipeline steps
            success = await self._run_pipeline_steps(teaser_id, selected_blocks)
            
            if success:
                # Generate the report
                report_path = await self._generate_report(teaser_id)
                
                # If report generation failed, mark as error
                if not report_path:
                    return False
                
                return True
            
            return False
        except Exception as e:
            print(f"Error in pipeline processing: {str(e)}")
            return False