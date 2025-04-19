import os
from typing import Dict, List, Optional, Any
import datetime
import json
import aiohttp
import asyncio  # Add import for asyncio
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from .base import Pipeline
from models import Teaser, TeaserStatus
from parser.pdf_parser import PDFParser
from parser.nlp import NLPProcessor

class TeaserProcessingPipeline(Pipeline):
    """
    Concrete implementation of the Pipeline for processing teaser documents
    """
    def __init__(self, db, nlp_processor=None):
        super().__init__(db)
        self.nlp_processor = nlp_processor or NLPProcessor()
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        # Get API key from environment variable
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Define the available building blocks
        self.building_blocks = {
            "teaser_summary": {
                "name": "Teaser Summary",
                "description": "A concise, bias-conscious summary of the key points in the teaser document."
            },
            "company_profile": {
                "name": "Company Profile",
                "description": "Products, business model, form, geography, status"
            },
            "customer_demand": {
                "name": "Customer & Demand Analysis",
                "description": "Target customers, market demand patterns"
            },
            "industry_landscape": {
                "name": "Industry & Competitive Landscape",
                "description": "Industry position, competitive analysis"
            },
            "commercial_strategy": {
                "name": "Commercial Strategy",
                "description": "Go-to-market approach, revenue strategy"
            },
            "talent_development": {
                "name": "Talent Development",
                "description": "Team structure, expertise, development plans"
            },
            "market_growth": {
                "name": "Market Growth and Trends",
                "description": "Analysis of market trajectory and trends"
            },
            "breadth_analysis": {
                "name": "Breadth Analysis",
                "description": "Assessment of market breadth, expansion potential"
            },
            "forces_analysis": {
                "name": "Forces Analysis",
                "description": "External forces affecting the business"
            },
            "moat_identification": {
                "name": "Moat Identification",
                "description": "Competitive advantages and barriers to entry"
            },
            "value_creation": {
                "name": "Key Value Creation Drivers & Risks",
                "description": "Factors driving value, associated risks"
            },
            "ownership_structure": {
                "name": "Compensation & Ownership Structure",
                "description": "Executive compensation, ownership analysis"
            },
            "related_party": {
                "name": "Related-party transactions",
                "description": "Assessment of related-party dealings"
            },
            "share_repurchases": {
                "name": "Share repurchases",
                "description": "History and strategy of share buybacks"
            },
            "dividends": {
                "name": "Dividends",
                "description": "Dividend history and policy"
            },
            "risk": {
                "name": "Risk",
                "description": "Risk assessment and mitigation strategies"
            },
            "synergies": {
                "name": "Synergies",
                "description": "Potential revenue and cost synergies"
            },
            "investment_criteria": {
                "name": "Investment Criteria",
                "description": "Key criteria for investment decisions"
            },
            "exit_perspective": {
                "name": "Exit perspective",
                "description": "Potential exit strategies and timelines"
            },
            "graveyard": {
                "name": "Graveyard",
                "description": "Failed competitors or previous attempts in this space"
            }
        }
        
        # Debug log - print if API key is available (without revealing the actual key)
        if self.openai_api_key:
            print(f"OpenAI API key is configured. First 5 chars: {self.openai_api_key[:5]}...")
        else:
            print("WARNING: OpenAI API key is not configured. GPT analysis will be skipped.")
        
    async def _run_pipeline_steps(self, teaser_id: int, selected_blocks: Optional[List[str]] = None) -> bool:
        """
        Run selected pipeline steps for teaser document processing
        
        Args:
            teaser_id: The ID of the teaser to process
            selected_blocks: Optional list of building block IDs to include in the processing
                             If None, all blocks will be processed
        """
        try:
            # Fetch the teaser from the database
            teaser = self.db.query(Teaser).filter(Teaser.id == teaser_id).first()
            if not teaser:
                print(f"Teaser with ID {teaser_id} not found")
                return False
                
            # If the teaser doesn't have entities yet, extract them
            if not teaser.entities and teaser.extracted_text:
                teaser.entities = self.nlp_processor.extract_entities(teaser.extracted_text)
                self.db.commit()
                
            # Initialize the GPT analysis dictionary if it doesn't exist
            if not hasattr(teaser, 'gpt_analysis') or teaser.gpt_analysis is None:
                teaser.gpt_analysis = {
                    "Teaser Summary": {},
                    "Outside-In View": {
                        "Core Status Quo": {},
                        "Future": {},
                        "Shareholder Friendliness": {}
                    },
                    "Context Basis": {}
                }
            
            # Process only if we have text and an API key
            if teaser.extracted_text and self.openai_api_key:
                print(f"Starting selective GPT analysis for teaser {teaser.id}")
                
                # If no specific blocks are selected, use all available blocks
                blocks_to_process = selected_blocks if selected_blocks else list(self.building_blocks.keys())
                print(f"Processing blocks: {blocks_to_process}")
                
                # Map blocks to their respective sections in the MECE structure
                section_mapping = {
                    "teaser_summary": ("Teaser Summary", "Bias conscious"),
                    
                    # Core Status Quo
                    "company_profile": ("Outside-In View", "Core Status Quo", "Company Profile"),
                    "customer_demand": ("Outside-In View", "Core Status Quo", "Customer & Demand Analysis"),
                    "industry_landscape": ("Outside-In View", "Core Status Quo", "Industry & Competitive Landscape"),
                    "commercial_strategy": ("Outside-In View", "Core Status Quo", "Commercial Strategy"),
                    "talent_development": ("Outside-In View", "Core Status Quo", "Talent Development"),
                    
                    # Future
                    "market_growth": ("Outside-In View", "Future", "Market Growth and Trends"),
                    "breadth_analysis": ("Outside-In View", "Future", "Breadth Analysis"),
                    "forces_analysis": ("Outside-In View", "Future", "Forces Analysis"),
                    "moat_identification": ("Outside-In View", "Future", "Moat Identification"),
                    "value_creation": ("Outside-In View", "Future", "Key Value Creation Drivers & Risks"),
                    
                    # Shareholder Friendliness
                    "ownership_structure": ("Outside-In View", "Shareholder Friendliness", "Compensation & Ownership Structure"),
                    "related_party": ("Outside-In View", "Shareholder Friendliness", "Related-party transactions"),
                    "share_repurchases": ("Outside-In View", "Shareholder Friendliness", "Share repurchases"),
                    "dividends": ("Outside-In View", "Shareholder Friendliness", "Dividends"),
                    
                    # Context Basis
                    "risk": ("Context Basis", "Risk"),
                    "synergies": ("Context Basis", "Synergies"),
                    "investment_criteria": ("Context Basis", "Investment Criteria"),
                    "exit_perspective": ("Context Basis", "Exit perspective"),
                    "graveyard": ("Context Basis", "Graveyard")
                }
                
                # Track rate limit status
                rate_limit_wait_time = 0
                
                # Process each selected block with an individual ChatGPT prompt
                for block_id in blocks_to_process:
                    if block_id not in self.building_blocks:
                        print(f"Warning: Unknown building block '{block_id}', skipping")
                        continue
                        
                    print(f"Processing building block: {self.building_blocks[block_id]['name']}")
                    
                    # Wait if we encountered a rate limit previously
                    if rate_limit_wait_time > 0:
                        print(f"Waiting for {rate_limit_wait_time} seconds due to rate limit...")
                        await asyncio.sleep(rate_limit_wait_time)
                        rate_limit_wait_time = 0
                    
                    # Generate focused prompt for this specific block
                    result, wait_time = await self._analyze_block_with_gpt(
                        text=teaser.extracted_text,
                        block_id=block_id,
                        block_name=self.building_blocks[block_id]['name'],
                        block_description=self.building_blocks[block_id]['description']
                    )
                    
                    # Update rate limit wait time if needed
                    if wait_time > 0:
                        rate_limit_wait_time = wait_time + 1  # Add a small buffer
                    
                    if result:
                        # Store the result in the appropriate location in the MECE structure
                        section_path = section_mapping.get(block_id)
                        if section_path:
                            # Navigate to the correct nested dictionary and update it
                            current = teaser.gpt_analysis
                            for i, key in enumerate(section_path):
                                if i < len(section_path) - 1:
                                    if key not in current:
                                        current[key] = {}
                                    current = current[key]
                                else:
                                    current[key] = result
                                    
                            # Save progress after each successful block analysis
                            self.db.commit()
                    else:
                        print(f"Failed to analyze block {block_id}")
                
                # Save the updated gpt_analysis to the database
                self.db.commit()
                print(f"GPT analysis completed for teaser {teaser.id}")
            else:
                if not teaser.extracted_text:
                    print(f"Skipping GPT analysis for teaser {teaser.id} - No extracted text available")
                if not self.openai_api_key:
                    print(f"Skipping GPT analysis for teaser {teaser.id} - No API key available")
            
            # Generate the report
            report_path = await self._generate_report(teaser_id)
            if report_path:
                # Update the teaser with the report path
                teaser.report_path = report_path
                teaser.status = TeaserStatus.COMPLETED
                self.db.commit()
                return True
            else:
                teaser.status = TeaserStatus.ERROR
                self.db.commit()
                return False
                
        except Exception as e:
            print(f"Error in pipeline steps: {str(e)}")
            # Mark as error in case of exception
            try:
                teaser = self.db.query(Teaser).filter(Teaser.id == teaser_id).first()
                if teaser:
                    teaser.status = TeaserStatus.ERROR
                    self.db.commit()
            except:
                pass
            return False

    async def _analyze_block_with_gpt(self, text: str, block_id: str, block_name: str, block_description: str) -> tuple[Optional[str], float]:
        """
        Analyze a specific section of the teaser using GPT
        
        Args:
            text: The teaser text to analyze
            block_id: The ID of the building block
            block_name: The name of the building block
            block_description: The description of what this block should contain
        
        Returns:
            Tuple[Optional[str], float]: The analysis result and wait time if rate limited (0 if no wait needed)
        """
        if not self.openai_api_key:
            return None, 0

        try:
            # Create a focused prompt for this specific section
            prompt = f"""
            You are a private equity expert analyzing a teaser document.
            
            Focus ONLY on the following section:
            
            {block_name}: {block_description}
            
            Based on the teaser document below, provide a detailed analysis for ONLY this section.
            Be specific and thorough in your analysis, focusing exclusively on the aspects mentioned in the description.
            If information for this section is not available in the document, indicate this with "Information not available in the teaser."
            
            Output your response as plain text without any formatting.
            
            Teaser Document:
            {text}
            """

            # Call the OpenAI API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4-turbo", # Using the latest available model
                "messages": [
                    {"role": "system", "content": "You are a private equity expert analyzing teaser documents."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2, # Lower temperature for more consistent, analytical output
                "max_tokens": 1500 # Enough tokens for a detailed section analysis
            }
            
            print(f"Analyzing {block_name} with GPT...")
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions", 
                        headers=headers, 
                        json=payload,
                        timeout=60  # Add a timeout of 60 seconds
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Error from OpenAI API for {block_name}: Status {response.status}, Response: {error_text}")
                            
                            # Check for rate limit errors
                            try:
                                error_data = json.loads(error_text)
                                if response.status == 429 and "error" in error_data:
                                    error_msg = error_data["error"].get("message", "")
                                    # Extract wait time from error message
                                    if "try again in" in error_msg:
                                        # Extract the wait time from "Please try again in X.XXXs"
                                        wait_time_str = error_msg.split("try again in")[1].split("s.")[0].strip()
                                        try:
                                            wait_time = float(wait_time_str)
                                            return None, wait_time
                                        except ValueError:
                                            # If we can't parse the wait time, use a default
                                            return None, 15.0
                            except:
                                pass
                            
                            return None, 0
                        
                        response_data = await response.json()
                
                except aiohttp.ClientError as e:
                    print(f"Network error when calling OpenAI API for {block_name}: {str(e)}")
                    return None, 0
                except Exception as e:
                    print(f"Unexpected error in API call for {block_name}: {str(e)}")
                    return None, 0
                    
            # Extract the text response from the API
            gpt_response = response_data['choices'][0]['message']['content']
            print(f"Successfully analyzed {block_name} section")
            
            return gpt_response.strip(), 0
                
        except Exception as e:
            print(f"Error in GPT analysis for {block_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, 0