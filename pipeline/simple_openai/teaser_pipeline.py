import os
from typing import Dict, List, Optional, Any
import datetime
import json
import aiohttp
import asyncio

from pipeline.base import Pipeline
from models import Teaser, TeaserStatus
from parser.pdf_parser import PDFParser
from parser.nlp import NLPProcessor
from document_generator.screening_report import generate_screening_report

class SimpleOpenAIPipeline(Pipeline):
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
                
            # Process only if we have text and an API key
            if teaser.extracted_text and self.openai_api_key:
                print(f"Starting selective GPT analysis for teaser {teaser.id}")
                
                # If no specific blocks are selected, use all available blocks
                blocks_to_process = selected_blocks if selected_blocks else list(self.building_blocks.keys())
                print(f"Processing blocks: {blocks_to_process}")
                
                # Only process blocks that exist in building_blocks
                valid_blocks = [block_id for block_id in blocks_to_process if block_id in self.building_blocks]
                if valid_blocks:
                    # Process all blocks in batched mode to save tokens
                    batch_results = await self._analyze_multiple_blocks_with_gpt(
                        text=teaser.extracted_text,
                        blocks_to_process=[
                            (block_id, 
                             self.building_blocks[block_id]['name'], 
                             self.building_blocks[block_id]['description'])
                            for block_id in valid_blocks
                        ]
                    )
                    
                    # Initialize the GPT analysis dictionary - keep it flat for simplicity
                    teaser.gpt_analysis = {}
                    
                    # Store all section results directly in the top level of gpt_analysis
                    processed_sections_count = 0
                    for block_id, block_name in [(b, self.building_blocks[b]['name']) for b in valid_blocks]:
                        # Check if we have content for this block_id
                        if block_id in batch_results and batch_results[block_id]:
                            content = batch_results[block_id]
                            processed_sections_count += 1
                            print(f"✅ Storing analysis for {block_name} ({len(content)} chars)")
                            
                            # Store content directly with the section name as key
                            teaser.gpt_analysis[block_name] = content
                        else:
                            print(f"⚠️ No content found for {block_name}")
                            # Store empty content
                            teaser.gpt_analysis[block_name] = ""
                    
                    print(f"Successfully processed and stored {processed_sections_count} out of {len(valid_blocks)} blocks")
                    
                    # Save the updated gpt_analysis to the database
                    self.db.commit()
                    print(f"GPT analysis completed and stored for teaser {teaser.id}")
                else:
                    print("No valid blocks selected for processing")
            else:
                if not teaser.extracted_text:
                    print(f"Skipping GPT analysis for teaser {teaser.id} - No extracted text available")
                if not self.openai_api_key:
                    print(f"Skipping GPT analysis for teaser {teaser.id} - No API key available")
            
            # Generate the report using the generate_screening_report function
            report_path = await generate_screening_report(teaser)
            if report_path:
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
            import traceback
            traceback.print_exc()
            # Mark as error in case of exception
            try:
                teaser = self.db.query(Teaser).filter(Teaser.id == teaser_id).first()
                if teaser:
                    teaser.status = TeaserStatus.ERROR
                    self.db.commit()
            except:
                pass
            return False

    async def _analyze_multiple_blocks_with_gpt(self, text: str, blocks_to_process: List[tuple]) -> Dict[str, str]:
        """
        Analyze multiple sections of the teaser using GPT with a shared context
        to optimize token usage by sending the teaser content only once.
        
        Args:
            text: The teaser text to analyze (sent only once)
            blocks_to_process: List of tuples containing (block_id, block_name, block_description)
            
        Returns:
            Dict[str, str]: Dictionary mapping block_ids to their analysis results
        """
        if not self.openai_api_key or not blocks_to_process:
            return {}

        results = {}
        try:
            # Create a comprehensive prompt that includes all sections to analyze
            # but sends the teaser text only once
            system_prompt = """Context: You are tasked with analyzing a private equity teaser document, focusing on specific sections that require detailed examination to inform investment decisions.

            Role: Act as a private equity expert with extensive knowledge of investment strategies, market trends, and financial metrics.

            Audience: Investors, analysts, or stakeholders interested in understanding the detailed components of a private equity opportunity.

            Task: For each section of the teaser document you provide, deliver a thorough analysis that explores the implications, strengths, weaknesses, and any missing information. If any section lacks relevant details, clearly indicate that.

            Visualization or output format: Text format, structured by sections for clarity.

            Format your response precisely as follows:

            ---SECTION: [Section Name]---

            [Your analysis for this section]

            ---SECTION: [Next Section Name]---

            [Your analysis for next section]

            And so on for each section requested.
            """
            
            # First message contains just the teaser text to establish context
            initial_message = f"""Here is the teaser document to analyze:

{text}

I'll now ask you to analyze specific sections of this document one by one.
            """
            
            # Second message contains the sections to analyze
            sections_message = "Please analyze the following sections:\n\n"
            for i, (block_id, block_name, block_description) in enumerate(blocks_to_process):
                sections_message += f"{i+1}. {block_name}: {block_description}\n"
            
            # Prepare API call
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_message},
                    {"role": "user", "content": sections_message}
                ],
                "temperature": 0.2,
                # Set max_tokens to a value that won't exceed the model's limits
                "max_tokens": min(4000, 1000 + (len(blocks_to_process) * 150))
            }
            
            print(f"Sending batched analysis request for {len(blocks_to_process)} sections...")
            print(f"Payload contains {sum(len(msg['content']) for msg in payload['messages'])} characters of context and instructions")
            print(f"Max tokens set to {payload['max_tokens']}")
            
            # Call the OpenAI API
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=180  # Increased timeout for batch processing
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Error from OpenAI API for batched analysis: Status {response.status}, Response: {error_text}")
                            return {}
                            
                        response_data = await response.json()
                        
                except aiohttp.ClientError as e:
                    print(f"Network error when calling OpenAI API for batched analysis: {str(e)}")
                    return {}
                except Exception as e:
                    print(f"Unexpected error in API call for batched analysis: {str(e)}")
                    return {}
            
            # Extract and parse the response
            full_response = response_data['choices'][0]['message']['content']
            print(f"Received response of {len(full_response)} characters")
            
            # Debugging: Save the full response to a file to examine it
            debug_file = f"reports/gpt_response_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(debug_file, "w") as f:
                f.write("REQUESTED BLOCKS:\n")
                for block_id, block_name, block_desc in blocks_to_process:
                    f.write(f"- {block_id}: {block_name} - {block_desc}\n")
                f.write("\n\nFULL RESPONSE:\n")
                f.write(full_response)
                
            print(f"Debug file saved to {debug_file}")
            
            # IMPROVED SECTION EXTRACTION:
            # Split the response by section markers directly
            section_splits = full_response.split("---SECTION: ")
            
            # The first split will be empty or contain non-section text, so skip it
            section_splits = section_splits[1:] if section_splits else []
            
            print(f"Found {len(section_splits)} sections in the GPT response")
            
            # Create a mapping of section names to block IDs for easy lookup
            section_name_to_block_id = {name: block_id for block_id, name, _ in blocks_to_process}
            
            # Process each section
            for section_text in section_splits:
                # Extract the section name and content
                section_parts = section_text.split("---", 1)
                if not section_parts:
                    continue
                
                # First part has the section name
                section_name_part = section_parts[0].strip()
                # Remove trailing dashes if they exist
                section_name = section_name_part.rstrip("-").strip()
                
                # The rest is the content
                content = section_parts[1].strip() if len(section_parts) > 1 else section_text
                
                # Find the block_id for this section name
                block_id = section_name_to_block_id.get(section_name)
                if block_id:
                    results[block_id] = content
                    print(f"✅ Stored analysis for '{section_name}' ({len(content)} chars)")
                else:
                    print(f"⚠️ Could not find block_id for section '{section_name}'")
                    # Try a fuzzy match
                    for name, id in section_name_to_block_id.items():
                        if section_name.lower() in name.lower() or name.lower() in section_name.lower():
                            results[id] = content
                            print(f"📌 Fuzzy matched section '{section_name}' to block '{name}'")
                            break
            
            print(f"Successfully extracted {len(results)} out of {len(blocks_to_process)} requested sections")
            return results
            
        except Exception as e:
            print(f"Error in batch GPT analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return results