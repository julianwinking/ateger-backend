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
                
            # Initialize the GPT analysis dictionary with proper nested structure
            if not hasattr(teaser, 'gpt_analysis') or teaser.gpt_analysis is None:
                teaser.gpt_analysis = {
                    "Teaser Summary": {
                        "Bias conscious": {}
                    },
                    "Outside-In View": {
                        "Core Status Quo": {
                            "Company Profile": {},
                            "Customer & Demand Analysis": {},
                            "Industry & Competitive Landscape": {},
                            "Commercial Strategy": {},
                            "Talent Development": {}
                        },
                        "Future": {
                            "Market Growth and Trends": {},
                            "Breadth Analysis": {},
                            "Forces Analysis": {},
                            "Moat Identification": {},
                            "Key Value Creation Drivers & Risks": {}
                        },
                        "Shareholder Friendliness": {
                            "Compensation & Ownership Structure": {},
                            "Related-party transactions": {},
                            "Share repurchases": {},
                            "Dividends": {}
                        }
                    },
                    "Context Basis": {
                        "Risk": {},
                        "Synergies": {},
                        "Investment Criteria": {},
                        "Exit perspective": {},
                        "Graveyard": {}
                    }
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
                    
                    # Store the results in the appropriate locations
                    for block_id, result in batch_results.items():
                        if result:
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
                                        # Store the analysis content directly
                                        if isinstance(current, dict):
                                            current[key] = result
                                        else:
                                            print(f"Warning: Cannot store result for {key} - parent is not a dict")
                                
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
                # Increase max tokens to accommodate multiple analyses
                "max_tokens": min(4000, 1000 + (len(blocks_to_process) * 500))
            }
            
            print(f"Sending batched analysis request for {len(blocks_to_process)} sections...")
            
            # Call the OpenAI API
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=120  # Increased timeout for batch processing
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
            
            # Parse the sections from the response
            for i, (block_id, block_name, block_description) in enumerate(blocks_to_process):
                # Look for the section marker
                section_marker = f"---SECTION: {block_name}---"
                alt_section_marker = f"SECTION: {block_name}"
                
                # Try to find either marker format
                start_idx = full_response.find(section_marker)
                if start_idx == -1:
                    start_idx = full_response.find(alt_section_marker)
                    if start_idx != -1:
                        # Account for different marker length
                        start_idx += len(alt_section_marker)
                    else:
                        # Try more flexible matching
                        for marker in [f"Section: {block_name}", block_name + ":", f"{i+1}. {block_name}"]:
                            start_idx = full_response.find(marker)
                            if start_idx != -1:
                                start_idx += len(marker)
                                break
                else:
                    start_idx += len(section_marker)
                
                if start_idx == -1:
                    print(f"Could not find section for {block_name} in the response")
                    continue
                
                # Find the end of this section (start of next section or end of response)
                end_idx = -1
                for next_block_id, next_block_name, _ in blocks_to_process[i+1:]:
                    for marker in [f"---SECTION: {next_block_name}---", f"SECTION: {next_block_name}",
                                 f"Section: {next_block_name}", next_block_name + ":"]:
                        end_idx = full_response.find(marker, start_idx)
                        if end_idx != -1:
                            break
                    if end_idx != -1:
                        break
                
                # Extract section content
                if end_idx != -1:
                    section_content = full_response[start_idx:end_idx].strip()
                else:
                    section_content = full_response[start_idx:].strip()
                
                results[block_id] = section_content
                print(f"Successfully extracted analysis for {block_name}")
            
            return results
            
        except Exception as e:
            print(f"Error in batch GPT analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return results
            
    async def _generate_report(self, teaser_id: int) -> Optional[str]:
        """
        Generate a PDF report from the teaser GPT analysis
        
        Args:
            teaser_id: The ID of the teaser to create a report for
            
        Returns:
            str: Path to the generated report file, or None if generation failed
        """
        try:
            # Fetch the teaser from the database
            teaser = self.db.query(Teaser).filter(Teaser.id == teaser_id).first()
            if not teaser:
                print(f"Teaser with ID {teaser_id} not found")
                return None
                
            # If we don't have GPT analysis, we can't generate a report
            if not teaser.gpt_analysis:
                print(f"No GPT analysis available for teaser {teaser_id}")
                return None
            
            # Print debug information about the structure
            print(f"Generating report for teaser {teaser_id} with analysis structure:")
            print(f"GPT analysis keys: {list(teaser.gpt_analysis.keys())}")
            for section_name, section_data in teaser.gpt_analysis.items():
                print(f"  Section: {section_name}")
                if isinstance(section_data, dict):
                    for subsection_name, subsection_data in section_data.items():
                        print(f"    Subsection: {subsection_name}")
                        if isinstance(subsection_data, dict):
                            print(f"      Contains {len(subsection_data)} sub-subsections: {list(subsection_data.keys())}")
                
            # Define the report filename
            report_filename = f"reports/teaser_{teaser_id}_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            # Create the PDF document
            doc = SimpleDocTemplate(
                report_filename,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            heading_style = styles['Heading1']
            subheading_style = styles['Heading2']
            subsubheading_style = styles['Heading3']
            normal_style = styles['Normal']
            
            # Make normal text slightly larger and add spacing after paragraphs
            normal_style.fontSize = 11
            normal_style.spaceAfter = 6
            
            # Container for the 'Flowable' objects
            elements = []
            
            # Add the title
            elements.append(Paragraph(f"Teaser Analysis: {teaser.filename}", title_style))
            elements.append(Spacer(1, 12))
            
            # Add the date
            elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y')}", normal_style))
            elements.append(Spacer(1, 24))
            
            # Process each section of the GPT analysis
            for section_name, section_data in teaser.gpt_analysis.items():
                print(f"Processing section: {section_name}")
                
                # Main section header
                elements.append(Paragraph(section_name, heading_style))
                elements.append(Spacer(1, 12))
                
                # Check if section_data is empty
                if not section_data:
                    elements.append(Paragraph("No data available for this section.", normal_style))
                    elements.append(Spacer(1, 12))
                    continue
                
                # Special handling for Teaser Summary section
                if section_name == "Teaser Summary":
                    print(f"  Teaser Summary contains: {list(section_data.keys())}")
                    
                    # Direct content or nested structure handling
                    for subsection, content in section_data.items():
                        print(f"    Processing subsection: {subsection}")
                        if content:  # Only add if content exists
                            elements.append(Paragraph(subsection, subheading_style))
                            elements.append(Paragraph(content, normal_style))
                            elements.append(Spacer(1, 12))
                        else:
                            print(f"    Skipping empty content for: {subsection}")
                
                # Handle nested dictionaries like Outside-In View
                elif isinstance(section_data, dict):
                    print(f"  Section {section_name} contains subsections: {list(section_data.keys())}")
                    
                    # Flag to track if any content was added in this section
                    section_has_content = False
                    
                    for subsection_name, subsection_data in section_data.items():
                        print(f"    Processing subsection: {subsection_name}")
                        subsection_has_content = False
                        
                        # Add subsection header
                        elements.append(Paragraph(subsection_name, subheading_style))
                        elements.append(Spacer(1, 6))
                        
                        if isinstance(subsection_data, dict):
                            print(f"      Subsection {subsection_name} contains: {list(subsection_data.keys())}")
                            
                            # Process sub-subsections
                            for subsubsection_name, content in subsection_data.items():
                                print(f"        Processing sub-subsection: {subsubsection_name}")
                                if content:  # Only add if content exists
                                    elements.append(Paragraph(subsubsection_name, subsubheading_style))
                                    elements.append(Paragraph(str(content), normal_style))
                                    elements.append(Spacer(1, 12))
                                    subsection_has_content = True
                                    section_has_content = True
                                else:
                                    print(f"        Skipping empty content in: {subsubsection_name}")
                            
                        elif subsection_data:  # Direct content in subsection
                            elements.append(Paragraph(str(subsection_data), normal_style))
                            elements.append(Spacer(1, 12))
                            subsection_has_content = True
                            section_has_content = True
                        
                        # If subsection had no content, add a placeholder
                        if not subsection_has_content:
                            elements.append(Paragraph("No data available for this subsection.", normal_style))
                            elements.append(Spacer(1, 12))
                    
                    # If section had no content at all, add a clear notice
                    if not section_has_content:
                        elements.append(Paragraph("No analysis content available for this section.", normal_style))
                        elements.append(Spacer(1, 12))
                
                # Handle direct content in the section (e.g., Context Basis items)
                elif isinstance(section_data, dict):
                    # Process each of the subsections (which might be direct content)
                    for subsection_name, content in section_data.items():
                        elements.append(Paragraph(subsection_name, subheading_style))
                        if content:
                            elements.append(Paragraph(str(content), normal_style))
                        else:
                            elements.append(Paragraph("No data available for this subsection.", normal_style))
                        elements.append(Spacer(1, 12))
                
                # Handle direct content (rare case)
                elif section_data:
                    elements.append(Paragraph(str(section_data), normal_style))
                    elements.append(Spacer(1, 12))
                else:
                    elements.append(Paragraph("No content available.", normal_style))
                    elements.append(Spacer(1, 12))
                
                # Add some space after each main section
                elements.append(Spacer(1, 24))
            
            # Add debug information to help troubleshoot
            elements.append(Paragraph("Debug Information", heading_style))
            elements.append(Paragraph(f"Total sections processed: {len(teaser.gpt_analysis)}", normal_style))
            
            # More detailed debug info
            section_details = []
            for section, data in teaser.gpt_analysis.items():
                if isinstance(data, dict):
                    subsections = []
                    for subsection, subdata in data.items():
                        if isinstance(subdata, dict):
                            subsubsections = list(subdata.keys())
                            subsections.append(f"{subsection} ({len(subsubsections)} items)")
                        else:
                            subsections.append(subsection)
                    section_details.append(f"{section}: {', '.join(subsections)}")
                else:
                    section_details.append(f"{section}")
            
            elements.append(Paragraph(f"GPT analysis structure: {', '.join(section_details)}", normal_style))
            
            # Build the PDF
            doc.build(elements)
            
            print(f"Generated report at {report_filename}")
            return report_filename
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            import traceback
            traceback.print_exc()
            return None