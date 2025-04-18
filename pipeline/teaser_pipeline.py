import os
from typing import Dict, List, Optional, Any
import datetime
import json
import aiohttp
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from .base import Pipeline
from models import Teaser
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
        
        # Debug log - print if API key is available (without revealing the actual key)
        if self.openai_api_key:
            print(f"OpenAI API key is configured. First 5 chars: {self.openai_api_key[:5]}...")
        else:
            print("WARNING: OpenAI API key is not configured. GPT analysis will be skipped.")
        
    async def _run_pipeline_steps(self, teaser: Teaser) -> bool:
        """
        Run all pipeline steps for teaser document processing
        """
        try:
            # For now, this is a placeholder where you would implement
            # the actual GPT or other processing steps
            
            # If the teaser doesn't have extracted text yet, we might need to extract it
            if not teaser.extracted_text:
                # This would need the original PDF file to be available
                # In a complete implementation, you'd store the PDF and access it here
                pass
                
            # If the teaser doesn't have entities yet, extract them
            if not teaser.entities and teaser.extracted_text:
                teaser.entities = self.nlp_processor.extract_entities(teaser.extracted_text)
                self.db.commit()
                
            # Call GPT analysis if we have an API key and extracted text
            if teaser.extracted_text:
                if self.openai_api_key:
                    print(f"Starting GPT analysis for teaser {teaser.id}")
                    mece_analysis = await self._analyze_with_gpt(teaser.extracted_text)
                    
                    # Store the analysis in the teaser object under gpt_analysis attribute
                    if mece_analysis:
                        print(f"GPT analysis completed successfully for teaser {teaser.id}")
                        teaser.gpt_analysis = mece_analysis
                        self.db.commit()
                    else:
                        print(f"GPT analysis failed for teaser {teaser.id}")
                else:
                    print(f"Skipping GPT analysis for teaser {teaser.id} - No API key available")
            else:
                print(f"Skipping GPT analysis for teaser {teaser.id} - No extracted text available")
            
            return True
            
        except Exception as e:
            print(f"Error in pipeline steps: {str(e)}")
            return False

    async def _analyze_with_gpt(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze the teaser text using GPT to generate structured content following MECE framework
        Returns a dictionary with sections matching the MECE structure
        """
        if not self.openai_api_key:
            print("No OpenAI API key provided. Skipping GPT analysis.")
            return None

        try:
            # Structure the MECE framework as a prompt
            prompt = f"""
            You are a private equity expert tasked with analyzing a teaser document. 
            Extract information from the document following the MECE (Mutually Exclusive, Collectively Exhaustive) framework.
            
            Analyze the document below and provide a structured analysis in JSON format with the following sections:
            
            1. Teaser Summary (Bias conscious):
               A concise, bias-conscious summary of the key points in the teaser document.
            
            2. Outside-In View:
               a. Core Status Quo:
                  - Company Profile: Products, business model, form, geography, status
                  - Customer & Demand Analysis: Target customers, market demand patterns
                  - Industry & Competitive Landscape: Industry position, competitive analysis
                  - Commercial Strategy: Go-to-market approach, revenue strategy
                  - Talent Development: Team structure, expertise, development plans
               
               b. Future:
                  - Market Growth and Trends: Analysis of market trajectory and trends
                  - Breadth Analysis: Assessment of market breadth, expansion potential
                  - Forces Analysis: External forces affecting the business
                  - Moat Identification: Competitive advantages and barriers to entry
                  - Key Value Creation Drivers & Risks: Factors driving value, associated risks
               
               c. Shareholder Friendliness:
                  - Compensation & Ownership Structure: Executive compensation, ownership analysis
                  - Related-party transactions: Assessment of related-party dealings
                  - Share repurchases: History and strategy of share buybacks
                  - Dividends: Dividend history and policy
            
            3. Context Basis:
               - Risk: Risk assessment and mitigation strategies
               - Synergies: Potential revenue and cost synergies
               - Investment Criteria: Key criteria for investment decisions
               - Exit perspective: Potential exit strategies and timelines
               - Graveyard: Failed competitors or previous attempts in this space
            
            Format your response as valid JSON that can be parsed programmatically.
            Always be clear about what is factual information from the document versus your analysis or interpretation.
            If information for a section is not available in the document, indicate this with "Information not available in the teaser."
            
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
                "max_tokens": 4000 # Ensure we have enough tokens for a comprehensive analysis
            }
            
            # Debug headers (without showing full API key)
            print(f"API Request Headers: Content-Type: {headers['Content-Type']}, Authorization: Bearer sk-...")
            print(f"API Request Payload: {json.dumps(payload, indent=2)[:200]}...")
            
            async with aiohttp.ClientSession() as session:
                print("Sending request to OpenAI API...")
                try:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions", 
                        headers=headers, 
                        json=payload,
                        timeout=60  # Add a timeout of 60 seconds
                    ) as response:
                        print(f"OpenAI API Response Status: {response.status}")
                        
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Error from OpenAI API: Status {response.status}, Response: {error_text}")
                            return None
                        
                        response_data = await response.json()
                        print("Successfully received response from OpenAI API")
                
                except aiohttp.ClientError as e:
                    print(f"Network error when calling OpenAI API: {str(e)}")
                    return None
                except Exception as e:
                    print(f"Unexpected error in API call: {str(e)}")
                    return None
                    
            # Extract the text response from the API
            gpt_response = response_data['choices'][0]['message']['content']
            
            # Parse the JSON response
            try:
                # The response might have markdown formatting with ```json ... ``` 
                # Strip that out if present
                if "```json" in gpt_response:
                    json_start = gpt_response.find("```json") + 7
                    json_end = gpt_response.find("```", json_start)
                    json_str = gpt_response[json_start:json_end].strip()
                    mece_analysis = json.loads(json_str)
                else:
                    # Try to parse the entire response as JSON
                    mece_analysis = json.loads(gpt_response)
                
                print("Successfully parsed JSON response from GPT")
                return mece_analysis
            
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from GPT response: {e}")
                print(f"Response (first 200 chars): {gpt_response[:200]}...")
                return None
                
        except Exception as e:
            print(f"Error in GPT analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    async def _generate_report(self, teaser: Teaser) -> Optional[str]:
        """
        Generate a PDF report based on the teaser's processed data following MECE structure
        (Mutually Exclusive, Collectively Exhaustive)
        """
        try:
            report_filename = f"teaser_{teaser.id}_report.pdf"
            report_path = os.path.join("reports", report_filename)
            
            # Create a PDF report using ReportLab
            doc = SimpleDocTemplate(
                report_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            # Add custom styles for MECE structure
            styles.add(styles['Heading1'].clone('SectionTitle', fontSize=16, spaceAfter=12, spaceBefore=12, textColor=colors.darkblue))
            styles.add(styles['Heading2'].clone('SubsectionTitle', fontSize=14, spaceAfter=6, spaceBefore=6))
            styles.add(styles['Heading3'].clone('SubSubsectionTitle', fontSize=12, spaceAfter=6, spaceBefore=6, fontName='Helvetica-Bold'))
            styles.add(styles['Normal'].clone('SectionText', fontSize=10))
            
            elements = []
            
            # Title and basic info
            title = Paragraph(f"Teaser Analysis Report: {teaser.filename}", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            date_text = Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
            elements.append(date_text)
            elements.append(Spacer(1, 24))
            
            # Check if we have GPT analysis
            gpt_analysis = getattr(teaser, 'gpt_analysis', None)
            
            # Debug the structure of gpt_analysis
            if gpt_analysis:
                print(f"GPT analysis structure for teaser {teaser.id}: {list(gpt_analysis.keys())}")
            else:
                print(f"No GPT analysis found for teaser {teaser.id}")
                
            # Section 1: Teaser Summary (Bias conscious)
            section_title = Paragraph("1. Teaser Summary (Bias conscious)", styles['SectionTitle'])
            elements.append(section_title)
            elements.append(Spacer(1, 12))
            
            if gpt_analysis and 'Teaser Summary' in gpt_analysis and 'Bias conscious' in gpt_analysis['Teaser Summary']:
                summary = gpt_analysis['Teaser Summary']['Bias conscious']
                summary_text = Paragraph(summary, styles['SectionText'])
            else:
                summary_text = Paragraph("This section provides a bias-conscious summary of the teaser document, highlighting key points while acknowledging potential biases in the presentation of information.", styles['SectionText'])
            elements.append(summary_text)
            elements.append(Spacer(1, 24))
            
            # Section 2: Outside-In View
            section_title = Paragraph("2. Outside-In View", styles['SectionTitle'])
            elements.append(section_title)
            elements.append(Spacer(1, 12))
            
            # --- Core Status Quo subsections ---
            subsection_title = Paragraph("Core Status Quo", styles['SubsectionTitle'])
            elements.append(subsection_title)
            elements.append(Spacer(1, 6))
            
            outside_in_view = gpt_analysis.get('Outside-In View', {}) if gpt_analysis else {}
            core_status_quo = outside_in_view.get('Core Status Quo', {}) if outside_in_view else {}
            
            # Company Profile
            elements.append(Paragraph("Company Profile", styles['SubSubsectionTitle']))
            if core_status_quo and 'Company Profile' in core_status_quo:
                company_profile = core_status_quo['Company Profile']
                elements.append(Paragraph(company_profile, styles['SectionText']))
            else:
                elements.append(Paragraph("Product, Business Model, Form, Geography, Status", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Customer & Demand Analysis
            elements.append(Paragraph("Customer & Demand Analysis", styles['SubSubsectionTitle']))
            if core_status_quo and 'Customer & Demand Analysis' in core_status_quo:
                customer_demand = core_status_quo['Customer & Demand Analysis']
                elements.append(Paragraph(customer_demand, styles['SectionText']))
            else:
                elements.append(Paragraph("Analysis of target customers and market demand patterns.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Industry & Competitive Landscape
            elements.append(Paragraph("Industry & Competitive Landscape", styles['SubSubsectionTitle']))
            if core_status_quo and 'Industry & Competitive Landscape' in core_status_quo:
                industry_landscape = core_status_quo['Industry & Competitive Landscape']
                elements.append(Paragraph(industry_landscape, styles['SectionText']))
            else:
                elements.append(Paragraph("Overview of the industry and competitive positioning.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Commercial Strategy
            elements.append(Paragraph("Commercial Strategy", styles['SubSubsectionTitle']))
            if core_status_quo and 'Commercial Strategy' in core_status_quo:
                commercial_strategy = core_status_quo['Commercial Strategy']
                elements.append(Paragraph(commercial_strategy, styles['SectionText']))
            else:
                elements.append(Paragraph("Go-to-market approach and revenue strategy.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Talent Development
            elements.append(Paragraph("Talent Development", styles['SubSubsectionTitle']))
            if core_status_quo and 'Talent Development' in core_status_quo:
                talent_development = core_status_quo['Talent Development']
                elements.append(Paragraph(talent_development, styles['SectionText']))
            else:
                elements.append(Paragraph("Team structure, expertise, and development plans.", styles['SectionText']))
            elements.append(Spacer(1, 12))
            
            # --- Future subsections ---
            subsection_title = Paragraph("Future", styles['SubsectionTitle'])
            elements.append(subsection_title)
            elements.append(Spacer(1, 6))
            
            future = outside_in_view.get('Future', {}) if outside_in_view else {}
            
            # Market Growth and Trends
            elements.append(Paragraph("Market Growth and Trends", styles['SubSubsectionTitle']))
            if future and 'Market Growth and Trends' in future:
                market_growth = future['Market Growth and Trends']
                elements.append(Paragraph(market_growth, styles['SectionText']))
            else:
                elements.append(Paragraph("Analysis of market trajectory and emerging trends.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Breadth Analysis
            elements.append(Paragraph("Breadth Analysis", styles['SubSubsectionTitle']))
            if future and 'Breadth Analysis' in future:
                breadth_analysis = future['Breadth Analysis']
                elements.append(Paragraph(breadth_analysis, styles['SectionText']))
            else:
                elements.append(Paragraph("Assessment of market breadth and potential for expansion.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Forces Analysis
            elements.append(Paragraph("Forces Analysis", styles['SubSubsectionTitle']))
            if future and 'Forces Analysis' in future:
                forces_analysis = future['Forces Analysis']
                elements.append(Paragraph(forces_analysis, styles['SectionText']))
            else:
                elements.append(Paragraph("Analysis of external forces affecting the business.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Moat Identification
            elements.append(Paragraph("Moat Identification", styles['SubSubsectionTitle']))
            if future and 'Moat Identification' in future:
                moat_identification = future['Moat Identification']
                elements.append(Paragraph(moat_identification, styles['SectionText']))
            else:
                elements.append(Paragraph("Competitive advantages and barriers to entry.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Key Value Creation Drivers & Risks
            elements.append(Paragraph("Key Value Creation Drivers & Risks", styles['SubSubsectionTitle']))
            if future and 'Key Value Creation Drivers & Risks' in future:
                value_creation = future['Key Value Creation Drivers & Risks']
                elements.append(Paragraph(value_creation, styles['SectionText']))
            else:
                elements.append(Paragraph("Factors that will drive value creation and associated risks.", styles['SectionText']))
            elements.append(Spacer(1, 12))
            
            # --- Shareholder Friendliness subsections ---
            subsection_title = Paragraph("Shareholder Friendliness", styles['SubsectionTitle'])
            elements.append(subsection_title)
            elements.append(Spacer(1, 6))
            
            shareholder = outside_in_view.get('Shareholder Friendliness', {}) if outside_in_view else {}
            
            # Compensation & Ownership Structure
            elements.append(Paragraph("Compensation & Ownership Structure", styles['SubSubsectionTitle']))
            if shareholder and 'Compensation & Ownership Structure' in shareholder:
                compensation = shareholder['Compensation & Ownership Structure']
                elements.append(Paragraph(compensation, styles['SectionText']))
            else:
                elements.append(Paragraph("Analysis of executive compensation and ownership structure.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Related-party transactions
            elements.append(Paragraph("Related-party transactions", styles['SubSubsectionTitle']))
            if shareholder and 'Related-party transactions' in shareholder:
                related_party = shareholder['Related-party transactions']
                elements.append(Paragraph(related_party, styles['SectionText']))
            else:
                elements.append(Paragraph("Assessment of related-party transactions and potential conflicts of interest.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Share repurchases
            elements.append(Paragraph("Share repurchases", styles['SubSubsectionTitle']))
            if shareholder and 'Share repurchases' in shareholder:
                share_repurchases = shareholder['Share repurchases']
                elements.append(Paragraph(share_repurchases, styles['SectionText']))
            else:
                elements.append(Paragraph("History and strategy of share repurchase programs.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Dividends
            elements.append(Paragraph("Dividends", styles['SubSubsectionTitle']))
            if shareholder and 'Dividends' in shareholder:
                dividends = shareholder['Dividends']
                elements.append(Paragraph(dividends, styles['SectionText']))
            else:
                elements.append(Paragraph("Dividend history and policy.", styles['SectionText']))
            elements.append(Spacer(1, 24))
            
            # Section 3: Context Basis
            section_title = Paragraph("3. Context Basis", styles['SectionTitle'])
            elements.append(section_title)
            elements.append(Spacer(1, 12))
            
            context_basis = gpt_analysis.get('Context Basis', {}) if gpt_analysis else {}
            
            # Risk
            elements.append(Paragraph("Risk", styles['SubSubsectionTitle']))
            if context_basis and 'Risk' in context_basis:
                risk = context_basis['Risk']
                elements.append(Paragraph(risk, styles['SectionText']))
            else:
                elements.append(Paragraph("Comprehensive risk assessment and mitigation strategies.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Synergies (Revenue, Cost)
            elements.append(Paragraph("Synergies (Revenue, Cost)", styles['SubSubsectionTitle']))
            if context_basis and 'Synergies' in context_basis:
                synergies = context_basis['Synergies']
                elements.append(Paragraph(synergies, styles['SectionText']))
            else:
                elements.append(Paragraph("Potential revenue and cost synergies.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Investment Criteria
            elements.append(Paragraph("Investment Criteria", styles['SubSubsectionTitle']))
            if context_basis and 'Investment Criteria' in context_basis:
                investment_criteria = context_basis['Investment Criteria']
                elements.append(Paragraph(investment_criteria, styles['SectionText']))
            else:
                elements.append(Paragraph("Key criteria for investment decision-making.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Exit perspective
            elements.append(Paragraph("Exit perspective", styles['SubSubsectionTitle']))
            if context_basis and 'Exit perspective' in context_basis:
                exit_perspective = context_basis['Exit perspective']
                elements.append(Paragraph(exit_perspective, styles['SectionText']))
            else:
                elements.append(Paragraph("Potential exit strategies and timelines.", styles['SectionText']))
            elements.append(Spacer(1, 6))
            
            # Graveyard
            elements.append(Paragraph("Graveyard", styles['SubSubsectionTitle']))
            if context_basis and 'Graveyard' in context_basis:
                graveyard = context_basis['Graveyard']
                elements.append(Paragraph(graveyard, styles['SectionText']))
            else:
                elements.append(Paragraph("Failed competitors or previous attempts in this space.", styles['SectionText']))
            elements.append(Spacer(1, 24))
            
            # If we have extracted entities from NLP processing, add them as an appendix
            if teaser.entities:
                appendix_title = Paragraph("Appendix: Extracted Entities", styles['SectionTitle'])
                elements.append(appendix_title)
                elements.append(Spacer(1, 12))
                
                for entity_type, entity_list in teaser.entities.items():
                    if entity_list:
                        entity_type_title = Paragraph(f"{entity_type}", styles['SubsectionTitle'])
                        elements.append(entity_type_title)
                        elements.append(Spacer(1, 6))
                        
                        # Create a table for entities
                        data = [["Text", "Start", "End"]]
                        for entity in entity_list:
                            data.append([entity["text"], str(entity["start_char"]), str(entity["end_char"])])
                            
                        table = Table(data, colWidths=[300, 100, 100])
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        
                        elements.append(table)
                        elements.append(Spacer(1, 12))
            
            # Build the PDF
            doc.build(elements)
            
            # Return the relative path to the report
            return report_path
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None