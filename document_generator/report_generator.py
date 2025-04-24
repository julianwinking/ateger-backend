import os
import datetime
import logging
import json
from typing import Dict, Optional, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("report_generator")

class ReportGenerator:
    """
    Class for generating PDF reports from analysis data.
    """

    @staticmethod
    async def generate_report(data: Dict[str, Any], filename: str) -> Optional[str]:
        """
        Generate a PDF report from the provided analysis data.

        Args:
            data: The analysis data to include in the report.
            filename: The name of the file to save the report as.

        Returns:
            str: Path to the generated report file, or None if generation failed.
        """
        try:
            logger.info(f"Starting report generation for {filename}")

            # Check reports directory
            reports_dir = "reports"
            logger.info(f"Checking if reports directory exists: {reports_dir}")
            if not os.path.exists(reports_dir):
                logger.info(f"Reports directory doesn't exist, creating: {reports_dir}")
                try:
                    os.makedirs(reports_dir, exist_ok=True)
                    logger.info(f"Successfully created reports directory")
                except Exception as dir_error:
                    logger.error(f"Failed to create reports directory: {str(dir_error)}")
                    return None

            # Define the report filename
            report_path = os.path.join(reports_dir, filename)
            logger.info(f"Report will be generated at: {report_path}")

            # Create the PDF document
            try:
                logger.info("Creating PDF document")
                doc = SimpleDocTemplate(
                    report_path,
                    pagesize=letter,
                    rightMargin=0.75 * inch,
                    leftMargin=0.75 * inch,
                    topMargin=0.75 * inch,
                    bottomMargin=0.75 * inch
                )

                # Create custom styles with improved spacing and formatting
                styles = getSampleStyleSheet()
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Title'],
                    fontSize=18,
                    alignment=TA_CENTER,
                    spaceAfter=12
                )
                heading_style = ParagraphStyle(
                    'CustomHeading1',
                    parent=styles['Heading1'],
                    fontSize=16,
                    spaceAfter=10,
                    spaceBefore=12
                )
                normal_style = ParagraphStyle(
                    'CustomNormal',
                    parent=styles['Normal'],
                    fontSize=10,
                    spaceAfter=6,
                    spaceBefore=2,
                    alignment=TA_JUSTIFY
                )

                # Container for the 'Flowable' objects
                elements = []

                # Add the title
                elements.append(Paragraph(f"Analysis Report: {filename}", title_style))
                elements.append(Spacer(1, 12))

                # Add the date
                elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y')}", normal_style))
                elements.append(Spacer(1, 24))

                # Process sections in the provided data
                for section_name, content in data.items():
                    logger.info(f"Processing section: {section_name}")

                    # Main section header
                    elements.append(Paragraph(section_name, heading_style))
                    elements.append(Spacer(1, 12))

                    # Add the content
                    if content and content.strip():
                        elements.append(Paragraph(content, normal_style))
                        logger.info(f"  Added content for {section_name}: {len(content)} chars")
                    else:
                        elements.append(Paragraph("No data available for this section.", normal_style))
                        logger.info(f"  No content found for {section_name}")

                    # Add spacing after section
                    elements.append(Spacer(1, 24))

                # Add a custom page template
                def add_page_template(canvas_obj, doc_obj):
                    # Add the company logo in the upper-right corner
                    logo_path = "backend/document_generator/images/ateger_logo_color.png"
                    try:
                        ImageReader(logo_path)
                        canvas_obj.drawImage(logo_path, 450, 750, width=100, height=50, preserveAspectRatio=True, mask='auto')
                    except Exception as e:
                        logger.warning(f"Could not add logo: {e}")

                    # Add page numbers in the lower-right corner
                    page_number_text = f"Page {doc_obj.page}"
                    canvas_obj.drawRightString(550, 20, page_number_text)

                # Build the PDF with the custom page template
                logger.info("Building PDF document")
                doc.build(elements, onFirstPage=add_page_template, onLaterPages=add_page_template)

                logger.info(f"Successfully generated report at {report_path}")
                return report_path

            except Exception as pdf_error:
                logger.error(f"Error creating PDF: {str(pdf_error)}")
                import traceback
                logger.error(traceback.format_exc())
                return None

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None