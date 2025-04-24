import asyncio
from typing import Optional
from models import Teaser
from document_generator.report_generator import ReportGenerator

async def generate_screening_report(teaser: Teaser) -> Optional[str]:
    """
    Generate a screening report for the given teaser.

    Args:
        teaser: The Teaser object containing analysis data.

    Returns:
        str: Path to the generated report file, or None if generation failed
    """
    # Define the filename for the report
    filename = f"screening_report_{teaser.id}.pdf"

    # Use the teaser's GPT analysis data to generate the report
    if teaser.gpt_analysis:
        report_path = await ReportGenerator.generate_report(teaser.gpt_analysis, filename)
        if report_path:
            print(f"Screening report generated successfully: {report_path}")
            return report_path
        else:
            print("Failed to generate screening report.")
            return None
    else:
        print("No GPT analysis data available for the teaser.")
        return None