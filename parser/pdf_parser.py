import pdfplumber
import pytesseract
from PIL import Image
import io
import os
import tempfile

class PDFParser:
    @staticmethod
    async def extract_text_from_pdf(file_content: bytes) -> str:
        """
        Extract text from a PDF file using pdfplumber.
        If text extraction fails or returns empty, use OCR as a fallback.
        """
        extracted_text = ""
        
        # First try with pdfplumber
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    extracted_text += page_text + "\n\n"
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                extracted_text += " | ".join([str(cell or "") for cell in row]) + "\n"
                            extracted_text += "\n"
        
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            
        # If no text was extracted, try OCR
        if not extracted_text.strip():
            extracted_text = await PDFParser.extract_text_with_ocr(file_content)
            
        return extracted_text.strip()
    
    @staticmethod
    async def extract_text_with_ocr(file_content: bytes) -> str:
        """
        Extract text from a PDF file using OCR (pytesseract).
        """
        extracted_text = ""
        
        # Create a temporary file to save the PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(file_content)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Use a temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                from pdf2image import convert_from_path
                
                # Convert PDF to images
                images = convert_from_path(temp_pdf_path, output_folder=temp_dir)
                
                # Extract text from each image
                for i, image in enumerate(images):
                    text = pytesseract.image_to_string(image)
                    extracted_text += text + "\n\n"
        
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            
        finally:
            # Clean up the temporary PDF file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                
        return extracted_text.strip()