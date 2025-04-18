import spacy
from typing import Dict, List, Tuple

class NLPProcessor:
    def __init__(self):
        # Load the spaCy model
        self.nlp = spacy.load("en_core_web_md")
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract named entities from text using spaCy
        Returns a dictionary with entity categories as keys and lists of entities as values
        """
        doc = self.nlp(text)
        
        # Initialize result dictionary
        entities = {
            "COMPANY": [],
            "ORGANIZATION": [],
            "PERSON": [],
            "LOCATION": [],
            "GPE": [],  # GeoPolitical Entity
            "MONEY": [],
            "PERCENT": [],
            "DATE": [],
            "INDUSTRY": [],
            "OTHER": []
        }
        
        # Map spaCy entity labels to our custom categories
        label_mapping = {
            "ORG": "ORGANIZATION",
            "PERSON": "PERSON",
            "LOC": "LOCATION",
            "GPE": "GPE",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT",
            "DATE": "DATE",
            "PRODUCT": "INDUSTRY",
        }
        
        # Extract entities and categorize them
        for ent in doc.ents:
            category = label_mapping.get(ent.label_, "OTHER")
            
            # Try to identify company names (usually organizations)
            if category == "ORGANIZATION" and any(term in ent.text.lower() for term in ["inc", "corp", "ltd", "llc", "company", "group"]):
                category = "COMPANY"
            
            # Create entity dictionary
            entity = {
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            }
            
            # Add to appropriate category
            entities[category].append(entity)
        
        # Filter out empty categories
        return {k: v for k, v in entities.items() if v}