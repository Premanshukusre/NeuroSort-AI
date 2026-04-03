"""
NeuroSort AI: Advanced NLP Preprocessing Pipeline.
Leverages spaCy for lemmatization, tokenization, and domain-agnostic cleaning.
"""

import re
import logging
import spacy
from typing import List, Optional

# Setup high-fidelity logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NeuroSortPipeline")

class DataPipeline:
    """
    Handles advanced NLP preprocessing for document classification.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Downloading...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess raw text for semantic classification.
        Steps: Lowercase, Regex Cleaning, spaCy Lemmatization, Stopword Removal.
        """
        if not text: return ""
        
        try:
            # 1. Lowercasing & Basic Cleaning
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            
            # 2. spaCy Pipeline
            doc = self.nlp(text)
            
            # 3. Token Filtering: Remove stopwords, punct, space, and short words
            # Use lemmatization to group related words
            tokens = [
                token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 2
            ]
            
            cleaned_text = " ".join(tokens)
            return cleaned_text.strip()
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return text

if __name__ == '__main__':
    pipeline = DataPipeline()
    sample = "The NeuroSort AI system classifies complex technical and medical reports."
    print(f"Original: {sample}")
    print(f"Cleaned: {pipeline.preprocess_text(sample)}")
