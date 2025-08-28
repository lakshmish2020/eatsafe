import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import logging

class OCRProcessor:
    """Handles OCR text extraction from food package images"""
    
    def __init__(self):
        """Initialize OCR processor with optimized settings"""
        # Tesseract configuration for better text extraction
        self.config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{},.;:-_%/'
        
    def extract_text(self, image, confidence_threshold=30):
        """
        Extract text from image using OCR
        
        Args:
            image: OpenCV image array
            confidence_threshold: Minimum confidence for text extraction
            
        Returns:
            str: Extracted text
        """
        try:
            # Convert to PIL Image for tesseract
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Get detailed OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=self.config)
            
            # Filter text by confidence
            filtered_text = []
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > confidence_threshold:
                    text = ocr_data['text'][i].strip()
                    if text:
                        filtered_text.append(text)
            
            # Join filtered text
            extracted_text = ' '.join(filtered_text)
            
            # Clean up the text
            cleaned_text = self._clean_ocr_text(extracted_text)
            
            return cleaned_text
            
        except Exception as e:
            logging.error(f"OCR extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text from image: {str(e)}")
    
    def _clean_ocr_text(self, text):
        """
        Clean and normalize OCR-extracted text
        
        Args:
            text: Raw OCR text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        replacements = {
            r'\b0\b': 'O',  # Zero to letter O
            r'\b1\b': 'I',  # One to letter I when isolated
            r'\b5\b': 'S',  # Five to S in some contexts
            r'\b8\b': 'B',  # Eight to B
            r'[|]': 'I',    # Pipe to I
            r'[`\']': "'",  # Backticks to apostrophe
            r'[""]': '"',   # Smart quotes to regular quotes
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove extra punctuation artifacts
        text = re.sub(r'[^\w\s,.:;()\[\]{}\-_%/]', '', text)
        
        # Normalize spaces around punctuation
        text = re.sub(r'\s*([,.;:])\s*', r'\1 ', text)
        text = re.sub(r'\s*([()])\s*', r' \1 ', text)
        
        return text.strip()
    
    def extract_ingredients_section(self, text):
        """
        Extract the ingredients section from OCR text
        
        Args:
            text: Full OCR text
            
        Returns:
            str: Ingredients section text
        """
        if not text:
            return ""
        
        # Common patterns for ingredients sections
        ingredient_patterns = [
            r'ingredients?[:\s]+(.*?)(?=nutrition|directions|allergen|contains|net weight|best before|expiry|$)',
            r'contains?[:\s]+(.*?)(?=nutrition|directions|allergen|net weight|best before|expiry|$)',
            r'made with[:\s]+(.*?)(?=nutrition|directions|allergen|net weight|best before|expiry|$)',
        ]
        
        text_lower = text.lower()
        
        for pattern in ingredient_patterns:
            matches = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if matches:
                ingredients_text = matches.group(1).strip()
                if len(ingredients_text) > 10:  # Reasonable minimum length
                    return ingredients_text
        
        # If no specific section found, look for comma-separated lists
        # that might be ingredients
        sentences = text.split('.')
        for sentence in sentences:
            if ',' in sentence and len(sentence) > 30:
                # Check if it looks like an ingredient list (has common food terms)
                food_terms = ['flour', 'sugar', 'salt', 'oil', 'water', 'milk', 'eggs', 'butter']
                if any(term in sentence.lower() for term in food_terms):
                    return sentence.strip()
        
        return text  # Return full text if no specific section identified
    
    def get_ocr_confidence(self, image):
        """
        Get overall confidence score for OCR on the image
        
        Args:
            image: Image to analyze
            
        Returns:
            float: Average confidence score
        """
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image
                
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            
            if confidences:
                return sum(confidences) / len(confidences)
            return 0
            
        except Exception:
            return 0
