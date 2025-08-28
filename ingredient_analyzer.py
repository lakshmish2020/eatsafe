import json
import os
import re
from typing import Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class IngredientAnalyzer:
    """Uses AI to analyze and summarize ingredients from OCR text"""
    
    def __init__(self):
        """Initialize the ingredient analyzer with OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Common allergens for identification
        self.common_allergens = [
            'milk', 'eggs', 'fish', 'shellfish', 'tree nuts', 'peanuts', 
            'wheat', 'soybeans', 'sesame', 'gluten', 'dairy', 'nuts'
        ]
    
    def analyze_ingredients(self, ocr_text: str) -> Dict[str, Any]:
        """
        Analyze ingredients from OCR text using AI
        
        Args:
            ocr_text: Raw OCR text from food package
            
        Returns:
            Dict containing ingredient analysis
        """
        try:
            # First, try to extract ingredients section
            ingredients_text = self._extract_ingredients_section(ocr_text)
            
            if not ingredients_text or len(ingredients_text.strip()) < 5:
                return self._empty_result("No ingredients text found in the image")
            
            # Use AI to analyze ingredients
            analysis = self._ai_analyze_ingredients(ingredients_text)
            
            if not analysis:
                return self._empty_result("AI analysis failed")
            
            # Enhance with additional processing
            enhanced_analysis = self._enhance_analysis(analysis, ingredients_text)
            
            return enhanced_analysis
            
        except Exception as e:
            return self._empty_result(f"Analysis error: {str(e)}")
    
    def _extract_ingredients_section(self, text: str) -> str:
        """Extract ingredients section from OCR text"""
        if not text:
            return ""
        
        text_lower = text.lower()
        
        # Patterns to find ingredients section
        patterns = [
            r'ingredients?[:\s]+(.*?)(?=nutrition|allergen|directions|contains|net\s*weight|best\s*before|expiry|storage|$)',
            r'contains?[:\s]+(.*?)(?=nutrition|allergen|directions|net\s*weight|best\s*before|expiry|storage|$)',
            r'made\s*with[:\s]+(.*?)(?=nutrition|allergen|directions|net\s*weight|best\s*before|expiry|storage|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients_section = match.group(1).strip()
                if len(ingredients_section) > 10:
                    return ingredients_section
        
        # If no clear section found, return the full text
        return text
    
    def _ai_analyze_ingredients(self, ingredients_text: str) -> Dict[str, Any]:
        """Use OpenAI to analyze ingredients"""
        try:
            prompt = f"""
            Analyze the following ingredients text from a food package and provide a comprehensive analysis.
            
            Ingredients text: "{ingredients_text}"
            
            Please provide the analysis in JSON format with the following structure:
            {{
                "ingredients": [
                    {{
                        "name": "ingredient name",
                        "description": "brief description of what this ingredient is"
                    }}
                ],
                "allergens": ["list of potential allergens"],
                "dietary_flags": ["vegetarian", "vegan", "gluten-free", etc.],
                "nutritional_insights": {{
                    "health_score": 1-10,
                    "categories": ["processed", "natural", "organic", etc.],
                    "key_nutrients": ["list of notable nutrients"],
                    "health_notes": "brief health assessment"
                }},
                "summary": "A brief summary of the product based on ingredients"
            }}
            
            Focus on:
            1. Identifying individual ingredients clearly
            2. Common allergens (milk, eggs, nuts, wheat, soy, etc.)
            3. Dietary compatibility (vegetarian, vegan, gluten-free)
            4. Health assessment based on ingredient quality
            5. Brief nutritional insights
            
            If the text doesn't contain clear ingredients, return empty arrays/null values appropriately.
            """
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a food science expert specializing in ingredient analysis. Provide accurate, helpful information about food ingredients."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if not content:
                raise Exception("Empty response from AI")
            result = json.loads(content)
            return result
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse AI response as JSON: {str(e)}")
        except Exception as e:
            raise Exception(f"AI analysis failed: {str(e)}")
    
    def _enhance_analysis(self, analysis: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Enhance AI analysis with additional processing"""
        
        # Ensure all required keys exist
        enhanced = {
            "ingredients": analysis.get("ingredients", []),
            "allergens": analysis.get("allergens", []),
            "dietary_flags": analysis.get("dietary_flags", []),
            "nutritional_insights": analysis.get("nutritional_insights", {}),
            "summary": analysis.get("summary", "")
        }
        
        # Additional allergen detection using text matching
        detected_allergens = self._detect_allergens_by_text(original_text)
        
        # Merge with AI-detected allergens
        all_allergens = list(set(enhanced["allergens"] + detected_allergens))
        enhanced["allergens"] = all_allergens
        
        # Validate health score
        if "health_score" in enhanced["nutritional_insights"]:
            score = enhanced["nutritional_insights"]["health_score"]
            if not isinstance(score, (int, float)) or score < 1 or score > 10:
                enhanced["nutritional_insights"]["health_score"] = 5  # Default moderate score
        
        return enhanced
    
    def _detect_allergens_by_text(self, text: str) -> List[str]:
        """Detect allergens by text matching"""
        detected = []
        text_lower = text.lower()
        
        allergen_patterns = {
            "milk": ["milk", "dairy", "lactose", "cream", "butter", "cheese", "whey", "casein"],
            "eggs": ["egg", "albumen", "lecithin"],
            "wheat": ["wheat", "flour", "gluten"],
            "soy": ["soy", "soya", "soybean"],
            "nuts": ["nuts", "almond", "walnut", "pecan", "hazelnut", "cashew", "pistachio"],
            "peanuts": ["peanut", "groundnut"],
            "fish": ["fish", "salmon", "tuna", "cod"],
            "shellfish": ["shellfish", "shrimp", "crab", "lobster"],
            "sesame": ["sesame", "tahini"]
        }
        
        for allergen, patterns in allergen_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                detected.append(allergen)
        
        return detected
    
    def _empty_result(self, message: str) -> Dict[str, Any]:
        """Return empty result structure with error message"""
        return {
            "ingredients": [],
            "allergens": [],
            "dietary_flags": [],
            "nutritional_insights": {
                "health_score": None,
                "categories": [],
                "key_nutrients": [],
                "health_notes": message
            },
            "summary": message
        }
    
    def get_ingredient_details(self, ingredient_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific ingredient
        
        Args:
            ingredient_name: Name of the ingredient
            
        Returns:
            Dict with ingredient details
        """
        try:
            prompt = f"""
            Provide detailed information about the food ingredient: "{ingredient_name}"
            
            Include:
            1. What it is (source, type)
            2. Common uses in food products
            3. Nutritional properties
            4. Any health considerations
            5. Allergen information if applicable
            
            Respond in JSON format:
            {{
                "name": "{ingredient_name}",
                "description": "what it is",
                "uses": "common uses",
                "nutrition": "nutritional properties",
                "health_notes": "health considerations",
                "allergen_info": "allergen information or null"
            }}
            """
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a food science expert. Provide accurate information about food ingredients."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            if not content:
                raise Exception("Empty response from AI")
            return json.loads(content)
            
        except Exception as e:
            return {
                "name": ingredient_name,
                "description": f"Unable to analyze ingredient: {str(e)}",
                "uses": "Unknown",
                "nutrition": "Unknown",
                "health_notes": "Analysis unavailable",
                "allergen_info": "None"
            }
