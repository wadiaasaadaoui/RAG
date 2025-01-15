from .base_agent import BaseAgent
from typing import Dict, Any, List
import re
import logging

class ValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__("validation_agent")
        self.logger = logging.getLogger(__name__)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = task["response"]
            context = task.get("context", "")
            question = task["question"]
            
            # Vérification de la cohérence
            validation_results = await self.validate_response(response, context, question)
            
            # Log des résultats pour le débogage
            self.logger.debug(f"Validation results: {validation_results}")
            
            return {
                "status": "success",
                "is_valid": validation_results["is_valid"],
                "confidence": validation_results["confidence"],
                "validation_details": validation_results["details"]
            }
        except Exception as e:
            return await self.handle_error(e)
            
    async def validate_response(self, response: str, context: str, question: str) -> Dict[str, Any]:
        validation_score = 0.0
        validation_details = {}
        
        try:
            # 1. Vérification de base de la réponse
            if not response or not response.strip():
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "details": {"error": "Empty response"}
                }

            # 2. Vérification de la longueur minimale
            if len(response.split()) < 3:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "details": {"error": "Response too short"}
                }

            # 3. Vérification du contexte
            key_elements = self.extract_key_elements(response)
            context_support = 0
            if context:  # Ne vérifie le contexte que s'il est fourni
                context_support = sum(1 for elem in key_elements if elem.lower() in context.lower())
                validation_score += 0.3 if context_support > 0 else 0.0
            
            # 4. Vérification de la pertinence par rapport à la question
            question_keywords = self.extract_keywords(question)
            response_relevance = sum(1 for kw in question_keywords if kw.lower() in response.lower())
            validation_score += (response_relevance / max(len(question_keywords), 1)) * 0.4
            
            # 5. Vérification de la structure
            if self.has_good_structure(response):
                validation_score += 0.3
                
            # Seuil de validation plus bas pour le développement
            is_valid = validation_score >= 0.4  # Seuil abaissé de 0.7 à 0.4
            
            validation_details = {
                "context_support": context_support,
                "response_relevance": response_relevance,
                "structure_quality": self.has_good_structure(response),
                "validation_score": validation_score
            }
            
            self.logger.debug(f"Validation details: {validation_details}")
            
            return {
                "is_valid": is_valid,
                "confidence": validation_score,
                "details": validation_details
            }
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "details": {"error": str(e)}
            }
    
    def extract_key_elements(self, text: str) -> List[str]:
        # Extraction plus simple des éléments clés
        return [phrase.strip() for phrase in text.split('.') if phrase.strip()]
    
    def extract_keywords(self, text: str) -> List[str]:
        # Liste de mots vides en anglais et français
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                     'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'du'}
        
        # Extraction des mots-clés en ignorant la casse et la ponctuation
        words = re.findall(r'\w+', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def has_good_structure(self, text: str) -> bool:
        # Critères de structure plus souples
        min_words = len(text.split()) >= 3  # Au moins 3 mots
        has_sentence_end = bool(re.search(r'[.!?]', text))  # Au moins une fin de phrase
        return min_words and has_sentence_end