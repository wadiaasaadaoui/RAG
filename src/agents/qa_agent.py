from .base_agent import BaseAgent
from ..generator.llm_interface import LLMInterface
from typing import Dict, Any, Optional
import asyncio
import logging

class QAAgent(BaseAgent):
    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__("qa_agent")
        self.llm = LLMInterface(config_path)
        self.logger = logging.getLogger(__name__)
        
    async def process_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Validation des entrées
            question = task.get("question")
            if not question or not question.strip():
                return {
                    "status": "error",
                    "error": "La question ne peut pas être vide"
                }
            
            context = task.get("context", "")
            template = task.get("template", "qa_default")
            
            self.logger.debug(f"Processing question: {question}")
            self.logger.debug(f"With context length: {len(context)}")
            
            # Génération de la réponse
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.llm.generate_response(
                        prompt=question,
                        context=context
                    )
                )
                
                return {
                    "status": "success",
                    "response": response,
                    "metadata": {
                        "question": question,
                        "context_length": len(context),
                        "template_used": template
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error generating response: {str(e)}")
                return {
                    "status": "error",
                    "error": f"Erreur lors de la génération: {str(e)}"
                }
                
        except Exception as e:
            return await self.handle_error(e)