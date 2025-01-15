from typing import Dict, List, Any, Optional
import asyncio
from .base_agent import BaseAgent
from .indexing_agent import IndexingAgent
from .research_agent import ResearchAgent
from .qa_agent import QAAgent
from .validation_agent import ValidationAgent
import yaml
from datetime import datetime

class AgentManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.agents: Dict[str, BaseAgent] = {
            "indexing": IndexingAgent(config_path),
            "research": ResearchAgent(config_path),
            "qa": QAAgent(self.config["model_config"]["model_name"]),
            "validation": ValidationAgent()
        }
        
    async def process_document(self, document: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if metadata is None:
                metadata = {}
            
            # Ajouter des métadonnées de traitement
            metadata.update({
                "processed_at": datetime.now().isoformat(),
                "processing_agent": "document_processor"
            })
            
            indexing_result = await self.agents["indexing"].process_task({
                "document": document,
                "metadata": metadata
            })
            
            if indexing_result["status"] == "error":
                return indexing_result
                
            return {
                "status": "success",
                "indexed_document": indexing_result,
                "processing_metadata": metadata
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": metadata
            }
        
    async def process_query(
        self,
        query: str,
        context: List[str],
        metadata_filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        try:
            research_result = await self.agents["research"].process_task({
                "query": query,
                "context": context,
                "metadata_filter": metadata_filter
            })
            
            if research_result["status"] == "error":
                return research_result
                
            relevant_contexts = [r["content"] for r in research_result["results"]]
            
            qa_result = await self.agents["qa"].process_task({
                "question": query,
                "context": "\n".join(relevant_contexts)
            })
            
            if qa_result["status"] == "error":
                return qa_result
                
            validation_result = await self.agents["validation"].process_task({
                "response": qa_result["response"],
                "context": "\n".join(relevant_contexts),
                "question": query
            })
            
            return {
                "status": "success",
                "response": qa_result["response"],
                "validation": validation_result,
                "sources": research_result["results"],
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "num_sources": len(research_result["results"]),
                    "filter_applied": bool(metadata_filter)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }  