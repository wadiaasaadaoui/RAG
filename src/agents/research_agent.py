from .base_agent import BaseAgent
from ..retriever.retrieval import Retriever
from typing import Dict, Any, List, Optional
import numpy as np
import yaml

class ResearchAgent(BaseAgent):
    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__("research_agent")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.retriever = Retriever(config_path)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = task["query"]
            query_vector = task.get("query_vector")
            k = task.get("k", self.config['vector_store']['settings']['max_results'])
            metadata_filter = task.get("metadata_filter")
            
            results = self.retriever.retrieve(
                query_vector=query_vector,
                k=k,
                filter_metadata=metadata_filter
            )
            
            return {
                "status": "success",
                "results": [
                    {
                        "content": content,
                        "similarity": float(sim),
                        "metadata": meta
                    } for content, sim, meta in results
                ],
                "metadata": {
                    "num_results": len(results),
                    "query": query,
                    "filter_applied": bool(metadata_filter)
                }
            }
        except Exception as e:
            return await self.handle_error(e)