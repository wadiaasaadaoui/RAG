from .base_agent import BaseAgent
from ..indexer.document_processor import DocumentProcessor
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

class IndexingAgent(BaseAgent):
    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__("indexing_agent")
        self.document_processor = DocumentProcessor(config_path)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            document = task["document"]
            metadata = task.get("metadata", {})
            
            # Validation du document
            if not document or not document.strip():
                return {
                    "status": "error",
                    "error": "Le document ne peut pas être vide"
                }
            
            # Ajouter des métadonnées par défaut
            default_metadata = {
                "processed_at": datetime.now().isoformat(),
                "document_type": task.get("document_type", "text"),
                "source": task.get("source", "unknown")
            }
            metadata.update(default_metadata)
            
            # Chunking et vectorisation
            chunks = self.document_processor.chunk_document(document)
            if not chunks:
                return {
                    "status": "error",
                    "error": "Échec du chunking du document"
                }
                
            vectors = self.document_processor.vectorize_chunks(chunks)
            if not isinstance(vectors, np.ndarray) or vectors.size == 0:
                return {
                    "status": "error",
                    "error": "Échec de la vectorisation"
                }
            
            vector_dim = vectors.shape[1] if vectors.size > 0 else 0
            
            return {
                "status": "success",
                "chunks": chunks,
                "vectors": vectors.tolist() if isinstance(vectors, np.ndarray) else vectors,
                "metadata": {
                    "num_chunks": len(chunks),
                    "vector_dim": vector_dim,
                    "document_metadata": metadata
                }
            }
        except Exception as e:
            return await self.handle_error(e)