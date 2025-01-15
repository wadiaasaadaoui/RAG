from typing import List
import numpy as np
import spacy
import yaml
from datetime import datetime

class DocumentProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.chunk_size = config['vector_store']['settings']['chunk_size']
        self.nlp = spacy.load("en_core_web_md")
        
    def chunk_document(self, text: str) -> List[str]:
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in doc.sents:
            if current_length + len(sent) > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(str(s) for s in current_chunk))
                current_chunk = [sent]
                current_length = len(sent)
            else:
                current_chunk.append(sent)
                current_length += len(sent)

        if current_chunk:
            chunks.append(' '.join(str(s) for s in current_chunk))

        return chunks

    def vectorize_chunks(self, chunks: List[str]) -> np.ndarray:
        return np.array([self.nlp(chunk).vector for chunk in chunks])

    def __call__(self, text: str) -> np.ndarray:
        chunks = self.chunk_document(text)
        return self.vectorize_chunks(chunks)