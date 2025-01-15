import pytest
import numpy as np
from src.retriever.retrieval import Retriever

def test_add_documents():
    retriever = Retriever()
    test_vectors = np.random.rand(2, 300)
    test_docs = ["Doc 1", "Doc 2"]
    
    retriever.add_documents(test_vectors, test_docs)
    # Vérifier l'ajout...

def test_retrieve():
    retriever = Retriever()
    # Test de récupération...