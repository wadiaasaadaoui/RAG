import pytest
from src.indexer.document_processor import DocumentProcessor

class TestDocumentProcessor:
    @pytest.fixture
    def processor(self):
        return DocumentProcessor()
    
    def test_chunk_document(self, processor):
        # Test document
        test_doc = """Ceci est un document de test.
        Il contient plusieurs phrases pour tester le chunking.
        Nous voulons voir comment le système fonctionne."""
        
        # Test le chunking
        chunks = processor.chunk_document(test_doc)
        
        # Assertions
        assert len(chunks) > 0
        print(f"Nombre de chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk}")
    
    def test_vectorize_chunks(self, processor):
        # Test chunks
        test_chunks = ["Premier test", "Deuxième test"]
        
        # Test la vectorisation
        vectors = processor.vectorize_chunks(test_chunks)
        
        # Assertions
        assert vectors.shape[0] == len(test_chunks)
        print(f"Shape des vecteurs: {vectors.shape}")