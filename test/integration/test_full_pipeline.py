import pytest
import asyncio
import logging
from src.agents.agent_manager import AgentManager

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture
def manager():
    return AgentManager()

@pytest.mark.asyncio
async def test_complete_pipeline(manager):
    """Test du pipeline RAG complet"""
    
    # Document de test
    test_doc = """Python is a programming language created by Guido van Rossum.
                  It is known for its simplicity and readability."""
    
    logger.info("Starting document processing test")
    try:
        # Test de l'indexation
        process_result = await manager.process_document(test_doc)
        logger.debug(f"Indexation result: {process_result}")
        
        assert process_result["status"] == "success", \
            f"L'indexation a échoué: {process_result.get('error', 'No error details')}"
            
        # Test d'une requête simple
        query = "Who created Python?"
        query_result = await manager.process_query(query, context=[test_doc])
        
        assert query_result["status"] == "success", \
            f"La requête a échoué: {query_result.get('error', 'No error details')}"
        
        response = query_result["response"]
        logger.info(f"Generated response: {response}")
        
        # Vérifications plus flexibles
        assert len(response) > 0, "La réponse est vide"
        assert "Python" in response, "La réponse ne mentionne pas Python"
        
        # Vérification de la validation
        validation = query_result["validation"]
        assert validation["is_valid"], "La validation a échoué"
        assert validation["confidence"] > 0.5, "Score de confiance trop bas"
        
        # Vérification des sources
        assert "sources" in query_result, "Pas de sources dans la réponse"
        assert len(query_result["sources"]) > 0, "Aucune source trouvée"
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_error_handling(manager):
    """Test de la gestion des erreurs"""
    
    # Test avec document vide
    empty_doc_result = await manager.process_document("")
    assert empty_doc_result["status"] == "error", \
        "Un document vide devrait produire une erreur"
    
    # Test avec document d'espaces
    whitespace_doc_result = await manager.process_document("    ")
    assert whitespace_doc_result["status"] == "error", \
        "Un document d'espaces devrait produire une erreur"
    
    # Test avec query vide
    empty_query_result = await manager.process_query("", context=["Some context"])
    assert empty_query_result["status"] == "error", \
        "Une requête vide devrait produire une erreur"
    
    logger.info("Error handling tests passed")

@pytest.mark.asyncio
async def test_basic_functionality(manager):
    """Test des fonctionnalités de base sans assertions strictes sur le contenu"""
    test_doc = "This is a test document about basic testing."
    
    # Test document processing
    process_result = await manager.process_document(test_doc)
    assert process_result["status"] == "success"
    
    # Test simple query
    query_result = await manager.process_query(
        "What is this document about?",
        context=[test_doc]
    )
    assert query_result["status"] == "success"
    assert isinstance(query_result["response"], str)
    assert len(query_result["response"]) > 0
    
    logger.info("Basic functionality test passed")