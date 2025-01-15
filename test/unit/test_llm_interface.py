import pytest
import logging
from src.generator.llm_interface import LLMInterface, ValueError

# Configuration du logging pour les tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def llm():
    """Fixture pour créer une instance de LLMInterface"""
    return LLMInterface()

def test_llm_initialization(llm):
    """Test l'initialisation du LLM"""
    assert llm is not None
    assert hasattr(llm, 'model')
    assert hasattr(llm, 'tokenizer')
    logger.info("LLM initialization successful")

def test_llm_generation(llm):
    """Test la génération de réponse"""
    test_question = "What is the capital of France?"
    test_context = "Paris is the capital of France."
    
    try:
        logger.info("Starting response generation test")
        response = llm.generate_response(
            prompt=test_question,
            context=test_context
        )
        
        # Vérifications de base
        assert isinstance(response, str), "La réponse doit être une chaîne de caractères"
        assert len(response) > 0, "La réponse ne doit pas être vide"
        
        # Log des résultats
        logger.info(f"Question: {test_question}")
        logger.info(f"Contexte: {test_context}")
        logger.info(f"Réponse: {response}")
        logger.info("Response generation test passed")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

def test_llm_error_handling(llm):
    """Test la gestion des erreurs de validation"""
    logger.info("Starting error handling tests")
    
    # Test avec un prompt vide
    with pytest.raises(ValueError) as exc_info:
        llm.generate_response("")
    assert str(exc_info.value) == "Le prompt ne peut pas être vide"
    
    # Test avec des espaces
    with pytest.raises(ValueError) as exc_info:
        llm.generate_response("   ")
    assert str(exc_info.value) == "Le prompt ne peut pas être vide"
    
    logger.info("Error handling tests passed successfully")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])