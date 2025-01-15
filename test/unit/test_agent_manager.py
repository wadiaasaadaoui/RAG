import pytest
from src.agents.agent_manager import AgentManager

@pytest.mark.asyncio
async def test_full_system():
    manager = AgentManager()
    
    # Test document processing
    test_doc = "Ceci est un document de test pour le système RAG."
    result = await manager.process_document(test_doc)
    print("Document Processing Result:", result)
    
    # Test query
    test_query = "Que contient le document?"
    result = await manager.process_query(
        query=test_query,
        context=[test_doc]
    )
    print("Query Result:", result)

# Enlever cette ligne car pytest-asyncio s'en occupera
# asyncio.run(test_full_system())