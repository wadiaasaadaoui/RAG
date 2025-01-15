from src.generator.llm_interface import LLMInterface

def test_llm():
    print("Initializing LLM Interface...")
    llm = LLMInterface()  # Utilisera les valeurs par défaut (gpt2)
    
    print("Testing generation...")
    prompt = "What is artificial intelligence?"
    response = llm.generate_response(prompt)
    
    print("\nResults:")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_llm()