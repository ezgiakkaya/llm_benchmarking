import os
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

def mask_api_key(api_key):
    """Mask API key for security"""
    if not api_key:
        return "Not set"
    return f"{api_key[:4]}...{api_key[-4:]}"

def test_rag_environment():
    print("\n=== RAG Environment Variables ===")
    
    # Pinecone Configuration
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index = os.getenv('PINECONE_INDEX_NAME')
    
    print("\nPinecone Configuration:")
    print(f"API Key: {mask_api_key(pinecone_api_key)}")
    print(f"Index Name: {pinecone_index}")
    
    # Groq Configuration
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    print("\nGroq Configuration:")
    print(f"API Key: {mask_api_key(groq_api_key)}")
    
    # Test Pinecone Connection
    print("\nTesting Pinecone Connection...")
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        print("✓ Successfully created Pinecone client instance")
        # List existing indexes
        indexes = pc.list_indexes().names()
        print(f"Available indexes: {indexes}")
    except Exception as e:
        print(f"✗ Error connecting to Pinecone: {str(e)}")
    
    print("\n=== Environment Test Complete ===")

if __name__ == "__main__":
    test_rag_environment() 