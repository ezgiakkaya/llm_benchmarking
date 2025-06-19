#!/usr/bin/env python3
"""
Test script to verify imports work correctly in Docker environment
"""
import sys
import os

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")

try:
    print("Testing import of core.llm_clients...")
    from core.llm_clients import query_groq, query_groq_with_rag, GROQ_MODELS
    print("✅ All imports successful!")
    print(f"GROQ_MODELS: {list(GROQ_MODELS.keys())}")
    print(f"query_groq function: {query_groq}")
    print(f"query_groq_with_rag function: {query_groq_with_rag}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Other error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()

print("Test completed.") 