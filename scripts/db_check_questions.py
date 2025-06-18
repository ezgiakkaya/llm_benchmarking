import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.database import questions_collection

# Print a summary of all LEC8_ questions and their versions

def main():
    print("LEC8_ questions in database (showing q_id, q_version, q_text):\n")
    query = {"q_id": {"$regex": "^LEC8_"}}
    for q in questions_collection.find(query, {"_id": 0, "q_id": 1, "q_version": 1, "q_text": 1}):
        print(f"q_id: {q.get('q_id')}, version: {q.get('q_version')}, text: {q.get('q_text')[:60]}...")

    count = questions_collection.count_documents(query)
    print(f"\nTotal LEC8_ question documents: {count}")

if __name__ == "__main__":
    main() 