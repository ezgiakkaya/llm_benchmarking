#!/usr/bin/env python3
"""
Script to view the latest 10 questions added to the database.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.database import questions_collection

def view_latest_questions(limit=10):
    """
    View the latest questions added to the database.
    
    Args:
        limit: Number of questions to display (default: 10)
    """
    print(f"🔍 Fetching the latest {limit} questions from the database...")
    print("=" * 80)
    
    # Get the latest questions by _id (which contains timestamp info)
    # Sort by _id in descending order to get the most recent first
    latest_questions = list(questions_collection.find().sort("_id", -1).limit(limit))
    
    if not latest_questions:
        print("❌ No questions found in the database.")
        return
    
    print(f"✅ Found {len(latest_questions)} questions:")
    print()
    
    for i, question in enumerate(latest_questions, 1):
        print(f"📝 Question #{i}")
        for key, value in question.items():
            print(f"   {key}: {value}")
        print("-" * 80)
    
    # Show summary statistics
    print("\n📊 Summary:")
    print(f"   Total questions in database: {questions_collection.count_documents({})}")
    
    # Count by type
    type_counts = {}
    for q in latest_questions:
        q_type = q.get('q_type', 'Unknown')
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    print("   Latest questions by type:")
    for q_type, count in type_counts.items():
        print(f"      {q_type}: {count}")
    
    # Count by version
    version_counts = {}
    for q in latest_questions:
        version = q.get('q_version', '1')
        version_counts[version] = version_counts.get(version, 0) + 1
    
    print("   Latest questions by version:")
    for version, count in version_counts.items():
        print(f"      Version {version}: {count}")

if __name__ == "__main__":
    # Allow command line argument for limit
    limit = 10
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid limit. Using default of 10.")
    
    view_latest_questions(limit) 