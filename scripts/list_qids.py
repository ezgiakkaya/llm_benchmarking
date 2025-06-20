#!/usr/bin/env python3
"""
A simple script to list all question IDs in the database,
optionally filtering by a prefix.
"""

import sys
import os
import argparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import questions_collection

def list_question_ids(prefix=None):
    """
    Prints a list of all question IDs, optionally filtered by a prefix.
    """
    query = {}
    if prefix:
        query = {'q_id': {'$regex': f'^{prefix}'}}
        print(f"🔍 Finding all question IDs with prefix '{prefix}'...")
    else:
        print("🔍 Finding all question IDs in the database...")


    q_ids = sorted(questions_collection.distinct('q_id', query))

    if not q_ids:
        print(f"❌ No questions found matching the criteria.")
        return

    print(f"✅ Found {len(q_ids)} matching question IDs:")
    for q_id in q_ids:
        print(f"  - {q_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List question IDs from the database.",
    )
    parser.add_argument(
        '--prefix', 
        type=str, 
        help="Optional prefix to filter question IDs (e.g., 'LEC1_')."
    )
    
    args = parser.parse_args()
    
    list_question_ids(args.prefix) 