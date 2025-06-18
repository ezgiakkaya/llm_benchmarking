#!/usr/bin/env python3
"""
Script to process lecture8.csv and generate all question versions.
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import questions_collection
from core.question_versioning import (
    generate_v1_original,
    generate_v2_reordered_mcq,
    generate_v3_mcq_with_nota,
    generate_v4_true_false_version
)

def process_lecture8():
    """Process lecture8.csv and generate all versions."""
    print("🚀 Starting Lecture 8 Question Processing...")
    print("=" * 70)
    
    # Read the CSV file
    try:
        df = pd.read_csv("lecture8.csv")
        # Robustly convert q_options from string to list
        def parse_options(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str) and x.strip().startswith('['):
                try:
                    return json.loads(x)
                except Exception as e:
                    print(f"⚠️  Could not parse q_options: {x}. Error: {e}")
                    return []
            return []
        df['q_options'] = df['q_options'].apply(parse_options)
        questions_list = df.to_dict('records')
        print(f"📊 Successfully read {len(questions_list)} questions from lecture8.csv")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    total_stored = 0
    total_replaced = 0
    
    for i, question in enumerate(questions_list, 1):
        original_q_id = question.get('q_id')
        if not original_q_id:
            print(f"⚠️  Skipping row {i} due to missing 'q_id'")
            continue

        print(f"\n📝 Processing question {i}/{len(questions_list)}: {original_q_id}")
        
        try:
            # Generate V1 (Original)
            v1_q = generate_v1_original(question)
            if v1_q:
                v1_q["original_q_id"] = original_q_id
                result = questions_collection.update_one(
                    {"q_id": v1_q["q_id"], "q_version": "1"},
                    {"$set": v1_q},
                    upsert=True
                )
                if result.upserted_id:
                    total_stored += 1
                elif result.modified_count:
                    total_replaced += 1
                print(f"  ✅ Stored V1")
            else:
                print(f"  ❌ Failed to generate V1")
                continue

            # Generate V2 (Reordered MCQ)
            v2_q = generate_v2_reordered_mcq(v1_q)
            if v2_q:
                v2_q["original_q_id"] = original_q_id
                result = questions_collection.update_one(
                    {"q_id": v2_q["q_id"], "q_version": "2"},
                    {"$set": v2_q},
                    upsert=True
                )
                if result.upserted_id:
                    total_stored += 1
                elif result.modified_count:
                    total_replaced += 1
                print(f"  ✅ Stored V2")
            else:
                print(f"  ❌ Failed to generate V2")

            # Generate V3 (MCQ with NOTA)
            v3_q = generate_v3_mcq_with_nota(v1_q)
            if v3_q:
                v3_q["original_q_id"] = original_q_id
                result = questions_collection.update_one(
                    {"q_id": v3_q["q_id"], "q_version": "3"},
                    {"$set": v3_q},
                    upsert=True
                )
                if result.upserted_id:
                    total_stored += 1
                elif result.modified_count:
                    total_replaced += 1
                print(f"  ✅ Stored V3")
            else:
                print(f"  ❌ Failed to generate V3")

            # Generate V4 (True/False)
            v4_q = generate_v4_true_false_version(v1_q)
            if v4_q:
                v4_q["original_q_id"] = original_q_id
                result = questions_collection.update_one(
                    {"q_id": v4_q["q_id"], "q_version": "4"},
                    {"$set": v4_q},
                    upsert=True
                )
                if result.upserted_id:
                    total_stored += 1
                elif result.modified_count:
                    total_replaced += 1
                print(f"  ✅ Stored V4")
            else:
                print(f"  ❌ Failed to generate V4")
                
        except Exception as e:
            print(f"  ❌ An unexpected error occurred while processing {original_q_id}: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Question version generation complete!")
    print(f"📊 Summary:")
    print(f"   - Processed: {len(questions_list)} questions")
    print(f"   - Stored: {total_stored} new versions")
    print(f"   - Replaced/Updated: {total_replaced} existing versions")

if __name__ == "__main__":
    process_lecture8() 