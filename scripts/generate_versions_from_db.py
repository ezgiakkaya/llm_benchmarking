#!/usr/bin/env python3
"""
Backend script to generate specific versions (V2, V3, V4) for questions
that already exist in the database. It reads the base question (V1) from
MongoDB, generates new versions, and stores them back in the database.
"""

import sys
import os
import argparse
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import questions_collection
from core.question_versioning import (
    generate_v1_original, 
    generate_v2_reordered_mcq,
    generate_v3_mcq_with_nota,
    generate_v4_true_false_version
)

def generate_versions_from_db(q_ids, versions_to_run=None):
    """
    Fetches V1 questions from the database by their IDs, generates
    new versions (V2, V3, V4), and stores them.
    """
    print(f"🚀 Starting version generation for {len(q_ids)} question ID(s)...")
    print("=" * 70)

    total_stored = 0
    total_replaced = 0

    for i, q_id in enumerate(q_ids, 1):
        print(f"\n📝 Processing question {i}/{len(q_ids)}: {q_id}")

        # 1. Fetch the base (V1) question from the database
        v1_question_raw = questions_collection.find_one({"q_id": q_id, "q_version": "1"})
        
        if not v1_question_raw:
            print(f"  ❌ Error: V1 for question ID '{q_id}' not found in the database. Skipping.")
            continue

        # 2. Standardize the V1 question (ensures correct format for generators)
        v1_q_standardized = generate_v1_original(v1_question_raw)
        if v1_q_standardized.get('q_correct_answer') == "N/A":
             print(f"  ❌ Error: V1 for '{q_id}' has an invalid or missing correct answer. Skipping.")
             continue

        # 3. Generate new versions
        all_possible_versions = {
            "V2": ("V2 (Reordered)", generate_v2_reordered_mcq),
            "V3": ("V3 (With NOTA)", generate_v3_mcq_with_nota),
            "V4": ("V4 (True/False)", generate_v4_true_false_version)
        }

        versions_to_generate = {}
        if versions_to_run:
            for v_key in versions_to_run:
                if v_key in all_possible_versions:
                    name, func = all_possible_versions[v_key]
                    versions_to_generate[name] = func
        else: 
            versions_to_generate = {name: func for name, func in all_possible_versions.values()}
        
        if not versions_to_generate:
            print("  ⚠️ No valid versions specified to generate. Skipping.")
            continue
            
        original_q_id = v1_q_standardized.get("original_q_id", v1_q_standardized["q_id"])

        for version_name, generator_func in versions_to_generate.items():
            try:
                new_version = generator_func(v1_q_standardized)
                
                if new_version:
                    # 4. Store the new version in the database
                    if '_id' in new_version:
                        del new_version['_id'] # Remove old _id to allow insertion
                        
                    new_version["original_q_id"] = original_q_id
                    new_version["created_at"] = datetime.now().isoformat()

                    result = questions_collection.update_one(
                        {"q_id": new_version["q_id"], "q_version": new_version["q_version"]},
                        {"$set": new_version},
                        upsert=True
                    )
                    
                    if result.upserted_id:
                        total_stored += 1
                        print(f"  ✅ Stored {version_name}")
                    elif result.modified_count > 0:
                        total_replaced += 1
                        print(f"  ✅ Replaced {version_name}")
                    else:
                        print(f"  ✅ {version_name} already exists and is up-to-date.")

                else:
                    print(f"  ⚠️  Could not generate {version_name} (function returned None).")
            
            except Exception as e:
                print(f"  ❌ An unexpected error occurred while generating {version_name}: {e}")

    print("\n" + "=" * 70)
    print("🎉 Version generation from database complete!")
    print(f"📊 Summary:")
    print(f"   - Stored: {total_stored} new versions")
    print(f"   - Replaced/Updated: {total_replaced} existing versions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate V2, V3, and V4 for questions directly from the database.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'q_ids', 
        nargs='*', 
        help="Zero or more question IDs to process."
    )
    parser.add_argument(
        '--file', 
        type=str, 
        help='Path to a file containing question IDs (one per line). This can be used with IDs from the command line.'
    )
    parser.add_argument(
        '--versions',
        nargs='+',
        choices=['V2', 'V3', 'V4'],
        help="Specify which versions to generate (e.g., --versions V4 or --versions V2 V3)."
    )
    
    args = parser.parse_args()
    

    q_ids_to_process = args.q_ids
    if args.file:
        try:
            with open(args.file, 'r') as f:
                file_qids = [line.strip() for line in f if line.strip()]
                q_ids_to_process.extend(file_qids)
        except FileNotFoundError:
            print(f"❌ Error: The file '{args.file}' was not found.")
            sys.exit(1)
            
    # Remove duplicates
    q_ids_to_process = sorted(list(set(q_ids_to_process)))

    if not q_ids_to_process:
        print("❌ No question IDs provided. Please provide IDs as arguments or via a file.")
        parser.print_help()
        sys.exit(1)
        
    generate_versions_from_db(q_ids_to_process, args.versions) 