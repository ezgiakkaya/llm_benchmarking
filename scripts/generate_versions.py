#!/usr/bin/env python3
"""
Backend script to process an input CSV of questions, generate all versions,
and store them in MongoDB, replacing existing ones.
"""

import sys
import os
import pandas as pd
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import questions_collection
from core.question_versioning import generate_all_versions_for_question

def store_question_versions(question_versions, original_q_id):
    """Store generated question versions in the database, replacing existing versions."""
    stored_count = 0
    replaced_count = 0
    
 
    existing_versions_cursor = questions_collection.find(
        {"original_q_id": original_q_id},
        {"q_id": 1, "q_version": 1}
    )
    existing_db_versions = {(v['q_id'], v['q_version']) for v in existing_versions_cursor}

    generated_version_ids = set()

    for version_dict in question_versions:
        try:
            # Add metadata
            version_dict["original_q_id"] = original_q_id
            version_dict["created_at"] = datetime.now().isoformat()
            
           
            generated_version_ids.add((version_dict['q_id'], version_dict['q_version']))

         
            result = questions_collection.update_one(
                {"q_id": version_dict["q_id"], "q_version": version_dict["q_version"]},
                {"$set": version_dict},
                upsert=True
            )
            
            if result.upserted_id is not None:
                stored_count += 1
            elif result.modified_count > 0:
                replaced_count += 1
            
        except Exception as e:
            print(f"❌ Error storing version {version_dict.get('q_version')} of {version_dict.get('q_id')}: {e}")


    versions_to_delete = existing_db_versions - generated_version_ids
    if versions_to_delete:
        print(f"🧹 Found {len(versions_to_delete)} old version(s) to remove...")
        for q_id_to_del, q_version_to_del in versions_to_delete:
            questions_collection.delete_one({"q_id": q_id_to_del, "q_version": q_version_to_del})
            print(f"   -> Removed old version: {q_id_to_del} v{q_version_to_del}")
            
    return stored_count, replaced_count

def process_questions_from_csv(file_path):
    """Main function to read a CSV and generate versions for all questions within it."""
    print("🚀 Starting Question Version Generation from CSV...")
    print("=" * 70)
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Input file not found at '{file_path}'")
        return

    try:
        df = pd.read_csv(file_path)
        # Ensure q_options is parsed correctly if it's a string representation of a list
        if 'q_options' in df.columns:
            df['q_options'] = df['q_options'].apply(lambda x: pd.io.json.loads(x) if isinstance(x, str) and x.startswith('[') else [])
        questions_list = df.to_dict('records')
        print(f"📊 Successfully read {len(questions_list)} questions from {file_path}")
    except Exception as e:
        print(f"❌ Error reading or parsing CSV file: {e}")
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
         
            generated_versions = generate_all_versions_for_question(question)
            
            if generated_versions:
                print(f"   -> Generated {len(generated_versions)} versions.")
            
                stored, replaced = store_question_versions(generated_versions, original_q_id)
                total_stored += stored
                total_replaced += replaced
                
                print(f"   -> Stored {stored} new versions, replaced {replaced} existing versions.")
            else:
                print(f"   ❌ Failed to generate any versions for {original_q_id}")
                
        except Exception as e:
            print(f"   ❌ An unexpected error occurred while processing {original_q_id}: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Question version generation complete!")
    print(f"📊 Summary:")
    print(f"   - Processed: {len(questions_list)} questions")
    print(f"   - Stored: {total_stored} new versions")
    print(f"   - Replaced/Updated: {total_replaced} existing versions")

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_versions.py <path_to_csv_file>")
        print("Example: python scripts/generate_versions.py data/mfa_questions.csv")
        sys.exit(1)
        
    csv_file_path = sys.argv[1]
    process_questions_from_csv(csv_file_path) 