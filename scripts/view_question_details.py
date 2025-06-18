import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.database import questions_collection

def view_question_details(q_id=None, q_version=None):
    """
    View detailed information about specific questions.
    If no parameters are provided, shows all LEC8_ questions.
    If q_id is provided, shows all versions of that question.
    If both q_id and q_version are provided, shows that specific version.
    """
    query = {}
    if q_id:
        query["q_id"] = q_id
    else:
        query["q_id"] = {"$regex": "^LEC8_"}
    
    if q_version is not None:
        query["q_version"] = q_version

    # Get all matching questions
    questions = list(questions_collection.find(query, {"_id": 0}))
    
    if not questions:
        print(f"No questions found matching the criteria.")
        return

    # Print details for each question
    for q in questions:
        print("\n" + "="*80)
        print(f"Question ID: {q['q_id']}")
        print(f"Version: {q['q_version']}")
        print("-"*80)
        print(f"Question Text:\n{q['q_text']}")
        print("-"*80)
        
        # Handle options based on question type
        if q['q_type'] == 'MCQ':
            print("Options:")
            try:
                # Try to parse options if they're stored as a JSON string
                options = q['q_options']
                if isinstance(options, str):
                    options = json.loads(options)
                for i, opt in enumerate(options):
                    print(f"{chr(ord('A') + i)}. {opt}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing options: {e}")
                print("Raw options:", q['q_options'])
        print("-"*80)
        print(f"Correct Answer: {q['q_correct_answer']}")
        print("="*80 + "\n")

    print(f"\nTotal questions found: {len(questions)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View detailed information about questions')
    parser.add_argument('--q_id', help='Question ID to view (e.g., LEC8_1)')
    parser.add_argument('--version', type=int, help='Specific version number to view')
    
    args = parser.parse_args()
    
    view_question_details(args.q_id, args.version) 