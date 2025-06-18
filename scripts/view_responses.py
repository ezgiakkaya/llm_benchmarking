import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.database import questions_collection, responses_collection

def view_llm_responses(q_id=None, model_name=None):
    """
    View LLM responses from the database.
    
    Args:
        q_id: Optional question ID to filter responses
        model_name: Optional model name to filter responses
    """
    print("🔍 Viewing LLM Responses...")
    print("=" * 80)
    
    # Build query
    query = {}
    if q_id:
        query["question_id"] = q_id
    if model_name:
        query["model_name"] = model_name
    
    # Get responses
    responses = list(responses_collection.find(query))
    
    if not responses:
        print("⚠️ No responses found matching the criteria.")
        return
    
    print(f"Found {len(responses)} responses")
    
    # Group responses by question
    questions = {}
    for response in responses:
        q_id = response.get("question_id")
        if not q_id:
            continue
        if q_id not in questions:
            questions[q_id] = []
        questions[q_id].append(response)
    
    # Print responses
    for q_id, q_responses in questions.items():
        print("\n" + "=" * 80)
        print(f"Question ID: {q_id}")
        
        # Get question details
        question = questions_collection.find_one({"q_id": q_id})
        if question:
            print(f"Question Text: {question.get('q_text', 'N/A')}")
            print(f"Question Type: {question.get('q_type', 'N/A')}")
            if question.get('q_type') == 'MCQ':
                print("\nOptions:")
                options = question.get('q_options', [])
                if isinstance(options, str):
                    try:
                        options = json.loads(options)
                    except json.JSONDecodeError:
                        print("Error parsing options JSON")
                for i, opt in enumerate(options):
                    print(f"{chr(ord('A') + i)}. {opt}")
            print(f"Correct Answer: {question.get('q_correct_answer', 'N/A')}")
        
        print("\nResponses:")
        for response in q_responses:
            print("\n" + "-" * 40)
            print(f"Model: {response.get('model_name', 'N/A')}")
            print(f"Version: {response.get('version', '1')}")
            print(f"Timestamp: {response.get('timestamp', 'N/A')}")
            
            # Print answer based on question type
            if 'mcq_answer' in response and response['mcq_answer']:
                mcq_data = response['mcq_answer']
                print(f"Selected Option: {mcq_data.get('selected_option', 'N/A')}")
                print(f"Confidence: {mcq_data.get('confidence', 0):.2f}")
                print(f"Explanation: {mcq_data.get('explanation', 'N/A')}")
            
            elif 'true_false_answer' in response and response['true_false_answer']:
                tf_data = response['true_false_answer']
                print(f"Answer: {tf_data.get('answer', 'N/A')}")
                print(f"Confidence: {tf_data.get('confidence', 0):.2f}")
                print(f"Explanation: {tf_data.get('explanation', 'N/A')}")
            
            elif 'short_answer' in response and response['short_answer']:
                sa_data = response['short_answer']
                print(f"Answer: {sa_data.get('answer', 'N/A')}")
                print(f"Confidence: {sa_data.get('confidence', 0):.2f}")
                print(f"Explanation: {sa_data.get('explanation', 'N/A')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View LLM responses from database')
    parser.add_argument('--q_id', help='Question ID to view (e.g., LEC8_001)')
    parser.add_argument('--model', help='Model name to filter responses')
    
    args = parser.parse_args()
    
    view_llm_responses(args.q_id, args.model) 