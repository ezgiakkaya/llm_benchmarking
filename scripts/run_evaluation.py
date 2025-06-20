#!/usr/bin/env python3
"""
Backend script to run evaluations by querying LLMs for all questions
in the database and storing their structured responses.
"""

import sys
import os
from datetime import datetime
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import questions_collection, responses_collection
from core.llm_clients import query_groq, GROQ_MODELS
from core.models import LLMResponse

def run_evaluation_for_all_questions(models_to_test=None):
    """
    Fetches all question versions from the database and queries the specified LLMs.
    
    Args:
        models_to_test (list, optional): A list of model names to test. 
                                         If None, all models in GROQ_MODELS will be tested.
    """
    print("🚀 Starting LLM Evaluation for All Questions...")
    print("=" * 70)

   
    if models_to_test:
        target_models = {k: v for k, v in GROQ_MODELS.items() if v['name'] in models_to_test}
        if not target_models:
            print(f"❌ Error: None of the specified models {models_to_test} are available.")
            return
    else:
        target_models = GROQ_MODELS
        
    print(f"🤖 Models to be tested: {', '.join([m['name'] for m in target_models.values()])}")
    
    all_questions = list(questions_collection.find())
    if not all_questions:
        print("⚠️ No questions found in the database. Aborting evaluation.")
        return

    print(f"📚 Found {len(all_questions)} total question versions to test against.")
    
    total_tests = len(all_questions) * len(target_models)
    print(f"📈 Total API calls to be made: {total_tests}")

    with tqdm(total=total_tests, desc="Running Evaluations") as progress_bar:
        for question in all_questions:
            for model_id, model_info in target_models.items():
                
                # Check if a response already exists for this specific question version and model
                existing_response = responses_collection.find_one({
                    "question_id": question["q_id"],
                    "version": str(question.get("q_version", "1")),
                    "model_name": model_info["name"]
                })
                
                if existing_response:
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(f"Skipped {question['q_id']} v{question.get('q_version', '1')} for {model_info['name']} (already exists)")
                    continue

                try:
                    # Use the structured query function from llm_clients
                    groq_response = query_groq(
                        question['q_text'],
                        model_id,
                        question['q_type'],
                        question.get('q_options')
                    )

                    if "error" not in groq_response:
                        # Create structured LLM response object
                        llm_response = LLMResponse(
                            question_id=question["q_id"],
                            question_text=question["q_text"],
                            question_type=question["q_type"],
                            model_name=model_info["name"],
                            timestamp=datetime.now().isoformat(),
                            version=str(question.get('q_version', "1"))
                        )
                        
                        # Add the appropriate answer based on type
                        if question['q_type'] == "MCQ":
                            llm_response.mcq_answer = groq_response["response"]
                        elif question['q_type'] == "True/False":
                            llm_response.true_false_answer = groq_response["response"]
                        else:
                            llm_response.short_answer = groq_response["response"]
                        
                        # Save the structured response to the database
                        responses_collection.insert_one(llm_response.model_dump())
                    else:
                        print(f"\n❌ Error from {model_info['name']} for {question['q_id']}: {groq_response['error']}")

                except Exception as e:
                    print(f"\n❌ Critical error testing {model_info['name']} on {question['q_id']}: {e}")
                
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Tested {question['q_id']} v{question.get('q_version', '1')} on {model_info['name']}")

    print("\n" + "=" * 70)
    print("🎉 LLM evaluation complete!")

if __name__ == "__main__":
    # Example: run 'python scripts/run_evaluation.py' to test all models.
    # Example: run 'python scripts/run_evaluation.py LLaMA3-8b-8192' to test a specific model.
    # Example: run 'python scripts/run_evaluation.py LLaMA3-8b-8192 Gemma-7b-It' to test multiple.
    
    models_to_test_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    
    run_evaluation_for_all_questions(models_to_test=models_to_test_arg) 