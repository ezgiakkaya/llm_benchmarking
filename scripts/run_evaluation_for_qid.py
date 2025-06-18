#!/usr/bin/env python3
import sys
import os
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import questions_collection, responses_collection
from core.llm_clients import query_groq, GROQ_MODELS
from core.models import LLMResponse

def run_evaluation_for_qid(q_id, models_to_test=None):
    print(f"🚀 Starting LLM Evaluation for Question ID: {q_id}")
    print("=" * 70)

    # Determine which models to run
    if models_to_test:
        target_models = {k: v for k, v in GROQ_MODELS.items() if v['name'] in models_to_test}
        if not target_models:
            print(f"❌ Error: None of the specified models {models_to_test} are available.")
            return
    else:
        target_models = GROQ_MODELS

    questions = list(questions_collection.find({"q_id": q_id}))
    if not questions:
        print(f"⚠️ No questions found in the database with q_id={q_id}. Aborting evaluation.")
        return

    print(f"📚 Found {len(questions)} version(s) for q_id={q_id} to test against.")
    total_tests = len(questions) * len(target_models)
    print(f"📈 Total API calls to be made: {total_tests}")

    with tqdm(total=total_tests, desc="Running Evaluations") as progress_bar:
        for question in questions:
            for model_id, model_info in target_models.items():
                existing_response = responses_collection.find_one({
                    "question_id": question["q_id"],
                    "version": str(question.get('q_version', "1")),
                    "model_name": model_info["name"]
                })
                if existing_response:
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(f"Skipped {question['q_id']} v{question.get('q_version', '1')} for {model_info['name']} (already exists)")
                    continue

                try:
                    groq_response = query_groq(
                        question['q_text'],
                        model_id,
                        question['q_type'],
                        question.get('q_options')
                    )

                    if "error" not in groq_response:
                        llm_response = LLMResponse(
                            question_id=question["q_id"],
                            question_text=question["q_text"],
                            question_type=question["q_type"],
                            model_name=model_info["name"],
                            timestamp=datetime.now().isoformat(),
                            version=str(question.get('q_version', "1"))
                        )
                        if question['q_type'] == "MCQ":
                            llm_response.mcq_answer = groq_response["response"]
                        elif question['q_type'] == "True/False":
                            llm_response.true_false_answer = groq_response["response"]
                        else:
                            llm_response.short_answer = groq_response["response"]
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
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_evaluation_for_qid.py <q_id>")
        sys.exit(1)
    q_id = sys.argv[1]
    models_to_test_arg = sys.argv[2:] if len(sys.argv) > 2 else None
    run_evaluation_for_qid(q_id, models_to_test=models_to_test_arg) 