import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.database import questions_collection, responses_collection
from core.llm_clients import query_groq


load_dotenv()


QUESTION_ID = "LEC8_020" 


MODELS = [
    "llama3-70b-8192",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b"
]

def get_question_from_db(q_id):
    """Retrieve the question from the database."""
    question = questions_collection.find_one({"q_id": q_id})
    if not question:
        print(f"No question found with ID: {q_id}")
        sys.exit(1)
   
    if "q_options" not in question:
        versions = list(questions_collection.find({"q_id": q_id}))
        if versions:
            question["q_options"] = versions[0].get("q_options", [])
    return question

def get_all_versions_from_db(q_id):
    """Retrieve all versions of the question from the database."""
    versions = list(questions_collection.find({"q_id": q_id}))
    if not versions:
        print(f"No questions found with ID: {q_id}")
        sys.exit(1)
    return versions

def format_prompt(question):
    """Format the prompt for the LLM."""
    options = question.get("q_options", [])
    if isinstance(options, str):
        import json
        try:
            options = json.loads(options)
        except Exception:
            options = []
    options_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
    return f"""Question: {question['q_text']}

Options:
{options_str}

IMPORTANT: Respond with ONLY the letter (A, B, C, or D) of the correct option."""

def save_response_to_db(q_id, q_version, model_id, response, selected_option):
    """Save the LLM response to the database."""
    response_doc = {
        "question_id": q_id,
        "version": str(q_version),
        "model_name": model_id,
        "response": str(response),
        "selected_option": selected_option,
        "timestamp": datetime.now().isoformat()
    }
    responses_collection.insert_one(response_doc)

def main():
   
    versions = get_all_versions_from_db(QUESTION_ID)
    for question in versions:
        version = question.get("q_version", "N/A")
        prompt = format_prompt(question)
        print(f"\nProcessing version: {version}")
        
        for model_id in MODELS:
            print(f"Querying model: {model_id}")
            result = query_groq(prompt, model_id, "MCQ", question.get("q_options", []))
            if "error" in result:
                print(f"Error querying {model_id}: {result['error']}")
                continue
            selected_option = result["response"].selected_option
            save_response_to_db(QUESTION_ID, version, model_id, result["response"], selected_option)
            print(f"Model {model_id} selected option: {selected_option}")

if __name__ == "__main__":
    main() 