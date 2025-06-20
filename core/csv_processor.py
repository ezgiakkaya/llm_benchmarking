import csv
import json
from typing import List, Dict, Any
from .database import questions_collection
from .question_versioning import generate_all_versions_for_question

def parse_csv_to_questions(csv_path: str) -> List[Dict[str, Any]]:
    """
    Parse a CSV file containing questions into a list of question dictionaries.
    Each question will be in the format expected by the question versioning system.
    """
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
          
            if isinstance(row['q_options'], str):
                try:
                  
                    cleaned_options = row['q_options'].replace('\\"', '"').strip()
                  
                    if cleaned_options.startswith('"') and cleaned_options.endswith('"'):
                        cleaned_options = cleaned_options[1:-1]
                    row['q_options'] = json.loads(cleaned_options)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse q_options for question {row['q_id']}: {str(e)}")
                    print(f"Raw options string: {row['q_options']}")
                    row['q_options'] = []

            if isinstance(row['q_correct_answer'], str):
                row['q_correct_answer'] = row['q_correct_answer'].replace('\\"', '').strip()
              
                if row['q_correct_answer'].startswith('"') and row['q_correct_answer'].endswith('"'):
                    row['q_correct_answer'] = row['q_correct_answer'][1:-1]
            
            questions.append(row)
    return questions

def save_questions_to_database(questions: List[Dict[str, Any]]) -> None:
    """
    Save questions to the database, generating versions for each question.
    """
    for question in questions:
        # Generate all versions for the question
        versions = generate_all_versions_for_question(question)
        
      
        for version in versions:
          
            if isinstance(version.get('q_options'), list):
                version['q_options'] = json.dumps(version['q_options'])
            
            # Insert into database
            questions_collection.insert_one(version)

def process_lecture_csv(csv_path: str) -> None:
    """
    Process a lecture CSV file: parse it, generate versions, and save to database.
    """
    print(f"Processing CSV file: {csv_path}")
    
    # Parse CSV to questions
    questions = parse_csv_to_questions(csv_path)
    print(f"Parsed {len(questions)} questions from CSV")
    
    # Save to database with versions
    save_questions_to_database(questions)
    print("Successfully saved all questions and their versions to database")

if __name__ == "__main__":
    # Example usage
    process_lecture_csv("lecture8.csv") 