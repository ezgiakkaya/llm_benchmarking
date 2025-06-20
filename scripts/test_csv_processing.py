import os
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.csv_processor import parse_csv_to_questions

def test_csv_parsing():
    """Test the CSV parsing functionality without database operations."""
    csv_path = os.path.join(project_root, "lecture8.csv")
    
    print(f"Testing CSV parsing with file: {csv_path}")
    print("-" * 50)
    
    try:
       
        questions = parse_csv_to_questions(csv_path)
        
       
        print(f"\nSuccessfully parsed {len(questions)} questions")
        
      
        if questions:
            print("\nSample question:")
            print(f"ID: {questions[0]['q_id']}")
            print(f"Text: {questions[0]['q_text']}")
            print(f"Type: {questions[0]['q_type']}")
            print(f"Topic: {questions[0]['topic_tag']}")
            print(f"Options: {questions[0]['q_options']}")
            print(f"Correct Answer: {questions[0]['q_correct_answer']}")
        
        return True
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_csv_parsing()
    sys.exit(0 if success else 1) 