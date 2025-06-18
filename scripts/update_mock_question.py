import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.database import questions_collection

def update_mock_question():
    # Define the updated question
    updated_question = {
        'q_id': 'LEC8_021',
        'q_text': 'which is the first letter of the alphabet?',
        'q_type': 'MCQ',
        'topic_tag': 'Case Studies',
        'q_options': ['B', 'D', 'A', 'C'],
        'q_correct_answer': 'A',
        'q_version': '1'
    }
    # Update the question in the database
    questions_collection.update_one(
        {'q_id': 'LEC8_021'},
        {'$set': updated_question}
    )
    print('✅ Updated question LEC8_021 with reordered options.')

if __name__ == '__main__':
    update_mock_question() 