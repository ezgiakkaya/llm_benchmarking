import csv
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.database import questions_collection

def insert_mock_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            # Expected columns: q_id, q_text, q_type, topic_tag, q_options, q_correct_answer
            q_id, q_text, q_type, topic_tag, q_options, q_correct_answer = row
            try:
                options = json.loads(q_options)
            except Exception:
                options = []
            question = {
                'q_id': q_id.strip(),
                'q_text': q_text.strip(),
                'q_type': q_type.strip(),
                'topic_tag': topic_tag.strip(),
                'q_options': options,
                'q_correct_answer': q_correct_answer.strip(),
                'q_version': '1'
            }
            print(f'Inserting: {question}')
            questions_collection.insert_one(question)
    print('✅ Done.')

if __name__ == '__main__':
    insert_mock_csv('mock.csv') 