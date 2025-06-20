import os
import pandas as pd
import json
from pathlib import Path
from core.database import get_db_connection

def import_questions_from_csv(csv_path: str):
    """Import questions from a CSV file into the database."""
    try:

        df = pd.read_csv(csv_path)
        

        conn = get_db_connection()
        cursor = conn.cursor()
        

        for _, row in df.iterrows():
       
            if isinstance(row['q_options'], str):
                try:
                    options = json.loads(row['q_options'])
                except json.JSONDecodeError:
                    options = row['q_options']
            else:
                options = row['q_options']
            
            # Insert the question into the database
            cursor.execute("""
                INSERT INTO questions (
                    q_id, q_text, q_type, topic_tag, q_options, q_correct_answer
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row['q_id'],
                row['q_text'],
                row['q_type'],
                row['topic_tag'],
                json.dumps(options),
                row['q_correct_answer']
            ))
        

        conn.commit()
        print(f"Successfully imported questions from {csv_path}")
        
    except Exception as e:
        print(f"Error importing questions from {csv_path}: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def main():
   
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*_questions.csv"))
    
    if not csv_files:
        print("No question CSV files found in the data directory")
        return
    
   
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        import_questions_from_csv(str(csv_file))

if __name__ == "__main__":
    main() 