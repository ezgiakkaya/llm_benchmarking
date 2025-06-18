import os
import pandas as pd
import json
from pathlib import Path
from core.database import get_db_connection

def import_questions_from_csv(csv_path: str):
    """Import questions from a CSV file into the database."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Process each row
        for _, row in df.iterrows():
            # Convert q_options from string to JSON if it's a string
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
        
        # Commit the changes
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
    # Get all CSV files from the data directory
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*_questions.csv"))
    
    if not csv_files:
        print("No question CSV files found in the data directory")
        return
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        import_questions_from_csv(str(csv_file))

if __name__ == "__main__":
    main() 