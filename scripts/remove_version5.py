import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.database import questions_collection

def remove_version5():
    result = questions_collection.delete_many({"q_version": "5"})
    print(f"Removed {result.deleted_count} version 5 questions from the database.")

if __name__ == "__main__":
    remove_version5() 