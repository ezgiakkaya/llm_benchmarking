import os
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.csv_processor import process_lecture_csv

def main():
  
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "lecture8.csv"

  
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(project_root, csv_path)
    

    process_lecture_csv(csv_path)

if __name__ == "__main__":
    main() 