from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["comp430_benchmark"]

questions_collection = db["questions"]
responses_collection = db["responses"]

def ensure_question_versioning():
    """Ensure all questions have original_q_id field for version tracking."""
    # Find all questions without original_q_id
    questions_without_version = questions_collection.find({"original_q_id": {"$exists": False}})
    
    for question in questions_without_version:
        # For version 1 questions, set original_q_id to their q_id
        if question.get("q_version") == "1" or "q_version" not in question:
            questions_collection.update_one(
                {"_id": question["_id"]},
                {"$set": {"original_q_id": question["q_id"], "q_version": "1"}}
            )
            print(f"Updated question {question['q_id']} with original_q_id")

print("📦 Initialized MongoDB connection.")
ensure_question_versioning() 