import sys
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.database import questions_collection, responses_collection

def get_option_letter(correct_answer: str, options: List[str]) -> str:
    """Convert a correct answer text to its corresponding option letter."""
    if not options:
        return ""
    try:
        index = options.index(correct_answer)
        return chr(ord('A') + index)
    except ValueError:
        return ""

def calculate_basic_accuracy(model_name: str = None, question_range: Tuple[str, str] = None) -> pd.DataFrame:
    """
    Calculate basic accuracy for LLM responses.
    
    Args:
        model_name: Optional filter for specific model. If None, calculates for all models.
        question_range: Optional tuple of (start_id, end_id) to filter questions.
    
    Returns:
        DataFrame with accuracy metrics per model and question type
    """
    print("📊 Calculating Basic Accuracy Metrics...")
    
    # Get all responses
    query = {"model_name": model_name} if model_name else {}
    responses = list(responses_collection.find(query))
    
    if not responses:
        print("⚠️ No responses found in database.")
        return pd.DataFrame()
    

    if question_range:
        start_id, end_id = question_range
        responses = [
            r for r in responses 
            if start_id <= r.get("question_id", "") <= end_id
        ]
        print(f"Filtered to {len(responses)} responses for questions {start_id} to {end_id}")
    
    results = []
    for response in responses:
        q_id = response.get("question_id")
        q_version = str(response.get("version", "1"))
        

        question = questions_collection.find_one({
            "q_id": q_id,
            "q_version": q_version
        })
        
        if not question:
            print(f"⚠️ Question not found for {q_id} v{q_version}")
            continue
        
        q_type = question.get("q_type")
        correct_answer = question.get("q_correct_answer")
        is_correct = False
        confidence = 0
        
        # Compare answers based on question type
        if q_type == "MCQ" and "mcq_answer" in response:
            mcq_data = response["mcq_answer"]
            model_answer = mcq_data.get("selected_option", "")
            confidence = mcq_data.get("confidence", 0)
            
            # Convert correct answer to option letter
            correct_option = get_option_letter(correct_answer, question.get("q_options", []))
            is_correct = (correct_option == model_answer)
            
        elif q_type == "True/False" and "true_false_answer" in response:
            tf_data = response["true_false_answer"]
            model_answer_bool = tf_data.get("answer", False)
            confidence = tf_data.get("confidence", 0)
            

            correct_bool = str(correct_answer).lower() == "true"
            is_correct = (correct_bool == model_answer_bool)
            
        elif q_type == "Short Answer" and "short_answer" in response:
            sa_data = response["short_answer"]
            model_answer = sa_data.get("answer", "")
            confidence = sa_data.get("confidence", 0)
            
            # Simple text comparison (case-insensitive)
            is_correct = str(correct_answer).lower().strip() == str(model_answer).lower().strip()
        
        results.append({
            "model": response.get("model_name", "unknown"),
            "q_id": q_id,
            "version": q_version,
            "q_type": q_type,
            "is_correct": is_correct,
            "confidence": confidence
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate metrics
    if not df.empty:
        # Overall accuracy
        overall_accuracy = df.groupby("model")["is_correct"].mean()
        
        # Accuracy by question type
        type_accuracy = df.groupby(["model", "q_type"])["is_correct"].mean().unstack()
        
        # Average confidence
        avg_confidence = df.groupby("model")["confidence"].mean()
        
        # Combine metrics
        metrics = pd.DataFrame({
            "Overall Accuracy": overall_accuracy,
            "Average Confidence": avg_confidence
        }).join(type_accuracy)
        
        # Format as percentages
        metrics = metrics.multiply(100).round(2)
        
        print("\n📈 Basic Accuracy Metrics:")
        print(metrics)
        
        # Save to CSV
        output_file = "reports/basic_accuracy_metrics.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        metrics.to_csv(output_file)
        print(f"\n✅ Metrics saved to {output_file}")
        
        # Print detailed results
        print("\n📝 Detailed Results by Question:")
        detailed_results = df.groupby(["model", "q_id", "version", "q_type"]).agg({
            "is_correct": "first",
            "confidence": "first"
        }).reset_index()
        print(detailed_results)
        
        return metrics
    else:
        print("❌ No valid results to analyze")
        return pd.DataFrame()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate LLM performance metrics')
    parser.add_argument('--model', help='Specific model to analyze (optional)')
    parser.add_argument('--start_id', default="LEC8_001", help='Start question ID (default: LEC8_001)')
    parser.add_argument('--end_id', default="LEC8_020", help='End question ID (default: LEC8_020)')
    
    args = parser.parse_args()
    
    calculate_basic_accuracy(args.model, (args.start_id, args.end_id)) 