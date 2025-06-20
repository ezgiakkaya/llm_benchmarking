#!/usr/bin/env python3
"""
Backend script to analyze all stored LLM responses, calculate accuracy and other metrics,
and generate a comprehensive performance report.
"""

import sys
import os
import pandas as pd
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import questions_collection, responses_collection

def get_option_letter(answer_text, q_options):
    """Convert answer text to its corresponding letter (A, B, C...)."""
    if not answer_text or not isinstance(q_options, list):
        return None
    try:
        index = q_options.index(answer_text)
        return chr(ord('A') + index)
    except ValueError:
        return None

def analyze_model_performance():
    """
    Analyzes model performance by comparing responses to correct answers.
    Generates and saves detailed and summary reports as CSV files.
    """
    print("🚀 Starting Performance Analysis...")
    print("=" * 70)

    all_responses = list(responses_collection.find())
    all_questions_list = list(questions_collection.find())
    
    if not all_responses:
        print("⚠️ No responses found in the database. Nothing to analyze.")
        return


    questions_lookup = {
        (q['q_id'], str(q.get('q_version', '1'))): q for q in all_questions_list
    }

    print(f"Found {len(all_responses)} responses and {len(all_questions_list)} questions to analyze.")
    
    results = []
    for response in all_responses:
        q_id = response.get("question_id")
        q_version = str(response.get("version", "1"))
        
        question = questions_lookup.get((q_id, q_version))
        
        if not question:
            print(f"Warning: Question not found for response on {q_id} v{q_version}. Skipping.")
            continue
            
        q_type = question.get("q_type")
        correct_answer = question.get("q_correct_answer")
        model_answer = None
        confidence = 0
        is_correct = False
        
        if q_type == "MCQ" and "mcq_answer" in response and response["mcq_answer"]:
            model_answer_option = response["mcq_answer"].get("selected_option")
            confidence = response["mcq_answer"].get("confidence", 0)
            correct_option = get_option_letter(correct_answer, question.get("q_options", []))
            is_correct = (correct_option == model_answer_option)
            model_answer = model_answer_option
            
        elif q_type == "True/False" and "true_false_answer" in response and response["true_false_answer"]:
            model_answer_bool = response["true_false_answer"].get("answer")
            confidence = response["true_false_answer"].get("confidence", 0)
            correct_bool = str(correct_answer).lower() == "true"
            is_correct = (correct_bool == model_answer_bool)
            model_answer = str(model_answer_bool)

        elif q_type == "Short Answer" and "short_answer" in response and response["short_answer"]:
            model_answer_text = response["short_answer"].get("answer", "")
            confidence = response["short_answer"].get("confidence", 0)
            is_correct = str(correct_answer).strip().lower() == str(model_answer_text).strip().lower()
            model_answer = model_answer_text

        results.append({
            "model": response.get("model_name"),
            "q_id": q_id,
            "version": q_version,
            "topic_tag": question.get("topic_tag"),
            "question_type": q_type,
            "is_correct": is_correct,
            "confidence": confidence,
            "correct_answer": correct_answer,
            "model_answer": model_answer,
            "question_text": question.get("q_text")
        })

    if not results:
        print("⚠️ Analysis resulted in no valid data points.")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Detailed Report
    detailed_report_path = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
    df.to_csv(detailed_report_path, index=False)
    print(f"✅ Saved detailed analysis to: {detailed_report_path}")

    # 2. Summary Report
    summary_df = df.groupby('model').agg(
        total_tests=('is_correct', 'count'),
        accuracy=('is_correct', 'mean'),
        avg_confidence=('confidence', 'mean')
    ).reset_index()
    summary_df['accuracy'] = summary_df['accuracy'] * 100
    summary_df['avg_confidence'] = summary_df['avg_confidence'] * 100
    
    summary_report_path = os.path.join(output_dir, f"summary_report_{timestamp}.csv")
    summary_df.to_csv(summary_report_path, index=False)
    print(f"✅ Saved summary report to: {summary_report_path}")
    
    # --- Print Summary to Console ---
    print("\n" + "=" * 70)
    print("📊 Model Performance Summary:")
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    analyze_model_performance() 