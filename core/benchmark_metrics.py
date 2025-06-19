import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

def calculate_basic_accuracy(responses: List[Dict], questions_collection) -> Dict[str, float]:
    """
    Calculate Basic Accuracy (Acc) for each model.
    
    Args:
        responses: List of response documents from database
        questions_collection: MongoDB collection of questions
        
    Returns:
        Dictionary mapping model names to accuracy scores
    """
    model_accuracies = {}
    model_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for response in responses:
        model_name = response.get("model_name", "Unknown")
        question_id = response.get("question_id")
        response_version = str(response.get("version", "1"))
        
        # Find the question with exact version match
        question = questions_collection.find_one({
            "q_id": question_id,
            "q_version": response_version
        })
        
        if not question:
            # Fallback to version 1 or no version field
            question = questions_collection.find_one({
                "q_id": question_id,
                "$or": [
                    {"q_version": "1"},
                    {"q_version": {"$exists": False}}
                ]
            })
        
        if not question:
            continue
            
        # Check if answer is correct
        is_correct = False
        q_type = question.get("q_type")
        correct_answer = question.get("q_correct_answer")
        
        if q_type == "MCQ" and "mcq_answer" in response:
            mcq_data = response["mcq_answer"]
            model_answer = mcq_data.get("selected_option", "")
            correct_option = get_option_letter(correct_answer, question.get("q_options", []))
            is_correct = correct_option == model_answer
            
        elif q_type == "True/False" and "true_false_answer" in response:
            tf_data = response["true_false_answer"]
            model_answer_bool = tf_data.get("answer", False)
            correct_bool = str(correct_answer).lower() == "true"
            is_correct = correct_bool == model_answer_bool
            
        elif q_type == "Short Answer" and "short_answer" in response:
            sa_data = response["short_answer"]
            model_answer = sa_data.get("answer", "")
            is_correct = str(correct_answer).lower().strip() == str(model_answer).lower().strip()
        
        model_stats[model_name]['total'] += 1
        if is_correct:
            model_stats[model_name]['correct'] += 1
    
    # Calculate accuracy percentages
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            model_accuracies[model] = (stats['correct'] / stats['total']) * 100
        else:
            model_accuracies[model] = 0.0
    
    return model_accuracies

def get_option_letter(answer_text, q_options):
    """Convert answer text to option letter (A, B, C, D)."""
    if not answer_text or not q_options:
        return None
    
    # Handle "Option X" format
    if answer_text.startswith("Option "):
        option_letter = answer_text.split(" ")[1]
        return option_letter if option_letter.isalpha() else None
    
    # Handle direct text match
    try:
        index = q_options.index(answer_text)
        return chr(ord('A') + index)
    except (ValueError, IndexError):
        return None

def calculate_permutation_robustness_score(responses: List[Dict], questions_collection) -> Dict[str, float]:
    """
    Calculate Permutation Robustness Score (PRS) for each model.
    
    This measures consistency when MCQ choices are reordered between V1 (original) and V2 (reordered).
    
    Args:
        responses: List of response documents from database
        questions_collection: MongoDB collection of questions
        
    Returns:
        Dictionary mapping model names to PRS scores
    """
    model_prs = {}
    model_consistency = defaultdict(lambda: defaultdict(dict))
    
    # Group responses by model and original question ID, tracking versions
    for response in responses:
        model_name = response.get("model_name", "Unknown")
        question_id = response.get("question_id")
        response_version = str(response.get("version", "1"))
        
        # Find the question with exact version match
        question = questions_collection.find_one({
            "q_id": question_id,
            "q_version": response_version
        })
        
        if not question or question.get("q_type") != "MCQ":
            continue
            
        if "mcq_answer" not in response:
            continue
            
        # Only consider V1 (original) and V2 (reordered) for PRS calculation
        if response_version not in ["1", "2"]:
            continue
            
        mcq_data = response["mcq_answer"]
        selected_option = mcq_data.get("selected_option", "")
        
        # Get the original question ID for grouping
        original_q_id = question.get("original_q_id", question_id)
        
        # Convert selected option to actual answer text
        q_options = question.get("q_options", [])
        selected_answer_text = None
        
        if selected_option and len(selected_option) == 1 and selected_option.isalpha():
            option_index = ord(selected_option.upper()) - ord('A')
            if 0 <= option_index < len(q_options):
                selected_answer_text = q_options[option_index]
        
        if selected_answer_text:
            model_consistency[model_name][original_q_id][response_version] = selected_answer_text
    
    # Calculate consistency for each model
    for model, question_responses in model_consistency.items():
        total_question_pairs = 0
        consistent_pairs = 0
        
        for original_q_id, version_responses in question_responses.items():
            # Check if we have both V1 and V2 responses for this question
            if "1" in version_responses and "2" in version_responses:
                total_question_pairs += 1
                # Check if the selected answer text is the same across versions
                if version_responses["1"] == version_responses["2"]:
                    consistent_pairs += 1
        
        if total_question_pairs > 0:
            model_prs[model] = (consistent_pairs / total_question_pairs) * 100
        else:
            model_prs[model] = 0.0
    
    return model_prs

def calculate_distractor_sensitivity_score(responses: List[Dict], questions_collection) -> Dict[str, float]:
    """
    Calculate Distractor Sensitivity Score (DSS) for each model.
    
    This measures how often models are misled by "None of the above" distractors in V3 questions.
    
    Args:
        responses: List of response documents from database
        questions_collection: MongoDB collection of questions
        
    Returns:
        Dictionary mapping model names to DSS scores
    """
    model_dss = {}
    model_stats = defaultdict(lambda: {'v3_questions': 0, 'misled_by_nota': 0})
    
    for response in responses:
        model_name = response.get("model_name", "Unknown")
        question_id = response.get("question_id")
        response_version = str(response.get("version", "1"))
        
        # Only consider V3 (None of the above) questions for DSS
        if response_version != "3":
            continue
        
        # Find the V3 question
        question = questions_collection.find_one({
            "q_id": question_id,
            "q_version": "3"
        })
        
        if not question or question.get("q_type") != "MCQ":
            continue
            
        if "mcq_answer" not in response:
            continue
            
        q_options = question.get("q_options", [])
        
        # Verify this is actually a V3 question with "None of the above"
        has_nota = any("none of the above" in option.lower() for option in q_options)
        
        if not has_nota:
            continue
            
        model_stats[model_name]['v3_questions'] += 1
        
        mcq_data = response["mcq_answer"]
        selected_option = mcq_data.get("selected_option", "")
        
        # Check if model selected "None of the above" option
        if selected_option and len(selected_option) == 1 and selected_option.isalpha():
            option_index = ord(selected_option.upper()) - ord('A')
            if 0 <= option_index < len(q_options):
                selected_answer = q_options[option_index]
                if "none of the above" in selected_answer.lower():
                    # Get the correct answer from the original V1 question to check if this was wrong
                    original_q_id = question.get("original_q_id", question_id)
                    v1_question = questions_collection.find_one({
                        "original_q_id": original_q_id,
                        "q_version": "1"
                    })
                    
                    if v1_question:
                        v1_correct_answer = v1_question.get("q_correct_answer")
                        # If the original correct answer wasn't "None of the above", then model was misled
                        if v1_correct_answer and "none of the above" not in v1_correct_answer.lower():
                            model_stats[model_name]['misled_by_nota'] += 1
    
    # Calculate DSS (lower is better - less sensitivity to distractors)
    for model, stats in model_stats.items():
        if stats['v3_questions'] > 0:
            model_dss[model] = (stats['misled_by_nota'] / stats['v3_questions']) * 100
        else:
            model_dss[model] = 0.0
    
    return model_dss

def calculate_rag_improvement_delta(responses: List[Dict], questions_collection) -> Dict[str, float]:
    """
    Calculate RAG Improvement Delta (ΔRAG) for each model.
    
    This measures the change in accuracy when using RAG vs standard approach.
    
    Args:
        responses: List of response documents from database
        questions_collection: MongoDB collection of questions
        
    Returns:
        Dictionary mapping model names to ΔRAG scores
    """
    model_deltas = {}
    
    # Separate RAG and non-RAG responses
    rag_responses = [r for r in responses if r.get("rag_enabled", False)]
    non_rag_responses = [r for r in responses if not r.get("rag_enabled", False)]
    
    # Calculate accuracy for RAG responses
    rag_accuracies = calculate_basic_accuracy(rag_responses, questions_collection)
    
    # Calculate accuracy for non-RAG responses
    non_rag_accuracies = calculate_basic_accuracy(non_rag_responses, questions_collection)
    
    # Calculate delta for each model
    all_models = set()
    for model in rag_accuracies.keys():
        base_model = model.replace(" (RAG)", "")
        all_models.add(base_model)
    for model in non_rag_accuracies.keys():
        all_models.add(model)
    
    for base_model in all_models:
        rag_model = f"{base_model} (RAG)"
        
        rag_acc = rag_accuracies.get(rag_model, 0)
        non_rag_acc = non_rag_accuracies.get(base_model, 0)
        
        if rag_acc > 0 or non_rag_acc > 0:
            model_deltas[base_model] = rag_acc - non_rag_acc
        else:
            model_deltas[base_model] = 0.0
    
    return model_deltas

def calculate_performance_by_topic(responses: List[Dict], questions_collection) -> pd.DataFrame:
    """
    Calculate model performance metrics grouped by topic, tracking question versions.
    
    Args:
        responses: List of response documents from database
        questions_collection: MongoDB collection of questions
        
    Returns:
        DataFrame with performance metrics by model and topic
    """
    results = []
    
    for response in responses:
        model_name = response.get("model_name", "Unknown")
        question_id = response.get("question_id")
        response_version = str(response.get("version", "1"))
        
        # Find the question with exact version match
        question = questions_collection.find_one({
            "q_id": question_id,
            "q_version": response_version
        })
        
        if not question:
            # Fallback to version 1 or no version field
            question = questions_collection.find_one({
                "q_id": question_id,
                "$or": [
                    {"q_version": "1"},
                    {"q_version": {"$exists": False}}
                ]
            })
        
        if not question:
            continue
            
        topic = question.get("topic_tag", "Unknown")
        q_version = question.get("q_version", "1")
        
        # Check if answer is correct
        is_correct = False
        confidence = 0
        q_type = question.get("q_type")
        correct_answer = question.get("q_correct_answer")
        
        if q_type == "MCQ" and "mcq_answer" in response:
            mcq_data = response["mcq_answer"]
            model_answer = mcq_data.get("selected_option", "")
            confidence = mcq_data.get("confidence", 0)
            correct_option = get_option_letter(correct_answer, question.get("q_options", []))
            is_correct = correct_option == model_answer
            
        elif q_type == "True/False" and "true_false_answer" in response:
            tf_data = response["true_false_answer"]
            model_answer_bool = tf_data.get("answer", False)
            confidence = tf_data.get("confidence", 0)
            correct_bool = str(correct_answer).lower() == "true"
            is_correct = correct_bool == model_answer_bool
            
        elif q_type == "Short Answer" and "short_answer" in response:
            sa_data = response["short_answer"]
            model_answer = sa_data.get("answer", "")
            confidence = sa_data.get("confidence", 0)
            is_correct = str(correct_answer).lower().strip() == str(model_answer).lower().strip()
        
        results.append({
            "model": model_name,
            "topic": topic,
            "question_version": q_version,
            "is_correct": is_correct,
            "confidence": confidence
        })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Group by model and topic
    topic_metrics = df.groupby(["model", "topic"]).agg({
        "is_correct": ["count", "mean", "sum"],
        "confidence": "mean",
        "question_version": lambda x: list(set(x))  # Track which versions were tested
    }).round(3)
    
    topic_metrics.columns = ["Total Questions", "Accuracy", "Correct Answers", "Avg Confidence", "Versions Tested"]
    topic_metrics["Accuracy"] = topic_metrics["Accuracy"] * 100
    topic_metrics["Avg Confidence"] = topic_metrics["Avg Confidence"] * 100
    
    return topic_metrics.reset_index()

def calculate_version_specific_metrics(responses: List[Dict], questions_collection) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy metrics broken down by question version.
    
    Args:
        responses: List of response documents from database
        questions_collection: MongoDB collection of questions
        
    Returns:
        Dictionary mapping model names to version-specific accuracy scores
    """
    version_metrics = {}
    model_version_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for response in responses:
        model_name = response.get("model_name", "Unknown")
        question_id = response.get("question_id")
        response_version = str(response.get("version", "1"))
        
        # Find the question with exact version match
        question = questions_collection.find_one({
            "q_id": question_id,
            "q_version": response_version
        })
        
        if not question:
            continue
            
        # Check if answer is correct
        is_correct = False
        q_type = question.get("q_type")
        correct_answer = question.get("q_correct_answer")
        
        if q_type == "MCQ" and "mcq_answer" in response:
            mcq_data = response["mcq_answer"]
            model_answer = mcq_data.get("selected_option", "")
            correct_option = get_option_letter(correct_answer, question.get("q_options", []))
            is_correct = correct_option == model_answer
            
        elif q_type == "True/False" and "true_false_answer" in response:
            tf_data = response["true_false_answer"]
            model_answer_bool = tf_data.get("answer", False)
            correct_bool = str(correct_answer).lower() == "true"
            is_correct = correct_bool == model_answer_bool
            
        elif q_type == "Short Answer" and "short_answer" in response:
            sa_data = response["short_answer"]
            model_answer = sa_data.get("answer", "")
            is_correct = str(correct_answer).lower().strip() == str(model_answer).lower().strip()
        
        model_version_stats[model_name][response_version]['total'] += 1
        if is_correct:
            model_version_stats[model_name][response_version]['correct'] += 1
    
    # Calculate accuracy percentages by version
    for model, version_stats in model_version_stats.items():
        version_metrics[model] = {}
        for version, stats in version_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                version_name = {
                    "1": "V1 (Original)",
                    "2": "V2 (Reordered)",
                    "3": "V3 (None of Above)",
                    "4": "V4 (True/False)"
                }.get(version, f"V{version}")
                version_metrics[model][version_name] = accuracy
            else:
                version_metrics[model][version_name] = 0.0
    
    return version_metrics

def calculate_comprehensive_benchmark_metrics(responses: List[Dict], questions_collection) -> Dict[str, Any]:
    """
    Calculate all benchmark metrics in one go, including version-specific tracking.
    
    Args:
        responses: List of response documents from database
        questions_collection: MongoDB collection of questions
        
    Returns:
        Dictionary containing all calculated metrics
    """
    return {
        "basic_accuracy": calculate_basic_accuracy(responses, questions_collection),
        "permutation_robustness": calculate_permutation_robustness_score(responses, questions_collection),
        "distractor_sensitivity": calculate_distractor_sensitivity_score(responses, questions_collection),
        "rag_improvement_delta": calculate_rag_improvement_delta(responses, questions_collection),
        "performance_by_topic": calculate_performance_by_topic(responses, questions_collection),
        "version_specific_metrics": calculate_version_specific_metrics(responses, questions_collection)
    } 