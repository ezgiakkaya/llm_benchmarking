import csv
import json
import random
import os
import time
import requests
from dotenv import load_dotenv
import re

from .llm_clients import llm_call_for_generation

load_dotenv()

def get_option_letter(index):
    """Converts a 0-based index to an option letter (0->A, 1->B)."""
    if 0 <= index < 26:
        return chr(ord('A') + index)
    return str(index + 1)

def standardize_mcq_input(question_dict):
    """
    Standardizes the input MCQ dictionary.
    Ensures q_options is a list and q_correct_answer is in "Option X" format.
    """
    q_copy = question_dict.copy()
    
    if isinstance(q_copy.get('q_options'), str):
        try:
            q_copy['q_options'] = json.loads(q_copy['q_options'])
        except json.JSONDecodeError:
            q_copy['q_options'] = []
    elif not isinstance(q_copy.get('q_options'), list):
        q_copy['q_options'] = []

    if q_copy.get('q_type') == "MCQ":
        options = q_copy.get('q_options', [])
        correct_answer_val = q_copy.get('q_correct_answer')

        if not correct_answer_val:
            q_copy['q_correct_answer'] = "N/A"
            return q_copy

        if isinstance(correct_answer_val, str) and correct_answer_val.startswith("Option ") and len(correct_answer_val.split(" ")) == 2:
             pass
        else:
            idx = None
            for i, opt in enumerate(options):
                if str(opt).strip().lower() == str(correct_answer_val).strip().lower():
                    idx = i
                    break
            if idx is not None:
                q_copy['q_correct_answer'] = f"Option {get_option_letter(idx)}"
            else:
                q_copy['q_correct_answer'] = "N/A"
    return q_copy

def get_correct_answer_text_from_option_format(options, option_format_answer):
    if not option_format_answer.startswith("Option "):
        return option_format_answer
    
    letter_or_num = option_format_answer.split(" ")[1]
    if 'A' <= letter_or_num.upper() <= 'Z':
        idx = ord(letter_or_num.upper()) - ord('A')
    elif letter_or_num.isdigit():
        idx = int(letter_or_num) -1
    else:
        return None
        
    if 0 <= idx < len(options):
        return options[idx]
    return None

def generate_v1_original(original_q):
    """Version 1: Original question, q_correct_answer standardized for MCQ."""
    v1_q = standardize_mcq_input(original_q.copy())
    v1_q['q_version'] = "1"
    return v1_q

def generate_v2_reordered_mcq(v1_q_standardized):
    """Version 2: MCQ with reordered choices."""
    if v1_q_standardized.get('q_type') != "MCQ" or not v1_q_standardized.get('q_options') or v1_q_standardized.get('q_correct_answer') == "N/A":
        return None

    v2_q = v1_q_standardized.copy()
    v2_q['q_version'] = "2"
    
    original_options = list(v2_q['q_options'])
    correct_answer_text = get_correct_answer_text_from_option_format(original_options, v2_q['q_correct_answer'])

    if correct_answer_text is None:
        return None

    shuffled_options = random.sample(original_options, len(original_options))
    v2_q['q_options'] = shuffled_options
    
    try:
        new_correct_index = shuffled_options.index(correct_answer_text)
        v2_q['q_correct_answer'] = f"Option {get_option_letter(new_correct_index)}"
    except ValueError:
        return None
        
    return v2_q

def generate_v3_mcq_with_nota(v1_q_standardized):
    """Version 3: MCQ with 'None of the above' added."""
    if v1_q_standardized.get('q_type') != "MCQ" or v1_q_standardized.get('q_correct_answer') == "N/A":
        return None

    v3_q = v1_q_standardized.copy()
    v3_q['q_version'] = "3"
    
    new_options = list(v3_q.get('q_options', [])) + ["None of the above"]
    v3_q['q_options'] = new_options
    return v3_q

def generate_v4_true_false_version(v1_q_standardized):
    """Version 4: True/False version derived from the original MCQ."""
    if v1_q_standardized.get('q_correct_answer') == "N/A": 
        return None

    v4_q = {
        'q_id': v1_q_standardized['q_id'],
        'q_version': "4",
        'q_type': "True/False",
        'topic_tag': v1_q_standardized.get('topic_tag', ''),
        'q_options': [],
    }
    
    original_question_text = v1_q_standardized['q_text']
    
    correct_answer_text = ""
    if v1_q_standardized['q_type'] == "MCQ":
        correct_answer_text = get_correct_answer_text_from_option_format(
            v1_q_standardized.get('q_options', []),
            v1_q_standardized['q_correct_answer']
        )
        if not correct_answer_text:
            return None
    elif v1_q_standardized['q_type'] == "True/False":
        correct_answer_text = v1_q_standardized['q_correct_answer']

    prompt = (
        f"You are an AI assistant. Convert the following information from a multiple-choice question (MCQ) "
        f"into a single, clear, and standalone True/False question. The True/False question you generate "
        f"MUST be answerable as 'True'. Base the statement on the provided correct answer to the MCQ.\n\n"
        f"Original MCQ Text: \"{original_question_text}\"\n"
        f"The Correct Answer to this MCQ is: \"{correct_answer_text}\"\n\n"
        f"Generate only the True/False question statement. Do not add any preamble like 'Here is the True/False question:'."
    )
    
    tf_question_text = llm_call_for_generation(prompt)
    
    if tf_question_text:
        v4_q['q_text'] = tf_question_text
        v4_q['q_correct_answer'] = "True"
    else:
        return None
        
    return v4_q

def generate_v5_rephrased_mcq(v1_q_standardized):
    """Version 5: Rephrased MCQ prompt using LLM."""
    if v1_q_standardized.get('q_type') != "MCQ" or v1_q_standardized.get('q_correct_answer') == "N/A":
        return None

    v5_q = v1_q_standardized.copy()
    v5_q['q_version'] = "5"

    original_text = v1_q_standardized['q_text']
    prompt = (
        f"You are an AI assistant. Rephrase the following multiple-choice question text. "
        f"The rephrased question should retain the original meaning, test the same knowledge, and be suitable for the same set of multiple-choice options. "
        f"Do not change the core subject or the difficulty significantly.\n\n"
        f"Original Question Text: \"{original_text}\"\n\n"
        f"Provide only the rephrased question text. Do not add any preamble."
    )
    
    rephrased_text = llm_call_for_generation(prompt)
    
    if rephrased_text:
        v5_q['q_text'] = rephrased_text
    else:
        return None 
        
    return v5_q

def write_questions_to_csv(questions_list, filename="generated_question_versions.csv"):
    """Writes a list of question dictionaries to a CSV file."""
    if not questions_list:
        return

    fieldnames = ['q_id', 'q_version', 'q_text', 'q_type', 'topic_tag', 'q_options', 'q_correct_answer']
    
    valid_questions = [q for q in questions_list if q is not None]
    if not valid_questions:
        return

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for q_dict in valid_questions:
            q_to_write = q_dict.copy()
            
            if isinstance(q_to_write.get('q_options'), list):
                q_to_write['q_options'] = json.dumps(q_to_write['q_options'])
            elif q_to_write.get('q_options') is None:
                 q_to_write['q_options'] = json.dumps([])

            if q_to_write.get('q_type') == "MCQ" and \
               isinstance(q_to_write.get('q_correct_answer'), str) and \
               q_to_write.get('q_correct_answer', '').startswith("Option ") and \
               isinstance(q_dict.get('q_options'), list):
                
                correct_answer_text = get_correct_answer_text_from_option_format(
                    q_dict.get('q_options'),
                    q_to_write.get('q_correct_answer')
                )
                if correct_answer_text is not None:
                    q_to_write['q_correct_answer'] = correct_answer_text
            
            writer.writerow(q_to_write)

def generate_all_versions_for_question(original_question_dict):
    """
    Takes a single original question dictionary, generates all its versions (V1-V5).
    Returns a list of question dictionaries for all successfully generated versions.
    For MCQs, q_correct_answer is converted to actual answer text before returning.
    LLM-dependent versions (V4, V5) are skipped if GROQ_API_KEY is not set.
    """
    processed_versions_for_return = []

    v1_q = generate_v1_original(original_question_dict)
    if not v1_q or v1_q.get('q_correct_answer') == "N/A":
        return [] 
    
    internal_generated_versions = [v1_q]

    v2_q = generate_v2_reordered_mcq(v1_q)
    if v2_q: 
        internal_generated_versions.append(v2_q)

    v3_q = generate_v3_mcq_with_nota(v1_q)
    if v3_q: 
        internal_generated_versions.append(v3_q)

    llm_generation_func = llm_call_for_generation
    
    if llm_generation_func:
        v4_q = generate_v4_true_false_version(v1_q)
        if v4_q: 
            internal_generated_versions.append(v4_q)

        v5_q = generate_v5_rephrased_mcq(v1_q)
        if v5_q: 
            internal_generated_versions.append(v5_q)
    
    for version_dict in internal_generated_versions:
        final_version_dict = version_dict.copy()
        if final_version_dict.get('q_type') == "MCQ":
            if isinstance(final_version_dict.get('q_correct_answer'), str) and \
               final_version_dict['q_correct_answer'].startswith("Option ") and \
               isinstance(final_version_dict.get('q_options'), list):
                
                correct_text = get_correct_answer_text_from_option_format(
                    final_version_dict['q_options'], 
                    final_version_dict['q_correct_answer']
                )
                if correct_text is not None:
                    final_version_dict['q_correct_answer'] = correct_text
        processed_versions_for_return.append(final_version_dict)
        
    return processed_versions_for_return

if __name__ == "__main__":
    # Example Usage:
    sample_question_tq008 = {
        '_id': '683723e792bb312c57e8a5b8', # Using string for _id for simplicity here
        'q_id': 'TQ008',
        'q_text': 'Which malware type disguises itself as legitimate software but also replicates across networks automatically?',
        'q_type': 'MCQ',
        'topic_tag': 'Malware',
        'q_options': [ 'Trojan', 'Worm', 'Ransomware', 'Backdoor' ], # List
        'q_correct_answer': 'Worm' # Text format
    }
    
    sample_question_222_1 = {
        '_id': '6837143392bb312c57e8a416',
        'q_id': '222_1',
        'q_text': 'Why might a classifier with 97% accuracy still be ineffective in a real-world NIDS?',
        'q_type': 'MCQ',
        'topic_tag': 'nids',
        'q_options': [
          'It may fail to detect rare but dangerous attacks',
          'It might be overfitting the training set',
          'The test data could be corrupted',
          'Real-world data contains only benign traffic'
        ],
        'q_correct_answer': 'Option A' # Already in Option X format
    }

    all_generated_versions = []
    
    # --- Process TQ008 ---
    print(f"--- Processing question q_id: {sample_question_tq008['q_id']} ---")
    v1_tq008 = generate_v1_original(sample_question_tq008)
    if v1_tq008:
        all_generated_versions.append(v1_tq008)
        print(f"  Generated V1: {v1_tq008.get('q_text')[:50]}..., Correct: {v1_tq008.get('q_correct_answer')}")

        v2_tq008 = generate_v2_reordered_mcq(v1_tq008)
        if v2_tq008: all_generated_versions.append(v2_tq008); print(f"  Generated V2 (reordered)")
        
        v3_tq008 = generate_v3_mcq_with_nota(v1_tq008)
        if v3_tq008: all_generated_versions.append(v3_tq008); print(f"  Generated V3 (w/ NOTA)")

        # LLM-dependent versions
        if llm_call_for_generation:
            v4_tq008 = generate_v4_true_false_version(v1_tq008)
            if v4_tq008: all_generated_versions.append(v4_tq008); print(f"  Generated V4 (T/F): {v4_tq008.get('q_text')[:50]}...")
            
            v5_tq008 = generate_v5_rephrased_mcq(v1_tq008)
            if v5_tq008: all_generated_versions.append(v5_tq008); print(f"  Generated V5 (Rephrased): {v5_tq008.get('q_text')[:50]}...")
        else:
            print("  Skipping LLM-dependent versions (V4, V5) as LLM function is not available.")
    else:
        print(f"  Failed to generate V1 for {sample_question_tq008['q_id']}")

    # --- Process 222_1 ---
    print(f"\n--- Processing question q_id: {sample_question_222_1['q_id']} ---")
    v1_222_1 = generate_v1_original(sample_question_222_1)
    if v1_222_1:
        all_generated_versions.append(v1_222_1)
        print(f"  Generated V1: {v1_222_1.get('q_text')[:50]}..., Correct: {v1_222_1.get('q_correct_answer')}")

        v2_222_1 = generate_v2_reordered_mcq(v1_222_1)
        if v2_222_1: all_generated_versions.append(v2_222_1); print(f"  Generated V2 (reordered)")
        
        v3_222_1 = generate_v3_mcq_with_nota(v1_222_1)
        if v3_222_1: all_generated_versions.append(v3_222_1); print(f"  Generated V3 (w/ NOTA)")
            
        if llm_call_for_generation:
            v4_222_1 = generate_v4_true_false_version(v1_222_1)
            if v4_222_1: all_generated_versions.append(v4_222_1); print(f"  Generated V4 (T/F): {v4_222_1.get('q_text')[:50]}...")
            
            v5_222_1 = generate_v5_rephrased_mcq(v1_222_1)
            if v5_222_1: all_generated_versions.append(v5_222_1); print(f"  Generated V5 (Rephrased): {v5_222_1.get('q_text')[:50]}...")
        else:
            print("  Skipping LLM-dependent versions (V4, V5) as LLM function is not available.")
    else:
        print(f"  Failed to generate V1 for {sample_question_222_1['q_id']}")

    # Write all collected versions to CSV
    if all_generated_versions:
        write_questions_to_csv(all_generated_versions, "generated_question_versions.csv")
    else:
        print("\nNo versions were generated for any question.")

    print("\nScript finished. Check 'generated_question_versions.csv'.")
    print("Remember to set your GROQ_API_KEY in a .env file for V4 and V5 generation.") 