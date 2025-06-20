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
    return str(index + 1) # Fallback for more than 26 options

def standardize_mcq_input(question_dict):
    """
    Standardizes the input MCQ dictionary.
    Ensures q_options is a list and q_correct_answer is in "Option X" format.
    """
    q_copy = question_dict.copy() # Work on a copy
    print(f"[DEBUG] q_id: {q_copy.get('q_id')}, q_type: {q_copy.get('q_type')}, q_options: {q_copy.get('q_options')}, q_correct_answer: {q_copy.get('q_correct_answer')}")
    
    # Ensure q_options is a list
    if isinstance(q_copy.get('q_options'), str):
        try:
            q_copy['q_options'] = json.loads(q_copy['q_options'])
        except json.JSONDecodeError:
            print(f"Warning: Could not parse q_options string for q_id {q_copy.get('q_id')}. Assuming empty list.")
            q_copy['q_options'] = []
    elif not isinstance(q_copy.get('q_options'), list):
        q_copy['q_options'] = []

    if q_copy.get('q_type') == "MCQ":
        options = q_copy.get('q_options', [])
        correct_answer_val = q_copy.get('q_correct_answer')

        if not correct_answer_val:
            print(f"Warning: No correct answer specified for MCQ q_id {q_copy.get('q_id')}")
            q_copy['q_correct_answer'] = "N/A" # Mark as N/A
            return q_copy

        # Check if correct_answer_val is already "Option X"
        if isinstance(correct_answer_val, str) and correct_answer_val.startswith("Option ") and len(correct_answer_val.split(" ")) == 2:
             # Potentially valid "Option X" format, keep it
             pass
        else: # Assume it's the answer text
            # Robust match: ignore case and whitespace
            idx = None
            for i, opt in enumerate(options):
                if str(opt).strip().lower() == str(correct_answer_val).strip().lower():
                    idx = i
                    break
            if idx is not None:
                q_copy['q_correct_answer'] = f"Option {get_option_letter(idx)}"
            else:
                print(f"Warning: Correct answer text '{correct_answer_val}' not found in options for q_id {q_copy.get('q_id')}. Options: {options}. Setting to N/A.")
                q_copy['q_correct_answer'] = "N/A" # Mark as N/A
    print(f"[DEBUG] After standardization: q_id: {q_copy.get('q_id')}, q_type: {q_copy.get('q_type')}, q_options: {q_copy.get('q_options')}, q_correct_answer: {q_copy.get('q_correct_answer')}")
    return q_copy

def get_correct_answer_text_from_option_format(options, option_format_answer):
    if not option_format_answer.startswith("Option "):
        # This might be an error or already text. For safety, return as is or handle.
        # print(f"Warning: Expected 'Option X' format, got '{option_format_answer}'")
        return option_format_answer # Or raise error/return None
    
    letter_or_num = option_format_answer.split(" ")[1]
    if 'A' <= letter_or_num.upper() <= 'Z':
        idx = ord(letter_or_num.upper()) - ord('A')
    elif letter_or_num.isdigit():
        idx = int(letter_or_num) -1
    else:
        # print(f"Warning: Could not parse option indicator '{letter_or_num}'")
        return None # Error case
        
    if 0 <= idx < len(options):
        return options[idx]
    # print(f"Warning: Index {idx} out of bounds for options list (len {len(options)})")
    return None



def generate_v1_original(original_q):
    """Version 1: Original question, q_correct_answer standardized for MCQ."""
    v1_q = standardize_mcq_input(original_q.copy())
    v1_q['q_version'] = "1"
    return v1_q

def generate_v2_reordered_mcq(v1_q_standardized):
    """Version 2: MCQ with reordered choices."""
    if v1_q_standardized.get('q_type') != "MCQ" or not v1_q_standardized.get('q_options') or v1_q_standardized.get('q_correct_answer') == "N/A":
        return None # Cannot reorder if not MCQ, no options, or correct answer is N/A

    v2_q = v1_q_standardized.copy()
    v2_q['q_version'] = "2"
    
    original_options = list(v2_q['q_options']) # Ensure it's a mutable copy
    correct_answer_text = get_correct_answer_text_from_option_format(original_options, v2_q['q_correct_answer'])

    if correct_answer_text is None:
        print(f"Warning (V2): Could not determine correct answer text for q_id {v2_q['q_id']}. Skipping reorder.")
        return None # Error in getting original correct text

    shuffled_options = random.sample(original_options, len(original_options))
    v2_q['q_options'] = shuffled_options
    
    try:
        new_correct_index = shuffled_options.index(correct_answer_text)
        v2_q['q_correct_answer'] = f"Option {get_option_letter(new_correct_index)}"
    except ValueError:
        # This should not happen if correct_answer_text was valid for original_options
        print(f"Error (V2): Original correct answer text '{correct_answer_text}' not found in shuffled options for q_id {v2_q['q_id']}.")
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
    # The original correct answer (e.g., "Option B") remains correct if it was one of the original options.
    # "None of the above" would only become correct if specifically designed so, which is not assumed here.
    return v3_q

def generate_v4_true_false_version(v1_q_standardized):
    """Version 4: True/False version derived from the original MCQ."""
    if v1_q_standardized.get('q_correct_answer') == "N/A": return None

    v4_q = {
        'q_id': v1_q_standardized['q_id'],
        'q_version': "4",
        'q_type': "True/False",
        'topic_tag': v1_q_standardized.get('topic_tag', ''),
        'q_options': [], # T/F questions have no options list here
    }
    
    original_question_text = v1_q_standardized['q_text']
    
    correct_answer_text = ""
    if v1_q_standardized['q_type'] == "MCQ":
        correct_answer_text = get_correct_answer_text_from_option_format(
            v1_q_standardized.get('q_options', []),
            v1_q_standardized['q_correct_answer']
        )
        if not correct_answer_text:
            print(f"Warning (V4): Could not get correct answer text for {v1_q_standardized['q_id']}. Skipping.")
            return None
    elif v1_q_standardized['q_type'] == "True/False": # If original is T/F
        correct_answer_text = v1_q_standardized['q_correct_answer'] # Should be "True" or "False"
        # To make a new T/F, we can try to make it affirm the original statement if True, or negate if False.
        # For simplicity, if original is T/F, we can just rephrase it slightly if needed, or mark as complex.
        # Current LLM prompt is geared towards MCQ to T/F.
        # Let's focus on MCQ to T/F for now. If input is T/F, maybe return as is or skip this version.
        print(f"Note (V4): Original question {v1_q_standardized['q_id']} is already True/False. LLM T/F generation might be redundant or produce similar.")


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
        v4_q['q_correct_answer'] = "True" # By design of the prompt
    else:
        print(f"Warning (V4): LLM failed to generate T/F question for q_id {v1_q_standardized['q_id']}.")
        return None # Failed to generate
        
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
        print(f"Warning (V5): LLM failed to rephrase question for q_id {v1_q_standardized['q_id']}. Using original text.")
        # Fallback to original text for this version if LLM fails, or return None
        # For now, let's use original text with a note, or return None to signify failure
        return None 
        
    return v5_q

def write_questions_to_csv(questions_list, filename="generated_question_versions.csv"):
    """Writes a list of question dictionaries to a CSV file."""
    if not questions_list:
        print("No questions to write to CSV.")
        return

    fieldnames = ['q_id', 'q_version', 'q_text', 'q_type', 'topic_tag', 'q_options', 'q_correct_answer']
    
    valid_questions = [q for q in questions_list if q is not None]
    if not valid_questions:
        print("No valid questions generated to write to CSV.")
        return

    print(f"Writing {len(valid_questions)} questions to {filename}...")

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for q_dict in valid_questions:
            q_to_write = q_dict.copy()
            
            # Convert q_options to JSON string
            if isinstance(q_to_write.get('q_options'), list):
                q_to_write['q_options'] = json.dumps(q_to_write['q_options'])
            elif q_to_write.get('q_options') is None:
                 q_to_write['q_options'] = json.dumps([])

            # For MCQ, if q_correct_answer is "Option X", convert it back to text for the CSV
            if q_to_write.get('q_type') == "MCQ" and \
               isinstance(q_to_write.get('q_correct_answer'), str) and \
               q_to_write.get('q_correct_answer', '').startswith("Option ") and \
               isinstance(q_dict.get('q_options'), list): # Use original q_dict options before JSON dump
                
                correct_answer_text = get_correct_answer_text_from_option_format(
                    q_dict.get('q_options'), # Original list of options from the non-copied q_dict
                    q_to_write.get('q_correct_answer')
                )
                if correct_answer_text is not None:
                    q_to_write['q_correct_answer'] = correct_answer_text
                else:
                    # This case should ideally not be hit if standardization and option format are correct
                    print(f"Warning (CSV Write): Could not convert '{q_to_write.get('q_correct_answer')}' back to text for q_id {q_to_write.get('q_id')}. Leaving as is.")
            
            writer.writerow(q_to_write)
    print(f"Successfully wrote questions to {filename}")

def generate_all_versions_for_question(original_question_dict):
    """
    Takes a single original question dictionary, generates all its versions (V1-V5).
    Returns a list of question dictionaries for all successfully generated versions.
    For MCQs, q_correct_answer is converted to actual answer text before returning.
    LLM-dependent versions (V4, V5) are skipped if GROQ_API_KEY is not set.
    """
    processed_versions_for_return = []

    # Standardize and create V1 first
    v1_q = generate_v1_original(original_question_dict) # v1_q has 'Option X' for MCQ correct answer
    if not v1_q or v1_q.get('q_correct_answer') == "N/A":
        print(f"Error: Could not generate or standardize V1 for q_id {original_question_dict.get('q_id')}. Skipping all versions for this question.")
        return [] 
    
    # Internal list to hold versions with "Option X" format for processing
    internal_generated_versions = [v1_q]

    # Generate V2 (reordered)
    v2_q = generate_v2_reordered_mcq(v1_q)
    if v2_q: internal_generated_versions.append(v2_q)

    # Generate V3 (with NOTA)
    v3_q = generate_v3_mcq_with_nota(v1_q)
    if v3_q: internal_generated_versions.append(v3_q)

    # LLM-dependent versions should now check if the llm_call function is available
    llm_generation_func = llm_call_for_generation
    
    if llm_generation_func:
        # Generate V4 (True/False) - This is not MCQ, so no conversion needed for its correct answer format
        v4_q = generate_v4_true_false_version(v1_q)
        if v4_q: internal_generated_versions.append(v4_q)

        # Generate V5 (Rephrased MCQ)
        v5_q = generate_v5_rephrased_mcq(v1_q)
        if v5_q: internal_generated_versions.append(v5_q)
    else:
        print(f"Info: LLM generation function not available. Skipping LLM-dependent versions (V4, V5) for q_id {v1_q.get('q_id')}.")
    
    # Convert MCQ correct answers from "Option X" to actual text for the final output list
    for version_dict in internal_generated_versions:
        # Make a copy to modify for the return list
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
                else:
                    # This would indicate an issue, potentially with options list or the "Option X" string
                    print(f"Warning (generate_all_versions): Could not convert '{final_version_dict['q_correct_answer']}' to text for q_id {final_version_dict['q_id']} V{final_version_dict['q_version']}. Keeping original.")
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