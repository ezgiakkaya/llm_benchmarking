import time
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import instructor
import groq
from datetime import datetime
from .models import MCQAnswer, TrueFalseAnswer, ShortAnswer, LLMResponse

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODELS = {
    "llama3-70b-8192": {
        "name": "Llama3 70B",
        "context_window": 8192,
        "max_completion_tokens": 8192
    },
    "gemma2-9b-it": {
        "name": "Gemma2 9B",
        "context_window": 8192,
        "max_completion_tokens": 8192
    },
    "deepseek-r1-distill-llama-70b": {
        "name": "Deepseek R1 Distill Llama 70B",
        "context_window": 8192,
        "max_completion_tokens": 8192
    }
}

client = instructor.from_groq(groq.Groq(api_key=GROQ_API_KEY))

def format_mcq_prompt(question_text: str, options: list, correct_answer: str) -> str:
    """Format prompt for MCQ questions"""
    lettered_options = []
    for i, opt_text in enumerate(options):
        letter = chr(ord('A') + i)
        lettered_options.append(f"{letter}. {opt_text}")
    options_string = "\n".join(lettered_options)
    
    return f"""Question: {question_text}

Options:
{options_string}

Correct Answer: {correct_answer}

Provide a detailed explanation for your answer choice and indicate your confidence level.
Your response should be structured and include:
1. A detailed explanation of your reasoning
2. The selected option (ONLY the letter: A, B, C, or D)
3. Your confidence level (between 0 and 1)

IMPORTANT: For the selected option, respond with ONLY the letter (A, B, C, or D), not the full text of the option."""

def format_true_false_prompt(question_text: str) -> str:
    """Format prompt for True/False questions"""
    return f"""Question: {question_text}

Provide a detailed explanation for your answer and indicate your confidence level.
Your response should be structured and include:
1. A detailed explanation of your reasoning
2. Your answer (True or False)
3. Your confidence level (between 0 and 1)"""

def format_short_answer_prompt(question_text: str) -> str:
    """Format prompt for short answer questions"""
    return f"""Question: {question_text}

Provide a concise answer and explanation, along with your confidence level.
Your response should be structured and include:
1. A detailed explanation of your reasoning
2. Your answer
3. Your confidence level (between 0 and 1)"""

def query_groq(prompt: str, model_id: str, question_type: str, question_options: list = None) -> dict:
    """Query Groq API with structured output using Instructor"""
    if model_id not in GROQ_MODELS:
        return {"error": f"Invalid model ID. Available models: {list(GROQ_MODELS.keys())}", "provider": "Groq"}
    
    if question_type == "MCQ" and not isinstance(question_options, list):
        question_options = []

    model_info = GROQ_MODELS[model_id]
    start = time.time()

    try:
        if question_type == "MCQ":
            formatted_prompt = format_mcq_prompt(prompt, question_options, question_options[0] if question_options else 'A')
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=MCQAnswer
            )
            if not isinstance(response.selected_option, str):
                response.selected_option = 'A'
            else:
                cleaned_option = response.selected_option.strip().upper()
                match = re.search(r'[A-D]', cleaned_option)
                if match:
                    response.selected_option = match.group(0)
                else:
                    for i, opt in enumerate(question_options):
                        if cleaned_option.lower() in opt.lower():
                            response.selected_option = chr(ord('A') + i)
                            break
                    else:
                        response.selected_option = 'A'
            answer = response
        elif question_type == "True/False":
            formatted_prompt = format_true_false_prompt(prompt)
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=TrueFalseAnswer
            )
            answer = response
        else:
            formatted_prompt = format_short_answer_prompt(prompt)
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=ShortAnswer
            )
            answer = response

        latency = round(time.time() - start, 3)
        
        return {
            "model": model_info["name"],
            "provider": "Groq",
            "response": answer,
            "latency_sec": latency,
            "model_id": model_id,
            "context_window": model_info["context_window"]
        }
    except Exception as e:
        return {
            "error": str(e),
            "provider": "Groq",
            "model": model_info["name"],
            "response_text": "",
            "llm_extracted_answer": "Error: Exception during API call"
        }

def query_openai(prompt: str, model: str = "gpt-3.5-turbo") -> dict:
    """Query OpenAI's API. Currently muted."""
    return {
        "error": "OpenAI client is currently muted.",
        "provider": "OpenAI",
        "model": "openai",
        "response_text": "",
        "llm_extracted_answer": "N/A_MUTED",
        "latency_sec": 0
    }

LLM_MODEL_FOR_GENERATION = "llama3-70b-8192"

HEADERS_GROQ = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def llm_call_for_generation(prompt_text, max_retries=3, retry_delay=5):
    """Generic LLM call for text generation tasks using Groq API."""
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found. Please set it in your .env file.")
        return None

    data = {
        "model": LLM_MODEL_FOR_GENERATION,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant specializing in educational content. Follow instructions precisely."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=HEADERS_GROQ,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            response_json = response.json()
            
            if "choices" in response_json and len(response_json["choices"]) > 0:
                generated_text = response_json["choices"][0]["message"]["content"].strip()
                return generated_text
            else:
                print(f"Warning: LLM response format issue. Full response: {response_json}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None
    return None

def query_groq_with_rag(prompt: str, model_id: str, question_type: str, question_options: list = None, rag_pipeline=None) -> dict:
    """Query Groq API with RAG-enhanced structured output using Instructor"""
    if model_id not in GROQ_MODELS:
        return {"error": f"Invalid model ID. Available models: {list(GROQ_MODELS.keys())}", "provider": "Groq"}
    
    if question_type == "MCQ" and not isinstance(question_options, list):
        question_options = []

    model_info = GROQ_MODELS[model_id]
    start = time.time()

    try:
        rag_context = ""
        rag_sources = []
        if rag_pipeline:
            try:
                rag_result = rag_pipeline.query_rag(prompt, top_k=3)
                if rag_result:
                    rag_context = f"\n\nRelevant Context from Course Materials:\n{rag_result['response']}"
                    rag_sources = rag_result.get('sources', [])
            except Exception as e:
                print(f"Warning: RAG retrieval failed: {str(e)}")
        
        if question_type == "MCQ":
            formatted_prompt = format_mcq_prompt_with_rag(prompt, question_options, question_options[0] if question_options else 'A', rag_context)
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=MCQAnswer
            )
            if not isinstance(response.selected_option, str):
                response.selected_option = 'A'
            else:
                cleaned_option = response.selected_option.strip().upper()
                match = re.search(r'[A-D]', cleaned_option)
                if match:
                    response.selected_option = match.group(0)
                else:
                    for i, opt in enumerate(question_options):
                        if cleaned_option.lower() in opt.lower():
                            response.selected_option = chr(ord('A') + i)
                            break
                    else:
                        response.selected_option = 'A'
            answer = response
        elif question_type == "True/False":
            formatted_prompt = format_true_false_prompt_with_rag(prompt, rag_context)
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=TrueFalseAnswer
            )
            answer = response
        else:
            formatted_prompt = format_short_answer_prompt_with_rag(prompt, rag_context)
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=ShortAnswer
            )
            answer = response

        latency = round(time.time() - start, 3)
        
        return {
            "model": model_info["name"],
            "provider": "Groq",
            "response": answer,
            "latency_sec": latency,
            "model_id": model_id,
            "context_window": model_info["context_window"],
            "rag_sources": rag_sources,
            "rag_enabled": bool(rag_pipeline)
        }
    except Exception as e:
        return {
            "error": str(e),
            "provider": "Groq",
            "model": model_info["name"],
            "response_text": "",
            "llm_extracted_answer": "Error: Exception during API call",
            "rag_enabled": bool(rag_pipeline)
        }

def format_mcq_prompt_with_rag(question_text: str, options: list, correct_answer: str, rag_context: str = "") -> str:
    """Format prompt for MCQ questions with RAG context"""
    lettered_options = []
    for i, opt_text in enumerate(options):
        letter = chr(ord('A') + i)
        lettered_options.append(f"{letter}. {opt_text}")
    options_string = "\n".join(lettered_options)
    
    return f"""Question: {question_text}

Options:
{options_string}

{rag_context}

Using the context provided from the course materials above (if available), provide a detailed explanation for your answer choice and indicate your confidence level.
Your response should be structured and include:
1. A detailed explanation of your reasoning (referencing the course materials when relevant)
2. The selected option (ONLY the letter: A, B, C, or D)
3. Your confidence level (between 0 and 1)

IMPORTANT: For the selected option, respond with ONLY the letter (A, B, C, or D), not the full text of the option."""

def format_true_false_prompt_with_rag(question_text: str, rag_context: str = "") -> str:
    """Format prompt for True/False questions with RAG context"""
    return f"""Question: {question_text}

{rag_context}

Using the context provided from the course materials above (if available), provide a detailed explanation for your answer and indicate your confidence level.
Your response should be structured and include:
1. A detailed explanation of your reasoning (referencing the course materials when relevant)
2. Your answer (True or False)
3. Your confidence level (between 0 and 1)"""

def format_short_answer_prompt_with_rag(question_text: str, rag_context: str = "") -> str:
    """Format prompt for short answer questions with RAG context"""
    return f"""Question: {question_text}

{rag_context}

Using the context provided from the course materials above (if available), provide a concise answer and explanation, along with your confidence level.
Your response should be structured and include:
1. A detailed explanation of your reasoning (referencing the course materials when relevant)
2. Your answer
3. Your confidence level (between 0 and 1)"""
