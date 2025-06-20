import streamlit as st
import pandas as pd
from datetime import datetime
import json
import io
from io import StringIO
import re
import sys
import os
from pathlib import Path
import shutil
import openai
from dotenv import load_dotenv
import PyPDF2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import questions_collection as questions
from core.database import responses_collection as responses
from core.llm_clients import query_groq, query_groq_with_rag, GROQ_MODELS
from core.models import LLMResponse
from core.question_versioning import generate_all_versions_for_question
from core.benchmark_metrics import calculate_comprehensive_benchmark_metrics

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")
    st.stop()

try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"⚠️ Error initializing OpenAI client: {str(e)}")
    st.stop()


def validate_question(question_data):
    required_fields = ["q_id", "q_text", "q_type", "topic_tag", "q_correct_answer"]
    for field in required_fields:
        if field not in question_data:
            return False, f"Missing required field: {field}"
    
    if question_data["q_type"] == "MCQ":
        if "q_options" not in question_data or not isinstance(question_data["q_options"], list):
            return False, "MCQ questions must have q_options as a list"
        if len(question_data["q_options"]) < 2:
            return False, "MCQ questions must have at least 2 options"
        if question_data["q_correct_answer"] not in question_data["q_options"]:
            return False, "Correct answer must be one of the options"
    
    return True, "Valid"

def get_option_letter(answer_text, q_options):
    """Convert answer text to option letter (A, B, C, D)."""
    if not answer_text or not q_options:
        return None
    try:
        index = q_options.index(answer_text)
        return chr(ord('A') + index)
    except:
        if isinstance(answer_text, str):
            match = re.search(r'Option\s+([A-D])', answer_text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            match = re.search(r'^[A-D]$', answer_text.strip().upper())
            if match:
                return match.group(0)
        return None

def calculate_accuracy_structured(model_responses, questions_collection):
    """Calculate accuracy metrics for structured responses."""
    results = []
    
    for response in model_responses:
        question_id = response.get("question_id")
        response_version = str(response.get("version", "1"))
        
        question = questions_collection.find_one({
            "q_id": question_id,
            "q_version": response_version
        })
        
        if not question:
            question = questions_collection.find_one({
                "q_id": question_id,
                "$or": [
                    {"q_version": "1"},
                    {"q_version": {"$exists": False}}
                ]
            })
        
        if not question:
            print(f"Warning: Question {question_id} v{response_version} not found in database")
            continue
            
        q_type = question.get("q_type")
        correct_answer = question.get("q_correct_answer")
        model_answer = None
        confidence = 0
        is_correct = False
        
        if q_type == "MCQ" and "mcq_answer" in response:
            mcq_data = response["mcq_answer"]
            model_answer = str(mcq_data.get("selected_option", ""))
            confidence = float(mcq_data.get("confidence", 0))
            
            correct_option = get_option_letter(correct_answer, question.get("q_options", []))
            is_correct = bool(correct_option == model_answer)
            
        elif q_type == "True/False" and "true_false_answer" in response:
            tf_data = response["true_false_answer"]
            model_answer_bool = tf_data.get("answer", False)
            model_answer = str(model_answer_bool)
            confidence = float(tf_data.get("confidence", 0))
            
            correct_bool = str(correct_answer).lower() == "true"
            is_correct = bool(correct_bool == model_answer_bool)
            
        elif q_type == "Short Answer" and "short_answer" in response:
            sa_data = response["short_answer"]
            model_answer = str(sa_data.get("answer", ""))
            confidence = float(sa_data.get("confidence", 0))
            
            is_correct = bool(str(correct_answer).lower().strip() == str(model_answer).lower().strip())
        
        results.append({
            "model": str(response.get("model_name", "")),
            "q_id": str(response.get("question_id", "")),
            "version": str(response_version),
            "question_text": str(question.get("q_text", "")),
            "question_type": str(q_type or ""),
            "topic_tag": str(question.get("topic_tag", "")),
            "correct_answer": str(correct_answer or ""),
            "model_answer": str(model_answer or ""),
            "confidence": float(confidence),
            "is_correct": bool(is_correct)
        })
    
    return pd.DataFrame(results)

st.set_page_config(
    page_title="COMP430 LLM Benchmark",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 COMP430 LLM Benchmark Dashboard")
st.markdown("**Advanced AI Model Comparison with Structured Responses & Confidence Scoring**")

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", ["📝 Upload Questions", "🔬 Run Tests", "🔬 Run Tests with RAG", "📊 Results Dashboard", "🏆 Benchmark", "📈 Evaluate Metrics", "📚 Create Questions", "💾 Manage CSV Files", "📚 RAG System"])

if page == "📝 Upload Questions":
    st.header("📝 Upload Questions")
    
    tab1, tab2 = st.tabs(["Single Question Upload", "Batch Upload"])
    
    with tab1:
        st.subheader("Upload Single Question")
        
        if 'options' not in st.session_state:
            st.session_state.options = []
        
        with st.form("question_form"):
            q_id = st.text_input("Question ID")
            q_text = st.text_area("Question Text")
            q_type = st.selectbox("Question Type", ["MCQ", "True/False", "Short Answer"])
            topic_tag = st.text_input("Topic Tag")
            
            if q_type == "MCQ":
                num_options = st.number_input("Number of Options", min_value=2, max_value=6, value=4)
                
                if 'prev_num_options' not in st.session_state:
                    st.session_state.prev_num_options = num_options
                elif st.session_state.prev_num_options != num_options:
                    st.session_state.options = []
                    st.session_state.prev_num_options = num_options
                
                for i in range(num_options):
                    option = st.text_input(
                        f"Option {i+1}",
                        key=f"option_{i}",
                        value=st.session_state.options[i] if i < len(st.session_state.options) else ""
                    )
                    if i < len(st.session_state.options):
                        st.session_state.options[i] = option
                    else:
                        st.session_state.options.append(option)
                
                correct_answer = st.selectbox(
                    "Correct Answer",
                    options=[f"Option {i+1}" for i in range(num_options)],
                    key="correct_answer"
                )
                
                selected_option_index = int(correct_answer.split()[-1]) - 1
                selected_option_text = st.session_state.options[selected_option_index] if selected_option_index < len(st.session_state.options) else ""
                
            elif q_type == "True/False":
                correct_answer = st.selectbox("Correct Answer", ["True", "False"])
            else:
                correct_answer = st.text_input("Correct Answer")
            
            submitted = st.form_submit_button("📤 Upload Question")
            
            if submitted:
                if q_type == "MCQ":
                    valid_options = [opt for opt in st.session_state.options if opt.strip()]
                    
                    if not valid_options:
                        st.error("Please enter at least one option for MCQ questions")
                    else:
                        question_data = {
                            "q_id": q_id,
                            "q_text": q_text,
                            "q_type": q_type,
                            "topic_tag": topic_tag,
                            "q_options": valid_options,
                            "q_correct_answer": selected_option_text,
                            "q_version": "1"
                        }
                        
                        is_valid, message = validate_question(question_data)
                        if is_valid:
                            questions.insert_one(question_data)
                            st.success("✅ Question uploaded successfully!")
                            st.session_state.options = []
                        else:
                            st.error(f"❌ Validation error: {message}")
                else:
                    question_data = {
                        "q_id": q_id,
                        "q_text": q_text,
                        "q_type": q_type,
                        "topic_tag": topic_tag,
                        "q_options": [] if q_type != "MCQ" else valid_options,
                        "q_correct_answer": correct_answer,
                        "q_version": "1"
                    }
                    
                    is_valid, message = validate_question(question_data)
                    if is_valid:
                        questions.insert_one(question_data)
                        st.success("✅ Question uploaded successfully!")
                    else:
                        st.error(f"❌ Validation error: {message}")
    
    with tab2:
        st.subheader("Batch Upload Questions")
        st.markdown("""
        ### 📋 CSV Format Requirements
        Upload a CSV file with the following columns:
        - `q_id`: Unique identifier for the question
        - `q_text`: The question text
        - `q_type`: Type of question (MCQ, True/False, Short Answer)
        - `topic_tag`: Topic category
        - `q_options`: For MCQ, provide options as a JSON array string (e.g., `["option1", "option2", "option3"]`)
        - `q_correct_answer`: The correct answer
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                questions_list = df.to_dict('records')
                
                success_count = 0
                error_count = 0
                error_messages = []
                
                for question in questions_list:
                    if question['q_type'] == 'MCQ':
                        try:
                            question['q_options'] = json.loads(question['q_options'])
                        except json.JSONDecodeError:
                            error_messages.append(f"Invalid JSON format for options in question {question['q_id']}")
                            error_count += 1
                            continue
                    
                    if 'q_version' not in question:
                        question['q_version'] = "1"
                    
                    is_valid, message = validate_question(question)
                    if is_valid:
                        questions.insert_one(question)
                        success_count += 1
                    else:
                        error_messages.append(f"Question {question['q_id']}: {message}")
                        error_count += 1
                
                st.success(f"✅ Successfully uploaded {success_count} questions!")
                if error_count > 0:
                    st.error(f"❌ Failed to upload {error_count} questions:")
                    for msg in error_messages:
                        st.error(msg)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    st.subheader("📚 Current Questions in Database")
    all_questions = list(questions.find())
    if all_questions:
        df = pd.DataFrame(all_questions)
        st.dataframe(df[['q_id', 'q_text', 'q_type', 'topic_tag']], use_container_width=True)
    else:
        st.info("No questions in the database yet.")

elif page == "🔬 Run Tests":
    st.header("🔬 Run LLM Tests")
    st.markdown("**Test questions with AI models and get structured responses with confidence scores**")
    
    all_questions = list(questions.find())
    if not all_questions:
        st.warning("⚠️ No questions found in the database. Please upload some questions first.")
    else:
        questions_to_test = []
        
        question_groups = {}
        for q in all_questions:
            base_id = q.get('original_q_id', q['q_id'])
            if base_id not in question_groups:
                question_groups[base_id] = []
            question_groups[base_id].append(q)
        
        st.subheader("🎯 Select Questions to Test")
        
        base_question_options = []
        for base_id, versions in question_groups.items():
            version_count = len(versions)
            base_q = versions[0]
            base_question_options.append({
                'base_id': base_id,
                'display': f"**{base_id}** ({base_q['q_type']}) - {version_count} versions: {base_q['q_text'][:50]}...",
                'versions': versions
            })
        
        test_mode = st.radio(
            "🔬 Testing Mode",
            ["Single Question", "Multiple Questions", "All Questions"],
            help="Choose whether to test a single question, multiple questions, or all questions"
        )
        
        if test_mode == "Single Question":
            selected_base_option = st.selectbox(
                "🎯 Select Question Group",
                base_question_options,
                format_func=lambda x: x['display']
            )
            
            if selected_base_option:
                versions = selected_base_option['versions']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    test_option = st.radio(
                        "🔬 Testing Options",
                        ["Test All Versions", "Test Specific Version"],
                        help="Choose whether to test all versions or a specific one"
                    )
                
                with col2:
                    if test_option == "Test Specific Version":
                        selected_version = st.selectbox(
                            "Select Version",
                            versions,
                            format_func=lambda x: f"Version {x.get('q_version', '1')}: {x.get('q_text', '')[:30]}..."
                        )
                        questions_to_test = [selected_version]
                    else:
                        questions_to_test = versions
        
        elif test_mode == "Multiple Questions":
            selected_base_options = st.multiselect(
                "🎯 Select Question Groups",
                base_question_options,
                format_func=lambda x: x['display']
            )
            
            if selected_base_options:
                questions_to_test = []
                for option in selected_base_options:
                    questions_to_test.extend(option['versions'])
        
            questions_to_test = all_questions
        
        if questions_to_test:
            st.subheader("📋 Questions to Test")
            st.write(f"Total questions to test: {len(questions_to_test)}")
            
            with st.expander("📋 View Questions to Test", expanded=True):
                for i, q in enumerate(questions_to_test):
                    st.write(f"**{i+1}. {q['q_id']}** (Version {q.get('q_version', '1')})")
                    st.write(f"Type: {q['q_type']} | Topic: {q.get('topic_tag', 'N/A')}")
                    st.write(f"Question: {q['q_text'][:100]}...")
                    st.write("---")
            
            if st.button("🚀 Run Tests", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner(f"🤖 Running AI model tests on {len(questions_to_test)} question(s)..."):
                    total_tests = len(questions_to_test) * len(GROQ_MODELS)
                    completed = 0
                    
                    for question in questions_to_test:
                        for model_id, model_info in GROQ_MODELS.items():
                            status_text.text(f"Querying {model_info['name']} for {question['q_id']} v{question.get('q_version', '1')}...")
                            
                            try:
                                groq_response = query_groq(
                                    question['q_text'],
                                    model_id,
                                    question['q_type'],
                                    question.get('q_options')
                                )
                                
                                if "error" not in groq_response:
                                    llm_response = LLMResponse(
                                        question_id=question["q_id"],
                                        question_text=question["q_text"],
                                        question_type=question["q_type"],
                                        model_name=model_info["name"],
                                        timestamp=datetime.now().isoformat(),
                                        version=str(question.get('q_version', "1"))
                                    )
                                    
                                    if question['q_type'] == "MCQ":
                                        llm_response.mcq_answer = groq_response["response"]
                                    elif question['q_type'] == "True/False":
                                        llm_response.true_false_answer = groq_response["response"]
                                    else:
                                        llm_response.short_answer = groq_response["response"]
                                    
                                    responses.insert_one(llm_response.model_dump())
                                    
                                completed += 1
                                progress_bar.progress(completed / total_tests)
                                
                            except Exception as e:
                                st.error(f"❌ Error testing {model_info['name']} on {question['q_id']}: {str(e)}")
                                completed += 1
                                progress_bar.progress(completed / total_tests)
                    
                    status_text.text("✅ All tests completed!")
                    
                st.subheader("📊 Test Results")
                
                for question in questions_to_test:
                    
                    latest_responses = list(responses.find(
                        {
                            "question_id": question["q_id"],
                            "version": str(question.get('q_version', "1"))
                        },
                        sort=[("timestamp", -1)],
                        limit=len(GROQ_MODELS)
                    ))
                    
                    if latest_responses:
                        accuracy_results = []
                        
                        for response in latest_responses:
                            model_name = response.get("model_name")
                            is_correct = False
                            confidence = 0
                            explanation = ""
                            model_answer = ""
                            
                            if question['q_type'] == "MCQ" and "mcq_answer" in response:
                                mcq_data = response["mcq_answer"]
                                model_answer = mcq_data.get("selected_option", "")
                                confidence = mcq_data.get("confidence", 0)
                                explanation = mcq_data.get("explanation", "")
                                
                                correct_option = get_option_letter(question['q_correct_answer'], question['q_options'])
                                is_correct = correct_option == model_answer
                                
                            elif question['q_type'] == "True/False" and "true_false_answer" in response:
                                tf_data = response["true_false_answer"]
                                model_answer_bool = tf_data.get("answer", False)
                                model_answer = str(model_answer_bool)
                                confidence = tf_data.get("confidence", 0)
                                explanation = tf_data.get("explanation", "")
                                
                                correct_bool = str(question['q_correct_answer']).lower() == "true"
                                is_correct = bool(correct_bool == model_answer_bool)
                                
                            elif question['q_type'] == "Short Answer" and "short_answer" in response:
                                sa_data = response["short_answer"]
                                model_answer = sa_data.get("answer", "")
                                confidence = sa_data.get("confidence", 0)
                                explanation = sa_data.get("explanation", "")
                                
                                is_correct = str(question['q_correct_answer']).lower().strip() == str(model_answer).lower().strip()
                            
                            accuracy_results.append({
                                "model": model_name,
                                "answer": model_answer,
                                "correct": is_correct,
                                "confidence": confidence,
                                "explanation": explanation
                            })
                        
                        for result in accuracy_results:
                            with st.expander(f"🤖 {result['model']} {'✅' if result['correct'] else '❌'}", expanded=len(questions_to_test) == 1):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Answer", result['answer'])
                                with col2:
                                    st.metric("Confidence", f"{result['confidence']:.1%}")
                                with col3:
                                    st.metric("Result", "✅ Correct" if result['correct'] else "❌ Incorrect")
                                
                                st.write("**Explanation:**")
                                st.write(result['explanation'])
                        
                        correct_count = sum(1 for r in accuracy_results if r['correct'])
                        avg_confidence = sum(r['confidence'] for r in accuracy_results) / len(accuracy_results)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{correct_count}/{len(accuracy_results)}")
                        with col2:
                            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                        with col3:
                            st.metric("Version", question.get('q_version', '1'))
                    else:
                        st.warning(f"⚠️ No responses found for {question['q_id']} v{question.get('q_version', '1')}")

elif page == "🔬 Run Tests with RAG":
    st.header("🔬 Run Tests with RAG")
    st.markdown("**Test questions with AI models enhanced by RAG document retrieval and get structured responses with confidence scores**")
    
    try:
        from core.rag_pipeline import RAGPipeline
        rag_pipeline = RAGPipeline()
        st.success("✅ RAG pipeline initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing RAG pipeline: {str(e)}")
        st.info("Please ensure your RAG environment is properly configured with Pinecone and documents are uploaded.")
        st.stop()
    
    all_questions = list(questions.find())
    if not all_questions:
        st.warning("⚠️ No questions found in the database. Please upload some questions first.")
    else:
        questions_to_test = []
        
        question_groups = {}
        for q in all_questions:
            base_id = q.get('original_q_id', q['q_id'])
            if base_id not in question_groups:
                question_groups[base_id] = []
            question_groups[base_id].append(q)
        
        st.subheader("🎯 Select Questions to Test")
        
        base_question_options = []
        for base_id, versions in question_groups.items():
            version_count = len(versions)
            base_question_options.append({
                'base_id': base_id,
                'display': f"**{base_id}** ({base_q['q_type']}) - {version_count} versions: {base_q['q_text'][:50]}...",
                'versions': versions
            })
        
        test_mode = st.radio(
            "🔬 Testing Mode",
            ["Single Question", "Multiple Questions", "All Questions"],
            help="Choose whether to test a single question, multiple questions, or all questions"
        )
        
        if test_mode == "Single Question":
            selected_base_option = st.selectbox(
                "🎯 Select Question Group",
                base_question_options,
                format_func=lambda x: x['display']
            )
            
            if selected_base_option:
                versions = selected_base_option['versions']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    test_option = st.radio(
                        "🔬 Testing Options",
                        ["Test All Versions", "Test Specific Version"],
                        help="Choose whether to test all versions or a specific one"
                    )
                
                with col2:
                    if test_option == "Test Specific Version":
                        selected_version = st.selectbox(
                            "Select Version",
                            versions,
                            format_func=lambda x: f"Version {x.get('q_version', '1')}: {x.get('q_text', '')[:30]}..."
                        )
                        questions_to_test = [selected_version]
                    else:
                        questions_to_test = versions
        
        elif test_mode == "Multiple Questions":
            selected_base_options = st.multiselect(
                "🎯 Select Question Groups",
                base_question_options,
                format_func=lambda x: x['display']
            )
            
            if selected_base_options:
                questions_to_test = []
                for option in selected_base_options:
                    questions_to_test.extend(option['versions'])
        
            questions_to_test = all_questions
        
        if questions_to_test:
            st.subheader("📋 Questions to Test")
            st.write(f"Total questions to test: {len(questions_to_test)}")
            
            with st.expander("📋 View Questions to Test", expanded=True):
                for i, q in enumerate(questions_to_test):
                    st.write(f"**{i+1}. {q['q_id']}** (Version {q.get('q_version', '1')})")
                    st.write(f"Type: {q['q_type']} | Topic: {q.get('topic_tag', 'N/A')}")
                    st.write(f"Question: {q['q_text'][:100]}...")
                    st.write("---")
            
            if st.button("🚀 Run Tests with RAG", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner(f"🤖 Running AI model tests with RAG on {len(questions_to_test)} question(s)..."):
                    total_tests = len(questions_to_test) * len(GROQ_MODELS)
                    completed = 0
                    
                    for question in questions_to_test:
                        for model_id, model_info in GROQ_MODELS.items():
                            status_text.text(f"Querying {model_info['name']} with RAG for {question['q_id']} v{question.get('q_version', '1')}...")
                            
                            try:
                                groq_response = query_groq_with_rag(
                                    question['q_text'],
                                    model_id,
                                    question['q_type'],
                                    question.get('q_options'),
                                    rag_pipeline
                                )
                                
                                if "error" not in groq_response:
                                    llm_response = LLMResponse(
                                        question_id=question["q_id"],
                                        question_text=question["q_text"],
                                        question_type=question["q_type"],
                                        model_name=f"{model_info['name']} (RAG)",
                                        timestamp=datetime.now().isoformat(),
                                        version=str(question.get('q_version', "1"))
                                    )
                                    
                                    if question['q_type'] == "MCQ":
                                        llm_response.mcq_answer = groq_response["response"]
                                    elif question['q_type'] == "True/False":
                                        llm_response.true_false_answer = groq_response["response"]
                                    else:
                                        llm_response.short_answer = groq_response["response"]
                                    
                                    response_dict = llm_response.model_dump()
                                    response_dict["rag_sources"] = groq_response.get("rag_sources", [])
                                    response_dict["rag_enabled"] = True
                                    responses.insert_one(response_dict)
                                    
                                completed += 1
                                progress_bar.progress(completed / total_tests)
                                
                            except Exception as e:
                                st.error(f"❌ Error testing {model_info['name']} with RAG on {question['q_id']}: {str(e)}")
                                completed += 1
                                progress_bar.progress(completed / total_tests)
                    
                    status_text.text("✅ All tests completed!")
                    
                st.subheader("📊 Test Results")
                
                for question in questions_to_test:
                    
                    latest_responses = list(responses.find(
                        {
                            "question_id": question["q_id"],
                            "version": str(question.get('q_version', "1")),
                            "rag_enabled": True
                        },
                        sort=[("timestamp", -1)],
                        limit=len(GROQ_MODELS)
                    ))
                    
                    if latest_responses:
                        accuracy_results = []
                        
                        for response in latest_responses:
                            model_name = response.get("model_name")
                            is_correct = False
                            confidence = 0
                            explanation = ""
                            model_answer = ""
                            rag_sources = response.get("rag_sources", [])
                            
                            if question['q_type'] == "MCQ" and "mcq_answer" in response:
                                mcq_data = response["mcq_answer"]
                                model_answer = mcq_data.get("selected_option", "")
                                confidence = mcq_data.get("confidence", 0)
                                explanation = mcq_data.get("explanation", "")
                                
                                correct_option = get_option_letter(question['q_correct_answer'], question['q_options'])
                                is_correct = correct_option == model_answer
                                
                            elif question['q_type'] == "True/False" and "true_false_answer" in response:
                                tf_data = response["true_false_answer"]
                                model_answer_bool = tf_data.get("answer", False)
                                model_answer = str(model_answer_bool)
                                confidence = tf_data.get("confidence", 0)
                                explanation = tf_data.get("explanation", "")
                                
                                correct_bool = str(question['q_correct_answer']).lower() == "true"
                                is_correct = bool(correct_bool == model_answer_bool)
                                
                            elif question['q_type'] == "Short Answer" and "short_answer" in response:
                                sa_data = response["short_answer"]
                                model_answer = sa_data.get("answer", "")
                                confidence = sa_data.get("confidence", 0)
                                explanation = sa_data.get("explanation", "")
                                
                                is_correct = str(question['q_correct_answer']).lower().strip() == str(model_answer).lower().strip()
                            
                            accuracy_results.append({
                                "model": model_name,
                                "answer": model_answer,
                                "correct": is_correct,
                                "confidence": confidence,
                                "explanation": explanation,
                                "rag_sources": rag_sources
                            })
                        
                        for result in accuracy_results:
                            with st.expander(f"🤖 {result['model']} {'✅' if result['correct'] else '❌'} (RAG-Enhanced)", expanded=len(questions_to_test) == 1):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Answer", result['answer'])
                                with col2:
                                    st.metric("Confidence", f"{result['confidence']:.1%}")
                                with col3:
                                    st.metric("Result", "✅ Correct" if result['correct'] else "❌ Incorrect")
                                
                                st.write("**Explanation:**")
                                st.write(result['explanation'])
                                
                                if result['rag_sources']:
                                    st.write("**📚 RAG Sources Used:**")
                                    for source in result['rag_sources']:
                                        st.write(f"- {source}")
                                else:
                                    st.write("**📚 RAG Sources:** No sources retrieved")
                        
                        correct_count = sum(1 for r in accuracy_results if r['correct'])
                        avg_confidence = sum(r['confidence'] for r in accuracy_results) / len(accuracy_results)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("RAG Accuracy", f"{correct_count}/{len(accuracy_results)}")
                        with col2:
                            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                        with col3:
                            st.metric("Version", question.get('q_version', '1'))
                        with col4:
                            total_sources = sum(len(r['rag_sources']) for r in accuracy_results)
                            st.metric("Total Sources", total_sources)
                    else:
                        st.warning(f"⚠️ No RAG responses found for {question['q_id']} v{question.get('q_version', '1')}")

elif page == "📊 Results Dashboard":
    st.header("📊 Results Dashboard")
    st.markdown("**Comprehensive analysis of AI model performance with structured insights**")

    active_groq_model_names = {model_info["name"] for model_info in GROQ_MODELS.values()}
    rag_model_names = {f"{model_info['name']} (RAG)" for model_info in GROQ_MODELS.values()}
    all_active_model_names = active_groq_model_names.union(rag_model_names)
    
    all_db_responses = list(responses.find())

    structured_responses = [
        r for r in all_db_responses
        if (r.get("model_name") in all_active_model_names or r.get("rag_enabled", False)) and 
        ("mcq_answer" in r or "true_false_answer" in r or "short_answer" in r)
    ]
    
    if not structured_responses:
        st.warning("⚠️ No structured test results found. Please run tests using the new structured format.")
    else:
        accuracy_df = calculate_accuracy_structured(structured_responses, questions)
        
        if accuracy_df.empty:
            st.warning("⚠️ No data available to display metrics.")
        else:
            rag_responses = [r for r in structured_responses if r.get("rag_enabled", False)]
            non_rag_responses = [r for r in structured_responses if not r.get("rag_enabled", False)]
            
            if rag_responses and non_rag_responses:
                st.subheader("🔍 RAG vs Standard Model Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🤖 Standard Responses", len(non_rag_responses))
                with col2:
                    st.metric("🔬 RAG-Enhanced Responses", len(rag_responses))
                
                rag_accuracy_df = calculate_accuracy_structured(rag_responses, questions)
                non_rag_accuracy_df = calculate_accuracy_structured(non_rag_responses, questions)
                
                if not rag_accuracy_df.empty and not non_rag_accuracy_df.empty:
                    rag_accuracy = (rag_accuracy_df["is_correct"].sum() / len(rag_accuracy_df)) * 100
                    non_rag_accuracy = (non_rag_accuracy_df["is_correct"].sum() / len(non_rag_accuracy_df)) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🎯 Standard Accuracy", f"{non_rag_accuracy:.1f}%")
                    with col2:
                        st.metric("🎯 RAG Accuracy", f"{rag_accuracy:.1f}%", 
                                 delta=f"{rag_accuracy - non_rag_accuracy:.1f}%")
            
            st.subheader("📈 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Total Tests", len(accuracy_df))
            with col2:
                total_correct = accuracy_df["is_correct"].sum()
                accuracy = (total_correct / len(accuracy_df)) * 100 if len(accuracy_df) > 0 else 0
                st.metric("✅ Overall Accuracy", f"{accuracy:.1f}%")
            with col3:
                avg_confidence = accuracy_df["confidence"].mean() * 100 if len(accuracy_df) > 0 else 0
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")
            with col4:
                models = accuracy_df["model"].nunique()
                st.metric("🤖 Models Tested", models)
        
            st.subheader("📊 Version Analysis")
            
            version_metrics = accuracy_df.groupby(["version"]).agg({
                "is_correct": ["count", "mean", "sum"],
                "confidence": "mean"
            }).round(3)
            
            version_metrics.columns = ["Total Tests", "Accuracy", "Correct Answers", "Avg Confidence"]
            version_metrics["Accuracy"] = version_metrics["Accuracy"] * 100
            version_metrics["Avg Confidence"] = version_metrics["Avg Confidence"] * 100
            
            st.write("**Performance by Version**")
            st.dataframe(version_metrics, use_container_width=True)
            
            st.write("**Model Accuracy by Version**")
            unique_models = accuracy_df["model"].unique()
            
            if len(unique_models) > 0:
                model_cols = st.columns(len(unique_models))
                
                for i, model in enumerate(unique_models):
                    with model_cols[i]:
                        model_data = accuracy_df[accuracy_df["model"] == model]
                        model_version_accuracy = model_data.groupby("version")["is_correct"].mean() * 100
                        
                        if not model_version_accuracy.empty:
                            st.write(f"**{model}**")
                            st.bar_chart(model_version_accuracy)
                        else:
                            st.info(f"No data for {model}")
            else:
                st.info("No model data available for version accuracy charts")
            
            st.write("**Model Confidence by Version**")
            if len(unique_models) > 0:
                confidence_cols = st.columns(len(unique_models))
                
                for i, model in enumerate(unique_models):
                    with confidence_cols[i]:
                        model_data = accuracy_df[accuracy_df["model"] == model]
                        model_version_confidence = model_data.groupby("version")["confidence"].mean() * 100
                        
                        if not model_version_confidence.empty:
                            st.write(f"**{model}**")
                            st.bar_chart(model_version_confidence)
                        else:
                            st.info(f"No confidence data for {model}")
            else:
                st.info("No model data available for version confidence charts")
            
            st.subheader("🏆 Model Performance Comparison")
            
            model_metrics = accuracy_df.groupby("model").agg({
                "is_correct": ["count", "mean", "sum"],
                "confidence": "mean"
            }).round(3)
            
            model_metrics.columns = ["Total Tests", "Accuracy", "Correct Answers", "Avg Confidence"]
            model_metrics["Accuracy"] = model_metrics["Accuracy"] * 100
            model_metrics["Avg Confidence"] = model_metrics["Avg Confidence"] * 100
            
            st.dataframe(model_metrics, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Accuracy by Model")
                accuracy_chart_data = accuracy_df.groupby("model")["is_correct"].mean() * 100
                st.bar_chart(accuracy_chart_data)
            
            with col2:
                st.subheader("🎯 Confidence by Model")
                confidence_chart_data = accuracy_df.groupby("model")["confidence"].mean() * 100
                st.bar_chart(confidence_chart_data)
            
            st.subheader("🔄 Model Performance Across Versions")
            
            model_version_metrics = accuracy_df.groupby(["model", "version"]).agg({
                "is_correct": ["count", "mean"],
                "confidence": "mean"
            }).round(3)
            
            model_version_metrics.columns = ["Questions", "Accuracy", "Avg Confidence"]
            model_version_metrics["Accuracy"] = model_version_metrics["Accuracy"] * 100
            model_version_metrics["Avg Confidence"] = model_version_metrics["Avg Confidence"] * 100
            
            st.dataframe(model_version_metrics, use_container_width=True)
            
            st.subheader("📚 Topic-wise Performance")
            
            topic_metrics = accuracy_df.groupby(["model", "topic_tag"]).agg({
                "is_correct": ["count", "mean"],
                "confidence": "mean"
            }).round(3)
            
            topic_metrics.columns = ["Questions", "Accuracy", "Avg Confidence"]
            topic_metrics["Accuracy"] = topic_metrics["Accuracy"] * 100
            topic_metrics["Avg Confidence"] = topic_metrics["Avg Confidence"] * 100
            
            st.dataframe(topic_metrics, use_container_width=True)
            
            st.subheader("📝 Question Type Analysis")
            
            qtype_metrics = accuracy_df.groupby(["model", "question_type"]).agg({
                "is_correct": ["count", "mean"],
                "confidence": "mean"
            }).round(3)
            
            qtype_metrics.columns = ["Questions", "Accuracy", "Avg Confidence"]
            qtype_metrics["Accuracy"] = qtype_metrics["Accuracy"] * 100
            qtype_metrics["Avg Confidence"] = qtype_metrics["Avg Confidence"] * 100
            
            st.dataframe(qtype_metrics, use_container_width=True)
            
            st.subheader("🔍 Detailed Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                selected_model = st.selectbox("Filter by Model", ["All"] + list(accuracy_df["model"].unique()))
            with col2:
                selected_topic = st.selectbox("Filter by Topic", ["All"] + list(accuracy_df["topic_tag"].unique()))
            with col3:
                selected_type = st.selectbox("Filter by Type", ["All"] + list(accuracy_df["question_type"].unique()))
            with col4:
                selected_version = st.selectbox("Filter by Version", ["All"] + sorted(list(accuracy_df["version"].unique())))
            
            filtered_df = accuracy_df.copy()
            if selected_model != "All":
                filtered_df = filtered_df[filtered_df["model"] == selected_model]
            if selected_topic != "All":
                filtered_df = filtered_df[filtered_df["topic_tag"] == selected_topic]
            if selected_type != "All":
                filtered_df = filtered_df[filtered_df["question_type"] == selected_type]
            if selected_version != "All":
                filtered_df = filtered_df[filtered_df["version"] == selected_version]
            
            display_columns = ["q_id", "version", "model", "question_type", "is_correct", "confidence", "model_answer"]
            st.dataframe(filtered_df[display_columns], use_container_width=True)
            
            st.subheader("📥 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export Model Performance"):
                    if not model_metrics.empty:
                        buffer = io.StringIO()
                        model_metrics.to_csv(buffer)
                        st.download_button(
                            label="📁 Download Model Metrics",
                            data=buffer.getvalue(),
                            file_name=f"model_performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No model metrics to export.")
            
            with col2:
                if st.button("📋 Export Version Analysis"):
                    if not version_metrics.empty:
                        buffer = io.StringIO()
                        version_metrics.to_csv(buffer)
                        st.download_button(
                            label="📁 Download Version Analysis",
                            data=buffer.getvalue(),
                            file_name=f"version_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No version metrics to export.")

elif page == "🏆 Benchmark":
    st.header("🏆 Benchmark")
    st.markdown("**Comprehensive benchmark metrics including Basic Accuracy, Permutation Robustness Score, Distractor Sensitivity Score, and RAG Improvement Delta**")

    all_db_responses = list(responses.find())
    
    if not all_db_responses:
        st.warning("⚠️ No test results found. Please run tests first to see benchmark metrics.")
        
        st.subheader("📊 Available Questions for Testing")
        
        distinct_questions = list(questions.find({
            "$or": [
                {"q_version": "1"},
                {"q_version": {"$exists": False}}
            ]
        }))
        
        if distinct_questions:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Total Questions", len(distinct_questions))
            with col2:
                mcq_count = len([q for q in distinct_questions if q.get("q_type") == "MCQ"])
                st.metric("🔤 MCQ Questions", mcq_count)
            with col3:
                topics = len(set(q.get("topic_tag", "Unknown") for q in distinct_questions))
                st.metric("📚 Unique Topics", topics)
            
            st.info("💡 Run tests using 'Run Tests' or 'Run Tests with RAG' to generate benchmark data.")
        else:
            st.warning("⚠️ No questions found in database. Please upload questions first.")
    
    else:
        active_groq_model_names = {model_info["name"] for model_info in GROQ_MODELS.values()}
        rag_model_names = {f"{model_info['name']} (RAG)" for model_info in GROQ_MODELS.values()}
        all_active_model_names = active_groq_model_names.union(rag_model_names)
        
        structured_responses = [
            r for r in all_db_responses
            if (r.get("model_name") in all_active_model_names or r.get("rag_enabled", False)) and 
            ("mcq_answer" in r or "true_false_answer" in r or "short_answer" in r)
        ]
        
        if not structured_responses:
            st.warning("⚠️ No structured test results found. Please run tests using the new structured format.")
        else:
            with st.spinner("🔄 Calculating benchmark metrics..."):
                metrics = calculate_comprehensive_benchmark_metrics(structured_responses, questions)
            
            rag_responses = [r for r in structured_responses if r.get("rag_enabled", False)]
            non_rag_responses = [r for r in structured_responses if not r.get("rag_enabled", False)]
            
            if rag_responses and non_rag_responses:
                st.subheader("🔍 RAG vs Standard Model Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🤖 Standard Responses", len(non_rag_responses))
                with col2:
                    st.metric("🔬 RAG-Enhanced Responses", len(rag_responses))
                
                from core.benchmark_metrics import calculate_basic_accuracy
                rag_accuracies = calculate_basic_accuracy(rag_responses, questions)
                non_rag_accuracies = calculate_basic_accuracy(non_rag_responses, questions)
                
                if rag_accuracies and non_rag_accuracies:
                    avg_rag_accuracy = sum(rag_accuracies.values()) / len(rag_accuracies)
                    avg_non_rag_accuracy = sum(non_rag_accuracies.values()) / len(non_rag_accuracies)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🎯 Standard Accuracy", f"{avg_non_rag_accuracy:.1f}%")
                    with col2:
                        st.metric("🎯 RAG Accuracy", f"{avg_rag_accuracy:.1f}%", 
                                 delta=f"{avg_rag_accuracy - avg_non_rag_accuracy:.1f}%")
            
            st.subheader("📈 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Total Tests", len(structured_responses))
            with col2:
                accuracy_df = calculate_accuracy_structured(structured_responses, questions)
                if not accuracy_df.empty:
                    overall_accuracy = (accuracy_df["is_correct"].sum() / len(accuracy_df)) * 100
                    st.metric("✅ Overall Accuracy", f"{overall_accuracy:.1f}%")
                else:
                    st.metric("✅ Overall Accuracy", "0.0%")
            with col3:
                if not accuracy_df.empty:
                    avg_confidence = accuracy_df["confidence"].mean() * 100
                    st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")
                else:
                    st.metric("🎯 Avg Confidence", "0.0%")
            with col4:
                unique_models = len(set(r.get("model_name", "") for r in structured_responses))
                st.metric("🤖 Models Tested", unique_models)
            
            st.subheader("🏆 Advanced Benchmark Metrics")
            
            st.write("### 📊 Basic Accuracy (Acc)")
            st.write("*Percentage of questions answered correctly by each model*")
            
            basic_acc = metrics["basic_accuracy"]
            if basic_acc:
                acc_df = pd.DataFrame(list(basic_acc.items()), columns=["Model", "Accuracy (%)"])
                acc_df = acc_df.sort_values("Accuracy (%)", ascending=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(acc_df.set_index("Model")["Accuracy (%)"])
                with col2:
                    st.dataframe(acc_df, use_container_width=True)
            else:
                st.info("No accuracy data available.")
            
            st.write("### 🔄 Permutation Robustness Score (PRS)")
            st.write("*Consistency when MCQ choices are reordered across different versions*")
            
            prs = metrics["permutation_robustness"]
            if prs:
                prs_df = pd.DataFrame(list(prs.items()), columns=["Model", "PRS (%)"])
                prs_df = prs_df.sort_values("PRS (%)", ascending=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(prs_df.set_index("Model")["PRS (%)"])
                with col2:
                    st.dataframe(prs_df, use_container_width=True)
                    
                avg_prs = prs_df["PRS (%)"].mean()
                if avg_prs >= 80:
                    st.success(f"🎯 Excellent robustness (avg: {avg_prs:.1f}%)")
                elif avg_prs >= 60:
                    st.warning(f"⚠️ Moderate robustness (avg: {avg_prs:.1f}%)")
                else:
                    st.error(f"❌ Low robustness (avg: {avg_prs:.1f}%)")
            else:
                st.info("No PRS data available. Need multiple question versions to calculate.")
            
            st.write("### 🎭 Distractor Sensitivity Score (DSS)")
            st.write("*How often models are misled by 'None of the above' distractors (lower is better)*")
            
            dss = metrics["distractor_sensitivity"]
            if dss:
                dss_df = pd.DataFrame(list(dss.items()), columns=["Model", "DSS (%)"])
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(dss_df.set_index("Model")["DSS (%)"])
                with col2:
                    st.dataframe(dss_df, use_container_width=True)
                    
                avg_dss = dss_df["DSS (%)"].mean()
                if avg_dss <= 20:
                    st.success(f"🎯 Low distractor sensitivity (avg: {avg_dss:.1f}%)")
                elif avg_dss <= 40:
                    st.warning(f"⚠️ Moderate distractor sensitivity (avg: {avg_dss:.1f}%)")
                else:
                    st.error(f"❌ High distractor sensitivity (avg: {avg_dss:.1f}%)")
            else:
                st.info("No DSS data available. Need questions with 'None of the above' options.")
            
            st.write("### 🚀 RAG Improvement Delta (ΔRAG)")
            st.write("*Change in accuracy when using RAG vs standard approach*")
            
            rag_delta = metrics["rag_improvement_delta"]
            if rag_delta:
                delta_df = pd.DataFrame(list(rag_delta.items()), columns=["Model", "ΔRAG (%)"])
                delta_df = delta_df.sort_values("ΔRAG (%)", ascending=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    chart_data = delta_df.set_index("Model")["ΔRAG (%)"]
                    st.bar_chart(chart_data)
                with col2:
                    for _, row in delta_df.iterrows():
                        delta_val = row["ΔRAG (%)"]
                        if delta_val > 0:
                            st.success(f"🟢 {row['Model']}: +{delta_val:.1f}%")
                        elif delta_val < 0:
                            st.error(f"🔴 {row['Model']}: {delta_val:.1f}%")
                        else:
                            st.info(f"⚪ {row['Model']}: {delta_val:.1f}%")
                
                avg_delta = delta_df["ΔRAG (%)"].mean()
                positive_models = len(delta_df[delta_df["ΔRAG (%)"] > 0])
                total_models = len(delta_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 Avg RAG Impact", f"{avg_delta:.1f}%")
                with col2:
                    st.metric("✅ Models Improved", f"{positive_models}/{total_models}")
                with col3:
                    improvement_rate = (positive_models / total_models) * 100 if total_models > 0 else 0
                    st.metric("🎯 Improvement Rate", f"{improvement_rate:.1f}%")
            else:
                st.info("No RAG delta data available. Need both standard and RAG test results.")
            
            st.subheader("📚 Performance by Topic")
            
            topic_performance = metrics["performance_by_topic"]
            if not topic_performance.empty:
                st.write("### 📊 Model Accuracy by Topic")
                
                topic_pivot = topic_performance.pivot(index="topic", columns="model", values="Accuracy")
                
                if not topic_pivot.empty:
                    st.dataframe(topic_pivot, use_container_width=True)
                    
                    st.write("### 📈 Topic Difficulty Ranking")
                    topic_difficulty = topic_performance.groupby("topic")["Accuracy"].mean().sort_values(ascending=True)
                    
                    difficulty_df = pd.DataFrame({
                        "Topic": topic_difficulty.index,
                        "Avg Accuracy (%)": topic_difficulty.values
                    }).reset_index(drop=True)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.bar_chart(topic_difficulty)
                    with col2:
                        st.dataframe(difficulty_df, use_container_width=True)
                        
                        easiest_topic = difficulty_df.iloc[-1]["Topic"]
                        hardest_topic = difficulty_df.iloc[0]["Topic"]
                        
                        st.success(f"🟢 Easiest: {easiest_topic}")
                        st.error(f"🔴 Hardest: {hardest_topic}")
                
                st.write("### 📋 Detailed Performance by Topic")
                st.dataframe(topic_performance, use_container_width=True)
            else:
                st.info("No topic performance data available.")
            
            st.subheader("🔢 Performance by Question Version")
            
            version_metrics = metrics["version_specific_metrics"]
            if version_metrics:
                st.write("### 📊 Accuracy by Question Version")
                st.write("*V1: Original, V2: Reordered MCQ, V3: MCQ with 'None of Above', V4: True/False*")
                
                version_data = []
                for model, versions in version_metrics.items():
                    for version, accuracy in versions.items():
                        version_data.append({
                            "Model": model,
                            "Version": version,
                            "Accuracy (%)": accuracy
                        })
                
                if version_data:
                    version_df = pd.DataFrame(version_data)
                    
                    version_pivot = version_df.pivot(index="Model", columns="Version", values="Accuracy (%)")
                    
                    if not version_pivot.empty:
                        st.dataframe(version_pivot, use_container_width=True)
                        
                        st.write("### 📈 Version Difficulty Analysis")
                        version_difficulty = version_df.groupby("Version")["Accuracy (%)"].agg(['mean', 'std', 'count']).round(2)
                        version_difficulty.columns = ["Avg Accuracy (%)", "Std Dev", "Test Count"]
                        version_difficulty = version_difficulty.sort_values("Avg Accuracy (%)", ascending=True)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.bar_chart(version_difficulty["Avg Accuracy (%)"])
                        with col2:
                            st.dataframe(version_difficulty, use_container_width=True)
                            
                            if not version_difficulty.empty:
                                easiest_version = version_difficulty.index[-1]
                                hardest_version = version_difficulty.index[0]
                                
                                st.success(f"🟢 Easiest: {easiest_version}")
                                st.error(f"🔴 Hardest: {hardest_version}")
                        
                        st.write("### 🔄 Model Consistency Across Versions")
                        
                        model_consistency = []
                        for model in version_pivot.index:
                            model_accuracies = version_pivot.loc[model].dropna()
                            if len(model_accuracies) > 1:
                                model_consistency.append({
                                    "Model": model,
                                    "Versions Tested": len(model_accuracies),
                                    "Accuracy Range": f"{model_accuracies.min():.1f}% - {model_accuracies.max():.1f}%"
                                })
                        
                        if model_consistency:
                            consistency_df = pd.DataFrame(model_consistency)
                            consistency_df = consistency_df.sort_values("Consistency Score", ascending=False)
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.bar_chart(consistency_df.set_index("Model")["Consistency Score"])
                            with col2:
                                st.dataframe(consistency_df, use_container_width=True)
                                
                                avg_consistency = consistency_df["Consistency Score"].mean()
                                if avg_consistency >= 80:
                                    st.success(f"🎯 High consistency (avg: {avg_consistency:.1f})")
                                elif avg_consistency >= 60:
                                    st.warning(f"⚠️ Moderate consistency (avg: {avg_consistency:.1f})")
                                else:
                                    st.error(f"❌ Low consistency (avg: {avg_consistency:.1f})")
                
                st.write("### 💡 Version-Specific Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📊 Version Characteristics:**")
                    st.write("- **V1 (Original)**: Baseline performance")
                    st.write("- **V2 (Reordered)**: Tests option order bias")
                    st.write("- **V3 (None of Above)**: Tests distractor sensitivity")
                    st.write("- **V4 (True/False)**: Tests format adaptation")
                
                with col2:
                    st.write("**🎯 Performance Expectations:**")
                    if version_data:
                        v1_avg = version_df[version_df["Version"] == "V1 (Original)"]["Accuracy (%)"].mean()
                        v2_avg = version_df[version_df["Version"] == "V2 (Reordered)"]["Accuracy (%)"].mean()
                        v3_avg = version_df[version_df["Version"] == "V3 (None of Above)"]["Accuracy (%)"].mean()
                        v4_avg = version_df[version_df["Version"] == "V4 (True/False)"]["Accuracy (%)"].mean()
                        
                        if not pd.isna(v1_avg) and not pd.isna(v2_avg):
                            if abs(v1_avg - v2_avg) < 5:
                                st.success("✅ Low option order bias")
                            else:
                                st.warning("⚠️ Significant option order bias detected")
                        
                        if not pd.isna(v3_avg):
                            if v3_avg < 80:
                                st.info("📊 'None of above' creates difficulty")
                            else:
                                st.success("✅ Resistant to distractor options")
                        
                        if not pd.isna(v4_avg):
                            if v4_avg > 85:
                                st.success("✅ Good format adaptation")
                            else:
                                st.info("📊 Format change affects performance")
            else:
                st.info("No version-specific data available. Test questions with multiple versions to see this analysis.")
            
            st.subheader("🏆 Model Ranking Summary")
            
            ranking_data = []
            for model in basic_acc.keys():
                base_model = model.replace(" (RAG)", "")
                
                ranking_data.append({
                    "Model": model,
                    "Basic Accuracy (%)": basic_acc.get(model, 0),
                    "PRS (%)": prs.get(model, 0),
                    "ΔRAG (%)": rag_delta.get(base_model, 0) if not model.endswith("(RAG)") else 0
                })
            
            if ranking_data:
                ranking_df = pd.DataFrame(ranking_data)
                
                ranking_df["Composite Score"] = (
                    ranking_df["Basic Accuracy (%)"] * 0.4 +  # 40% weight
                    ranking_df["PRS (%)"] * 0.3 +              # 30% weight
                    (100 - ranking_df["DSS (%)"]) * 0.2 +      # 20% weight (inverted)
                    ranking_df["ΔRAG (%)"] * 0.1               # 10% weight
                )
                
                ranking_df = ranking_df.sort_values("Composite Score", ascending=False)
                ranking_df["Rank"] = range(1, len(ranking_df) + 1)
                
                display_cols = ["Rank", "Model", "Composite Score", "Basic Accuracy (%)", "PRS (%)", "DSS (%)", "ΔRAG (%)"]
                st.dataframe(ranking_df[display_cols].round(2), use_container_width=True)
                
                top_model = ranking_df.iloc[0]
                st.success(f"🏆 **Top Performer:** {top_model['Model']} (Score: {top_model['Composite Score']:.1f})")
            
            st.subheader("📥 Export Benchmark Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export All Metrics"):
                    export_data = {
                        "Basic Accuracy": basic_acc,
                        "Permutation Robustness": prs,
                        "Distractor Sensitivity": dss,
                        "RAG Improvement Delta": rag_delta
                    }
                    
                    buffer = io.StringIO()
                    buffer.write("COMP430 LLM Benchmark Report\n")
                    buffer.write("=" * 50 + "\n\n")
                    
                    for metric_name, metric_data in export_data.items():
                        buffer.write(f"{metric_name}:\n")
                        for model, score in metric_data.items():
                            buffer.write(f"  {model}: {score:.2f}%\n")
                        buffer.write("\n")
                    
                    if version_metrics:
                        buffer.write("Version-Specific Performance:\n")
                        for model, versions in version_metrics.items():
                            buffer.write(f"  {model}:\n")
                            for version, accuracy in versions.items():
                                buffer.write(f"    {version}: {accuracy:.2f}%\n")
                        buffer.write("\n")
                    
                    st.download_button(
                        label="📁 Download Benchmark Report",
                        data=buffer.getvalue(),
                        file_name=f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("📋 Export Topic Performance"):
                    if not topic_performance.empty:
                        buffer = io.StringIO()
                        topic_performance.to_csv(buffer, index=False)
                        st.download_button(
                            label="📁 Download Topic Analysis",
                            data=buffer.getvalue(),
                            file_name=f"topic_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No topic performance data to export.")
            
            with col3:
                if st.button("🔢 Export Version Analysis"):
                    if version_metrics:
                        version_export_data = []
                        for model, versions in version_metrics.items():
                            for version, accuracy in versions.items():
                                version_export_data.append({
                                    "Model": model,
                                    "Version": version,
                                    "Accuracy (%)": accuracy
                                })
                        
                        if version_export_data:
                            version_export_df = pd.DataFrame(version_export_data)
                            buffer = io.StringIO()
                            version_export_df.to_csv(buffer, index=False)
                            st.download_button(
                                label="📁 Download Version Analysis",
                                data=buffer.getvalue(),
                                file_name=f"version_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("⚠️ No version data to export.")
                    else:
                        st.warning("⚠️ No version metrics to export.")

elif page == "📈 Evaluate Metrics":
    st.header("📈 Evaluate Model Metrics Across Question Versions")
    st.markdown("**Advanced metrics evaluation including version analysis and robustness scoring**")

    st.info("🚧 Advanced metrics evaluation (PRS, DSS, RS) coming soon! Currently showing basic version analysis.")
    
    st.subheader("📊 Available Questions for Analysis")
    
    pipeline = [
        {
            "$match": {
                "$or": [
                    {"q_version": "1"},
                    {"q_version": {"$exists": False}}
                ]
            }
        },
        {
            "$group": {
                "_id": "$q_id", 
                "doc": {"$first": "$$ROOT"}
            }
        },
        {
            "$replaceRoot": { "newRoot": "$doc" }
        }
    ]
    
    distinct_questions = list(questions.aggregate(pipeline))
    
    if distinct_questions:
        for q in distinct_questions:
            version_count = questions.count_documents({
                "$or": [
                    {"q_id": q["q_id"]},
                    {"original_q_id": q["q_id"]}
                ]
            })
            
            response_count = responses.count_documents({"question_id": q["q_id"]})
            
            with st.expander(f"📝 {q['q_id']}: {q['q_text'][:60]}..."):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Versions", version_count)
                with col2:
                    st.metric("Responses", response_count)
                with col3:
                    st.metric("Type", q.get("q_type", "Unknown"))
    else:
        st.warning("⚠️ No questions found for metrics evaluation.")

elif page == "📚 Create Questions":
    st.header("📚 Create Questions from PDFs")
    st.markdown("Upload lecture PDFs to automatically generate questions")
    
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s)")
        
        for file in uploaded_files:
            st.write(f"- {file.name}")
        
        if st.button("Generate Questions", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    filename = Path(file.name).stem
                    lecture_number = filename.split('_')[0] if '_' in filename else filename
                    lecture_number = lecture_number[1:] if lecture_number.startswith('L') else lecture_number
                    
                    prompt = f"""I will give you a lecture slides from {filename}. You will read it and you will prepare 20 MCQ's. 
                    You will prepare the questions to:
                    1- measure the memorization of the student
                    2- measure the understanding of the student of the core concepts
                    3- measure the robustness of a student to different distractor, tricky questions
                    4- measure the students attention in details.
                    
                    Make sure the questions are well difficult and challenging, well prepared, options are well prepared, tricky. Do not keep the questions  short, make them long
                    if needed add intentional distractors such as (i.e. if something is true try to convince a student who is not 
                    studied very well to the topic fall in to the trap in the opposite option), you can add close options to 
                    distract a student.
                    
                    IMPORTANT: Your response must start with a CSV header row exactly as shown below, followed by the data rows.
                    Do not include any text before or after the CSV data. The CSV must have these exact columns:
                    
                    q_id,q_text,q_type,topic_tag,q_options,q_correct_answer
                    
                    Example format for {filename}:
                    q_id,q_text,q_type,topic_tag,q_options,q_correct_answer
                    LEC{lecture_number}_001,"What is the main purpose of X?",MCQ,"Topic 1","[""Option A"",""Option B"",""Option C"",""Option D""]","Option A"
                    LEC{lecture_number}_002,"Which statement about Y is true?",MCQ,"Topic 1","[""Option A"",""Option B"",""Option C"",""Option D""]","Option B"
                    
                    Rules for the CSV:
                    1. q_id must be unique and follow the format LEC{lecture_number}_XXX where XXX is a three-digit number (e.g., for {filename} use LEC{lecture_number}_001, LEC{lecture_number}_002, etc.)
                    2. q_text must be enclosed in double quotes
                    3. q_type must be MCQ
                    4. topic_tag should be a relevant topic from the lecture
                    5. q_options must be a JSON array string with exactly 4 options
                    6. q_correct_answer must be one of the options exactly as written
                    7. Do not include any commas within the q_text or q_options fields
                    8. Make sure all fields are properly escaped and quoted
                    
                    Lecture slides:
                    """
                    prompt += text
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are an expert in creating educational questions. You must respond with a valid CSV format as specified."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=2000
                        )
                        
                        questions_text = response.choices[0].message.content.strip()
                        
                        csv_start = questions_text.find("q_id,q_text")
                        if csv_start == -1:
                            st.error(f"Invalid response format. Response received:\n{questions_text[:500]}...")
                            continue
                        
                        csv_data = questions_text[csv_start:].strip()
                        
                        if not csv_data.startswith("q_id,q_text,q_type,topic_tag,q_options,q_correct_answer"):
                            st.error("Invalid CSV header format")
                            continue
                        
                        try:
                            df = pd.read_csv(StringIO(csv_data))
                            
                            required_columns = ["q_id", "q_text", "q_type", "topic_tag", "q_options", "q_correct_answer"]
                            if not all(col in df.columns for col in required_columns):
                                st.error(f"Missing required columns. Found columns: {list(df.columns)}")
                                continue
                            
                            if not all(df["q_type"] == "MCQ"):
                                st.error("All questions must be of type MCQ")
                                continue
                            
                            csv_filename = f"{Path(file.name).stem}_questions.csv"
                            csv_path = Path("data") / csv_filename
                            df.to_csv(csv_path, index=False)
                            
                            progress = (i + 1) / len(uploaded_files)
                            progress_bar.progress(progress)
                            
                        except pd.errors.ParserError as e:
                            st.error(f"Error parsing CSV data: {str(e)}")
                            continue
                        
                    except openai.AuthenticationError:
                        st.error("⚠️ OpenAI API key is invalid. Please check your API key in the .env file.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Error generating questions for {file.name}: {str(e)}")
                        continue
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            status_text.text("✅ All files processed!")
            
            if st.button("Import Questions to Database"):
                try:
                    csv_files = list(Path("data").glob("*_questions.csv"))
                    
                    if not csv_files:
                        st.warning("No question files found in the data directory.")
                        st.stop()
                    
                    for csv_file in csv_files:
                        try:
                            df = pd.read_csv(csv_file)
                            
                            questions_list = df.to_dict('records')
                            
                            success_count = 0
                            error_count = 0
                            error_messages = []
                            
                            for question in questions_list:
                                if question['q_type'] == 'MCQ':
                                    try:
                                        question['q_options'] = json.loads(question['q_options'])
                                    except json.JSONDecodeError:
                                        error_messages.append(f"Invalid JSON format for options in question {question['q_id']}")
                                        error_count += 1
                                        continue
                                
                                if 'q_version' not in question:
                                    question['q_version'] = "1"
                                
                                is_valid, message = validate_question(question)
                                if is_valid:
                                    questions.insert_one(question)
                                    success_count += 1
                                else:
                                    error_messages.append(f"Question {question['q_id']}: {message}")
                                    error_count += 1
                            
                            st.success(f"✅ Successfully imported {success_count} questions from {csv_file.name}!")
                            if error_count > 0:
                                st.error(f"❌ Failed to import {error_count} questions from {csv_file.name}:")
                                for msg in error_messages:
                                    st.error(msg)
                            
                            if success_count > 0:
                                st.subheader("📋 Imported Questions Summary")
                                
                                imported_questions = list(questions.find(
                                    {"q_id": {"$in": [q["q_id"] for q in questions_list if "q_id" in q]}},
                                    sort=[("q_id", 1)]
                                ))
                                
                                summary_data = []
                                for q in imported_questions:
                                    summary_data.append({
                                        "Question ID": q["q_id"],
                                        "Type": q["q_type"],
                                        "Topic": q["topic_tag"],
                                        "Question": q["q_text"][:100] + "..." if len(q["q_text"]) > 100 else q["q_text"],
                                        "Options": str(q["q_options"]) if "q_options" in q else "N/A",
                                        "Correct Answer": q["q_correct_answer"]
                                    })
                                
                                if summary_data:
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Questions", len(summary_data))
                                    with col2:
                                        topics = len(set(q["Topic"] for q in summary_data))
                                        st.metric("Unique Topics", topics)
                                    with col3:
                                        types = len(set(q["Type"] for q in summary_data))
                                        st.metric("Question Types", types)
                            
                            csv_file.unlink()
                            
                        except Exception as e:
                            st.error(f"Error processing {csv_file.name}: {str(e)}")
                    
                    st.success("✅ All files processed!")
                    
                except Exception as e:
                    st.error(f"Error importing questions: {str(e)}")

elif page == "💾 Manage CSV Files":
    st.header("💾 Manage CSV Files")
    st.markdown("View and import questions from CSV files in the data folder")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    csv_files = list(data_dir.glob("*_questions.csv"))
    
    if not csv_files:
        st.info("No CSV files found in the data directory. Generate questions from PDFs first.")
    else:
        st.subheader("📁 Available CSV Files")
        
        for csv_file in csv_files:
            with st.expander(f"📄 {csv_file.name}", expanded=True):
                try:
                    df = pd.read_csv(csv_file)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Questions", len(df))
                    with col2:
                        topics = len(df["topic_tag"].unique())
                        st.metric("Unique Topics", topics)
                    with col3:
                        types = len(df["q_type"].unique())
                        st.metric("Question Types", types)
                    
                    st.subheader("📋 Questions Preview")
                    preview_df = df[["q_id", "q_type", "topic_tag", "q_text"]].copy()
                    preview_df["q_text"] = preview_df["q_text"].apply(lambda x: x[:100] + "..." if len(str(x)) > 100 else x)
                    st.dataframe(preview_df, use_container_width=True)
                    
                    if st.button(f"Import {csv_file.name} to Database", key=f"import_{csv_file.name}"):
                        try:
                            questions_list = df.to_dict('records')
                            
                            success_count = 0
                            error_count = 0
                            error_messages = []
                            
                            for question in questions_list:
                                if question['q_type'] == 'MCQ':
                                    try:
                                        question['q_options'] = json.loads(question['q_options'])
                                    except json.JSONDecodeError:
                                        error_messages.append(f"Invalid JSON format for options in question {question['q_id']}")
                                        error_count += 1
                                        continue
                                
                                if 'q_version' not in question:
                                    question['q_version'] = "1"
                                
                                is_valid, message = validate_question(question)
                                if is_valid:
                                    questions.insert_one(question)
                                    success_count += 1
                                else:
                                    error_messages.append(f"Question {question['q_id']}: {message}")
                                    error_count += 1
                            
                            st.success(f"✅ Successfully imported {success_count} questions!")
                            if error_count > 0:
                                st.error(f"❌ Failed to import {error_count} questions:")
                                for msg in error_messages:
                                    st.error(msg)
                            
                            csv_file.unlink()
                            
                        except Exception as e:
                            st.error(f"Error importing questions: {str(e)}")
                    
                    if st.button(f"Delete {csv_file.name}", key=f"delete_{csv_file.name}"):
                        try:
                            csv_file.unlink()
                            st.success(f"✅ {csv_file.name} deleted successfully!")
                        except Exception as e:
                            st.error(f"Error deleting file: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error reading {csv_file.name}: {str(e)}")
        
        if len(csv_files) > 1:
            if st.button("Import All Files to Database", type="primary"):
                total_success = 0
                total_errors = 0
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        
                        questions_list = df.to_dict('records')
                        
                        success_count = 0
                        error_count = 0
                        
                        for question in questions_list:
                            if question['q_type'] == 'MCQ':
                                try:
                                    question['q_options'] = json.loads(question['q_options'])
                                except json.JSONDecodeError:
                                    error_count += 1
                                    continue
                            
                            if 'q_version' not in question:
                                question['q_version'] = "1"
                            
                            is_valid, _ = validate_question(question)
                            if is_valid:
                                questions.insert_one(question)
                                success_count += 1
                            else:
                                error_count += 1
                        
                        total_success += success_count
                        total_errors += error_count
                        
                        csv_file.unlink()
                        
                    except Exception as e:
                        st.error(f"Error processing {csv_file.name}: {str(e)}")
                
                st.success(f"✅ Import complete! Successfully imported {total_success} questions.")
                if total_errors > 0:
                    st.error(f"❌ Failed to import {total_errors} questions.")
                

elif page == "📚 RAG System":
    st.header("📚 RAG System")
    st.markdown("Upload lecture PDFs and query them using RAG")
    
    from core.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    
    tab1, tab2 = st.tabs(["📤 Upload PDFs", "🔍 Query System"])
    
    with tab1:
        st.subheader("Upload Lecture PDFs")
        
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            for file in uploaded_files:
                st.write(f"- {file.name}")
            
            if st.button("Process PDFs", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    try:
                        temp_path = Path("data/processed_pdfs") / file.name
                        with open(temp_path, "wb") as f:
                            f.write(file.getvalue())
                        
                        chunks = rag.process_pdf(temp_path)
                        
                        if chunks:
                            embedded_chunks = rag.create_embeddings(chunks)
                            
                            rag.store_vectors(embedded_chunks)
                            
                            st.success(f"✅ Successfully processed {file.name}")
                        else:
                            st.error(f"❌ No content extracted from {file.name}")
                        
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                status_text.text("✅ All files processed!")
    
    with tab2:
        st.subheader("Query the RAG System")
        
        query = st.text_area("Enter your question about the lecture content")
        
        if query:
            if st.button("Get Answer", type="primary"):
                with st.spinner("Searching and generating answer..."):
                    try:
                        result = rag.query_rag(query)
                        
                        if result:
                            st.subheader("Answer")
                            st.write(result['response'])
                            
                            st.subheader("Sources")
                            for source in result['sources']:
                                st.write(f"- {source}")
                            
                            st.subheader("Statistics")
                            st.json(result['stats'])
                        else:
                            st.error("❌ Error getting answer. Please try again.")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("**COMP430 LLM Benchmark Dashboard** - Advanced AI Model Testing & Analysis Platform") 