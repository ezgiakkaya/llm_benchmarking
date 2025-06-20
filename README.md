# COMP430 LLM Benchmarking Project

## Overview

This is a comprehensive LLM (Large Language Model) benchmarking platform developed for the COMP430 course at Koç University. The project provides advanced AI model comparison capabilities with structured responses, confidence scoring, and RAG (Retrieval-Augmented Generation) enhancement.

## 🌐 Live Demo

**Visit and try the website at:** [https://llmbenchmarking-6ano9.kinsta.app/](https://llmbenchmarking-6ano9.kinsta.app/)

## Features

### 🚀 Core Functionality
- **Multi-Model Testing**: Compare performance across different LLM models (Groq models including LLaMA, Gemma, DeepSeek)
- **Question Types**: Support for Multiple Choice Questions (MCQ), True/False, and Short Answer questions
- **Version Management**: Generate and test multiple versions of questions (original, reordered, with distractors, format changes)
- **RAG Integration**: Enhanced model performance through Retrieval-Augmented Generation

### 📊 Advanced Analytics
- **Comprehensive Metrics**: Basic Accuracy, Permutation Robustness Score (PRS), Distractor Sensitivity Score (DSS)
- **RAG Improvement Delta**: Measure performance gains from RAG enhancement
- **Topic-wise Analysis**: Performance breakdown by subject areas
- **Version Comparison**: Analyze model consistency across question variations

### 🛠️ Management Tools
- **Question Upload**: Single and batch question upload capabilities
- **PDF Processing**: Automatic question generation from lecture PDFs using OpenAI GPT-4
- **Database Management**: MongoDB integration for scalable data storage
- **Export Functionality**: Download results and analysis reports

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python with FastAPI-style architecture
- **Database**: MongoDB
- **LLM Integration**: Groq API
- **RAG System**: Pinecone vector database with OpenAI embeddings
- **Deployment**: Kinsta hosting platform

## Project Structure

```
COMP430_PROJECT/
├── app/
│   └── main.py              # Main Streamlit application
├── core/
│   ├── database.py          # Database connections and collections
│   ├── llm_clients.py       # LLM API integrations
│   ├── models.py            # Data models and schemas
│   ├── question_versioning.py # Question version generation
│   ├── benchmark_metrics.py # Performance calculation algorithms
│   └── rag_pipeline.py      # RAG implementation
├── scripts/                 # Utility and analysis scripts
└── data/                    # Data storage directory
```

## Getting Started

### Prerequisites
- Python 3.8+
- MongoDB instance
- Groq API key
- OpenAI API key (for question generation)
- Pinecone API key (for RAG functionality)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd COMP430_PROJECT
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
MONGODB_URI=your_mongodb_connection_string
```

5. Run the application:
```bash
streamlit run app/main.py
```

## Usage

### Web Interface
1. **Upload Questions**: Add questions individually or via CSV batch upload
2. **Generate from PDFs**: Upload lecture PDFs to auto-generate questions
3. **Run Tests**: Execute LLM evaluations with or without RAG enhancement
4. **View Results**: Analyze performance through comprehensive dashboards
5. **Export Data**: Download detailed reports and metrics

### Command Line Scripts
- `scripts/run_evaluation.py`: Run evaluations for all questions
- `scripts/calculate_metrics.py`: Calculate performance metrics
- `scripts/generate_versions.py`: Generate question versions
- `scripts/analyze_results.py`: Generate analysis reports

## Academic Context

This project was developed as part of the COMP430 course at Koç University, focusing on:
- Large Language Model evaluation methodologies
- Benchmarking techniques for AI systems
- Educational technology applications
- Advanced natural language processing

## Contributing

This is an academic project for COMP430 at Koç University. For questions or contributions, please contact the course instructors or project maintainers.

## License

This project is developed for academic purposes at Koç University.

---

**Koç University - COMP430 Course Project**  
**Live Demo**: [https://llmbenchmarking-6ano9.kinsta.app/](https://llmbenchmarking-6ano9.kinsta.app/)
