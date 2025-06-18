import os
from pathlib import Path
import PyPDF2
from typing import List, Dict, Any
import json
from transformers import AutoModel
import pinecone
from groq import Groq
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self):
        # Validate environment variables
        required_env_vars = {
            'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
            'PINECONE_INDEX_NAME': os.getenv('PINECONE_INDEX_NAME'),
            'GROQ_API_KEY': os.getenv('GROQ_API_KEY')
        }
        
        missing_vars = [var for var, value in required_env_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        try:
            # Initialize Pinecone with new API (Pinecone class)
            print(f"Initializing Pinecone (new API, Pinecone class)")
            pc = Pinecone(api_key=required_env_vars['PINECONE_API_KEY'])
            index_name = required_env_vars['PINECONE_INDEX_NAME']
            # List indexes
            print("Listing existing Pinecone indexes...")
            existing_indexes = pc.list_indexes().names()
            print(f"Found indexes: {existing_indexes}")
            if index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=768,  # Dimension for jina embeddings
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-west-2")
                )
                print(f"Successfully created index: {index_name}")
            print(f"Connecting to Pinecone index: {index_name}")
            self.index = pc.Index(index_name)
            print("Successfully connected to Pinecone index")
            
            # Initialize embedding model
            print("Initializing embedding model...")
            self.embedding_model = AutoModel.from_pretrained(
                'jinaai/jina-embeddings-v2-base-en',
                trust_remote_code=True
            )
            print("Successfully initialized embedding model")
            
            # Initialize Groq client
            print("Initializing Groq client...")
            self.groq_client = Groq(api_key=required_env_vars['GROQ_API_KEY'])
            print("Successfully initialized Groq client")
            
            # Initialize directories
            self.data_dir = Path("data")
            self.embeddings_dir = self.data_dir / "embeddings"
            self.processed_dir = self.data_dir / "processed_pdfs"
            self.vector_store_dir = self.data_dir / "vector_store"
            
            # Create directories if they don't exist
            for dir_path in [self.data_dir, self.embeddings_dir, self.processed_dir, self.vector_store_dir]:
                dir_path.mkdir(exist_ok=True)
            print("Successfully initialized all directories")
                
        except Exception as e:
            print(f"Error initializing RAG pipeline: {str(e)}")
            raise
    
    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Process a PDF file and return chunks of text with metadata."""
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    # Split text into chunks (you can adjust the chunk size)
                    chunk_size = 1000
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i + chunk_size]
                        if chunk.strip():  # Only add non-empty chunks
                            chunks.append({
                                'text': chunk,
                                'metadata': {
                                    'source': pdf_path.name,
                                    'page': page_num + 1,
                                    'chunk_index': i // chunk_size
                                }
                            })
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for text chunks."""
        embedded_chunks = []
        
        for chunk in chunks:
            try:
                # Create embedding
                embedding = self.embedding_model.encode(chunk['text']).tolist()
                
                # Add embedding to chunk data
                chunk_with_embedding = {
                    'id': f"{chunk['metadata']['source']}_{chunk['metadata']['page']}_{chunk['metadata']['chunk_index']}",
                    'values': embedding,
                    'metadata': {
                        'text': chunk['text'],
                        'source': chunk['metadata']['source'],
                        'page': chunk['metadata']['page'],
                        'chunk_index': chunk['metadata']['chunk_index']
                    }
                }
                embedded_chunks.append(chunk_with_embedding)
            
            except Exception as e:
                print(f"Error creating embedding for chunk: {str(e)}")
                continue
        
        return embedded_chunks
    
    def store_vectors(self, embedded_chunks: List[Dict[str, Any]]):
        """Store vectors in Pinecone."""
        try:
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(embedded_chunks), batch_size):
                batch = embedded_chunks[i:i + batch_size]
                self.index.upsert(vectors=batch)
        
        except Exception as e:
            print(f"Error storing vectors: {str(e)}")
    
    def query_rag(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Prepare context
            matched_info = ' '.join(item['metadata']['text'] for item in results['matches'])
            sources = [item['metadata']['source'] for item in results['matches']]
            
            # Create system prompt
            context = f"Information: {matched_info} and the sources: {sources}"
            sys_prompt = f"""
            Instructions:
            - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
            - Utilize the context provided for accurate and specific information.
            - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
            - Cite your sources
            Context: {context}
            """
            
            # Query Groq with fallback models
            models_to_try = [
                "llama3-70b-8192",
                "gemma2-9b-it",
                "deepseek-r1-distill-llama-70b"
            ]
            
            last_error = None
            for model_id in models_to_try:
                try:
                    completion = self.groq_client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": query}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    return {
                        'response': completion.choices[0].message.content,
                        'sources': sources,
                        'stats': {
                            'model_used': model_id,
                            'total_tokens': completion.usage.total_tokens,
                            'prompt_tokens': completion.usage.prompt_tokens,
                            'completion_tokens': completion.usage.completion_tokens
                        }
                    }
                except Exception as e:
                    print(f"Error with model {model_id}: {str(e)}")
                    last_error = e
                    continue
            
            print(f"All models failed. Last error: {str(last_error)}")
            return None
        
        except Exception as e:
            print(f"Error in RAG query: {str(e)}")
            return None
    
    def process_directory(self, pdf_dir: Path):
        """Process all PDFs in a directory."""
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            
            # Process PDF
            chunks = self.process_pdf(pdf_file)
            
            # Create embeddings
            embedded_chunks = self.create_embeddings(chunks)
            
            # Store vectors
            self.store_vectors(embedded_chunks)
            
            print(f"Completed processing {pdf_file.name}") 