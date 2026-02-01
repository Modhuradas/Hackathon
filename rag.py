'''
Docstring for rag
1. Load EU directive PDF
2. Split into chunks
3. Create embeddings(Using API)
4. Store in vector DB
Enable search
'''
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Loading environment variables
load_dotenv()

PDF_PATH = r"C:\Users\modhu\OneDrive\Desktop\Hackathon\EU_2023_Dir.pdf"  # Your EU directive PDF
CHROMA_DB_DIR = "./chroma_db"  # Where vector database will be stored
CHUNK_SIZE = 1000  # Size of text chunks (characters)
CHUNK_OVERLAP = 200  # Overlap between chunks (for context continuity)

import re

def chunk_by_articles(documents):
    """
    Chunk documents by article sections
    Attempts to keep each article as a complete chunk
    """
    chunks = []
    
    for doc in documents:
        text = doc.page_content
        page = doc.metadata.get('page', 'Unknown')
        
        # Split on article markers
        # Matches: "Article 1", "Article 2", etc.
        article_pattern = r'(Article\s+\d+)'
        
        # Split but keep the delimiter (article marker)
        parts = re.split(f'({article_pattern})', text)
        
        current_article = None
        current_text = ""
        
        for part in parts:
            # Check if this part is an article marker
            if re.match(article_pattern, part):
                # Save previous article if exists
                if current_article and current_text.strip():
                    chunks.append({
                        'content': current_text.strip(),
                        'metadata': {
                            'page': page,
                            'article': current_article,
                            'source': 'EU_Green_Claims_Directive'
                        }
                    })
                
                # Start new article
                current_article = part
                current_text = part + " "
            else:
                current_text += part
        
        # Don't forget the last article
        if current_article and current_text.strip():
            chunks.append({
                'content': current_text.strip(),
                'metadata': {
                    'page': page,
                    'article': current_article,
                    'source': 'EU_Green_Claims_Directive'
                }
            })
    
    # Convert to Document objects
    from langchain.schema import Document
    doc_chunks = [
        Document(page_content=chunk['content'], metadata=chunk['metadata'])
        for chunk in chunks
    ]
    
    print(f"✅ Created {len(doc_chunks)} article-based chunks")
    
    # Show sample of what we got
    articles = [chunk['metadata'].get('article') for chunk in chunks if chunk['metadata'].get('article')]
    unique_articles = set(articles)
    print(f"📋 Found {len(unique_articles)} unique articles: {sorted(unique_articles)[:10]}...")
    
    return doc_chunks




def load_and_chunk_pdf(pdf_path):
    """
    Loads PDF and split into chunks
    Args: pdf_path: Path to the PDF file 
    Returns: ist of document chunks
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages")
    # First try article-based chunking
    try:
        chunks = chunk_by_articles(documents)
        
        # If we got reasonable number of chunks, use them
        if len(chunks) > 10:  # Sanity check
            print(f"✅ Using article-based chunking")
            return chunks
    except Exception as e:
        print(f"⚠️ Article-based chunking failed: {e}")
        print(f"Falling back to character-based chunking...")
    
    # Split into chunks 
    # # Fallback: use old method if article chunking fails
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks")
    
    return chunks
#next I am creating the vectore databse from document chunks 
def create_vector_store(chunks):
    """    
    Args:
        chunks: List of document chunks
        
    Returns:
        Chroma vector store
    """
    print("⏳ This will take 1-2 minutes and cost ~$0.10-0.20")
    
    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # chpice of embedder
    )
    #sends each chunk to OpenAI API to convert to embeddings (vectors)
    # Create vector database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings, #using the embedding choice
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"Vector database created and saved to {CHROMA_DB_DIR}")
    
    return vectorstore

#loading the vector database that we just created so that we dont have to pay everytime we need
#it and then create it 
def load_vector_store():
    
    print(f"Loading existing vector database from {CHROMA_DB_DIR}")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    print("✅ Vector database loaded")
    
    return vectorstore
def search_directive(query, vectorstore=None, k=3):
    """
    Searches the EU directive for relevant content
    
    Args:
        query: Search query string
        vectorstore: Vector store (will load if not provided)
        k: Number of results to return
        
    Returns:
        List of relevant document chunks
    """
    # Load vector store if not provided
    if vectorstore is None:
        vectorstore = load_vector_store()
    
    # Search
    results = vectorstore.similarity_search(query, k=k)
    
    return results
def setup_rag():
    """
    Main setup function - creates vector database if it doesn't exist
    
    Returns:
        Chroma vector store
    """
    # Check if vector database already exists
    if os.path.exists(CHROMA_DB_DIR):
        print("✅ Vector database already exists, loading it...")
        return load_vector_store()
    
    # Create new vector database
    print("🆕 Vector database not found, creating new one...")
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            f"Please make sure EU_2023_Dir.pdf is in the same folder as this script"
        )
    
    # Load and chunk PDF
    chunks = load_and_chunk_pdf(PDF_PATH)
    
    # Create vector database
    vectorstore = create_vector_store(chunks)
    
    return vectorstore
# Test code - runs when you execute this file directly
if __name__ == "__main__":
    print("🚀 Testing RAG System\n")
    
    # Setup (create or load vector database)
    vectorstore = setup_rag()
    
    print("\n" + "="*50)
    print("Testing searches:")
    print("="*50 + "\n")
    
    # Test searches
    test_queries = [
        "What does the directive say about vague environmental claims?",
        "What are the requirements for substantiation of claims?",
        "What penalties exist for greenwashing?"
    ]
    
    for query in test_queries:
        print(f"🔍 Query: {query}")
        results = search_directive(query, vectorstore)
        
        print(f"📋 Found {len(results)} relevant chunks:\n")
        
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"  Content preview: {doc.page_content[:200]}...")
            print()
        
        print("-" * 50 + "\n")
    
    print("✅ RAG system test complete!")
    #python rag.py