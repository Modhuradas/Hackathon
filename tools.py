
"""
Tools for AI Agents
This module defines tools that agents can use to interact with the RAG system
"""

from langchain.tools import tool
from rag import search_directive, load_vector_store, setup_rag
import os

# Global variable for lazy loading
_vectorstore = None

def get_vectorstore():
    """
    Lazy load the vector store
    Only creates/loads when first needed
    """
    global _vectorstore
    
    if _vectorstore is None:
        print("📂 Initializing vector database...")
        
        # Check if database exists
        if os.path.exists("./chroma_db"):
            try:
                print("✅ Loading existing vector database...")
                _vectorstore = load_vector_store()
                print("✅ Vector database loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading existing DB: {e}")
                print("🆕 Creating new vector database...")
                _vectorstore = setup_rag()
        else:
            print("🆕 Vector database not found, creating new one...")
            _vectorstore = setup_rag()
    
    return _vectorstore

@tool
def search_eu_directive(query: str) -> str:
    """
    Search the EU Green Claims Directive for relevant information.
    
    Use this tool when you need to find information about:
    - Environmental regulations
    - Greenwashing rules
    - Claim substantiation requirements
    - Specific articles or sections
    - Examples of violations
    
    Args:
        query: A search query describing what information you need.
               Examples: "vague environmental claims", "substantiation requirements"
    
    Returns:
        Relevant text chunks from the EU directive
    """
    
    # Get vector store (lazy load)
    vectorstore = get_vectorstore()
    
    # Search the directive
    results = search_directive(query, vectorstore, k=5)
    
    # Format results into readable text
    formatted_results = []
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get('page', 'Unknown')
        article = doc.metadata.get('article', 'Unknown Article')
        content = doc.page_content
        formatted_results.append(
            f"[Result {i} - {article} - Page {page}]\n{content}\n"
        )
    
    # Join all results with separators
    return "\n" + "="*50 + "\n".join(formatted_results)