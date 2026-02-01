"""
Tools for AI Agents
This module defines tools that agents can use to interact with the RAG system
"""
from langchain.tools import tool
from rag import search_directive, load_vector_store

# Load or create vector store (for cloud deployment)
import os

if os.path.exists("./chroma_db"):
    vectorstore = load_vector_store()
else:
    print("🆕 Creating vector database for first time...")
    vectorstore = setup_rag()
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
    
    # Search the directive
    results = search_directive(query, vectorstore, k=5)
    
    # Format results into readable text
    formatted_results = []
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get('page', 'Unknown')
        content = doc.page_content
        formatted_results.append(
            f"[Result {i} - Page {page}]\n{content}\n"
        )
    
    # Join all results with separators
    return "\n" + "="*50 + "\n".join(formatted_results)