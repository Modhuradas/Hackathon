from rag import load_vector_store, search_directive

vectorstore = load_vector_store()

# Test different searches
test_queries = [
    "Article 7 future environmental performance commitments",
    "Article 6 comparative claims between products",
    "Article 10 environmental labelling certification schemes",
    "Article 5 generic environmental statements vague claims",
    "Article 12 verification conformity assessment"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    results = search_directive(query, vectorstore, k=3)
    
    for i, doc in enumerate(results, 1):
        article = doc.metadata.get('article', 'No article metadata')
        print(f"\nResult {i}: {article}")
        print(f"Preview: {doc.page_content[:200]}...")