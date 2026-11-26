from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


# Examples usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    for r in rag_search.vectorstore.query(query, top_k=3):
        meta = r["metadata"]
        print(meta["source_file"], meta["page"], round(float(r["distance"]), 4))
    
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary: ", summary)

