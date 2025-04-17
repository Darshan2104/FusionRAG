import os
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()  
groq_api_key = os.environ.get("GROQ_API_KEY")
llm = Groq(api_key=groq_api_key)


embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents (In practice, load your product datasheets or knowledge base)
documents = [
    "The IM72D128 microphone has an IP57 rating, providing dust and water resistance.",
    "MEMS microphones convert sound waves into electrical signals using a diaphragm.",
    "The 100V Linear FET offers ultra-low RDS(on) and high efficiency for power applications.",
    "IM73A135V01 is a dust and water resistant analog MEMS microphone with high SNR.",
    "The IM69D130 microphone uses a PDM interface compatible with STM32 microcontrollers."
]

# Embed documents and build FAISS index
doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Using Inner Product for cosine similarity
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

def embed_text(text):
    """Generate normalized embedding for text."""
    emb = embed_model.encode([text], convert_to_numpy=True)[0]
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb

def vector_search(query_emb, top_k=3):
    """Search FAISS index for top_k documents."""
    faiss.normalize_L2(query_emb.reshape(1, -1))
    distances, indices = index.search(query_emb.reshape(1, -1), top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append((idx, dist, documents[idx]))
    return results

def generate_sub_queries(original_query, n=4):
    """Use Groq LLM to generate multiple sub-queries."""
    prompt = (
        f"Generate {n} diverse search queries based on the original query:\n"
        f"\"{original_query}\"\n\nQueries:\n"
    )
    response = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a query generator. Generate diverse search queries."},
            {"role": "user", "content": prompt}
        ],
        model= "llama-3.3-70b-versatile",  # Model specified here
        max_tokens=150,
        temperature=0.7,
        n=1
    )
    text = response.choices[0].message.content.strip()
    # Parse queries assuming line breaks or numbered listsssss
    queries = [line.strip("0123456789.- ") for line in text.split("\n") if line.strip()]
    return queries if queries else [original_query]

def reciprocal_rank_fusion(ranked_lists, k=60):
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for ranked_docs in ranked_lists:
        for rank, (doc_id, score, _) in enumerate(ranked_docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused

def rag_fusion_answer(query):
    # Step 1: Generate sub-queries
    sub_queries = generate_sub_queries(query)

    # Step 2: Retrieve documents for each sub-query
    ranked_lists = []
    for sq in sub_queries:
        emb = embed_text(sq)
        results = vector_search(emb, top_k=5)
        ranked_lists.append(results)

    # Step 3: Fuse rankings with RRF
    fused_docs = reciprocal_rank_fusion(ranked_lists)

    # Step 4: Prepare context from top fused documents
    top_docs_text = "\n\n".join(documents[doc_id] for doc_id, _ in fused_docs[:5])

    # Step 5: Generate final answer using Groq LLM
    final_prompt = (
        f"Use the following context to answer the question:\n{top_docs_text}\n\n"
        f"Question: {query}\nAnswer:"
    )
    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful technical assistant."},
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=300,
        temperature=0.5,
        n=1
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    user_query = "What is the IP rating and durability of the IM72D128 microphone?"
    answer = rag_fusion_answer(user_query)
    print("Final Answer:\n", answer)
