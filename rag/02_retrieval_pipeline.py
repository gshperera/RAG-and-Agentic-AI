from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "How much did Microseft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k":5})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")