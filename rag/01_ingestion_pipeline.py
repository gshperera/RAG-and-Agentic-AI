import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from /{docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your files."
        )

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. please add files."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocuments {i + 1}:")
        print(f"\tSource: {doc.metadata['source']}")
        print(f"\tContent Length: {len(doc.page_content)} characters")
        print(f"\tMetadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk 0{i + 1} ---")
        print(f"\tSource: {chunk.metadata['source']}")
        print(f"\tLength: {len(chunk.page_content)} characters")
        print("Content: ")
        print(chunk.page_content)
        print("_" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # If DB already exists -> Load
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print(f"Loaded {vector_store._collection.count()} documents")
        return vector_store

    print("Creating embeddings and storing in ChromaDB...")
    # Create ChromaDB vector store
    print("--- Creating vecor store ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vector_store


def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")

    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # Step 01: Load documents
    documents = load_documents(docs_path)

    # Step 02: Split documents into chunks
    chunks = split_documents(documents)

    # Step 03: Create or Load vector store
    vector_store = create_vector_store(chunks, persistent_directory)

    print("\n Ingestion complete!✅ Documents are ready")

    return vector_store


if __name__ == "__main__":
    main()
