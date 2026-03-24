from langchain_core.documents import Document

# Create a Document instance with:
# 1. page_content: The actual text content about Python
# 2. metadata: A dictionary containing additional information about this document
Document(
    page_content="""Python is an interpreted high-level general-purpose programming language that lets you work quickly
and integrate systems more effectively.""",
    metadata={"doc_id": 123456, "doc_source": "About Python"},
)

# Document object without metadata
Document(
    page_content="""Python is an interpreted high-level general-purpose programming language that lets you work quickly
and integrate systems more effectively."""
)
