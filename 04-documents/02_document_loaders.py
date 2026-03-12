from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# --PDF Loader--
loader = PyPDFLoader(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
)

# Call the load() method to:
# 1. Download the PDF if needed
# 2. Extract text from each page
# 3. Create a list of Document objects, one for each page of the PDF
# Each Document will contain the text content of a page and metadata including page number
document = loader.load()

document[2].page_content


# --URL and Website Loader--
web_loader = WebBaseLoader(
    "https://docs.langchain.com/oss/python/integrations/document_loaders"
)

# Call the load() method to:
# 1. Send an HTTP request to the specified URL
# 2. Download the HTML content
# 3. Parse the HTML to extract meaningful text
# 4. Create a list of Document objects containing the extracted content
web_data = web_loader.load()
