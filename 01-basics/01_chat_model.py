from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# -- CHAT MODEL --
llm = ChatGroq(model="llama-3.1-8b-instant")

print(llm.invoke("Who is man's best friend?"))
