from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import (
    JsonOutputParser,
    CommaSeparatedListOutputParser,
)
from pydantic import BaseModel, Field

load_dotenv()

# -- CHAT MODEL --
llm = ChatGroq(model="llama-3.1-8b-instant")


# -- PROMPT TEMPLATES --

# String Prompt Templates

# prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")

# # create a dictionary to store the correspoinding input to placeholders in prompt
# input = {"adjective": "funny", "topic": "cats"}

# prompt.invoke(input)

# Chat Prompt Templates

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant"),
#         ("user", "Tell me a joke about {topic}"),
#     ]
# )

# input = {"topic": "cats"}

# prompt.invoke(input)

# MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant"),
#         MessagesPlaceholder("msg"),  # This will be replaced with one or more messages
#     ]
# )

# input = {"msg": [HumanMessage(content="What is the day after Tuesday?")]}
# # input = {"msg": [("user", "What is the day after Tuesday?")]}

# prompt.invoke(input)

# You can wrap the prompt and the chat model and pass them into a chain, which acan invoke the message.
# chain = prompt | llm
# response = chain.invoke(input=input)
# print(response)

# -- OUTPUT PARSERS --


# JSON parser
# Define your desired data structure
# class Joke(BaseModel):
#     setup: str = Field(description="question to set up a joke")
#     punchline: str = Field(description="answer to resolve the joke")


# output_parser = JsonOutputParser(pydantic_object=Joke)

# format_instructions = output_parser.get_format_instructions()

# prompt = PromptTemplate(
#     template="Answer the user query.\n{format_instructions}\n{query}\n",
#     input_variables=["query"],
#     partial_variables={"format_instructions": format_instructions},
# )

# chain = prompt | llm | output_parser

# joke_query = "Tell me a joke."
# chain.invoke({"query": joke_query})

# Comma-separated list parser

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm | output_parser

chain.invoke({"subject": "ice cream flavours"})

## -- EXERCISE --

json_parser = JsonOutputParser()

format_instructions = """
    RESPONSE FORMAT: Return ONLY a single JSON object, no markdown, no examples, no extra keys. It must look exactly like:
    {
        "title": "movie title",
        "director": "director name",
        "year": 2000,
        "genre": "movie genre"
    }

    IMPORTANT: Your response must be *only* that JSON. Do NOT include any illustrative or example JSON.
"""

prompt = PromptTemplate(
    template="""You are a JSON-only assistant.

        Task: Generate info about the movie "{movie_name}" in JSON format.

        {format_instructions}
    """,
    input_variables=["movie_name"],
    partial_variables={"format_instructions": format_instructions},
)

movie_chain = prompt | llm | json_parser

result = movie_chain.invoke({"movie_name": "The matrix"})

print(result)

print(result["title"])
print(f"Director: {result['director']}")
