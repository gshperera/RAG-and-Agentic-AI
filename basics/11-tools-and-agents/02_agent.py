from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
import math

llm = ChatGroq(model="llama-3.1-8b-instant")

python_repl = PythonREPL()

python_calculator = Tool(
    name="Python Calculator",
    func=python_repl.run,
    description="Useful for when you need to perform calculations or execute Python code. Input should be valid Python code.",
)


@tool
def search_weather(location: str):
    """Search for the current weather in the specified location."""
    # In a real application, this would call a weather API
    return f"The weather in {location} is currently sunny and 72°F."


# -- Toolkits --
tools = [python_calculator, search_weather]

# --Agent Creating--

# Create the ReAct agent prompt template
# The ReAct prompt needs to instruct the model to follow the thought-action-observation pattern
prompt_template = """You are an agent who has access to the following tools:
{tools}

The available tools are: {tool_names}

To use a tool, please use the following format:
```
Thought: I need to figure out what to do
Action: tool_name
Action Input: the input to the tool
```

After you use a tool, the observation will be provided to you:
```
Observation: result of the tool
```

Then you should continue with the thought-action-observation cycle until you have enough information to respond to the user's request directly.
When you have the final answer, respond in this format:
```
Thought: I know the answer
Final Answer: the final answer to the original query
```

Remember, when using the Python Claculator tool, the input must be valid Python code.

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Create the agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

result = agent_executor.invoke({"input": "What is the square root of 256?"})
print(result["output"])
print(result)

queries = [
    "What's 345 * 789?",
    "Calculate the square root of 144",
    "What's the weather in Miami?",
    "If it's sunny in Chicago, what would be a good outdoor activity?",
    "Generate a list of prime numbers below 50 and calculate their sum",
]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"QUERY: {query}")
    print(f"{'=' * 60}")

    result = agent_executor.invoke({"input": query})

    print(f"\nFINAL ANSWER: {result['output']}")
