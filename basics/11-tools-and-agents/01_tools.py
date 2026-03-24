from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL

# The @tool decorator is a convenient way to define tools, but you can also use the Tool class directly

# Ths provides an environment where Python code can be executed as strings
python_repl = PythonREPL()

# Create a Tool using the Tool class
# This wraps the Python REPL functionality as a tool that can be used by agents
python_calculator = Tool(
    # The name of the tool - this helps agents identify when to use this tool
    name="Python Calculator",
    # The function that will be called when the tool is used
    # python_repl.run takes a string of Python code and executes it
    func=python_repl.run,
    # A description of what the tool does and how to use it
    # This helps the agent understand when and how to use this tool
    description="Useful for when you need to perform calculations or execute Python code. Input should be valid Python code.",
)

result = python_calculator.invoke("a = 3; b = 1; print(a+b)")

# We can also creat custome tools using the @tool decorator


@tool
def search_weather(location: str):
    """Search for the current weather in the specified location."""
    # In a real application, this would call a weather API
    return f"The weather in {location} is currently sunny and 72°F."


# -- Toolkits --
# Toolkits are collections of tools that are designed to be used together for specific tasks.
tools = [python_calculator, search_weather]
