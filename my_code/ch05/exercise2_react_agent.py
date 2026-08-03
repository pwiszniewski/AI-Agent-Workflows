import asyncio

from agents import Agent, Runner, function_tool

import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# model_name = 'gemini-3-flash-preview'
model_name = 'gemini-3.6-flash'

@function_tool
def travel_back(year: int, years: int) -> str:
    """
    Travel back in time by a given number of years from the start year.
    """
    print(f"Time travel back by {years} years")
    return f"Current year in time: {year - years}"

@function_tool
def travel_forward(year: int, years: int) -> str:
    """Travel forward in time by a given number of years from the start year."""
    print(f"Time travel forward by {years} years")
    return f"Current year in time: {year + years}"

# Define an agent that always explains its reasoning step by step
cot_agent = Agent(
    name="TimeTravelerCoT",
    instructions=(
        """
        You are a time travel problem solver.
        Work out the solution step by step, then give the final answer.
        Number each line of thought before the final answer.
        You must use the tools to calculate dates. 
        After using a tool, reflect on the result and continue reasoning. 
        After gathering information, provide the final answer.
        """
    ),
    tools=[travel_back, travel_forward],
    model=OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    ),  
)

# Example time travel question
question = (
    "Starting in 2025, you travel 10 years to the past, then 5 years to the future. "
    "What year do you end up in?"
)

# Run the agent (using await in an async context, or Runner.run_sync in a script)
result = asyncio.run(Runner.run(cot_agent, input=question, max_turns=25,))
print(result.final_output)
