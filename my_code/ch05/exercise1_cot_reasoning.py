import asyncio

from agents import Agent, Runner

import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'

# Define an agent that always explains its reasoning step by step
cot_agent = Agent(
    name="TimeTravelerCoT",
    instructions=(
        "You are a time travel problem solver. "
        "Work out the solution step by step, then give the final answer." 
        "Number each line of thought before the final answer."
        ""
    ),
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
result = asyncio.run(Runner.run(cot_agent, input=question))
print(result.final_output)
