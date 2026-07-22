from agents import Agent, Runner
from agents import function_tool
import asyncio

import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


_journal = []

@function_tool
def record_event(entry: str) -> dict:
    """Add a new travel event to the journal."""
    _journal.append(entry)
    print(f"Event recorded: {entry}")
    return {"status": "recorded", "entry": entry}

@function_tool
def load_journal() -> dict:
    """Load the current travel journal entries."""
    print("Loading journal entries...")
    return {"status": "loaded", "journal": "\n".join(_journal)}

agent = Agent(
    name="Time Tracker Agent",
    instructions="""You are a time tracking journaling agent.
Always use the 'load_journal' tool at the start to get past entries.
For a new event, call 'record_event' to save it.
If asked for a summary or to show the journal, output all recorded events.""",
    tools=[record_event, load_journal],
    model=OpenAIChatCompletionsModel(
                    model="gemini-3.5-flash",
                    openai_client=client,
                ),
)

# Simulate a series of historical travel events
travel_events = [
    "Traveled to Ancient Rome and watched a gladiator fight",
    "Visited the signing of the Declaration of Independence in 1776",
    "Witnessed the moon landing in 1969",
]

async def main():
    print("Recording travels:")
    for event in travel_events:
        await Runner.run(starting_agent=agent, input=event)
    # Ask the agent to summarize the adventures
    result = await Runner.run(agent, "Show my travel history")
    print("\nFinal Journal:")
    print(result.final_output)

asyncio.run(main())

# Recording travels:
# Loading journal entries...
# Event recorded: Traveled to Ancient Rome and watched a gladiator fight
# Loading journal entries...
# Event recorded: Visited the signing of the Declaration of Independence in 1776
# Loading journal entries...
# Event recorded: Witnessed the moon landing in 1969
# Loading journal entries...

# Final Journal:
# Here is your travel history:

# * Traveled to Ancient Rome and watched a gladiator fight
# * Visited the signing of the Declaration of Independence in 1776
# * Witnessed the moon landing in 1969