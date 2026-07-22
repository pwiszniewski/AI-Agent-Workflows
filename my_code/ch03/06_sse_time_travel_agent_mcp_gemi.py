from agents import Agent, Runner
from agents.mcp import MCPServerSse, MCPServerSseParams
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

from pathlib import Path
SCRIPT = Path(__file__).with_name(
    "01_claude_mcp_server.py").resolve()

# Simulate a series of historical travel events
travel_events = [
    "Traveled to Ancient Rome and watched a gladiator fight",
    "Visited the signing of the Declaration of Independence in 1776",
    "Witnessed the moon landing in 1969",
]

async def main():
    async with MCPServerSse(
        name="Time Tracker Server",
        params={
            "url": "http://localhost:8000/sse",
        },
    ) as time_tracker_server:
        agent = Agent(
            name="Assistant",
            instructions="""
            You are a time-travel journaling agent.
            Always use the 'load_journal' tool at the start to get past entries.
            For a new event, call 'record_event' to save it.
            If asked for a summary or to show the journal, output all recorded events.
            """,
            mcp_servers=[time_tracker_server],
            model=OpenAIChatCompletionsModel(
                    # model="gemini-3.5-flash",
                    model="gemini-3.1-flash-lite",
                    openai_client=client,
                ),
        )
        print("Recording travels")
        for event in travel_events:
            await Runner.run(starting_agent=agent, input=event)
        result = await Runner.run(
            starting_agent=agent, 
            input="Show my travel history")
        print(result.final_output)


if __name__ == "__main__":    
    asyncio.run(main())
