import asyncio
from pathlib import Path

from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerStdio, MCPServerStdioParams

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

SANDBOX = os.path.dirname(os.path.abspath(__file__))

SCRIPT = Path(__file__).with_name(
    "01_research_tools_mcp_server.py").resolve()

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
    # print(f"Time travel back by {years} years")
    return f"Current year in time: {year - years}"

@function_tool
def travel_forward(year: int, years: int) -> str:
    """Travel forward in time by a given number of years from the start year."""
    # print(f"Time travel forward by {years} years")
    return f"Current year in time: {year + years}"

thinking_srv = MCPServerStdio(    
    name="sequential-thinking",
    params={
        "command": "npx",
        "args": [
            "-y", 
            "@modelcontextprotocol/server-sequential-thinking"
            ],
    },
)

instructions = """
You are helpful planning assistant.
use the SequentialThinking tool for every reasoning step before choosing a time travel tool.
"""
agent = Agent(
    name="Assistant",
    instructions=instructions,
    tools=[travel_back, travel_forward],
    mcp_servers=[thinking_srv],
    model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        ),    
)

async def main():
    async with thinking_srv:
        tools = await thinking_srv.list_tools()
        print("Available tools:", tools)    
        goal = """
    Discover and output the tool and functions you have available.
    """
        print("Running...", goal)
        result = await Runner.run(agent, goal, max_turns=25,)
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())