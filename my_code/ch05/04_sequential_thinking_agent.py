import asyncio
from pathlib import Path

from agents import Agent, Runner
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
model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'

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
"""
agent = Agent(
    name="Assistant",
    instructions=instructions,
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
        result = await Runner.run(agent, goal)
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())