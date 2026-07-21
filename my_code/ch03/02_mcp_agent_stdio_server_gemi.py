
import asyncio
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

SCRIPT = Path(__file__).with_name(
    "01_claude_mcp_server.py").resolve()

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


async def main():
    async with MCPServerStdio(
        name="Research Tools",
        params=MCPServerStdioParams(
            command="mcp",
            args=["run", str(SCRIPT)],
        ),
    ) as research_server:
        agent = Agent(
            name="Assistant",
            instructions="Use the research tools to perform research.",
            mcp_servers=[research_server],
            model=OpenAIChatCompletionsModel(
                    model="gemini-3.5-flash",
                    openai_client=client,
                ),

        )

        print("Running: Get the available research sources")
        result = await Runner.run(
            starting_agent=agent, 
            input="Get the available research sources")
        print(result.final_output)

if __name__ == "__main__":    
    asyncio.run(main())

# OUTPUT
# Running: Get the available research sources
# The available research sources are:

# 1. Wikipedia
# 2. Google
# 3. YouTube