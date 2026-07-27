
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
model_name = 'gemini-3.1-flash-lite'
# model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'


from pydantic import BaseModel
from typing import List
class ResearchSourcesModel(BaseModel):
        research_sources: List[str]
        """A list of research sources to use for research."""

servers = [
    MCPServerStdio(
        name="Research Tools",
        params=MCPServerStdioParams(
            command="mcp",
            args=["run", str(SCRIPT)],
        ),
    ),
    MCPServerStdio(
        name="sequential-thinking",
        params={
            "command": "npx",
            "args": [
                "-y", 
                "@modelcontextprotocol/server-sequential-thinking"
            ],
        },
    ),
    MCPServerStdio(
        name="filesystem",
        params={
            "command": "npx",
            "args": [
                "-y", 
                "@modelcontextprotocol/server-filesystem@latest", SANDBOX
            ],
        },
    ),
]

research_agent = Agent(
    name="Research Agent",
    instructions="""
        You are a research assistant.
        Your role is to find research sources.
        Always hand off to planning agent
        """,
    model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        ),
    output_type=ResearchSourcesModel
)
thinking_agent = Agent(
    name="Thinking Agent",
    instructions="""
        You are a research assistant.
        Your role is to plan the research.
        Always hand off to filesystem agent
        """,
    model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        ),
)

filesystem_agent = Agent(
    name="Filesystem Agent",
    instructions="""
    You are a filesystem assistant.
    Your role is to write the output as a text file to 'out_exercise3.txt'
    Never make up or invent any ouput.
    """,
    model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=client,
            ),
)

async def main():
    async with (
        servers[0] as research_srv,
        servers[1] as thinking_srv,
        servers[2] as fs_srv,
    ):

        research_agent.mcp_servers = [research_srv]
        research_agent.handoffs = [thinking_agent]
        thinking_agent.mcp_servers = [thinking_srv]
        thinking_agent.handoffs = [filesystem_agent]
        filesystem_agent.mcp_servers = [fs_srv]

        goal = """
            Produce a research plan to find the book:
            'The Hitchhiker's Guide to the Galaxy'
            """
        print("Running...", goal)

        
        print("Running...", goal)
        result = await Runner.run(
            research_agent,
            goal, 
            max_turns=25,
            )
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())